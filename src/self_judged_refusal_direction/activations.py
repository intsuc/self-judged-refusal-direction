from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from torch import nn

from self_judged_refusal_direction.errors import ArtifactError, InvariantError
from self_judged_refusal_direction.hashing import canonical_json_bytes, object_sha256
from self_judged_refusal_direction.schema import ActivationKey, JudgeLabel

ClassLabel = Literal["REFUSAL", "NON_REFUSAL"]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def accumulator_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value not in {torch.float32, torch.float64}:
            raise InvariantError("activation accumulator dtype must be float32 or float64")
        return value
    try:
        return _DTYPES[value.removeprefix("torch.")]
    except KeyError as error:
        raise InvariantError("activation accumulator dtype must be float32 or float64") from error


@dataclass(frozen=True)
class ActivationMoments:
    count: int
    mean: torch.Tensor
    m2: torch.Tensor

    def __post_init__(self) -> None:
        if self.count < 0:
            raise InvariantError("activation count must be non-negative")
        if self.mean.device.type != "cpu" or self.m2.device.type != "cpu":
            raise InvariantError("activation moments must reside on CPU")
        if self.mean.shape != self.m2.shape or self.mean.ndim != 1:
            raise InvariantError("activation moment tensors must be one-dimensional and shape-aligned")

    @property
    def variance(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(self.m2)
        return self.m2 / self.count

    @property
    def sample_variance(self) -> torch.Tensor:
        if self.count < 2:
            return torch.zeros_like(self.m2)
        return self.m2 / (self.count - 1)


class OnlineWelford:
    def __init__(self, *, dtype: str | torch.dtype = torch.float64, dimension: int | None = None):
        self.dtype = accumulator_dtype(dtype)
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=self.dtype, device="cpu") if dimension is not None else None
        self.m2 = torch.zeros(dimension, dtype=self.dtype, device="cpu") if dimension is not None else None

    def update(self, values: torch.Tensor) -> None:
        batch = values.detach().to(device="cpu", dtype=self.dtype)
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        if batch.ndim != 2:
            raise InvariantError("Welford updates must have shape [samples, hidden_size]")
        if batch.shape[0] == 0:
            return
        if not torch.isfinite(batch).all():
            raise InvariantError("activation batch contains NaN or Inf")
        if self.mean is None or self.m2 is None:
            self.mean = torch.zeros(batch.shape[1], dtype=self.dtype, device="cpu")
            self.m2 = torch.zeros_like(self.mean)
        if batch.shape[1:] != self.mean.shape:
            raise InvariantError("activation hidden size changed during collection")
        batch_count = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        centered = batch - batch_mean
        batch_m2 = (centered * centered).sum(dim=0)
        self.merge(count=batch_count, mean=batch_mean, m2=batch_m2)

    def merge(self, *, count: int, mean: torch.Tensor, m2: torch.Tensor) -> None:
        if count < 0:
            raise InvariantError("merged activation count must be non-negative")
        if count == 0:
            return
        incoming_mean = mean.detach().to(device="cpu", dtype=self.dtype)
        incoming_m2 = m2.detach().to(device="cpu", dtype=self.dtype)
        if incoming_mean.ndim != 1 or incoming_mean.shape != incoming_m2.shape:
            raise InvariantError("merged activation moments are shape-incompatible")
        if not torch.isfinite(incoming_mean).all() or not torch.isfinite(incoming_m2).all():
            raise InvariantError("merged activation moments contain NaN or Inf")
        if self.mean is None or self.m2 is None:
            self.mean = torch.zeros_like(incoming_mean)
            self.m2 = torch.zeros_like(incoming_m2)
        if incoming_mean.shape != self.mean.shape:
            raise InvariantError("merged activation hidden size does not match")
        if self.count == 0:
            self.count = count
            self.mean.copy_(incoming_mean)
            self.m2.copy_(incoming_m2)
            return
        total = self.count + count
        delta = incoming_mean - self.mean
        self.mean.add_(delta * (count / total))
        self.m2.add_(incoming_m2 + delta.square() * (self.count * count / total))
        self.count = total

    def snapshot(self) -> ActivationMoments:
        if self.mean is None or self.m2 is None:
            raise InvariantError("activation accumulator has no observations")
        return ActivationMoments(count=self.count, mean=self.mean.clone(), m2=self.m2.clone())


@dataclass(frozen=True)
class ActivationStatistics:
    refusal: Mapping[ActivationKey, ActivationMoments]
    non_refusal: Mapping[ActivationKey, ActivationMoments]

    def for_label(self, label: ClassLabel | JudgeLabel | str) -> Mapping[ActivationKey, ActivationMoments]:
        value = label.value if isinstance(label, JudgeLabel) else str(label)
        if value == JudgeLabel.REFUSAL.value:
            return self.refusal
        if value == JudgeLabel.NON_REFUSAL.value:
            return self.non_refusal
        raise InvariantError("activation statistics only contain REFUSAL and NON_REFUSAL classes")

    @property
    def keys(self) -> tuple[ActivationKey, ...]:
        return tuple(sorted(set(self.refusal) | set(self.non_refusal), key=lambda key: key.layer))

    def paired(self, key: ActivationKey) -> tuple[ActivationMoments, ActivationMoments]:
        try:
            return self.refusal[key], self.non_refusal[key]
        except KeyError as error:
            raise InvariantError(f"both activation classes are required for {key.storage_key}") from error


@dataclass(frozen=True)
class _CaptureState:
    labels: tuple[str, ...]


class ActivationCollector:
    def __init__(
        self,
        block_modules: Sequence[nn.Module],
        *,
        layers: Literal["all"] | Sequence[int] = "all",
        dtype: str | torch.dtype = torch.float64,
    ):
        blocks = tuple(block_modules)
        selected_layers = (
            tuple(range(len(blocks))) if layers == "all" else tuple(dict.fromkeys(int(layer) for layer in layers))
        )
        if not selected_layers or any(layer < 0 or layer >= len(blocks) for layer in selected_layers):
            raise InvariantError("candidate layer index is outside the transformer block sequence")
        self.block_modules = blocks
        self.layers = selected_layers
        self.dtype = accumulator_dtype(dtype)
        self._moments: dict[str, dict[ActivationKey, OnlineWelford]] = {
            JudgeLabel.REFUSAL.value: {},
            JudgeLabel.NON_REFUSAL.value: {},
        }
        self._capture: _CaptureState | None = None
        self._seen_layers: set[int] = set()

    def reset(self) -> None:
        if self._capture is not None:
            raise InvariantError("cannot reset an active activation collector")
        self._moments = {JudgeLabel.REFUSAL.value: {}, JudgeLabel.NON_REFUSAL.value: {}}

    @contextmanager
    def capture(
        self,
        labels: Sequence[JudgeLabel | str | None],
    ) -> Iterator[None]:
        if self._capture is not None:
            raise InvariantError("activation captures cannot be nested")
        label_values = tuple(
            label.value if isinstance(label, JudgeLabel) else str(label) if label is not None else ""
            for label in labels
        )
        self._capture = _CaptureState(labels=label_values)
        self._seen_layers = set()
        handles = [self.block_modules[layer].register_forward_pre_hook(self._hook(layer)) for layer in self.layers]
        completed = False
        try:
            yield
            completed = True
        finally:
            for handle in handles:
                handle.remove()
            missing = set(self.layers) - self._seen_layers
            self._capture = None
            self._seen_layers = set()
        if completed and missing:
            raise InvariantError(f"activation read points were not reached for layers: {sorted(missing)}")

    def collect(
        self,
        forward: Callable[[], object],
        labels: Sequence[JudgeLabel | str | None],
    ) -> object:
        with self.capture(labels):
            return forward()

    def _hook(self, layer: int):
        def hook(_module: nn.Module, inputs: tuple[object, ...]) -> None:
            if self._capture is None:
                raise InvariantError("activation hook ran without capture state")
            if layer in self._seen_layers:
                raise InvariantError(f"activation layer {layer} ran more than once in one capture")
            self._seen_layers.add(layer)
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise InvariantError("transformer block input does not begin with a tensor")
            activation = inputs[0]
            if activation.ndim != 3:
                raise InvariantError("transformer block activation must have shape [batch, sequence, hidden]")
            if activation.shape[1] < 1:
                raise InvariantError("transformer block activation sequence must be non-empty")
            if activation.shape[0] != len(self._capture.labels):
                raise InvariantError("activation batch size does not match label count")
            accepted = [
                index
                for index, label in enumerate(self._capture.labels)
                if label in {JudgeLabel.REFUSAL.value, JudgeLabel.NON_REFUSAL.value}
            ]
            if not accepted:
                return
            selected = activation[:, -1, :].detach().to(device="cpu", dtype=self.dtype)
            key = ActivationKey(layer=layer)
            for label in (JudgeLabel.REFUSAL.value, JudgeLabel.NON_REFUSAL.value):
                label_rows = [index for index in accepted if self._capture.labels[index] == label]
                if not label_rows:
                    continue
                accumulator = self._moments[label].setdefault(key, OnlineWelford(dtype=self.dtype))
                accumulator.update(selected[label_rows])

        return hook

    def statistics(self) -> ActivationStatistics:
        def snapshots(label: str) -> dict[ActivationKey, ActivationMoments]:
            return {key: accumulator.snapshot() for key, accumulator in self._moments[label].items()}

        return ActivationStatistics(
            refusal=snapshots(JudgeLabel.REFUSAL.value),
            non_refusal=snapshots(JudgeLabel.NON_REFUSAL.value),
        )


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".safetensors")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_activation_statistics(path: str | Path, statistics: ActivationStatistics) -> None:
    tensors: dict[str, torch.Tensor] = {}
    entries: list[dict[str, object]] = []
    for label in (JudgeLabel.REFUSAL.value, JudgeLabel.NON_REFUSAL.value):
        for key, moments in sorted(
            statistics.for_label(label).items(),
            key=lambda item: item[0].layer,
        ):
            prefix = f"{label}/{key.storage_key}"
            tensors[f"{prefix}/mean"] = moments.mean.contiguous()
            tensors[f"{prefix}/m2"] = moments.m2.contiguous()
            entries.append({"label": label, "key": key.storage_key, "count": moments.count})
    payload = {"entries": entries}
    _atomic_safetensors(
        Path(path),
        tensors,
        {
            "activation_statistics": canonical_json_bytes(payload).decode(),
            "content_profile_sha256": object_sha256(payload),
        },
    )


def load_activation_statistics(path: str | Path) -> ActivationStatistics:
    target = Path(path)
    if not target.is_file():
        raise ArtifactError(f"activation statistics do not exist: {target}")
    loaded: dict[str, dict[ActivationKey, ActivationMoments]] = {
        JudgeLabel.REFUSAL.value: {},
        JudgeLabel.NON_REFUSAL.value: {},
    }
    try:
        with safe_open(target, framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
            payload_text = metadata.get("activation_statistics")
            if payload_text is None:
                raise ArtifactError("activation statistics metadata is missing")
            payload = json.loads(payload_text)
            if object_sha256(payload) != metadata.get("content_profile_sha256"):
                raise ArtifactError("activation statistics metadata is invalid")
            for entry in payload.get("entries", []):
                label = entry["label"]
                if label not in loaded:
                    raise ArtifactError("activation statistics contain an unsupported class label")
                key = ActivationKey.parse(entry["key"])
                prefix = f"{label}/{key.storage_key}"
                loaded[label][key] = ActivationMoments(
                    count=int(entry["count"]),
                    mean=stream.get_tensor(f"{prefix}/mean"),
                    m2=stream.get_tensor(f"{prefix}/m2"),
                )
    except ArtifactError:
        raise
    except (KeyError, OSError, SafetensorError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid activation statistics: {target}") from error
    return ActivationStatistics(
        refusal=loaded[JudgeLabel.REFUSAL.value],
        non_refusal=loaded[JudgeLabel.NON_REFUSAL.value],
    )
