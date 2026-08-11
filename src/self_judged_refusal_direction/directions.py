from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from self_judged_refusal_direction.activations import ActivationStatistics
from self_judged_refusal_direction.artifacts import ArtifactMetadata, ArtifactProfile, ArtifactStore
from self_judged_refusal_direction.errors import ArtifactError, InvariantError
from self_judged_refusal_direction.hashing import canonical_json_bytes, file_sha256, object_sha256, tensor_sha256
from self_judged_refusal_direction.schema import ActivationKey, DirectionCandidate

_DIRECTION_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass(frozen=True)
class CandidateBundle:
    directions: torch.Tensor
    candidates: tuple[DirectionCandidate, ...]

    def __post_init__(self) -> None:
        if self.directions.device.type != "cpu" or self.directions.ndim != 2:
            raise InvariantError("candidate directions must be a two-dimensional CPU tensor")
        if self.directions.shape[0] != len(self.candidates):
            raise InvariantError("candidate tensor and metadata counts do not match")
        for index, candidate in enumerate(self.candidates):
            if candidate.direction_index != index:
                raise InvariantError("candidate direction indices must be contiguous and ordered")

    @property
    def metadata(self) -> tuple[DirectionCandidate, ...]:
        return self.candidates

    def direction(self, candidate: DirectionCandidate | str) -> torch.Tensor:
        candidate_id = candidate.candidate_id if isinstance(candidate, DirectionCandidate) else candidate
        for metadata in self.candidates:
            if metadata.candidate_id == candidate_id:
                return self.directions[metadata.direction_index]
        raise KeyError(candidate_id)


def _direction_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value not in set(_DIRECTION_DTYPES.values()):
            raise InvariantError("direction dtype must be float32 or float64")
        return value
    try:
        return _DIRECTION_DTYPES[value.removeprefix("torch.")]
    except KeyError as error:
        raise InvariantError("direction dtype must be float32 or float64") from error


def _boundary_token(boundary_tokens: Mapping[ActivationKey | str, str] | None, key: ActivationKey) -> str:
    if boundary_tokens is None:
        return ""
    if key in boundary_tokens:
        return boundary_tokens[key]
    return boundary_tokens.get(key.storage_key, "")


def build_direction_candidates(
    statistics: ActivationStatistics,
    *,
    boundary_tokens: Mapping[ActivationKey | str, str] | None = None,
    minimum_norm: float = 1e-8,
    dtype: str | torch.dtype = torch.float32,
) -> CandidateBundle:
    if minimum_norm <= 0:
        raise InvariantError("minimum direction norm must be positive")
    output_dtype = _direction_dtype(dtype)
    directions: list[torch.Tensor] = []
    candidates: list[DirectionCandidate] = []
    hidden_size: int | None = None
    for key in statistics.keys:
        refusal, non_refusal = statistics.paired(key)
        if refusal.count < 1 or non_refusal.count < 1:
            raise InvariantError(f"both activation classes require observations for {key.storage_key}")
        if refusal.mean.shape != non_refusal.mean.shape:
            raise InvariantError(f"activation class shapes differ for {key.storage_key}")
        if hidden_size is None:
            hidden_size = refusal.mean.numel()
        elif refusal.mean.numel() != hidden_size:
            raise InvariantError("activation hidden size changed across direction candidates")
        refusal_mean = refusal.mean.to(torch.float64)
        non_refusal_mean = non_refusal.mean.to(torch.float64)
        refusal_variance = refusal.variance.to(torch.float64)
        non_refusal_variance = non_refusal.variance.to(torch.float64)
        raw_direction = refusal_mean - non_refusal_mean
        norm_tensor = torch.linalg.vector_norm(raw_direction)
        numerical_finite = bool(
            torch.isfinite(refusal_mean).all()
            and torch.isfinite(non_refusal_mean).all()
            and torch.isfinite(refusal_variance).all()
            and torch.isfinite(non_refusal_variance).all()
            and torch.isfinite(norm_tensor)
        )
        norm = float(norm_tensor.item()) if numerical_finite else 0.0
        eligible = numerical_finite and norm >= minimum_norm
        unit_direction = raw_direction / norm_tensor if eligible else torch.zeros_like(raw_direction)
        if eligible:
            refusal_projected_mean = float(torch.dot(refusal_mean, unit_direction).item())
            non_refusal_projected_mean = float(torch.dot(non_refusal_mean, unit_direction).item())
            refusal_projected_variance = float(torch.dot(unit_direction.square(), refusal_variance.clamp_min(0)).item())
            non_refusal_projected_variance = float(
                torch.dot(unit_direction.square(), non_refusal_variance.clamp_min(0)).item()
            )
            pooled_variance = max(
                0.5 * (refusal_projected_variance + non_refusal_projected_variance),
                torch.finfo(torch.float64).eps,
            )
            standardized_separation = abs(refusal_projected_mean - non_refusal_projected_mean) / pooled_variance**0.5
        else:
            refusal_projected_mean = 0.0
            non_refusal_projected_mean = 0.0
            refusal_projected_variance = 0.0
            non_refusal_projected_variance = 0.0
            standardized_separation = 0.0
        stored_direction = unit_direction.to(dtype=output_dtype, device="cpu").contiguous()
        candidate_id = object_sha256(
            {
                "activation_key": key.storage_key,
                "direction_sha256": tensor_sha256(stored_direction),
                "refusal_count": refusal.count,
                "non_refusal_count": non_refusal.count,
            }
        )
        index = len(directions)
        directions.append(stored_direction)
        candidates.append(
            DirectionCandidate(
                candidate_id=candidate_id,
                phase=key.phase,
                layer=key.layer,
                relative_position=key.relative_position,
                direction_index=index,
                norm=norm,
                refusal_count=refusal.count,
                non_refusal_count=non_refusal.count,
                standardized_separation=standardized_separation,
                refusal_projected_mean=refusal_projected_mean,
                non_refusal_projected_mean=non_refusal_projected_mean,
                refusal_projected_variance_diagonal=refusal_projected_variance,
                non_refusal_projected_variance_diagonal=non_refusal_projected_variance,
                boundary_token=_boundary_token(boundary_tokens, key),
                finite=numerical_finite,
            )
        )
    if directions:
        tensor = torch.stack(directions)
    else:
        tensor = torch.empty((0, hidden_size or 0), dtype=output_dtype, device="cpu")
    return CandidateBundle(directions=tensor, candidates=tuple(candidates))


def build_candidates(
    statistics: ActivationStatistics,
    *,
    boundary_tokens: Mapping[ActivationKey | str, str] | None = None,
    minimum_norm: float = 1e-8,
    dtype: str | torch.dtype = torch.float32,
) -> CandidateBundle:
    return build_direction_candidates(
        statistics,
        boundary_tokens=boundary_tokens,
        minimum_norm=minimum_norm,
        dtype=dtype,
    )


def rank_stage_a(
    candidates: CandidateBundle | Sequence[DirectionCandidate],
    *,
    top_m: int,
    minimum_norm: float = 1e-8,
) -> tuple[DirectionCandidate, ...]:
    if top_m < 1:
        raise InvariantError("Stage A top_m must be positive")
    if minimum_norm <= 0:
        raise InvariantError("minimum direction norm must be positive")
    metadata = candidates.candidates if isinstance(candidates, CandidateBundle) else tuple(candidates)
    eligible = [
        candidate
        for candidate in metadata
        if candidate.finite
        and candidate.norm >= minimum_norm
        and candidate.refusal_count > 0
        and candidate.non_refusal_count > 0
    ]
    eligible.sort(
        key=lambda candidate: (
            -candidate.standardized_separation,
            -candidate.norm,
            candidate.phase,
            candidate.layer,
            candidate.relative_position,
            candidate.candidate_id,
        )
    )
    return tuple(eligible[:top_m])


def _atomic_safetensors(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    *,
    private: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".safetensors")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.chmod(temporary, 0o600 if private else 0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_candidates(path: str | Path, bundle: CandidateBundle, *, private: bool = False) -> None:
    candidate_values = [asdict(candidate) for candidate in bundle.candidates]
    payload = {"schema_version": 1, "candidates": candidate_values}
    _atomic_safetensors(
        Path(path),
        {"directions": bundle.directions.contiguous()},
        {
            "candidate_bundle": canonical_json_bytes(payload).decode(),
            "candidate_bundle_sha256": object_sha256(payload),
        },
        private=private,
    )


def load_candidates(path: str | Path) -> CandidateBundle:
    target = Path(path)
    if not target.is_file():
        raise ArtifactError(f"candidate directions do not exist: {target}")
    try:
        with safe_open(target, framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
            payload_text = metadata.get("candidate_bundle")
            if payload_text is None:
                raise ArtifactError("candidate bundle metadata is missing")
            payload = json.loads(payload_text)
            if payload.get("schema_version") != 1 or object_sha256(payload) != metadata.get("candidate_bundle_sha256"):
                raise ArtifactError("candidate bundle metadata is invalid")
            directions = stream.get_tensor("directions")
        candidates = tuple(DirectionCandidate(**value) for value in payload.get("candidates", []))
        bundle = CandidateBundle(directions=directions, candidates=candidates)
    except ArtifactError:
        raise
    except (KeyError, OSError, SafetensorError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid candidate directions: {target}") from error
    if not torch.isfinite(bundle.directions).all():
        raise ArtifactError("candidate direction tensor contains NaN or Inf")
    return bundle


def save_direction(
    path: str | Path,
    direction: torch.Tensor,
    *,
    metadata: Mapping[str, Any] | None = None,
    private: bool = False,
) -> None:
    value = direction.detach().to(device="cpu").contiguous()
    if value.ndim != 1 or not torch.isfinite(value).all():
        raise InvariantError("saved direction must be a finite one-dimensional tensor")
    payload = dict(metadata or {})
    _atomic_safetensors(
        Path(path),
        {"direction": value},
        {
            "direction_metadata": canonical_json_bytes(payload).decode(),
            "direction_sha256": tensor_sha256(value),
        },
        private=private,
    )


def load_direction(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise ArtifactError(f"direction does not exist: {target}")
    try:
        with safe_open(target, framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
            direction = stream.get_tensor("direction")
            payload = json.loads(metadata.get("direction_metadata", "{}"))
            if not isinstance(payload, dict):
                raise ArtifactError("direction metadata must be an object")
            if metadata.get("direction_sha256") != tensor_sha256(direction):
                raise ArtifactError("direction tensor hash mismatch")
    except ArtifactError:
        raise
    except (KeyError, OSError, SafetensorError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid direction artifact: {target}") from error
    if direction.ndim != 1 or not torch.isfinite(direction).all():
        raise ArtifactError("direction tensor is not finite and one-dimensional")
    return direction, payload


def write_candidate_artifacts(
    store: ArtifactStore,
    bundle: CandidateBundle,
    *,
    profile: ArtifactProfile,
) -> None:
    save_candidates(store.paths.candidates, bundle)
    metadata = ArtifactMetadata(
        schema_version=1,
        artifact_type="direction_candidates",
        private=False,
        record_count=len(bundle.candidates),
        content_sha256=file_sha256(store.paths.candidates),
        profile=profile,
    )
    store.write_json(store.metadata_path(store.paths.candidates), asdict(metadata), private=False)
    store.write_jsonl(
        store.paths.candidate_metadata,
        bundle.candidates,
        artifact_type="direction_candidate_metadata",
        profile=profile,
        private=False,
    )


def load_candidate_artifacts(
    store: ArtifactStore,
    *,
    expected_profile: ArtifactProfile,
) -> CandidateBundle:
    binary_metadata = store.validate(
        store.paths.candidates,
        artifact_type="direction_candidates",
        expected_profile=expected_profile,
    )
    bundle = load_candidates(store.paths.candidates)
    rows = tuple(
        store.read_jsonl(
            store.paths.candidate_metadata,
            artifact_type="direction_candidate_metadata",
            expected_profile=expected_profile,
        )
    )
    sidecar_candidates = tuple(DirectionCandidate(**row) for row in rows)
    if binary_metadata.record_count != len(bundle.candidates) or sidecar_candidates != bundle.candidates:
        raise ArtifactError("candidate tensor and metadata artifacts do not match")
    return bundle


def write_stage_a_ranking(
    store: ArtifactStore,
    ranking: Sequence[DirectionCandidate],
) -> None:
    store.write_json(
        store.paths.stage_a_ranking,
        [
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "standardized_separation": candidate.standardized_separation,
                "norm": candidate.norm,
            }
            for rank, candidate in enumerate(ranking, start=1)
        ],
        private=False,
    )
