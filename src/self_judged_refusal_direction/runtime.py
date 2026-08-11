from __future__ import annotations

import gc
from contextlib import AbstractContextManager
from typing import Any, Self

import torch
from torch import nn

from self_judged_refusal_direction.config import ProjectConfig
from self_judged_refusal_direction.errors import CompatibilityError, InvariantError
from self_judged_refusal_direction.hashing import object_sha256, tensor_sha256
from self_judged_refusal_direction.models.base import ArchitectureAdapter
from self_judged_refusal_direction.models.registry import adapter_for_config
from self_judged_refusal_direction.schema import CompatibilityReport, PromptRecord, TargetTrajectory


def _parameter_versions(model: nn.Module) -> tuple[tuple[str, int, int], ...]:
    return tuple((name, id(parameter), parameter._version) for name, parameter in model.named_parameters())


def _hook_signature(model: nn.Module) -> tuple[tuple[str, int, int, int], ...]:
    return tuple(
        (
            name,
            len(module._forward_pre_hooks),
            len(module._forward_hooks),
            len(module._backward_hooks),
        )
        for name, module in model.named_modules()
    )


def _hook_count(signature: tuple[tuple[str, int, int, int], ...]) -> int:
    return sum(pre + forward + backward for _, pre, forward, backward in signature)


def _tensor_sample_checksum(tensor: torch.Tensor) -> str:
    if tensor.device.type == "meta" or tensor.numel() < 1:
        raise CompatibilityError("major checkpoint tensor is not materialized")
    source = tensor.detach().reshape(-1)
    sample_count = min(64, source.numel())
    denominator = max(1, sample_count - 1)
    positions = [index * (source.numel() - 1) // denominator for index in range(sample_count)]
    indices = torch.tensor(positions, device=source.device)
    return tensor_sha256(source.index_select(0, indices))


def _checkpoint_checksum(config: ProjectConfig, adapter: ArchitectureAdapter, model: nn.Module) -> str:
    writers = adapter.residual_writers(model)
    targets = [adapter.text_embedding(model), adapter.lm_head(model)]
    if writers:
        writer_indices = {
            0,
            min(1, len(writers) - 1),
            len(writers) // 2,
            min(len(writers) - 1, len(writers) // 2 + 1),
            max(0, len(writers) - 2),
            len(writers) - 1,
        }
        targets.extend(writers[index] for index in sorted(writer_indices))
    targets.extend(adapter.multimodal_projections(model))
    tensors: dict[tuple[Any, ...], tuple[list[str], torch.Tensor]] = {}
    for target in targets:
        weight = getattr(target.module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise CompatibilityError(f"major checkpoint target has no weight: {target.name}")
        if weight.device.type == "meta":
            raise CompatibilityError(f"major checkpoint target is not materialized: {target.name}")
        storage = weight.untyped_storage()
        identity = (
            str(weight.device),
            storage.data_ptr(),
            storage.nbytes(),
            weight.storage_offset(),
            tuple(weight.shape),
            tuple(weight.stride()),
            str(weight.dtype),
        )
        if identity in tensors:
            tensors[identity][0].append(target.name)
        else:
            tensors[identity] = ([target.name], weight)
    entries = [
        {
            "names": tuple(names),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "sample_sha256": _tensor_sample_checksum(tensor),
        }
        for names, tensor in tensors.values()
    ]
    return object_sha256(
        {
            "model_id": config.model.id,
            "revision": config.model.revision,
            "major_tensors": entries,
        }
    )


class _ModelRuntime:
    def __init__(self, config: ProjectConfig, adapter: ArchitectureAdapter | None = None):
        self.config = config
        self.adapter = adapter or adapter_for_config(config)
        self._model: nn.Module | None = None
        self._processor: Any | None = None
        self._compatibility_report: CompatibilityReport | None = None
        self._chat_template_hash: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise InvariantError("model runtime is not loaded")
        return self._model

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise InvariantError("model runtime is not loaded")
        return self._processor

    @property
    def compatibility_report(self) -> CompatibilityReport:
        if self._compatibility_report is None:
            raise InvariantError("model runtime is not loaded")
        return self._compatibility_report

    @property
    def chat_template_hash(self) -> str:
        if self._chat_template_hash is None:
            raise InvariantError("model runtime is not loaded")
        return self._chat_template_hash

    def load(self) -> Self:
        if self.loaded:
            return self
        self.config.validate()
        processor = self.adapter.load_processor(self.config.model)
        model: nn.Module | None = None
        try:
            model = self.adapter.load_model(self.config.model)
            model.eval()
            model.requires_grad_(False)
            report = self.adapter.compatibility_report(model, processor)
            if not report.compatible:
                raise CompatibilityError("; ".join(report.errors))
            self._model = model
            self._processor = processor
            self._compatibility_report = report
            self._chat_template_hash = self.adapter.chat_template_hash(processor)
            return self
        except Exception:
            del model
            del processor
            self._release_memory()
            raise

    def generate_target(
        self,
        prompt: PromptRecord | str,
        *,
        prompt_id: str | None = None,
        split: str | None = None,
        seed: int | None = None,
    ) -> TargetTrajectory:
        from self_judged_refusal_direction.generation import generate_target_trajectory

        return generate_target_trajectory(
            self,
            prompt,
            prompt_id=prompt_id,
            split=split,
            seed=seed,
        )

    def close(self) -> None:
        try:
            self._before_close()
        finally:
            self._compatibility_report = None
            self._chat_template_hash = None
            self._processor = None
            self._model = None
            self._release_memory()

    def _before_close(self) -> None:
        return None

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self.load()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class BaseModelRuntime(_ModelRuntime):
    def __init__(self, config: ProjectConfig, adapter: ArchitectureAdapter | None = None):
        super().__init__(config, adapter)
        self._loaded_revision: str | None = None
        self._checkpoint_checksum: str | None = None
        self._parameter_version_signature: tuple[tuple[str, int, int], ...] | None = None
        self._hook_count_signature: tuple[tuple[str, int, int, int], ...] | None = None

    @property
    def model(self) -> nn.Module:
        model = super().model
        if self._loaded_revision is not None:
            self.assert_unchanged(verify_checksum=False)
        return model

    @property
    def revision(self) -> str:
        if self._loaded_revision is None:
            raise InvariantError("base model runtime is not loaded")
        return self._loaded_revision

    @property
    def checkpoint_checksum(self) -> str:
        if self._checkpoint_checksum is None:
            raise InvariantError("base model runtime is not loaded")
        self.assert_unchanged(verify_checksum=False)
        return self._checkpoint_checksum

    @property
    def hook_count(self) -> int:
        if self._hook_count_signature is None:
            raise InvariantError("base model runtime is not loaded")
        self.assert_unchanged(verify_checksum=False)
        return _hook_count(self._hook_count_signature)

    def load(self) -> Self:
        if self.loaded:
            self.assert_unchanged()
            return self
        try:
            super().load()
            model = self._model
            if model is None:
                raise InvariantError("base model runtime did not load a model")
            self._loaded_revision = self.config.model.revision
            self._parameter_version_signature = _parameter_versions(model)
            self._hook_count_signature = _hook_signature(model)
            self._checkpoint_checksum = _checkpoint_checksum(self.config, self.adapter, model)
            return self
        except Exception:
            self._clear_invariants()
            self._compatibility_report = None
            self._chat_template_hash = None
            self._processor = None
            self._model = None
            self._release_memory()
            raise

    def assert_unchanged(self, *, verify_checksum: bool = True) -> None:
        model = self._model
        if (
            model is None
            or self._loaded_revision is None
            or self._checkpoint_checksum is None
            or self._parameter_version_signature is None
            or self._hook_count_signature is None
        ):
            raise InvariantError("base model runtime is not loaded")
        if self.config.model.revision != self._loaded_revision:
            raise InvariantError("base model revision changed after load")
        if _parameter_versions(model) != self._parameter_version_signature:
            raise InvariantError("base model parameters changed after load")
        if _hook_signature(model) != self._hook_count_signature:
            raise InvariantError("base model hook count changed after load")
        if verify_checksum and _checkpoint_checksum(self.config, self.adapter, model) != self._checkpoint_checksum:
            raise InvariantError("base model checkpoint checksum changed after load")

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._clear_invariants()

    def _before_close(self) -> None:
        if self.loaded:
            self.assert_unchanged()

    def _clear_invariants(self) -> None:
        self._loaded_revision = None
        self._checkpoint_checksum = None
        self._parameter_version_signature = None
        self._hook_count_signature = None


class IntervenedModelRuntime(_ModelRuntime):
    def __init__(
        self,
        config: ProjectConfig,
        *,
        direction: torch.Tensor | None = None,
        weight_edit_plan: Any | None = None,
        install_temporary: bool | None = None,
        adapter: ArchitectureAdapter | None = None,
    ):
        super().__init__(config, adapter)
        if direction is not None and weight_edit_plan is not None:
            raise InvariantError("specify either a direction or a weight edit plan")
        self._direction = direction
        self._weight_edit_plan = weight_edit_plan
        self._install_temporary = (
            direction is not None or weight_edit_plan is not None if install_temporary is None else install_temporary
        )
        self._intervention_context: AbstractContextManager[Any] | None = None

    @property
    def weight_edit_plan(self) -> Any:
        if self._weight_edit_plan is None:
            raise InvariantError("intervened runtime has no weight edit plan")
        return self._weight_edit_plan

    @property
    def checkpoint_checksum(self) -> str:
        return _checkpoint_checksum(self.config, self.adapter, self.model)

    def load(self) -> Self:
        if self.loaded:
            return self
        try:
            super().load()
            if self._weight_edit_plan is None:
                if self._direction is None:
                    if self._install_temporary:
                        raise InvariantError("temporary intervention requires a direction or weight edit plan")
                    return self
                self._weight_edit_plan = self.adapter.build_weight_edit_plan(self.model, self._direction)
            validate = getattr(self._weight_edit_plan, "validate_shapes", None)
            if not callable(validate):
                raise InvariantError("weight edit plan has no validate_shapes method")
            validate(self.model)
            if not self._install_temporary:
                return self
            install = getattr(self._weight_edit_plan, "install_export_equivalent_hooks", None)
            if not callable(install):
                raise InvariantError("weight edit plan has no export-equivalent intervention")
            context = install(self.model)
            if not isinstance(context, AbstractContextManager):
                raise InvariantError("weight edit plan intervention is not a context manager")
            context.__enter__()
            self._intervention_context = context
            return self
        except Exception:
            self.close()
            raise

    def _before_close(self) -> None:
        if self._intervention_context is None:
            return
        context = self._intervention_context
        self._intervention_context = None
        context.__exit__(None, None, None)


__all__ = ["BaseModelRuntime", "IntervenedModelRuntime"]
