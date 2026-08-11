from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import torch
from torch import nn

from self_judged_refusal_direction.errors import InvariantError
from self_judged_refusal_direction.hashing import object_sha256, tensor_sha256


class ProjectionKind(StrEnum):
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    VECTOR = "VECTOR"


class HookKind(StrEnum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


@dataclass(frozen=True)
class RuntimeSite:
    module_name: str
    hook_kind: HookKind
    vector_key: str

    def as_dict(self) -> dict[str, str]:
        return {
            "module_name": self.module_name,
            "hook_kind": self.hook_kind.value,
            "vector_key": self.vector_key,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeSite:
        return cls(
            module_name=str(value["module_name"]),
            hook_kind=HookKind(value["hook_kind"]),
            vector_key=str(value["vector_key"]),
        )


@dataclass(frozen=True)
class EditOp:
    parameter_name: str
    projection_kind: ProjectionKind
    vector_key: str
    runtime_sites: tuple[RuntimeSite, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "projection_kind": self.projection_kind.value,
            "vector_key": self.vector_key,
            "runtime_sites": [site.as_dict() for site in self.runtime_sites],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EditOp:
        return cls(
            parameter_name=str(value["parameter_name"]),
            projection_kind=ProjectionKind(value["projection_kind"]),
            vector_key=str(value["vector_key"]),
            runtime_sites=tuple(RuntimeSite.from_dict(site) for site in value.get("runtime_sites", ())),
        )


@dataclass(frozen=True)
class WeightEditPlan:
    vectors: dict[str, torch.Tensor]
    operations: tuple[EditOp, ...]

    @property
    def direction(self) -> torch.Tensor:
        return self.vector("direction")

    @property
    def plan_hash(self) -> str:
        return object_sha256(
            {
                "plan": self.to_metadata(),
                "vectors": {key: tensor_sha256(value) for key, value in sorted(self.vectors.items())},
            }
        )

    def vector(self, key: str) -> torch.Tensor:
        try:
            return self.vectors[key]
        except KeyError as error:
            raise InvariantError(f"weight edit plan references unknown vector: {key}") from error

    def to_metadata(self) -> dict[str, Any]:
        return {
            "operations": [operation.as_dict() for operation in self.operations],
            "vector_keys": sorted(self.vectors),
        }

    def tensor_state(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().to(device="cpu").contiguous().clone() for key, value in self.vectors.items()}

    def serialize(self) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        return self.to_metadata(), self.tensor_state()

    @classmethod
    def deserialize(
        cls,
        metadata: Mapping[str, Any],
        tensors: Mapping[str, torch.Tensor],
    ) -> WeightEditPlan:
        if set(metadata) != {"operations", "vector_keys"}:
            raise InvariantError("weight edit plan metadata has unexpected fields")
        expected_keys = {str(key) for key in metadata["vector_keys"]}
        actual_keys = set(tensors)
        if expected_keys != actual_keys:
            raise InvariantError("weight edit plan vector keys do not match serialized tensors")
        return cls(
            vectors={key: tensors[key].detach().to(device="cpu").contiguous().clone() for key in sorted(tensors)},
            operations=tuple(EditOp.from_dict(value) for value in metadata["operations"]),
        )

    @classmethod
    def from_adapter(
        cls,
        adapter: Any,
        model: nn.Module,
        direction: torch.Tensor,
    ) -> WeightEditPlan:
        builder = WeightEditPlanBuilder(model, direction)
        embedding = adapter.text_embedding(model)
        _validate_module_target(model, embedding)
        builder.add_embedding(embedding.name)
        head = adapter.lm_head(model)
        _validate_module_target(model, head)
        builder.add_lm_head(head.name)
        for writer in adapter.residual_writers(model):
            _validate_module_target(model, writer)
            if _get_module(model, writer.post_norm_name) is not writer.post_norm:
                raise InvariantError(f"adapter post norm target does not match model: {writer.post_norm_name}")
            dual = adapter.dual_direction(writer.post_norm, builder.vectors["direction"])
            builder.add_residual_writer(writer.name, dual_direction=dual)
        for projection in adapter.multimodal_projections(model):
            _validate_module_target(model, projection)
            builder.add_multimodal_projection(projection.name)
        return builder.build()

    def validate_shapes(self, model: nn.Module) -> None:
        if "direction" not in self.vectors:
            raise InvariantError("weight edit plan has no direction vector")
        for key, vector in self.vectors.items():
            if vector.ndim != 1 or vector.numel() == 0:
                raise InvariantError(f"projection vector must be non-empty and rank one: {key}")
            if not vector.is_floating_point() or not torch.isfinite(vector).all():
                raise InvariantError(f"projection vector must be finite and floating point: {key}")
            norm = torch.linalg.vector_norm(vector.float()).item()
            if abs(norm - 1.0) > 1e-5:
                raise InvariantError(f"projection vector must have unit norm: {key}")
        aliases: dict[tuple[Any, ...], tuple[ProjectionKind, str]] = {}
        for operation in self.operations:
            parameter = _get_parameter(model, operation.parameter_name)
            if parameter.device.type == "meta":
                raise InvariantError(f"edit parameter must be materialized: {operation.parameter_name}")
            if not parameter.is_floating_point():
                raise InvariantError(f"edit parameter must be floating point: {operation.parameter_name}")
            vector = self.vector(operation.vector_key)
            if operation.projection_kind is ProjectionKind.RIGHT:
                if parameter.ndim != 2 or parameter.shape[1] != vector.numel():
                    raise InvariantError(f"right projection shape mismatch: {operation.parameter_name}")
            elif operation.projection_kind is ProjectionKind.LEFT:
                if parameter.ndim != 2 or parameter.shape[0] != vector.numel():
                    raise InvariantError(f"left projection shape mismatch: {operation.parameter_name}")
            elif operation.projection_kind is ProjectionKind.VECTOR and (
                parameter.ndim != 1 or parameter.numel() != vector.numel()
            ):
                raise InvariantError(f"vector projection shape mismatch: {operation.parameter_name}")
            identity = _alias_identity(parameter)
            signature = (operation.projection_kind, operation.vector_key)
            previous = aliases.get(identity)
            if previous is not None and previous != signature:
                raise InvariantError(f"aliased parameter has conflicting edit operations: {operation.parameter_name}")
            aliases[identity] = signature
            for site in operation.runtime_sites:
                _get_module(model, site.module_name)
                self.vector(site.vector_key)

    @contextmanager
    def temporary(self, model: nn.Module) -> Iterator[nn.Module]:
        self.validate_shapes(model)
        handles: list[Any] = []
        sites: set[tuple[str, HookKind, str]] = set()
        try:
            for operation in self.operations:
                for site in operation.runtime_sites:
                    identity = (site.module_name, site.hook_kind, site.vector_key)
                    if identity in sites:
                        continue
                    sites.add(identity)
                    module = _get_module(model, site.module_name)
                    vector = self.vector(site.vector_key)
                    if site.hook_kind is HookKind.INPUT:
                        handles.append(module.register_forward_pre_hook(_projection_input_hook(vector)))
                    else:
                        handles.append(module.register_forward_hook(_projection_output_hook(vector)))
            yield model
        finally:
            for handle in reversed(handles):
                handle.remove()

    def install_export_equivalent_hooks(self, model: nn.Module):
        return self.temporary(model)

    @contextmanager
    def activation_addition(
        self,
        model: nn.Module,
        module_name: str,
        coefficient: float,
        vector_key: str = "direction",
    ) -> Iterator[nn.Module]:
        vector = self.vector(vector_key)
        with activation_addition(model, module_name, vector, coefficient):
            yield model

    def apply_in_place(
        self,
        model: nn.Module,
        *,
        chunk_rows: int = 4096,
    ) -> tuple[str, ...]:
        if chunk_rows < 1:
            raise InvariantError("edit chunk size must be positive")
        self.validate_shapes(model)
        edited: list[str] = []
        seen: dict[tuple[Any, ...], tuple[ProjectionKind, str]] = {}
        with torch.no_grad():
            for operation in self.operations:
                parameter = _get_parameter(model, operation.parameter_name)
                identity = _alias_identity(parameter)
                signature = (operation.projection_kind, operation.vector_key)
                if identity in seen:
                    if seen[identity] != signature:
                        raise InvariantError(
                            f"aliased parameter has conflicting edit operations: {operation.parameter_name}"
                        )
                    continue
                seen[identity] = signature
                vector = self.vector(operation.vector_key)
                _apply_projection(parameter, operation.projection_kind, vector, chunk_rows)
                edited.append(operation.parameter_name)
        tie_weights = getattr(model, "tie_weights", None)
        if callable(tie_weights):
            tie_weights()
        return tuple(edited)


class WeightEditPlanBuilder:
    def __init__(self, model: nn.Module, direction: torch.Tensor):
        self.model = model
        self.vectors: dict[str, torch.Tensor] = {"direction": _unit_vector(direction)}
        self.operations: list[EditOp] = []
        self.aliases: dict[tuple[Any, ...], int] = {}

    def add_embedding(
        self,
        module_name: str,
        parameter_name: str | None = None,
    ) -> WeightEditPlanBuilder:
        name = parameter_name or _member_name(module_name, "weight")
        self._add_operation(
            name,
            ProjectionKind.RIGHT,
            "direction",
            RuntimeSite(module_name, HookKind.OUTPUT, "direction"),
        )
        return self

    def add_lm_head(
        self,
        module_name: str,
        parameter_name: str | None = None,
    ) -> WeightEditPlanBuilder:
        name = parameter_name or _member_name(module_name, "weight")
        self._add_operation(
            name,
            ProjectionKind.RIGHT,
            "direction",
            RuntimeSite(module_name, HookKind.INPUT, "direction"),
        )
        return self

    def add_residual_writer(
        self,
        module_name: str,
        *,
        effective_scale_name: str | None = None,
        effective_scale: torch.Tensor | None = None,
        dual_direction: torch.Tensor | None = None,
        weight_name: str | None = None,
        bias_name: str | None = None,
        vector_key: str | None = None,
    ) -> WeightEditPlanBuilder:
        key = vector_key or f"dual:{module_name}"
        if key in self.vectors:
            raise InvariantError(f"duplicate projection vector key: {key}")
        if dual_direction is not None:
            if effective_scale_name is not None or effective_scale is not None:
                raise InvariantError("dual direction cannot be combined with an effective scale")
            self.vectors[key] = _unit_vector(dual_direction)
        else:
            scale = self._resolve_scale(effective_scale_name, effective_scale)
            self.vectors[key] = _unit_vector(scale * self.vectors["direction"])
        weight = weight_name or _member_name(module_name, "weight")
        self._add_operation(
            weight,
            ProjectionKind.LEFT,
            key,
            RuntimeSite(module_name, HookKind.OUTPUT, key),
        )
        bias = self._resolve_bias_name(module_name, bias_name)
        if bias is not None:
            self._add_operation(bias, ProjectionKind.VECTOR, key)
        return self

    def add_multimodal_projection(
        self,
        module_name: str,
        *,
        weight_name: str | None = None,
        bias_name: str | None = None,
    ) -> WeightEditPlanBuilder:
        weight = weight_name or _member_name(module_name, "weight")
        self._add_operation(
            weight,
            ProjectionKind.LEFT,
            "direction",
            RuntimeSite(module_name, HookKind.OUTPUT, "direction"),
        )
        bias = self._resolve_bias_name(module_name, bias_name)
        if bias is not None:
            self._add_operation(bias, ProjectionKind.VECTOR, "direction")
        return self

    def build(self) -> WeightEditPlan:
        plan = WeightEditPlan(
            vectors={key: value.clone() for key, value in self.vectors.items()},
            operations=tuple(self.operations),
        )
        plan.validate_shapes(self.model)
        return plan

    def _resolve_scale(
        self,
        effective_scale_name: str | None,
        effective_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if (effective_scale_name is None) == (effective_scale is None):
            raise InvariantError("exactly one residual writer effective scale source is required")
        source = (
            _get_parameter(self.model, effective_scale_name) if effective_scale_name is not None else effective_scale
        )
        assert source is not None
        result = source.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if result.shape != self.vectors["direction"].shape:
            raise InvariantError("residual writer effective scale shape does not match direction")
        if not torch.isfinite(result).all():
            raise InvariantError("residual writer effective scale must be finite")
        return result

    def _resolve_bias_name(self, module_name: str, bias_name: str | None) -> str | None:
        if bias_name is not None:
            _get_parameter(self.model, bias_name)
            return bias_name
        module = _get_module(self.model, module_name)
        bias = getattr(module, "bias", None)
        return _member_name(module_name, "bias") if isinstance(bias, nn.Parameter) else None

    def _add_operation(
        self,
        parameter_name: str,
        projection_kind: ProjectionKind,
        vector_key: str,
        runtime_site: RuntimeSite | None = None,
    ) -> None:
        parameter = _get_parameter(self.model, parameter_name)
        identity = _alias_identity(parameter)
        site_tuple = () if runtime_site is None else (runtime_site,)
        if identity in self.aliases:
            index = self.aliases[identity]
            existing = self.operations[index]
            if existing.projection_kind is not projection_kind or existing.vector_key != vector_key:
                raise InvariantError(f"aliased parameter has conflicting edit operations: {parameter_name}")
            sites = tuple(dict.fromkeys((*existing.runtime_sites, *site_tuple)))
            self.operations[index] = replace(existing, runtime_sites=sites)
            return
        self.aliases[identity] = len(self.operations)
        self.operations.append(
            EditOp(
                parameter_name=parameter_name,
                projection_kind=projection_kind,
                vector_key=vector_key,
                runtime_sites=site_tuple,
            )
        )


@contextmanager
def activation_addition(
    model: nn.Module,
    module_name: str,
    vector: torch.Tensor,
    coefficient: float,
) -> Iterator[nn.Module]:
    if not torch.isfinite(torch.as_tensor(coefficient)):
        raise InvariantError("activation addition coefficient must be finite")
    direction = _unit_vector(vector)
    module = _get_module(model, module_name)
    handle = module.register_forward_pre_hook(_addition_input_hook(direction, coefficient))
    try:
        yield model
    finally:
        handle.remove()


def _member_name(module_name: str, member: str) -> str:
    return f"{module_name}.{member}" if module_name else member


def _validate_module_target(model: nn.Module, target: Any) -> None:
    name = getattr(target, "name", None)
    module = getattr(target, "module", None)
    if not isinstance(name, str) or not isinstance(module, nn.Module):
        raise InvariantError("adapter returned an invalid module target")
    if _get_module(model, name) is not module:
        raise InvariantError(f"adapter module target does not match model: {name}")


def _get_module(model: nn.Module, name: str) -> nn.Module:
    try:
        return model if not name else model.get_submodule(name)
    except (AttributeError, KeyError) as error:
        raise InvariantError(f"unknown module in weight edit plan: {name}") from error


def _get_parameter(model: nn.Module, name: str | None) -> nn.Parameter:
    if name is None:
        raise InvariantError("parameter name is required")
    try:
        parameter = model.get_parameter(name)
    except (AttributeError, KeyError) as error:
        raise InvariantError(f"unknown parameter in weight edit plan: {name}") from error
    if not isinstance(parameter, nn.Parameter):
        raise InvariantError(f"weight edit target is not a parameter: {name}")
    return parameter


def _alias_identity(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        storage.data_ptr(),
        storage.nbytes(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def _unit_vector(vector: torch.Tensor) -> torch.Tensor:
    result = vector.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
    if result.numel() == 0 or not torch.isfinite(result).all():
        raise InvariantError("projection vector must be non-empty and finite")
    norm = torch.linalg.vector_norm(result)
    if not torch.isfinite(norm) or norm.item() <= 0.0:
        raise InvariantError("projection vector norm must be positive and finite")
    return result / norm


def _project_last_dimension(value: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != vector.numel():
        raise InvariantError("runtime projection shape does not match activation")
    source = value.float()
    direction = vector.to(device=value.device, dtype=torch.float32)
    projected = source - torch.sum(source * direction, dim=-1, keepdim=True) * direction
    return projected.to(dtype=value.dtype)


def _replace_first_tensor(values: tuple[Any, ...], transform) -> tuple[Any, ...]:
    if not values or not isinstance(values[0], torch.Tensor):
        raise InvariantError("runtime hook expected a tensor as the first positional value")
    return (transform(values[0]), *values[1:])


def _projection_input_hook(vector: torch.Tensor):
    def hook(_module: nn.Module, inputs: tuple[Any, ...]):
        return _replace_first_tensor(inputs, lambda value: _project_last_dimension(value, vector))

    return hook


def _projection_output_hook(vector: torch.Tensor):
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
        if isinstance(output, torch.Tensor):
            return _project_last_dimension(output, vector)
        if isinstance(output, tuple):
            return _replace_first_tensor(output, lambda value: _project_last_dimension(value, vector))
        raise InvariantError("runtime output hook expected a tensor or tensor-first tuple")

    return hook


def _addition_input_hook(vector: torch.Tensor, coefficient: float):
    def add(value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != vector.numel():
            raise InvariantError("activation addition shape does not match activation")
        direction = vector.to(device=value.device, dtype=value.dtype)
        return value + coefficient * direction

    def hook(_module: nn.Module, inputs: tuple[Any, ...]):
        return _replace_first_tensor(inputs, add)

    return hook


def _apply_projection(
    parameter: nn.Parameter,
    kind: ProjectionKind,
    vector: torch.Tensor,
    chunk_size: int,
) -> None:
    direction = vector.to(device=parameter.device, dtype=torch.float32)
    if kind is ProjectionKind.RIGHT:
        for start in range(0, parameter.shape[0], chunk_size):
            stop = min(start + chunk_size, parameter.shape[0])
            block = parameter[start:stop].float()
            coefficients = torch.sum(block * direction.unsqueeze(0), dim=1, keepdim=True)
            block = block - coefficients * direction.unsqueeze(0)
            parameter[start:stop].copy_(block.to(dtype=parameter.dtype))
        return
    if kind is ProjectionKind.LEFT:
        for start in range(0, parameter.shape[1], chunk_size):
            stop = min(start + chunk_size, parameter.shape[1])
            block = parameter[:, start:stop].float()
            coefficients = torch.sum(direction.unsqueeze(1) * block, dim=0, keepdim=True)
            block = block - direction.unsqueeze(1) * coefficients
            parameter[:, start:stop].copy_(block.to(dtype=parameter.dtype))
        return
    source = parameter.float()
    projected = source - direction * torch.dot(direction, source)
    parameter.copy_(projected.to(dtype=parameter.dtype))
