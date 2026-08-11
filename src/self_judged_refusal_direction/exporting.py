from __future__ import annotations

import os
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import accelerate
import safetensors
import torch
import transformers
from torch import nn

from self_judged_refusal_direction.artifacts import ArtifactMetadata, ArtifactPaths, ArtifactProfile, ArtifactStore
from self_judged_refusal_direction.config import ProjectConfig
from self_judged_refusal_direction.editing import ProjectionKind, WeightEditPlan
from self_judged_refusal_direction.errors import CompatibilityError, InvariantError
from self_judged_refusal_direction.hashing import file_sha256, object_sha256, tensor_sha256
from self_judged_refusal_direction.models.base import ArchitectureAdapter
from self_judged_refusal_direction.reload_check import (
    ReloadCheckReport,
    parameter_count,
    parameter_shapes,
    run_fresh_reload_check,
)


@dataclass(frozen=True)
class ProbeEquivalence:
    passed: bool
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float


@dataclass(frozen=True)
class OrthogonalityCheck:
    parameter_name: str
    projection_kind: str
    vector_key: str
    max_abs_projection: float
    passed: bool


@dataclass(frozen=True)
class ExportResult:
    output_dir: str
    edited_parameter_names: tuple[str, ...]
    temporary_permanent_equivalence: ProbeEquivalence
    orthogonality: tuple[OrthogonalityCheck, ...]
    reload: ReloadCheckReport | None
    deferred_reload: DeferredReload | None
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DeferredReload:
    model_dir: str
    probe_inputs: dict[str, torch.Tensor]
    expected_logits: torch.Tensor
    expected_parameter_shapes: dict[str, tuple[int, ...]]
    expected_parameter_count: int
    tied_parameter_pairs: tuple[tuple[str, str], ...]
    device_map: str | None
    dtype: str | None
    attention_implementation: str | None
    atol: float
    rtol: float
    timeout_seconds: float
    processor_fingerprints: dict[str, str]
    adapter_name: str
    verify_target_trajectory: bool
    target_trajectory_max_new_tokens: int
    revision: str
    low_cpu_mem_usage: bool


def export_edited_model(
    model: nn.Module,
    processor: Any,
    adapter: ArchitectureAdapter,
    plan: WeightEditPlan,
    config: ProjectConfig,
    probe_inputs: Mapping[str, torch.Tensor],
    *,
    output_dir: str | Path | None = None,
    validation_metrics: Mapping[str, Any] | None = None,
    test_metrics: Mapping[str, Any] | None = None,
    probe_atol: float | None = None,
    probe_rtol: float | None = None,
    orthogonality_atol: float | None = None,
    verify_unchanged_parameters: bool = True,
    reload_timeout_seconds: float = 1800.0,
    defer_reload: bool = True,
    verify_reload_target_trajectory: bool = True,
    reload_target_trajectory_max_new_tokens: int = 256,
) -> ExportResult:
    config.validate()
    _validate_export_config(config)
    if not defer_reload:
        raise InvariantError("fresh reload must be deferred until the edited model is released")
    if reload_target_trajectory_max_new_tokens < 1:
        raise InvariantError("fresh reload target trajectory length must be positive")
    target = Path(output_dir) if output_dir is not None else ArtifactPaths(config.run.output_dir).exported_model
    target = target.resolve()
    if target.exists() and any(target.iterdir()):
        raise InvariantError(f"export directory must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.requires_grad_(False)
    plan.validate_shapes(model)
    compatibility_before = adapter.compatibility_report(model, None)
    if not compatibility_before.compatible:
        raise CompatibilityError("; ".join(compatibility_before.errors))
    shapes_before = parameter_shapes(model)
    count_before = parameter_count(model)
    tied_pairs = _tied_parameter_pairs(model, adapter)
    untouched_before = _untouched_hashes(model, plan) if verify_unchanged_parameters else {}
    atol, rtol = _probe_tolerances(model, probe_atol, probe_rtol)
    with plan.install_export_equivalent_hooks(model):
        temporary_logits = _probe_logits(model, adapter, probe_inputs)
    edited_parameter_names = plan.apply_in_place(model, chunk_rows=config.export.edit_chunk_rows)
    permanent_logits = _probe_logits(model, adapter, probe_inputs)
    equivalence = _compare_logits(temporary_logits, permanent_logits, atol, rtol)
    if not equivalence.passed:
        raise InvariantError("temporary intervention and permanent edit probe logits do not match")
    orthogonality_tolerance = _orthogonality_tolerance(model, orthogonality_atol)
    orthogonality = verify_plan_orthogonality(
        model,
        plan,
        atol=orthogonality_tolerance,
        chunk_size=config.export.edit_chunk_rows,
    )
    if verify_unchanged_parameters:
        _verify_untouched_hashes(model, untouched_before)
    shapes_after = parameter_shapes(model)
    count_after = parameter_count(model)
    if shapes_after != shapes_before or count_after != count_before:
        raise InvariantError("permanent edit changed model parameter topology")
    _verify_tied_pairs(model, tied_pairs)
    compatibility_after = adapter.compatibility_report(model, None)
    if not compatibility_after.compatible:
        raise CompatibilityError("; ".join(compatibility_after.errors))
    adapter.save_model_and_processor(model, processor, target, config)
    processor_fingerprints = adapter.processor_fingerprints(processor)
    reload_request = DeferredReload(
        model_dir=str(target),
        probe_inputs={name: value.detach().to(device="cpu").contiguous() for name, value in probe_inputs.items()},
        expected_logits=permanent_logits,
        expected_parameter_shapes=shapes_after,
        expected_parameter_count=count_after,
        tied_parameter_pairs=tied_pairs,
        device_map=config.model.device_map,
        dtype=config.model.dtype,
        attention_implementation=config.model.attention_implementation,
        atol=atol,
        rtol=rtol,
        timeout_seconds=reload_timeout_seconds,
        processor_fingerprints=processor_fingerprints,
        adapter_name=config.model.adapter,
        verify_target_trajectory=verify_reload_target_trajectory,
        target_trajectory_max_new_tokens=reload_target_trajectory_max_new_tokens,
        revision=config.model.revision,
        low_cpu_mem_usage=config.model.low_cpu_mem_usage,
    )
    reload_report = None if defer_reload else _run_deferred_reload(reload_request)
    manifest = _build_manifest(
        config=config,
        adapter=adapter,
        plan=plan,
        edited_parameter_names=edited_parameter_names,
        equivalence=equivalence,
        orthogonality=orthogonality,
        reload_report=reload_report,
        parameter_shapes_hash=object_sha256(shapes_after),
        parameter_count_value=count_after,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        chat_template_hash=adapter.chat_template_hash(processor),
        processor_fingerprints=processor_fingerprints,
        untouched_parameters_verified=verify_unchanged_parameters,
    )
    write_export_manifest(target, manifest)
    _write_public_text(target / "README.md", _readme(config))
    return ExportResult(
        output_dir=str(target),
        edited_parameter_names=edited_parameter_names,
        temporary_permanent_equivalence=equivalence,
        orthogonality=orthogonality,
        reload=reload_report,
        deferred_reload=reload_request if defer_reload else None,
        manifest=manifest,
    )


def complete_deferred_reload(result: ExportResult) -> ExportResult:
    if result.reload is not None or result.deferred_reload is None:
        raise InvariantError("export result has no pending fresh reload check")
    report = _run_deferred_reload(result.deferred_reload)
    manifest = dict(result.manifest)
    manifest["fresh_reload"] = asdict(report)
    write_export_manifest(result.output_dir, manifest)
    return replace(result, reload=report, deferred_reload=None, manifest=manifest)


def _run_deferred_reload(request: DeferredReload) -> ReloadCheckReport:
    return run_fresh_reload_check(
        request.model_dir,
        request.probe_inputs,
        request.expected_logits,
        request.expected_parameter_shapes,
        request.expected_parameter_count,
        request.tied_parameter_pairs,
        device_map=request.device_map,
        dtype=request.dtype,
        attention_implementation=request.attention_implementation,
        atol=request.atol,
        rtol=request.rtol,
        timeout_seconds=request.timeout_seconds,
        expected_processor_fingerprints=request.processor_fingerprints,
        adapter_name=request.adapter_name,
        verify_target_trajectory=request.verify_target_trajectory,
        target_trajectory_max_new_tokens=request.target_trajectory_max_new_tokens,
        revision=request.revision,
        low_cpu_mem_usage=request.low_cpu_mem_usage,
    )


def export_model(*args: Any, **kwargs: Any) -> ExportResult:
    return export_edited_model(*args, **kwargs)


def verify_plan_orthogonality(
    model: nn.Module,
    plan: WeightEditPlan,
    *,
    atol: float,
    chunk_size: int = 4096,
) -> tuple[OrthogonalityCheck, ...]:
    if atol < 0 or chunk_size < 1:
        raise InvariantError("orthogonality tolerance and chunk size must be valid")
    results: list[OrthogonalityCheck] = []
    seen: set[tuple[Any, ...]] = set()
    for operation in plan.operations:
        parameter = model.get_parameter(operation.parameter_name)
        identity = _storage_identity(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        vector = plan.vector(operation.vector_key)
        residual = _maximum_projection(parameter, operation.projection_kind, vector, chunk_size)
        check = OrthogonalityCheck(
            parameter_name=operation.parameter_name,
            projection_kind=operation.projection_kind.value,
            vector_key=operation.vector_key,
            max_abs_projection=residual,
            passed=residual <= atol,
        )
        if not check.passed:
            raise InvariantError(f"edited parameter is not orthogonal to its plan vector: {operation.parameter_name}")
        results.append(check)
    return tuple(results)


def _validate_export_config(config: ProjectConfig) -> None:
    if not config.export.safe_serialization:
        raise InvariantError("model export requires safe serialization")
    if not config.export.include_processor:
        raise InvariantError("model export requires processor files")
    if config.export.include_raw_thinking:
        raise InvariantError("raw thinking must not be included in model export")
    if config.export.push_to_hub:
        raise InvariantError("model export cannot push to the Hub")
    if not config.export.verify_fresh_process:
        raise InvariantError("model export requires fresh process verification")
    if config.export.edit_compute_dtype != "float32":
        raise InvariantError("model export requires float32 edit compute")


def _probe_tolerances(
    model: nn.Module,
    atol: float | None,
    rtol: float | None,
) -> tuple[float, float]:
    dtype = next(model.parameters()).dtype
    low_precision = dtype in {torch.bfloat16, torch.float16}
    resolved_atol = (5e-2 if low_precision else 2e-5) if atol is None else atol
    resolved_rtol = (5e-2 if low_precision else 2e-5) if rtol is None else rtol
    if resolved_atol < 0 or resolved_rtol < 0:
        raise InvariantError("probe tolerances must be non-negative")
    return resolved_atol, resolved_rtol


def _orthogonality_tolerance(model: nn.Module, value: float | None) -> float:
    if value is not None:
        if value < 0:
            raise InvariantError("orthogonality tolerance must be non-negative")
        return value
    dtype = next(model.parameters()).dtype
    return 2e-2 if dtype in {torch.bfloat16, torch.float16} else 2e-5


def _probe_logits(
    model: nn.Module,
    adapter: ArchitectureAdapter,
    probe_inputs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if not probe_inputs or any(not isinstance(value, torch.Tensor) for value in probe_inputs.values()):
        raise InvariantError("probe inputs must be a non-empty tensor mapping")
    embedding = adapter.text_embedding(model).module
    weight = getattr(embedding, "weight", None)
    device = weight.device if isinstance(weight, torch.Tensor) else next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in probe_inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise InvariantError("model probe did not return logits")
    return logits.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _compare_logits(
    temporary: torch.Tensor,
    permanent: torch.Tensor,
    atol: float,
    rtol: float,
) -> ProbeEquivalence:
    if temporary.shape != permanent.shape:
        raise InvariantError("temporary and permanent probe logits have different shapes")
    difference = torch.abs(temporary - permanent)
    relative = difference / torch.clamp(torch.abs(permanent), min=1e-7)
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    max_rel = float(relative.max().item()) if relative.numel() else 0.0
    return ProbeEquivalence(
        passed=bool(torch.allclose(temporary, permanent, atol=atol, rtol=rtol)),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        atol=atol,
        rtol=rtol,
    )


def _maximum_projection(
    parameter: torch.Tensor,
    kind: ProjectionKind,
    vector: torch.Tensor,
    chunk_size: int,
) -> float:
    direction = vector.to(device=parameter.device, dtype=torch.float32)
    maximum = 0.0
    with torch.inference_mode():
        if kind is ProjectionKind.RIGHT:
            for start in range(0, parameter.shape[0], chunk_size):
                block = parameter[start : start + chunk_size].float()
                value = torch.sum(block * direction.unsqueeze(0), dim=1)
                maximum = max(maximum, float(torch.max(torch.abs(value)).item()))
            return maximum
        if kind is ProjectionKind.LEFT:
            for start in range(0, parameter.shape[1], chunk_size):
                block = parameter[:, start : start + chunk_size].float()
                value = torch.sum(direction.unsqueeze(1) * block, dim=0)
                maximum = max(maximum, float(torch.max(torch.abs(value)).item()))
            return maximum
        return float(torch.abs(torch.dot(direction, parameter.float())).item())


def _all_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return dict(model.named_parameters(remove_duplicate=False))


def _storage_identity(parameter: torch.Tensor) -> tuple[Any, ...]:
    storage = parameter.untyped_storage()
    return (
        str(parameter.device),
        storage.data_ptr(),
        storage.nbytes(),
        parameter.storage_offset(),
        tuple(parameter.shape),
        tuple(parameter.stride()),
        str(parameter.dtype),
    )


def _untouched_hashes(model: nn.Module, plan: WeightEditPlan) -> dict[str, str]:
    parameters = _all_named_parameters(model)
    edited_identities = {
        _storage_identity(model.get_parameter(operation.parameter_name)) for operation in plan.operations
    }
    return {
        name: tensor_sha256(parameter)
        for name, parameter in parameters.items()
        if _storage_identity(parameter) not in edited_identities
    }


def _verify_untouched_hashes(model: nn.Module, expected: Mapping[str, str]) -> None:
    parameters = _all_named_parameters(model)
    if not set(expected) <= set(parameters):
        raise InvariantError("permanent edit removed an unedited parameter")
    changed = [name for name, digest in expected.items() if tensor_sha256(parameters[name]) != digest]
    if changed:
        raise InvariantError(f"permanent edit changed unplanned parameters: {changed}")


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return _storage_identity(left) == _storage_identity(right)


def _tied_parameter_pairs(
    model: nn.Module,
    adapter: ArchitectureAdapter,
) -> tuple[tuple[str, str], ...]:
    embedding_name = f"{adapter.text_embedding(model).name}.weight"
    head_name = f"{adapter.lm_head(model).name}.weight"
    embedding = model.get_parameter(embedding_name)
    head = model.get_parameter(head_name)
    return ((embedding_name, head_name),) if _same_storage(embedding, head) else ()


def _verify_tied_pairs(model: nn.Module, pairs: tuple[tuple[str, str], ...]) -> None:
    if not all(_same_storage(model.get_parameter(left), model.get_parameter(right)) for left, right in pairs):
        raise InvariantError("permanent edit did not preserve tied parameters")


def _build_manifest(
    *,
    config: ProjectConfig,
    adapter: ArchitectureAdapter,
    plan: WeightEditPlan,
    edited_parameter_names: tuple[str, ...],
    equivalence: ProbeEquivalence,
    orthogonality: tuple[OrthogonalityCheck, ...],
    reload_report: ReloadCheckReport | None,
    parameter_shapes_hash: str,
    parameter_count_value: int,
    validation_metrics: Mapping[str, Any] | None,
    test_metrics: Mapping[str, Any] | None,
    chat_template_hash: str,
    processor_fingerprints: Mapping[str, str],
    untouched_parameters_verified: bool,
) -> dict[str, Any]:
    rules = []
    for operation in plan.operations:
        rule = operation.as_dict()
        rule["vector_sha256"] = tensor_sha256(plan.vector(operation.vector_key))
        rules.append(rule)
    metadata = plan.metadata
    return {
        "schema_version": 1,
        "base_model_id": config.model.id,
        "base_revision": config.model.revision,
        "config_hash": config.config_hash,
        "target_profile_hash": config.target_profile_hash,
        "judge_profile_hash": config.judge_profile_hash,
        "chat_template_hash": chat_template_hash,
        **dict(processor_fingerprints),
        "adapter": adapter.__class__.__name__,
        "target_generation": {"thinking_enabled": True},
        "judge_profile": {
            "thinking_enabled": False,
            "backend": "enum_trie",
            "semantics": "REFUSAL iff thought considers refusal OR final answer refuses",
        },
        "direction_semantics": "trajectory-level refusal direction: thinking consideration OR final refusal",
        "direction_source_phase": metadata.get("phase"),
        "direction_source_layer": metadata.get("layer"),
        "direction_source_relative_position": metadata.get("relative_position"),
        "direction_sha256": tensor_sha256(plan.direction),
        "weight_edit_plan_sha256": plan.plan_hash,
        "selected_projection_rank": 1,
        "edited_parameter_names": list(edited_parameter_names),
        "projection_rules": rules,
        "parameter_shapes_hash": parameter_shapes_hash,
        "parameter_count": parameter_count_value,
        "untouched_parameters_verified": untouched_parameters_verified,
        "text_only_direction_discovery": bool(metadata.get("text_only_direction_discovery", True)),
        "multimodal_behavior_validated": bool(metadata.get("multimodal_behavior_validated", False)),
        "temporary_permanent_equivalence": asdict(equivalence),
        "orthogonality": [asdict(value) for value in orthogonality],
        "fresh_reload": asdict(reload_report) if reload_report is not None else {"status": "PENDING"},
        "validation_metrics": dict(validation_metrics or {}),
        "test_metrics": dict(test_metrics or {}),
        "privacy": {
            "raw_thinking_included": False,
            "push_to_hub": False,
        },
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "safetensors": safetensors.__version__,
        },
    }


def write_export_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> None:
    target = Path(output_dir) / "edit_manifest.json"
    value = dict(manifest)
    ArtifactStore.write_json(target, value, private=False)
    try:
        profile = ArtifactProfile(
            model_id=str(value["base_model_id"]),
            model_revision=str(value["base_revision"]),
            config_hash=str(value["config_hash"]),
            target_profile_hash=str(value["target_profile_hash"]),
            judge_profile_hash=str(value["judge_profile_hash"]),
            chat_template_hash=str(value["chat_template_hash"]),
        )
    except KeyError as error:
        raise InvariantError(f"export manifest profile field is missing: {error.args[0]}") from error
    metadata = ArtifactMetadata(
        schema_version=1,
        artifact_type="edit_manifest",
        private=False,
        record_count=1,
        content_sha256=file_sha256(target),
        profile=profile,
    )
    ArtifactStore.write_json(ArtifactStore.metadata_path(target), asdict(metadata), private=False)


def _readme(config: ProjectConfig) -> str:
    return f"""# Direction-edited model

Base model: `{config.model.id}` at `{config.model.revision}`.

The removed rank-1 direction is a trajectory-level refusal direction associated
with either refusal consideration in generated thinking or actual refusal in the
final answer. Target generations used for discovery and evaluation keep thinking
enabled. Classification uses the unchanged base checkpoint with thinking disabled.

This directory contains no raw thinking artifacts and is not published automatically.

```python
from transformers import AutoModelForMultimodalLM

model = AutoModelForMultimodalLM.from_pretrained(
    "path/to/exported_model",
    trust_remote_code=False,
)
```
"""


def _write_public_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    os.chmod(temporary, 0o644)
    temporary.replace(path)


__all__ = [
    "DeferredReload",
    "ExportResult",
    "OrthogonalityCheck",
    "ProbeEquivalence",
    "complete_deferred_reload",
    "export_edited_model",
    "export_model",
    "verify_plan_orthogonality",
    "write_export_manifest",
]
