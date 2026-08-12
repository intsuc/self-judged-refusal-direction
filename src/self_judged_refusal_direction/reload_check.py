from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from self_judged_refusal_direction.config import ModelConfig, TargetGenerationConfig
from self_judged_refusal_direction.errors import InvariantError
from self_judged_refusal_direction.generation import (
    generation_config_hash,
    generation_kwargs,
    resolved_generation_kwargs,
)
from self_judged_refusal_direction.hashing import canonical_json_bytes, object_sha256
from self_judged_refusal_direction.models.base import ArchitectureAdapter
from self_judged_refusal_direction.models.registry import adapter_for_config


@dataclass(frozen=True)
class ReloadCheckReport:
    status: str
    model_class: str
    model_module: str
    parameter_count: int
    parameter_shapes_hash: str
    tied_weights_preserved: bool
    probe_logits_match: bool
    probe_max_abs_error: float
    probe_max_rel_error: float
    processor_reload_verified: bool
    target_trajectory_required: bool
    target_thinking_enabled: bool
    target_trajectory_passed: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @property
    def passed(self) -> bool:
        trajectory_passed = not self.target_trajectory_required or (
            self.processor_reload_verified and self.target_trajectory_passed
        )
        return self.status == "OK" and self.tied_weights_preserved and self.probe_logits_match and trajectory_passed

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReloadCheckReport:
        return cls(**value)


def parameter_shapes(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(parameter.shape) for name, parameter in model.named_parameters(remove_duplicate=False)}


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_fresh_reload_check(
    model_dir: str | Path,
    probe_inputs: dict[str, torch.Tensor],
    expected_logits: torch.Tensor,
    expected_parameter_shapes: dict[str, tuple[int, ...]],
    expected_parameter_count: int,
    tied_parameter_pairs: tuple[tuple[str, str], ...],
    *,
    device_map: str | None,
    dtype: str | None,
    attention_implementation: str | None,
    atol: float,
    rtol: float,
    timeout_seconds: float = 1800.0,
    expected_processor_fingerprints: dict[str, str],
    verify_target_trajectory: bool = True,
    target_trajectory_max_new_tokens: int = 256,
    target_generation: TargetGenerationConfig,
    expected_generation_config_hash: str,
    seed: int,
    revision: str,
) -> ReloadCheckReport:
    target = Path(model_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="self-judged-refusal-direction-reload-") as temporary_name:
        temporary = Path(temporary_name)
        tensor_path = temporary / "probe.safetensors"
        request_path = temporary / "request.json"
        response_path = temporary / "response.json"
        tensors = {
            **{f"input.{name}": tensor.detach().to(device="cpu").contiguous() for name, tensor in probe_inputs.items()},
            "expected_logits": expected_logits.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        }
        save_file(tensors, tensor_path)
        request = {
            "model_dir": str(target),
            "tensor_path": str(tensor_path),
            "expected_parameter_shapes": {
                name: list(shape) for name, shape in sorted(expected_parameter_shapes.items())
            },
            "expected_parameter_count": expected_parameter_count,
            "tied_parameter_pairs": [list(pair) for pair in tied_parameter_pairs],
            "device_map": device_map,
            "dtype": dtype,
            "attention_implementation": attention_implementation,
            "atol": atol,
            "rtol": rtol,
            "expected_processor_fingerprints": expected_processor_fingerprints,
            "verify_target_trajectory": verify_target_trajectory,
            "target_trajectory_max_new_tokens": target_trajectory_max_new_tokens,
            "target_generation": asdict(target_generation),
            "expected_generation_config_hash": expected_generation_config_hash,
            "seed": seed,
            "revision": revision,
        }
        request_path.write_bytes(canonical_json_bytes(request) + b"\n")
        environment = dict(os.environ)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        command = [
            sys.executable,
            "-m",
            "self_judged_refusal_direction.reload_check",
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        stream_output = sys.stderr.isatty()
        completed = subprocess.run(
            command,
            capture_output=not stream_output,
            check=False,
            env=environment,
            text=True,
            timeout=timeout_seconds,
        )
        if not response_path.is_file():
            detail = (completed.stderr or completed.stdout or "").strip()
            if not detail:
                detail = f"exit code {completed.returncode}"
            raise InvariantError(f"fresh reload did not produce a report: {detail}")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        report = ReloadCheckReport.from_dict(response)
        if completed.returncode != 0 or not report.passed:
            detail = report.error or "fresh reload verification failed"
            raise InvariantError(detail)
        return report


def perform_reload_check(request: dict[str, Any]) -> ReloadCheckReport:
    dtype = str(request.get("dtype") or "float32").removeprefix("torch.")
    _torch_dtype(dtype)
    device_map = str(request.get("device_map") or "cpu")
    attention_implementation = str(request.get("attention_implementation") or "sdpa")
    model_config = ModelConfig(
        id=str(request["model_dir"]),
        revision=str(request["revision"]),
        dtype=dtype,
        device_map=device_map,
        attention_implementation=attention_implementation,
    )
    adapter = adapter_for_config(model_config)
    with tqdm(
        total=1,
        desc="Reloading exported model",
        unit="model",
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        model = adapter.load_model(model_config)
        progress.update()
    model.eval()
    model.requires_grad_(False)
    actual_shapes = parameter_shapes(model)
    expected_shapes = {
        name: tuple(int(dimension) for dimension in shape)
        for name, shape in request["expected_parameter_shapes"].items()
    }
    if actual_shapes != expected_shapes:
        raise InvariantError("fresh reload parameter shapes do not match the edited model")
    actual_count = parameter_count(model)
    if actual_count != int(request["expected_parameter_count"]):
        raise InvariantError("fresh reload parameter count does not match the edited model")
    pairs = tuple(tuple(pair) for pair in request["tied_parameter_pairs"])
    ties_preserved = all(
        _same_parameter_storage(model.get_parameter(left), model.get_parameter(right)) for left, right in pairs
    )
    if not ties_preserved:
        raise InvariantError("fresh reload did not preserve tied parameters")
    stored = load_file(request["tensor_path"], device="cpu")
    inputs = {key.removeprefix("input."): value for key, value in stored.items() if key.startswith("input.")}
    expected_logits = stored["expected_logits"].float()
    input_device = _input_device(model)
    inputs = {name: value.to(input_device) for name, value in inputs.items()}
    with tqdm(
        total=1,
        desc="Probing reloaded model",
        unit="probe",
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
        progress.update()
    logits = _extract_logits(output).detach().to(device="cpu", dtype=torch.float32)
    if logits.shape != expected_logits.shape:
        raise InvariantError("fresh reload probe logits shape does not match")
    difference = torch.abs(logits - expected_logits)
    relative = difference / torch.clamp(torch.abs(expected_logits), min=1e-7)
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    max_rel = float(relative.max().item()) if relative.numel() else 0.0
    logits_match = bool(
        torch.allclose(
            logits,
            expected_logits,
            atol=float(request["atol"]),
            rtol=float(request["rtol"]),
        )
    )
    if not logits_match:
        raise InvariantError("fresh reload probe logits do not match the saved edited model")
    model_type = type(model)
    if not model_type.__module__.startswith("transformers."):
        raise InvariantError("fresh reload used a non-Transformers model implementation")
    processor_reload_verified = False
    raw_target_generation = request.get("target_generation")
    if not isinstance(raw_target_generation, dict):
        raise InvariantError("fresh reload target generation config is invalid")
    target_generation = TargetGenerationConfig(**raw_target_generation)
    target_thinking_enabled = target_generation.thinking_enabled
    target_trajectory_passed = False
    target_trajectory_required = bool(request.get("verify_target_trajectory", True))
    actual_generation_config_hash = generation_config_hash(
        target_generation,
        resolved_generation_kwargs(model, target_generation),
    )
    if actual_generation_config_hash != str(request["expected_generation_config_hash"]):
        raise InvariantError("fresh reload generation configuration does not match")
    if target_trajectory_required:
        processor = adapter.load_processor(model_config)
        actual_fingerprints = adapter.processor_fingerprints(processor)
        expected_fingerprints = dict(request["expected_processor_fingerprints"])
        if actual_fingerprints != expected_fingerprints:
            raise InvariantError("fresh reload processor or tokenizer fingerprint does not match")
        processor_reload_verified = True
        with tqdm(
            total=1,
            desc="Generating reload trajectory",
            unit="trajectory",
            dynamic_ncols=True,
            disable=None,
        ) as progress:
            target_trajectory_passed = _run_target_trajectory_probe(
                model,
                processor,
                adapter,
                int(request["target_trajectory_max_new_tokens"]),
                target_generation,
                int(request["seed"]),
            )
            progress.update()
    return ReloadCheckReport(
        status="OK",
        model_class=model_type.__name__,
        model_module=model_type.__module__,
        parameter_count=actual_count,
        parameter_shapes_hash=object_sha256(actual_shapes),
        tied_weights_preserved=ties_preserved,
        probe_logits_match=logits_match,
        probe_max_abs_error=max_abs,
        probe_max_rel_error=max_rel,
        processor_reload_verified=processor_reload_verified,
        target_trajectory_required=target_trajectory_required,
        target_thinking_enabled=target_thinking_enabled,
        target_trajectory_passed=target_trajectory_passed,
    )


def _run_target_trajectory_probe(
    model: torch.nn.Module,
    processor: Any,
    adapter: ArchitectureAdapter,
    max_new_tokens: int,
    target_generation: TargetGenerationConfig,
    seed: int,
) -> bool:
    if max_new_tokens < 1:
        raise InvariantError("fresh reload target trajectory length must be positive")
    probe_generation = replace(
        target_generation,
        max_new_tokens=min(max_new_tokens, target_generation.max_new_tokens),
    )
    messages: list[dict[str, str]] = []
    if probe_generation.system_prompt is not None:
        messages.append({"role": "system", "content": probe_generation.system_prompt})
    messages.append({"role": "user", "content": "Reply with exactly OK."})
    rendered = adapter.render_target_chat(
        processor,
        messages,
        config=probe_generation,
        prefill_thinking=True,
    )
    if not isinstance(rendered, dict) and not hasattr(rendered, "items"):
        raise InvariantError("fresh reload target chat did not return model inputs")
    inputs = dict(rendered)
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise InvariantError("fresh reload target chat did not return one token sequence")
    prefix_ids = input_ids[0].detach().to(device="cpu")
    input_device = _input_device(model)
    inputs = {
        name: value.to(input_device) if isinstance(value, torch.Tensor) else value for name, value in inputs.items()
    }
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise InvariantError("fresh reload target model does not support generation")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = generate(**inputs, **generation_kwargs(probe_generation))
    sequences = getattr(output, "sequences", output)
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2 or sequences.shape[0] != 1:
        raise InvariantError("fresh reload target generation did not return one token sequence")
    sequence = sequences[0].detach().to(device="cpu")
    if not torch.equal(sequence[: prefix_ids.numel()], prefix_ids):
        raise InvariantError("fresh reload target generation changed its input prefix")
    parsed = adapter.parse_target_trajectory(
        processor,
        sequence[prefix_ids.numel() :],
        prefix_ids=prefix_ids,
        thinking_enabled=probe_generation.thinking_enabled,
    )
    if not parsed.terminal_found:
        raise InvariantError("fresh reload target trajectory did not reach an official terminal boundary")
    return True


def _torch_dtype(value: str) -> torch.dtype:
    normalized = value.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise InvariantError(f"unsupported reload dtype: {value}")
    return dtype


def _input_device(model: torch.nn.Module) -> torch.device:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    embeddings = get_input_embeddings() if callable(get_input_embeddings) else None
    weight = getattr(embeddings, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.device
    return next(model.parameters()).device


def _extract_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise InvariantError("model probe did not return logits")
    return logits


def _same_parameter_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
    )


def _write_report(path: Path, value: ReloadCheckReport) -> None:
    path.write_bytes(canonical_json_bytes(value.as_dict()) + b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    arguments = parser.parse_args(argv)
    response_path = Path(arguments.response)
    try:
        request = json.loads(Path(arguments.request).read_text(encoding="utf-8"))
        report = perform_reload_check(request)
    except Exception as error:
        report = ReloadCheckReport(
            status="ERROR",
            model_class="",
            model_module="",
            parameter_count=0,
            parameter_shapes_hash="",
            tied_weights_preserved=False,
            probe_logits_match=False,
            probe_max_abs_error=0.0,
            probe_max_rel_error=0.0,
            processor_reload_verified=False,
            target_trajectory_required=False,
            target_thinking_enabled=False,
            target_trajectory_passed=False,
            error=f"{type(error).__name__}: {error}",
        )
        _write_report(response_path, report)
        return 1
    _write_report(response_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
