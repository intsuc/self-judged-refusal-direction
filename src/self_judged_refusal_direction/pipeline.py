from __future__ import annotations

import dataclasses
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import safetensors
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from self_judged_refusal_direction.activations import (
    ActivationCollector,
    load_activation_statistics,
    save_activation_statistics,
)
from self_judged_refusal_direction.artifacts import ArtifactMetadata, ArtifactProfile, ArtifactStore
from self_judged_refusal_direction.ce_loss import (
    CEInput,
    CELoss,
    ce_evaluation_from_losses,
    completed_non_refusal_completion_inputs,
    compute_ce_loss,
    raw_text_ce_inputs,
)
from self_judged_refusal_direction.checkpoint import PrivateCheckpoint
from self_judged_refusal_direction.config import ArtifactStage, ProjectConfig, load_config
from self_judged_refusal_direction.data import (
    ingest_and_split_prompts,
    ingest_texts,
    records_by_split,
)
from self_judged_refusal_direction.decoding import EnumTrieConstrainedDecoder
from self_judged_refusal_direction.directions import (
    CandidateBundle,
    build_candidates,
    load_candidate_artifacts,
    load_direction,
    rank_activation_screening,
    save_direction,
    write_candidate_artifacts,
)
from self_judged_refusal_direction.editing import WeightEditPlan
from self_judged_refusal_direction.errors import ArtifactError, InvariantError, NonFiniteMetricError, PipelineError
from self_judged_refusal_direction.evaluation import (
    apply_pilot_filters,
    evaluate_behavior,
    judgment_counts,
    mean_next_token_kl,
    metrics_dict,
    parser_statistics,
    select_candidate,
)
from self_judged_refusal_direction.generation import (
    TargetTrajectoryGenerator,
    generation_config_hash,
    resolved_batch_generation_kwargs,
)
from self_judged_refusal_direction.hashing import file_sha256, object_sha256, tensor_sha256
from self_judged_refusal_direction.judge_validation import (
    judge_validation_fixture_hash,
    judge_validation_passed,
    load_judge_validation_cases,
    run_judge_validation,
    validate_judge_validation_results,
)
from self_judged_refusal_direction.judging import TrajectoryJudge
from self_judged_refusal_direction.models.registry import adapter_for_config
from self_judged_refusal_direction.prompting import (
    judge_messages,
    target_messages,
)
from self_judged_refusal_direction.prompting import judge_profile_hash as current_judge_profile_hash
from self_judged_refusal_direction.runtime import BaseModelRuntime, IntervenedModelRuntime
from self_judged_refusal_direction.schema import (
    CandidateMetrics,
    CEEvaluation,
    DirectionCandidate,
    JudgeInput,
    JudgeLabel,
    JudgeResult,
    JudgeValidationCase,
    JudgeValidationResult,
    LabeledTrajectory,
    PromptRecord,
    TargetTrajectory,
)


def _progress[T](
    values: Iterable[T],
    *,
    desc: str,
    unit: str,
    total: int | None = None,
    leave: bool = True,
) -> Iterable[T]:
    return tqdm(
        values,
        desc=desc,
        unit=unit,
        total=total,
        leave=leave,
        dynamic_ncols=True,
        disable=None,
    )


def _load(
    path: str | Path,
    *,
    require_judge_validation: bool = True,
) -> tuple[ProjectConfig, ArtifactStore]:
    config = load_config(path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(config.run.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.run.seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    store = ArtifactStore(config)
    store.initialize_run()
    if require_judge_validation:
        _require_judge_validation(store)
    return config, store


def _profile(
    store: ArtifactStore,
    *,
    stage: ArtifactStage,
    target: bool = False,
    chat_template_hash: str | None = None,
    judge_profile_hash: str | None = None,
    judge_fixture_hash: str | None = None,
    judge_validation_hash: str | None = None,
    baseline_generation_hash: str | None = None,
    baseline_judgment_hash: str | None = None,
    label_selection_hash: str | None = None,
    activation_extraction_hash: str | None = None,
    direction_construction_hash: str | None = None,
    candidate_evaluation_hash: str | None = None,
    acceptance_policy_hash: str | None = None,
) -> ArtifactProfile:
    return store.profile(
        stage=stage,
        target=target,
        chat_template_hash=chat_template_hash,
        judge_profile_hash=judge_profile_hash,
        judge_fixture_hash=judge_fixture_hash,
        judge_validation_hash=judge_validation_hash,
        baseline_generation_hash=baseline_generation_hash,
        baseline_judgment_hash=baseline_judgment_hash,
        label_selection_hash=label_selection_hash,
        activation_extraction_hash=activation_extraction_hash,
        direction_construction_hash=direction_construction_hash,
        candidate_evaluation_hash=candidate_evaluation_hash,
        acceptance_policy_hash=acceptance_policy_hash,
    )


def _judge_validation_proof(
    runtime: BaseModelRuntime,
    cases: Sequence[JudgeValidationCase],
    results: Sequence[JudgeValidationResult],
) -> str:
    return object_sha256(
        {
            "fixture_hash": judge_validation_fixture_hash(cases),
            "results": [result.as_dict() for result in results],
            "checkpoint_checksum": runtime.checkpoint_checksum,
            "chat_template_hash": runtime.chat_template_hash,
            "judge_profile_hash": current_judge_profile_hash(),
            "processor_fingerprints": runtime.adapter.processor_fingerprints(runtime.processor),
        }
    )


def _load_judge_validation(
    store: ArtifactStore,
) -> tuple[tuple[JudgeValidationCase, ...], list[JudgeValidationResult], ArtifactMetadata]:
    cases = load_judge_validation_cases()
    expected_profile = _profile(
        store,
        stage="judge_validation",
        judge_profile_hash=current_judge_profile_hash(),
        judge_fixture_hash=judge_validation_fixture_hash(cases),
    )
    metadata = store.validate(
        store.paths.judge_results,
        artifact_type="judge_validation_results",
        expected_profile=expected_profile,
    )
    if metadata.profile.chat_template_hash is None or metadata.profile.judge_validation_hash is None:
        raise ArtifactError("judge validation provenance is incomplete")
    rows = store.read_jsonl(
        store.paths.judge_results,
        artifact_type="judge_validation_results",
        expected_profile=expected_profile,
    )
    results = [JudgeValidationResult.from_dict(row) for row in rows]
    validate_judge_validation_results(cases, results)
    if not judge_validation_passed(cases, results):
        raise PipelineError(f"judge validation did not pass; results: {store.paths.judge_results}")
    return cases, results, metadata


def _require_judge_validation(
    store: ArtifactStore,
    runtime: BaseModelRuntime | None = None,
) -> str:
    cases, results, metadata = _load_judge_validation(store)
    validation_hash = metadata.profile.judge_validation_hash
    if validation_hash is None:
        raise ArtifactError("judge validation hash is missing")
    if runtime is not None:
        if metadata.profile.chat_template_hash != runtime.chat_template_hash:
            raise ArtifactError("judge validation used a different chat template")
        if validation_hash != _judge_validation_proof(runtime, cases, results):
            raise ArtifactError("judge validation proof does not match the loaded base model")
    return validation_hash


def _evaluate_judge_validation(
    runtime: BaseModelRuntime,
) -> tuple[
    tuple[JudgeValidationCase, ...],
    tuple[JudgeValidationResult, ...],
    str,
    str,
]:
    cases = load_judge_validation_cases()
    results = run_judge_validation(runtime, cases)
    validation_hash = _judge_validation_proof(runtime, cases, results)
    return cases, results, validation_hash, runtime.chat_template_hash


def _write_judge_validation(
    store: ArtifactStore,
    cases: Sequence[JudgeValidationCase],
    results: Sequence[JudgeValidationResult],
    validation_hash: str,
    chat_template_hash: str,
) -> None:
    passed = judge_validation_passed(cases, results)
    store.write_jsonl(
        store.paths.judge_results,
        results,
        artifact_type="judge_validation_results",
        profile=_profile(
            store,
            stage="judge_validation",
            chat_template_hash=chat_template_hash,
            judge_profile_hash=current_judge_profile_hash(),
            judge_fixture_hash=judge_validation_fixture_hash(cases),
            judge_validation_hash=validation_hash,
        ),
        private=False,
    )
    matched = sum(result.status == "OK" and result.actual_label == result.expected_label for result in results)
    if not passed:
        errors = sum(result.status == "ERROR" for result in results)
        mismatches = len(results) - matched - errors
        raise PipelineError(
            f"judge validation failed with {mismatches} mismatches and {errors} errors; "
            f"results: {store.paths.judge_results}"
        )
    print(f"Judge validation: PASS ({matched}/{len(results)}); results: {store.paths.judge_results}")


def validate_judge(config_path: str) -> None:
    config, store = _load(config_path, require_judge_validation=False)
    load_judge_validation_cases()
    with BaseModelRuntime(config) as runtime:
        evidence = _evaluate_judge_validation(runtime)
    _write_judge_validation(store, *evidence)


def _ensure_judge_validation(config_path: str) -> None:
    config, store = _load(config_path, require_judge_validation=False)
    load_judge_validation_cases()
    with BaseModelRuntime(config) as runtime:
        try:
            _require_judge_validation(store, runtime)
        except ArtifactError, PipelineError:
            evidence = _evaluate_judge_validation(runtime)
        else:
            return
    _write_judge_validation(store, *evidence)


def _check_error_rate(count: int, total: int, config: ProjectConfig, phase: str) -> None:
    if count < 0 or total < 1 or count > total:
        raise InvariantError(f"{phase} error accounting is invalid")
    rate = count / total
    maximum = config.acceptance.max_error_rate
    if rate > maximum:
        raise PipelineError(
            f"{phase} produced {count}/{total} ERROR records ({rate:.2%}); maximum rate is {maximum:.2%}"
        )


def _target_generation_profile_hash(config: ProjectConfig, runtime: BaseModelRuntime | IntervenedModelRuntime) -> str:
    generate_kwargs = resolved_batch_generation_kwargs(
        runtime.model,
        runtime.processor,
        config.target_generation,
    )
    return generation_config_hash(config.target_generation, generate_kwargs)


def _implementation_hash(*values: Any) -> str:
    sources: dict[str, str] = {}
    for value in values:
        module = inspect.getmodule(value)
        source = inspect.getsourcefile(value)
        if module is None or source is None:
            raise InvariantError("runtime implementation source is unavailable")
        sources[module.__name__] = file_sha256(source)
    return object_sha256(sources)


def _activation_extraction_hash(config: ProjectConfig) -> str:
    adapter = adapter_for_config(config)
    return _implementation_hash(
        _collect_activation_statistics,
        ActivationCollector,
        adapter.activation_read_points,
        type(adapter),
    )


def _baseline_generation_hash(
    config: ProjectConfig,
    runtime: BaseModelRuntime,
    records: Sequence[PromptRecord],
) -> str:
    return object_sha256(
        {
            "prompt_plan": [dataclasses.asdict(record) for record in records],
            "checkpoint_checksum": runtime.checkpoint_checksum,
            "processor_fingerprints": runtime.adapter.processor_fingerprints(runtime.processor),
            "chat_template_hash": runtime.chat_template_hash,
            "target_generation_profile_hash": _target_generation_profile_hash(config, runtime),
            "target_implementation_hash": _implementation_hash(
                TargetTrajectoryGenerator,
                type(runtime.adapter),
                BaseModelRuntime,
            ),
            "seed": config.run.seed,
        }
    )


def _validate_checkpointed_trajectory(
    trajectory: TargetTrajectory,
    record: PromptRecord,
    config: ProjectConfig,
    generation_profile_hash: str,
) -> None:
    if (
        trajectory.prompt_id != record.prompt_id
        or trajectory.original_prompt != record.original_prompt
        or trajectory.split != record.split
        or trajectory.seed != config.run.seed
        or trajectory.model_revision != config.model.revision
        or trajectory.generation_config_hash != generation_profile_hash
    ):
        raise ArtifactError("checkpoint trajectory does not match its prompt or generation profile")
    values = dataclasses.asdict(trajectory)
    trajectory_hash = values.pop("trajectory_hash")
    if trajectory_hash != object_sha256(values):
        raise ArtifactError("checkpoint trajectory hash does not match its content")
    if trajectory.parser_status == "OK":
        if trajectory.error_code is not None or trajectory.error_detail is not None:
            raise ArtifactError("successful checkpoint trajectory contains an error")
    elif not trajectory.error_code or not trajectory.error_detail:
        raise ArtifactError("failed checkpoint trajectory has incomplete diagnostics")


def _fixed_batches[T](values: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _generate_target_batch(
    runtime: BaseModelRuntime | IntervenedModelRuntime,
    prompts: Sequence[PromptRecord],
    config: ProjectConfig,
    generation_profile_hash: str,
) -> list[TargetTrajectory]:
    trajectories = runtime.generate_targets(prompts)
    if len(trajectories) != len(prompts):
        raise InvariantError("target generation batch returned an unexpected number of trajectories")
    for trajectory, prompt in zip(trajectories, prompts, strict=True):
        _validate_checkpointed_trajectory(
            trajectory,
            prompt,
            config,
            generation_profile_hash,
        )
    return trajectories


def _validate_checkpointed_judgment(result: JudgeResult, trajectory: TargetTrajectory) -> None:
    if result.trajectory_hash != trajectory.trajectory_hash:
        raise ArtifactError("checkpoint judgment references a different trajectory")
    if result.status == "OK":
        if result.label not in {"REFUSAL", "NON_REFUSAL", "UNCERTAIN"} or result.error_code is not None:
            raise ArtifactError("successful checkpoint judgment is invalid")
    elif result.label is not None or not result.error_code:
        raise ArtifactError("failed checkpoint judgment is invalid")


def _checkpointed_target_trajectories(
    runtime: BaseModelRuntime | IntervenedModelRuntime,
    config: ProjectConfig,
    prompts: Sequence[PromptRecord],
    *,
    directory: Path,
    identity: str,
    desc: str,
) -> list[TargetTrajectory]:
    generation_profile_hash = _target_generation_profile_hash(config, runtime)
    prompt_keys = [prompt.prompt_id for prompt in prompts]
    with PrivateCheckpoint(
        directory,
        identity=identity,
        prompt_keys=prompt_keys,
    ) as checkpoint:
        entries = checkpoint.load()
        trajectory_by_ordinal: dict[int, TargetTrajectory] = {}
        for entry in entries:
            trajectory = TargetTrajectory.from_dict(entry.payload)
            _validate_checkpointed_trajectory(
                trajectory,
                prompts[entry.ordinal],
                config,
                generation_profile_hash,
            )
            trajectory_by_ordinal[entry.ordinal] = trajectory
        error_count = sum(TargetTrajectory.from_dict(entry.payload).parser_status == "ERROR" for entry in entries)
        progress = tqdm(
            total=len(prompts),
            initial=len(entries),
            desc=desc,
            unit="prompt",
            dynamic_ncols=True,
            disable=None,
        )
        progress.set_postfix(errors=error_count)
        try:
            for ordinals in _fixed_batches(range(len(prompts)), config.target_generation.batch_size):
                if all(ordinal in trajectory_by_ordinal for ordinal in ordinals):
                    continue
                batch_prompts = [prompts[ordinal] for ordinal in ordinals]
                trajectories = _generate_target_batch(
                    runtime,
                    batch_prompts,
                    config,
                    generation_profile_hash,
                )
                for ordinal, trajectory in zip(ordinals, trajectories, strict=True):
                    checkpointed = trajectory_by_ordinal.get(ordinal)
                    if checkpointed is not None and trajectory != checkpointed:
                        raise ArtifactError("regenerated target batch does not match its checkpoint")
                for ordinal, prompt, trajectory in zip(ordinals, batch_prompts, trajectories, strict=True):
                    if ordinal in trajectory_by_ordinal:
                        continue
                    checkpoint.write(ordinal, prompt.prompt_id, trajectory.as_dict())
                    trajectory_by_ordinal[ordinal] = trajectory
                    if trajectory.parser_status == "ERROR":
                        error_count += 1
                        progress.set_postfix(errors=error_count)
                    progress.update()
        finally:
            progress.close()
        return [TargetTrajectory.from_dict(entry.payload) for entry in checkpoint.require_complete()]


def _checkpointed_judgments(
    runtime: BaseModelRuntime,
    trajectories: Sequence[TargetTrajectory],
    *,
    directory: Path,
    identity: str,
    desc: str,
) -> list[JudgeResult]:
    prompt_keys = [trajectory.trajectory_hash for trajectory in trajectories]
    with PrivateCheckpoint(
        directory,
        identity=identity,
        prompt_keys=prompt_keys,
    ) as checkpoint:
        entries = checkpoint.load()
        for entry in entries:
            result = JudgeResult.from_dict(entry.payload)
            _validate_checkpointed_judgment(result, trajectories[entry.ordinal])
        completed = {entry.ordinal for entry in entries}
        error_count = sum(JudgeResult.from_dict(entry.payload).status == "ERROR" for entry in entries)
        judge = TrajectoryJudge(runtime.adapter, runtime.model, runtime.processor)
        progress = tqdm(
            total=len(trajectories),
            initial=len(entries),
            desc=desc,
            unit="trajectory",
            dynamic_ncols=True,
            disable=None,
        )
        progress.set_postfix(errors=error_count)
        try:
            for ordinal, trajectory in enumerate(trajectories):
                if ordinal in completed:
                    continue
                result = judge.classify(trajectory)
                _validate_checkpointed_judgment(result, trajectory)
                checkpoint.write(ordinal, trajectory.trajectory_hash, result.as_dict())
                if result.status == "ERROR":
                    error_count += 1
                    progress.set_postfix(errors=error_count)
                progress.update()
        finally:
            progress.close()
            del judge
        return [JudgeResult.from_dict(entry.payload) for entry in checkpoint.require_complete()]


def _remove_artifact(store: ArtifactStore, path: Path) -> None:
    path.unlink(missing_ok=True)
    store.metadata_path(path).unlink(missing_ok=True)


def _discard_checkpoint(directory: Path) -> None:
    try:
        (directory / "checkpoint.sqlite3-journal").unlink(missing_ok=True)
        (directory / "checkpoint.sqlite3").unlink(missing_ok=True)
        directory.rmdir()
    except OSError:
        return


def _directory_file_hashes(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactError(f"artifact bundle is not a directory: {directory}")
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ArtifactError(f"artifact bundle contains an unsupported entry: {path}")
        if path.is_file():
            result[str(path.relative_to(directory))] = file_sha256(path)
    return result


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError(f"{name} is invalid")
    return value


def _write_error_diagnostics(
    store: ArtifactStore,
    *,
    phase: str,
    total: int,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    artifact_type: str,
    profile: ArtifactProfile,
) -> None:
    if rows:
        store.write_jsonl(
            path,
            rows,
            artifact_type=artifact_type,
            profile=profile,
            private=True,
        )
        counts = Counter(str(row["error_code"]) for row in rows)
        summary = ", ".join(f"{code}={count}" for code, count in sorted(counts.items()))
        print(f"{phase}: {len(rows)}/{total} ERROR ({summary}); details: {path}", file=sys.stderr)
        for row in rows[:25]:
            fields = [
                f"prompt_id={row['prompt_id']}",
                f"error_code={row['error_code']}",
            ]
            for name in ("candidate_id", "kind", "generation_truncated"):
                if name in row:
                    fields.append(f"{name}={row[name]}")
            print(f"{phase}: {' '.join(fields)}", file=sys.stderr)
        if len(rows) > 25:
            print(f"{phase}: showing 25/{len(rows)} ERROR records", file=sys.stderr)
    else:
        _remove_artifact(store, path)


def _record_error_diagnostics(
    config: ProjectConfig,
    store: ArtifactStore,
    *,
    phase: str,
    total: int,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    artifact_type: str,
    profile: ArtifactProfile,
) -> None:
    _write_error_diagnostics(
        store,
        phase=phase,
        total=total,
        rows=rows,
        path=path,
        artifact_type=artifact_type,
        profile=profile,
    )
    _check_error_rate(len(rows), total, config, phase)


def _write_json_artifact(
    store: ArtifactStore,
    path: Path,
    value: Any,
    *,
    artifact_type: str,
    profile: ArtifactProfile,
    private: bool = False,
) -> None:
    store.write_json(path, value, private=private)
    metadata = ArtifactMetadata(
        artifact_type=artifact_type,
        private=private,
        content_sha256=file_sha256(path),
        profile=profile,
    )
    store.write_json(store.metadata_path(path), metadata.as_dict(), private=private)


def _read_json_artifact(
    store: ArtifactStore,
    path: Path,
    *,
    artifact_type: str,
    profile: ArtifactProfile,
) -> Any:
    store.validate(path, artifact_type=artifact_type, expected_profile=profile)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _move_inputs(inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def _validate_static_context_budget(config: ProjectConfig, runtime: BaseModelRuntime) -> None:
    judge_budget_probe = JudgeInput(
        original_prompt="",
        trajectory="\n",
        generation_truncated=False,
        input_hash="context-preflight",
    )
    judge_rendered = runtime.adapter.render_judge_chat(runtime.processor, judge_messages(judge_budget_probe))
    judge_ids = tuple(int(value) for value in judge_rendered["input_ids"][0].tolist())
    decoder = EnumTrieConstrainedDecoder.compile(
        runtime.processor.tokenizer,
        tuple(label.value for label in JudgeLabel),
        (judge_ids,),
    )
    context_window = runtime.adapter.context_window(runtime.model)
    judge_required = (
        len(judge_ids) + config.data.max_text_tokens + config.target_generation.max_new_tokens + decoder.max_new_tokens
    )
    target_rendered = runtime.adapter.render_target_chat(
        runtime.processor,
        target_messages("", config.target_generation.system_prompt),
        config=config.target_generation,
        prefill_thinking=True,
    )
    target_required = (
        int(target_rendered["input_ids"].shape[-1])
        + config.data.max_text_tokens
        + config.target_generation.max_new_tokens
    )
    if target_required > context_window:
        raise PipelineError(
            f"configured target generation budget is {target_required} tokens; context window is {context_window}"
        )
    if judge_required > context_window:
        raise PipelineError(
            f"configured full-trajectory judge budget is {judge_required} tokens; context window is {context_window}"
        )


def _baseline_generation_directory(store: ArtifactStore, generation_hash: str) -> Path:
    return store.paths.data / "baseline_generations" / generation_hash


def _read_baseline_generation_manifest(
    store: ArtifactStore,
    *,
    enforce_attempt: bool = True,
) -> tuple[dict[str, Any], ArtifactProfile, Path]:
    value = _read_json(store.paths.baseline_generation)
    required = {
        "profile",
        "prompt_splits_sha256",
        "test_prompts_sha256",
        "baseline_trajectories_sha256",
        "record_count",
        "error_count",
    }
    if not isinstance(value, dict) or set(value) != required or not isinstance(value["profile"], dict):
        raise ArtifactError("baseline generation manifest is invalid")
    try:
        profile = ArtifactProfile(**value["profile"])
    except TypeError as error:
        raise ArtifactError("baseline generation manifest profile is invalid") from error
    expected = _profile(
        store,
        stage="baseline_generation",
        target=True,
        chat_template_hash=profile.chat_template_hash,
        baseline_generation_hash=profile.baseline_generation_hash,
    )
    if profile != expected or not profile.chat_template_hash:
        raise ArtifactError("baseline generation manifest profile does not match this run")
    generation_hash = _require_sha256(profile.baseline_generation_hash, "baseline generation hash")
    if enforce_attempt:
        if not store.paths.baseline_generation_attempt.is_file():
            raise ArtifactError("active baseline generation has no matching attempt")
        attempt = _read_json(store.paths.baseline_generation_attempt)
        if attempt != {"generation_hash": generation_hash}:
            raise ArtifactError("active baseline generation does not match the latest attempt")
    directory = _baseline_generation_directory(store, generation_hash)
    splits_path = directory / "splits.private.jsonl"
    test_prompts_path = directory / "test_prompts.private.jsonl"
    trajectories_path = directory / "baseline_trajectories.private.jsonl"
    if type(value["record_count"]) is not int or value["record_count"] < 1:
        raise ArtifactError("baseline generation record count is invalid")
    if type(value["error_count"]) is not int or not 0 <= value["error_count"] <= value["record_count"]:
        raise ArtifactError("baseline generation error count is invalid")
    store.validate(
        splits_path,
        artifact_type="prompt_splits",
        expected_profile=_profile(store, stage="baseline_generation"),
    )
    store.validate(
        test_prompts_path,
        artifact_type="test_prompts",
        expected_profile=_profile(store, stage="baseline_generation"),
    )
    store.validate(
        trajectories_path,
        artifact_type="baseline_trajectories",
        expected_profile=profile,
    )
    if value["prompt_splits_sha256"] != file_sha256(splits_path):
        raise ArtifactError("baseline generation prompt split hash does not match")
    if value["test_prompts_sha256"] != file_sha256(test_prompts_path):
        raise ArtifactError("baseline generation test prompt hash does not match")
    if value["baseline_trajectories_sha256"] != file_sha256(trajectories_path):
        raise ArtifactError("baseline generation trajectory hash does not match")
    try:
        trajectory_rows = list(
            store.read_jsonl(
                trajectories_path,
                artifact_type="baseline_trajectories",
                expected_profile=profile,
            )
        )
        trajectory_errors = sum(TargetTrajectory.from_dict(row).parser_status == "ERROR" for row in trajectory_rows)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError("baseline generation trajectory rows are invalid") from error
    if len(trajectory_rows) != value["record_count"] or trajectory_errors != value["error_count"]:
        raise ArtifactError("baseline generation counts do not match its trajectories")
    return value, profile, directory


def _load_prompt_records(store: ArtifactStore, config: ProjectConfig) -> list[PromptRecord]:
    del config
    _, _, directory = _read_baseline_generation_manifest(store)
    rows = store.read_jsonl(
        directory / "splits.private.jsonl",
        artifact_type="prompt_splits",
        expected_profile=_profile(store, stage="baseline_generation"),
    )
    records = [PromptRecord.from_dict(row) for row in rows if row.get("split") in {"train", "validation"}]
    if not records:
        raise ArtifactError("development prompt split artifact is empty")
    return records


def _baseline_artifact_profile(
    store: ArtifactStore,
    chat_template_hash: str,
    *,
    enforce_error_rate: bool = True,
) -> ArtifactProfile:
    value, profile, _ = _read_baseline_generation_manifest(store)
    if profile.chat_template_hash != chat_template_hash:
        raise ArtifactError("baseline generation used a different chat template")
    if enforce_error_rate:
        _check_error_rate(
            value["error_count"],
            value["record_count"],
            store.config,
            "baseline trajectory generation",
        )
    return profile


def _load_baseline_trajectories(
    store: ArtifactStore,
    chat_template_hash: str,
    *,
    enforce_error_rate: bool = True,
) -> list[TargetTrajectory]:
    profile = _baseline_artifact_profile(
        store,
        chat_template_hash,
        enforce_error_rate=enforce_error_rate,
    )
    manifest, _, directory = _read_baseline_generation_manifest(store)
    rows = store.read_jsonl(
        directory / "baseline_trajectories.private.jsonl",
        artifact_type="baseline_trajectories",
        expected_profile=profile,
    )
    trajectories = [TargetTrajectory.from_dict(row) for row in rows]
    if len(trajectories) != manifest["record_count"]:
        raise ArtifactError("baseline trajectory count does not match its manifest")
    return trajectories


def _load_baseline_judgments(
    store: ArtifactStore,
    chat_template_hash: str,
    *,
    enforce_error_rate: bool = True,
) -> list[JudgeResult]:
    validation_hash = _require_judge_validation(store)
    baseline_profile = _baseline_artifact_profile(store, chat_template_hash)
    value = _read_json(store.paths.baseline_judgment)
    required = {"profile", "judgment_hash", "content_sha256", "record_count", "error_count"}
    if not isinstance(value, dict) or set(value) != required or not isinstance(value["profile"], dict):
        raise ArtifactError("baseline judgment manifest is invalid")
    try:
        profile = ArtifactProfile(**value["profile"])
    except TypeError as error:
        raise ArtifactError("baseline judgment manifest profile is invalid") from error
    judgment_hash = _require_sha256(value["judgment_hash"], "baseline judgment hash")
    expected_profile = _profile(
        store,
        stage="baseline_judgment",
        target=True,
        chat_template_hash=chat_template_hash,
        judge_profile_hash=current_judge_profile_hash(),
        judge_validation_hash=validation_hash,
        baseline_generation_hash=baseline_profile.baseline_generation_hash,
        baseline_judgment_hash=judgment_hash,
    )
    if profile != expected_profile:
        raise ArtifactError("baseline judgment manifest profile does not match this run")
    if not store.paths.baseline_judgment_attempt.is_file():
        raise ArtifactError("active baseline judgment has no matching attempt")
    attempt = _read_json(store.paths.baseline_judgment_attempt)
    if attempt != {"judgment_hash": judgment_hash}:
        raise ArtifactError("active baseline judgment does not match the latest attempt")
    path = store.paths.data / "baseline_judgment_generations" / judgment_hash / "judgments.jsonl"
    if type(value["record_count"]) is not int or value["record_count"] < 1:
        raise ArtifactError("baseline judgment record count is invalid")
    if type(value["error_count"]) is not int or not 0 <= value["error_count"] <= value["record_count"]:
        raise ArtifactError("baseline judgment error count is invalid")
    rows = store.read_jsonl(
        path,
        artifact_type="baseline_judgments",
        expected_profile=profile,
    )
    if value["content_sha256"] != file_sha256(path):
        raise ArtifactError("baseline judgment content hash does not match")
    judgments = [JudgeResult.from_dict(row) for row in rows]
    if len(judgments) != value["record_count"]:
        raise ArtifactError("baseline judgment count does not match its manifest")
    if sum(result.status == "ERROR" for result in judgments) != value["error_count"]:
        raise ArtifactError("baseline judgment error count does not match its results")
    if enforce_error_rate:
        _check_error_rate(value["error_count"], value["record_count"], store.config, "baseline trajectory judging")
    return judgments


def _load_labeled(
    store: ArtifactStore,
    chat_template_hash: str,
    split: Literal["train", "validation"],
) -> list[LabeledTrajectory]:
    validation_hash = _require_judge_validation(store)
    baseline_profile = _baseline_artifact_profile(store, chat_template_hash)
    _load_baseline_judgments(store, chat_template_hash)
    judgment_manifest = _read_json(store.paths.baseline_judgment)
    judgment_hash = _require_sha256(judgment_manifest.get("judgment_hash"), "baseline judgment hash")
    selection_hash = _label_selection_hash(store.config, judgment_hash)
    path = store.paths.labeled_train if split == "train" else store.paths.labeled_validation
    rows = store.read_jsonl(
        path,
        artifact_type=f"labeled_{split}",
        expected_profile=_profile(
            store,
            stage="label_selection",
            target=True,
            chat_template_hash=chat_template_hash,
            judge_profile_hash=current_judge_profile_hash(),
            judge_validation_hash=validation_hash,
            baseline_generation_hash=baseline_profile.baseline_generation_hash,
            baseline_judgment_hash=judgment_hash,
            label_selection_hash=selection_hash,
        ),
    )
    return [LabeledTrajectory(**row) for row in rows]


def _label_selection_hash(config: ProjectConfig, judgment_hash: str) -> str:
    return object_sha256(
        {
            "stage_config_hash": config.stage_config_hash("label_selection"),
            "baseline_judgment_hash": judgment_hash,
            "implementation_hash": _implementation_hash(
                _eligible_baseline_trajectory,
                _require_healthy_baseline_capacity,
                judge_baseline_trajectories,
            ),
        }
    )


def _eligible_baseline_trajectory(trajectory: TargetTrajectory) -> bool:
    return (
        trajectory.parser_status == "OK"
        and not trajectory.generation_truncated
        and bool(trajectory.final_answer.strip())
    )


def _require_healthy_baseline_capacity(
    trajectories: Sequence[TargetTrajectory],
    config: ProjectConfig,
) -> None:
    required = {
        "train": 2 * config.data.train_per_class,
        "validation": 2 * config.data.validation_per_class,
    }
    for split, minimum in required.items():
        available = sum(
            trajectory.split == split and _eligible_baseline_trajectory(trajectory) for trajectory in trajectories
        )
        if available < minimum:
            raise PipelineError(
                f"{split} has {available} complete non-empty baseline trajectories; "
                f"at least {minimum} are required for two refusal classes"
            )


def inspect_model(config_path: str) -> None:
    config, store = _load(config_path, require_judge_validation=False)
    with BaseModelRuntime(config) as runtime:
        _validate_static_context_budget(config, runtime)
        _target_generation_profile_hash(config, runtime)
        fingerprints = runtime.adapter.processor_fingerprints(runtime.processor)
        report = dataclasses.asdict(runtime.compatibility_report)
        if not report["errors"]:
            del report["errors"]
        report.update(
            {
                "model_id": config.model.id,
                "model_revision": config.model.revision,
                "model_profile_hash": config.stage_config_hash("judge_validation"),
                "chat_template_hash": runtime.chat_template_hash,
                **fingerprints,
            }
        )
        store.write_json(store.paths.root / "model_compatibility.json", report, private=False)
        environment_path = store.paths.root / "environment.json"
        environment = _read_json(environment_path)
        environment.update({"chat_template_hash": runtime.chat_template_hash, **fingerprints})
        store.write_json(environment_path, environment, private=False)


def generate_baseline_trajectories(config_path: str) -> None:
    config, store = _load(config_path)
    store.write_json(
        store.paths.baseline_generation_attempt,
        {
            "request_hash": object_sha256(
                {
                    "data": config.data,
                    "seed": config.run.seed,
                    "implementation_hash": _implementation_hash(ingest_and_split_prompts),
                }
            )
        },
        private=False,
    )
    trajectories: list[TargetTrajectory]
    generation_hash: str
    generation_profile_hash: str
    chat_template_hash: str
    active_generation = False
    with BaseModelRuntime(config) as runtime:
        _require_judge_validation(store, runtime)
        _validate_static_context_budget(config, runtime)
        tokenizer = runtime.processor.tokenizer
        records = ingest_and_split_prompts(
            config,
            token_counter=lambda prompt: len(tokenizer.encode(prompt, add_special_tokens=False)),
        )
        if not records:
            raise PipelineError("no prompts were ingested from data.prompt_files")
        grouped = records_by_split(records)
        if len(grouped["test"]) > config.data.max_test_prompts:
            retained_test = sorted(grouped["test"], key=lambda item: item.prompt_id)[: config.data.max_test_prompts]
            records = [item for item in records if item.split != "test"] + retained_test
        grouped = records_by_split(records)
        records = [*grouped["train"], *grouped["validation"], *grouped["test"]]
        baseline_records = [*grouped["train"], *grouped["validation"]]
        if not baseline_records:
            raise PipelineError("prompt split contains no discovery or validation records")
        generation_hash = _baseline_generation_hash(config, runtime, records)
        generation_profile_hash = _target_generation_profile_hash(config, runtime)
        chat_template_hash = runtime.chat_template_hash
        active_profile: ArtifactProfile | None = None
        if store.paths.baseline_generation.is_file():
            _, active_profile, _ = _read_baseline_generation_manifest(store, enforce_attempt=False)
            if active_profile.chat_template_hash != chat_template_hash:
                raise ArtifactError("active baseline generation used a different chat template")
        store.write_json(
            store.paths.baseline_generation_attempt,
            {"generation_hash": generation_hash},
            private=False,
        )
        if active_profile is not None and active_profile.baseline_generation_hash == generation_hash:
            trajectories = _load_baseline_trajectories(
                store,
                chat_template_hash,
                enforce_error_rate=False,
            )
            if len(trajectories) != len(baseline_records):
                raise ArtifactError("active baseline generation has incomplete prompt coverage")
            for record, trajectory in zip(baseline_records, trajectories, strict=True):
                _validate_checkpointed_trajectory(trajectory, record, config, generation_profile_hash)
            active_generation = True
        if not active_generation:
            checkpoint_directory = store.paths.baseline_generation_checkpoint / generation_hash
            trajectories = _checkpointed_target_trajectories(
                runtime,
                config,
                baseline_records,
                directory=checkpoint_directory,
                identity=generation_hash,
                desc="Generating baseline trajectories",
            )
    baseline_profile = _profile(
        store,
        stage="baseline_generation",
        target=True,
        chat_template_hash=chat_template_hash,
        baseline_generation_hash=generation_hash,
    )
    error_rows = [
        {
            "prompt_id": trajectory.prompt_id,
            "error_code": trajectory.error_code,
            "error_detail": trajectory.error_detail,
            "generation_truncated": trajectory.generation_truncated,
            "generated_token_count": len(trajectory.raw_generated_token_ids),
        }
        for trajectory in trajectories
        if trajectory.parser_status == "ERROR"
    ]
    _record_error_diagnostics(
        config,
        store,
        phase="baseline trajectory generation",
        total=len(trajectories),
        rows=error_rows,
        path=store.paths.baseline_generation_errors,
        artifact_type="baseline_generation_errors",
        profile=baseline_profile,
    )
    if active_generation:
        return
    bundle_parent = store.paths.data / "baseline_generations"
    bundle_parent.mkdir(parents=True, exist_ok=True)
    staging = bundle_parent / f".{generation_hash}.staging"
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():
            raise ArtifactError(f"baseline staging path is invalid: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    splits_path = staging / "splits.private.jsonl"
    test_prompts_path = staging / "test_prompts.private.jsonl"
    trajectories_path = staging / "baseline_trajectories.private.jsonl"
    store.write_jsonl(
        splits_path,
        records,
        artifact_type="prompt_splits",
        profile=_profile(store, stage="baseline_generation"),
        private=True,
    )
    store.write_jsonl(
        test_prompts_path,
        (record for record in records if record.split == "test"),
        artifact_type="test_prompts",
        profile=_profile(store, stage="baseline_generation"),
        private=True,
    )
    metadata = store.write_jsonl(
        trajectories_path,
        trajectories,
        artifact_type="baseline_trajectories",
        profile=baseline_profile,
        private=True,
    )
    final_directory = _baseline_generation_directory(store, generation_hash)
    if final_directory.exists():
        if _directory_file_hashes(final_directory) != _directory_file_hashes(staging):
            raise ArtifactError("existing immutable baseline generation bundle differs")
        shutil.rmtree(staging)
    else:
        staging.rename(final_directory)
    ArtifactStore._fsync_directory(bundle_parent)
    store.write_json(
        store.paths.baseline_generation,
        {
            "profile": baseline_profile.as_dict(),
            "prompt_splits_sha256": file_sha256(final_directory / "splits.private.jsonl"),
            "test_prompts_sha256": file_sha256(final_directory / "test_prompts.private.jsonl"),
            "baseline_trajectories_sha256": metadata.content_sha256,
            "record_count": len(trajectories),
            "error_count": len(error_rows),
        },
        private=False,
    )
    _discard_checkpoint(store.paths.baseline_generation_checkpoint / generation_hash)


def judge_baseline_trajectories(config_path: str) -> None:
    config, store = _load(config_path)
    baseline_manifest, baseline_profile, _ = _read_baseline_generation_manifest(store)
    store.write_json(
        store.paths.baseline_judgment_attempt,
        {
            "request_hash": object_sha256(
                {
                    "baseline_generation_hash": baseline_profile.baseline_generation_hash,
                    "baseline_trajectory_hash": baseline_manifest["baseline_trajectories_sha256"],
                    "judge_profile_hash": current_judge_profile_hash(),
                    "implementation_hash": _implementation_hash(
                        TrajectoryJudge,
                        EnumTrieConstrainedDecoder,
                    ),
                }
            )
        },
        private=False,
    )
    judgments: list[JudgeResult]
    judgment_hash: str
    judgment_profile: ArtifactProfile
    active_judgment = False
    with BaseModelRuntime(config) as runtime:
        validation_hash = _require_judge_validation(store, runtime)
        trajectories = _load_baseline_trajectories(store, runtime.chat_template_hash)
        _require_healthy_baseline_capacity(trajectories, config)
        baseline_profile = _baseline_artifact_profile(store, runtime.chat_template_hash)
        judgment_profile = _profile(
            store,
            stage="baseline_judgment",
            target=True,
            chat_template_hash=runtime.chat_template_hash,
            judge_profile_hash=current_judge_profile_hash(),
            judge_validation_hash=validation_hash,
            baseline_generation_hash=baseline_profile.baseline_generation_hash,
        )
        judgment_hash = object_sha256(
            {
                "baseline_generation_hash": baseline_profile.baseline_generation_hash,
                "trajectory_hashes": [trajectory.trajectory_hash for trajectory in trajectories],
                "checkpoint_checksum": runtime.checkpoint_checksum,
                "processor_fingerprints": runtime.adapter.processor_fingerprints(runtime.processor),
                "chat_template_hash": runtime.chat_template_hash,
                "judge_profile_hash": current_judge_profile_hash(),
                "judge_validation_hash": validation_hash,
                "judge_implementation_hash": _implementation_hash(
                    TrajectoryJudge,
                    EnumTrieConstrainedDecoder,
                    type(runtime.adapter),
                    BaseModelRuntime,
                ),
            }
        )
        judgment_profile = dataclasses.replace(
            judgment_profile,
            baseline_judgment_hash=judgment_hash,
        )
        store.write_json(
            store.paths.baseline_judgment_attempt,
            {"judgment_hash": judgment_hash},
            private=False,
        )
        if store.paths.baseline_judgment.is_file():
            manifest = _read_json(store.paths.baseline_judgment)
            if not isinstance(manifest, dict):
                raise ArtifactError("active baseline judgment manifest is invalid")
            if manifest.get("judgment_hash") == judgment_hash:
                judgments = _load_baseline_judgments(
                    store,
                    runtime.chat_template_hash,
                    enforce_error_rate=False,
                )
                if len(judgments) != len(trajectories):
                    raise ArtifactError("active baseline judgment has incomplete trajectory coverage")
                for result, trajectory in zip(judgments, trajectories, strict=True):
                    _validate_checkpointed_judgment(result, trajectory)
                active_judgment = True
        if not active_judgment:
            checkpoint_directory = store.paths.baseline_judgment_checkpoint / judgment_hash
            prompt_keys = [trajectory.trajectory_hash for trajectory in trajectories]
            with PrivateCheckpoint(
                checkpoint_directory,
                identity=judgment_hash,
                prompt_keys=prompt_keys,
            ) as checkpoint:
                entries = checkpoint.load()
                for entry in entries:
                    result = JudgeResult.from_dict(entry.payload)
                    _validate_checkpointed_judgment(result, trajectories[entry.ordinal])
                completed = {entry.ordinal for entry in entries}
                error_count = sum(JudgeResult.from_dict(entry.payload).status == "ERROR" for entry in entries)
                judge = TrajectoryJudge(runtime.adapter, runtime.model, runtime.processor)
                progress = tqdm(
                    total=len(trajectories),
                    initial=len(entries),
                    desc="Judging baseline trajectories",
                    unit="trajectory",
                    dynamic_ncols=True,
                    disable=None,
                )
                progress.set_postfix(errors=error_count)
                try:
                    for ordinal, trajectory in enumerate(trajectories):
                        if ordinal in completed:
                            continue
                        result = judge.classify(trajectory)
                        _validate_checkpointed_judgment(result, trajectory)
                        checkpoint.write(ordinal, trajectory.trajectory_hash, result.as_dict())
                        if result.status == "ERROR":
                            error_count += 1
                            progress.set_postfix(errors=error_count)
                        progress.update()
                finally:
                    progress.close()
                    del judge
                judgments = [JudgeResult.from_dict(entry.payload) for entry in checkpoint.require_complete()]
    trajectory_by_hash = {trajectory.trajectory_hash: trajectory for trajectory in trajectories}
    error_rows = [
        {
            "prompt_id": trajectory_by_hash[result.trajectory_hash].prompt_id,
            "trajectory_hash": result.trajectory_hash,
            "error_code": result.error_code,
        }
        for result in judgments
        if result.status == "ERROR"
    ]
    _record_error_diagnostics(
        config,
        store,
        phase="baseline trajectory judging",
        total=len(judgments),
        rows=error_rows,
        path=store.paths.baseline_judgment_errors,
        artifact_type="baseline_judgment_errors",
        profile=judgment_profile,
    )
    by_hash = {item.trajectory_hash: item for item in judgments}
    labeled: dict[str, list[LabeledTrajectory]] = {"train": [], "validation": []}
    for trajectory in trajectories:
        if not _eligible_baseline_trajectory(trajectory):
            continue
        result = by_hash[trajectory.trajectory_hash]
        if result.status != "OK" or result.label not in {"REFUSAL", "NON_REFUSAL"}:
            continue
        if trajectory.split not in labeled:
            continue
        labeled_split = cast(Literal["train", "validation"], trajectory.split)
        labeled[labeled_split].append(
            LabeledTrajectory(
                prompt_id=trajectory.prompt_id,
                label=result.label,
                trajectory_hash=trajectory.trajectory_hash,
            )
        )
    required = {
        "train": config.data.train_per_class,
        "validation": config.data.validation_per_class,
    }
    selected_by_split: dict[str, list[LabeledTrajectory]] = {}
    for split, values in labeled.items():
        selected: list[LabeledTrajectory] = []
        for label in ("REFUSAL", "NON_REFUSAL"):
            cohort = sorted((item for item in values if item.label == label), key=lambda item: item.prompt_id)
            if len(cohort) < required[split]:
                raise PipelineError(f"{split} has {len(cohort)} {label} trajectories; {required[split]} are required")
            selected.extend(cohort[: required[split]])
        selected_by_split[split] = sorted(selected, key=lambda item: item.prompt_id)
    if not active_judgment:
        bundle_parent = store.paths.data / "baseline_judgment_generations"
        bundle_parent.mkdir(parents=True, exist_ok=True)
        staging = bundle_parent / f".{judgment_hash}.staging"
        if staging.exists():
            if staging.is_symlink() or not staging.is_dir():
                raise ArtifactError(f"baseline judgment staging path is invalid: {staging}")
            shutil.rmtree(staging)
        staging.mkdir(mode=0o700)
        judgment_path = staging / "judgments.jsonl"
        metadata = store.write_jsonl(
            judgment_path,
            judgments,
            artifact_type="baseline_judgments",
            profile=judgment_profile,
            private=False,
        )
        final_directory = bundle_parent / judgment_hash
        if final_directory.exists():
            if _directory_file_hashes(final_directory) != _directory_file_hashes(staging):
                raise ArtifactError("existing immutable baseline judgment bundle differs")
            shutil.rmtree(staging)
        else:
            staging.rename(final_directory)
        ArtifactStore._fsync_directory(bundle_parent)
        store.write_json(
            store.paths.baseline_judgment,
            {
                "profile": judgment_profile.as_dict(),
                "judgment_hash": judgment_hash,
                "content_sha256": metadata.content_sha256,
                "record_count": len(judgments),
                "error_count": len(error_rows),
            },
            private=False,
        )
        _discard_checkpoint(store.paths.baseline_judgment_checkpoint / judgment_hash)
    label_profile = _profile(
        store,
        stage="label_selection",
        target=True,
        chat_template_hash=judgment_profile.chat_template_hash,
        judge_profile_hash=current_judge_profile_hash(),
        judge_validation_hash=judgment_profile.judge_validation_hash,
        baseline_generation_hash=judgment_profile.baseline_generation_hash,
        baseline_judgment_hash=judgment_hash,
        label_selection_hash=_label_selection_hash(config, judgment_hash),
    )
    for split, selected in selected_by_split.items():
        path = store.paths.labeled_train if split == "train" else store.paths.labeled_validation
        store.write_jsonl(
            path,
            selected,
            artifact_type=f"labeled_{split}",
            profile=label_profile,
            private=False,
        )


def collect_activations(config_path: str) -> None:
    config, store = _load(config_path)
    with BaseModelRuntime(config) as runtime:
        validation_hash = _require_judge_validation(store, runtime)
        baseline_profile = _baseline_artifact_profile(store, runtime.chat_template_hash)
        trajectories = _load_baseline_trajectories(store, runtime.chat_template_hash)
        labeled = _load_labeled(store, runtime.chat_template_hash, "train")
        judgment_manifest = _read_json(store.paths.baseline_judgment)
        judgment_hash = _require_sha256(judgment_manifest.get("judgment_hash"), "baseline judgment hash")
        selection_hash = _label_selection_hash(config, judgment_hash)
        trajectory_by_hash = {item.trajectory_hash: item for item in trajectories}
        statistics = _collect_activation_statistics(config, runtime, labeled, trajectory_by_hash)
        save_activation_statistics(store.paths.activation_statistics, statistics)
        activation_profile = _profile(
            store,
            stage="activation_extraction",
            target=True,
            chat_template_hash=runtime.chat_template_hash,
            judge_profile_hash=current_judge_profile_hash(),
            judge_validation_hash=validation_hash,
            baseline_generation_hash=baseline_profile.baseline_generation_hash,
            baseline_judgment_hash=judgment_hash,
            label_selection_hash=selection_hash,
            activation_extraction_hash=_activation_extraction_hash(config),
        )
        metadata = ArtifactMetadata(
            artifact_type="activation_statistics",
            private=True,
            content_sha256=file_sha256(store.paths.activation_statistics),
            profile=activation_profile,
        )
        store.write_json(
            store.metadata_path(store.paths.activation_statistics),
            metadata.as_dict(),
            private=True,
        )


def _collect_activation_statistics(
    config: ProjectConfig,
    runtime: BaseModelRuntime,
    labeled: Sequence[LabeledTrajectory],
    trajectory_by_hash: Mapping[str, TargetTrajectory],
):
    model = runtime.model
    layers = cast(Literal["all"] | Sequence[int], config.search.layers)
    collector = ActivationCollector(
        runtime.adapter.activation_read_points(model),
        layers=layers,
        dtype=config.search.accumulator_dtype,
    )
    for item in _progress(
        labeled,
        desc="Collecting activations",
        unit="trajectory",
    ):
        trajectory = trajectory_by_hash[item.trajectory_hash]
        rendered = runtime.adapter.render_target_chat(
            runtime.processor,
            target_messages(trajectory.original_prompt, config.target_generation.system_prompt),
            config=config.target_generation,
            prefill_thinking=True,
        )
        inputs = _move_inputs(dict(rendered), runtime.adapter.input_device(model))
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise InvariantError("activation input_ids must contain one sequence")
        if input_ids.shape[1] < 1:
            raise InvariantError("activation input sequence must be non-empty")
        with collector.capture([item.label]), torch.inference_mode():
            model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
    return collector.statistics()


def build_direction_candidates(config_path: str) -> None:
    config, store = _load(config_path)
    activation_profile = _activation_dependent_profile(store)
    adapter = adapter_for_config(config)
    processor = adapter.load_processor(config.model)
    _validate_activation_chat_profile(activation_profile, adapter.chat_template_hash(processor))
    del processor
    store.validate(
        store.paths.activation_statistics,
        artifact_type="activation_statistics",
        expected_profile=activation_profile,
    )
    statistics = load_activation_statistics(store.paths.activation_statistics)
    bundle = build_candidates(statistics, dtype="float32")
    ranking = rank_activation_screening(
        bundle,
        keep=max(len(bundle.candidates), 1),
        layer_count=statistics.layer_count,
    )
    if not ranking:
        raise PipelineError("activation screening produced no numerically valid direction candidates")
    profile = _derived_profile(
        store,
        activation_profile,
        stage="direction_construction",
        direction_construction_hash=_direction_construction_hash(),
    )
    write_candidate_artifacts(store, bundle, profile=profile)
    _write_json_artifact(
        store,
        store.paths.activation_screening_ranking,
        [candidate.candidate_id for candidate in ranking],
        artifact_type="activation_screening_ranking",
        profile=profile,
    )


def _direction_construction_hash() -> str:
    return _implementation_hash(build_candidates, rank_activation_screening)


def _candidate_evaluation_hash(config: ProjectConfig) -> str:
    return object_sha256(
        {
            "stage_config_hash": config.stage_config_hash("candidate_evaluation"),
            "implementation_hash": _implementation_hash(
                _screen_candidates_by_kl,
                _evaluate_candidates_phase,
                evaluate_behavior,
                mean_next_token_kl,
                completed_non_refusal_completion_inputs,
                compute_ce_loss,
                ce_evaluation_from_losses,
                CEEvaluation,
                type(adapter_for_config(config)),
            ),
        }
    )


def _derived_profile(
    store: ArtifactStore,
    upstream: ArtifactProfile,
    *,
    stage: ArtifactStage,
    direction_construction_hash: str | None = None,
    candidate_evaluation_hash: str | None = None,
    acceptance_policy_hash: str | None = None,
) -> ArtifactProfile:
    return _profile(
        store,
        stage=stage,
        target=upstream.target_generation_config_hash is not None,
        chat_template_hash=upstream.chat_template_hash,
        judge_profile_hash=upstream.judge_profile_hash,
        judge_fixture_hash=upstream.judge_fixture_hash,
        judge_validation_hash=upstream.judge_validation_hash,
        baseline_generation_hash=upstream.baseline_generation_hash,
        baseline_judgment_hash=upstream.baseline_judgment_hash,
        label_selection_hash=upstream.label_selection_hash,
        activation_extraction_hash=upstream.activation_extraction_hash,
        direction_construction_hash=(
            direction_construction_hash
            if direction_construction_hash is not None
            else upstream.direction_construction_hash
        ),
        candidate_evaluation_hash=(
            candidate_evaluation_hash if candidate_evaluation_hash is not None else upstream.candidate_evaluation_hash
        ),
        acceptance_policy_hash=acceptance_policy_hash,
    )


def _activation_dependent_profile(store: ArtifactStore) -> ArtifactProfile:
    validation_hash = _require_judge_validation(store)
    _, baseline_profile, _ = _read_baseline_generation_manifest(store)
    chat_template_hash = baseline_profile.chat_template_hash
    if chat_template_hash is None:
        raise ArtifactError("baseline generation has no chat-template profile")
    _baseline_artifact_profile(store, chat_template_hash)
    _load_baseline_judgments(store, chat_template_hash)
    judgment_manifest = _read_json(store.paths.baseline_judgment)
    judgment_hash = _require_sha256(judgment_manifest.get("judgment_hash"), "baseline judgment hash")
    selection_hash = _label_selection_hash(store.config, judgment_hash)
    metadata = store.validate(
        store.paths.activation_statistics,
        artifact_type="activation_statistics",
        expected_profile=_profile(
            store,
            stage="activation_extraction",
            target=True,
            chat_template_hash=chat_template_hash,
            judge_profile_hash=current_judge_profile_hash(),
            judge_validation_hash=validation_hash,
            baseline_generation_hash=baseline_profile.baseline_generation_hash,
            baseline_judgment_hash=judgment_hash,
            label_selection_hash=selection_hash,
            activation_extraction_hash=_activation_extraction_hash(store.config),
        ),
    )
    if (
        not metadata.private
        or metadata.profile.chat_template_hash is None
        or metadata.profile.baseline_generation_hash is None
        or metadata.profile.baseline_judgment_hash is None
        or metadata.profile.label_selection_hash is None
        or metadata.profile.activation_extraction_hash is None
    ):
        raise ArtifactError("activation statistics privacy or chat-template profile is invalid")
    return metadata.profile


def _direction_dependent_profile(store: ArtifactStore) -> ArtifactProfile:
    activation_profile = _activation_dependent_profile(store)
    profile = _derived_profile(
        store,
        activation_profile,
        stage="direction_construction",
        direction_construction_hash=_direction_construction_hash(),
    )
    store.validate(
        store.paths.candidates,
        artifact_type="direction_candidates",
        expected_profile=profile,
    )
    return profile


def _selection_dependent_profile(store: ArtifactStore) -> ArtifactProfile:
    direction_profile = _direction_dependent_profile(store)
    evaluation_profile = _derived_profile(
        store,
        direction_profile,
        stage="candidate_evaluation",
        candidate_evaluation_hash=_candidate_evaluation_hash(store.config),
    )
    profile = _derived_profile(
        store,
        evaluation_profile,
        stage="candidate_selection",
        acceptance_policy_hash=store.config.acceptance_policy_hash,
    )
    store.validate(
        store.paths.selected_direction,
        artifact_type="selected_direction",
        expected_profile=profile,
    )
    return profile


def _validate_activation_chat_profile(profile: ArtifactProfile, chat_template_hash: str) -> None:
    if (
        profile.chat_template_hash != chat_template_hash
        or profile.judge_profile_hash != current_judge_profile_hash()
        or profile.judge_validation_hash is None
    ):
        raise ArtifactError("activation-derived artifacts use a different runtime profile")


def evaluate_candidates(config_path: str) -> None:
    config, store = _load(config_path)
    direction_profile = _direction_dependent_profile(store)
    profile = _derived_profile(
        store,
        direction_profile,
        stage="candidate_evaluation",
        candidate_evaluation_hash=_candidate_evaluation_hash(config),
    )
    selection_profile = _derived_profile(
        store,
        profile,
        stage="candidate_selection",
        acceptance_policy_hash=config.acceptance_policy_hash,
    )
    bundle = load_candidate_artifacts(store, expected_profile=direction_profile)
    screening = _read_json_artifact(
        store,
        store.paths.activation_screening_ranking,
        artifact_type="activation_screening_ranking",
        profile=direction_profile,
    )
    if not isinstance(screening, list) or any(not isinstance(candidate_id, str) for candidate_id in screening):
        raise ArtifactError("activation screening ranking is invalid")
    by_id = {item.candidate_id: item for item in bundle.candidates}
    try:
        ranking = [by_id[candidate_id] for candidate_id in screening]
    except KeyError as error:
        raise ArtifactError("activation screening ranking references an unknown candidate") from error
    with BaseModelRuntime(config) as profile_runtime:
        _require_judge_validation(store, profile_runtime)
        chat_template_hash = profile_runtime.chat_template_hash
        _validate_activation_chat_profile(profile, chat_template_hash)
    validation = _load_labeled(store, chat_template_hash, "validation")
    trajectories = _load_baseline_trajectories(store, chat_template_hash)
    baseline_trajectories = {item.prompt_id: item for item in trajectories}
    records = {item.prompt_id: item for item in _load_prompt_records(store, config)}
    baseline_prompt_plan = [records[item.prompt_id] for item in trajectories]
    pilot_labels = _balanced_subset(validation, config.search.pilot_prompts_per_class)
    kl_prompts = [records[item.prompt_id] for item in pilot_labels if item.label == "NON_REFUSAL"]
    ranking, screened_mean_kl, kl_screening_identity = _screen_candidates_by_kl(
        config,
        store,
        bundle,
        ranking,
        kl_prompts,
        artifact_profile=profile,
    )
    pilot_metrics, pilot_evaluation_identity = _evaluate_candidates_phase(
        config,
        store,
        bundle,
        ranking,
        pilot_labels,
        records,
        baseline_trajectories,
        baseline_prompt_plan,
        artifact_profile=profile,
        results_profile=selection_profile,
        evaluation_phase="pilot_evaluation",
        screened_mean_kl=screened_mean_kl,
    )
    pilot_eligible = [item for item in pilot_metrics if item.hard_filter_passed]
    if not pilot_eligible:
        raise PipelineError(
            "no pilot evaluation candidate passed causal and quality screening; "
            + _candidate_failure_summary(pilot_metrics, store.paths.pilot_evaluation_results)
        )
    full_validation_ids = [
        item.candidate_id
        for item in sorted(pilot_eligible, key=_pilot_ranking_key, reverse=True)[: config.search.pilot_evaluation_keep]
    ]
    full_validation_candidates = [by_id[candidate_id] for candidate_id in full_validation_ids]
    full_validation_metrics, full_validation_identity = _evaluate_candidates_phase(
        config,
        store,
        bundle,
        full_validation_candidates,
        validation,
        records,
        baseline_trajectories,
        baseline_prompt_plan,
        artifact_profile=profile,
        results_profile=selection_profile,
        evaluation_phase="full_validation",
        screened_mean_kl=None,
    )
    try:
        selected_metrics, selected_candidate = select_candidate(full_validation_metrics, by_id)
    except PipelineError:
        raise PipelineError(
            "no full validation candidate passed the configured hard filters; "
            + _candidate_failure_summary(full_validation_metrics, store.paths.full_validation_results)
        ) from None
    selected_direction = bundle.direction(selected_candidate)
    selected_metadata = {
        "candidate": dataclasses.asdict(selected_candidate),
        "full_validation_metrics": metrics_dict(selected_metrics),
    }
    save_direction(store.paths.selected_direction, selected_direction, metadata=selected_metadata, private=False)
    direction_sidecar = ArtifactMetadata(
        artifact_type="selected_direction",
        private=False,
        content_sha256=file_sha256(store.paths.selected_direction),
        profile=selection_profile,
    )
    store.write_json(
        store.metadata_path(store.paths.selected_direction),
        direction_sidecar.as_dict(),
        private=False,
    )
    _write_json_artifact(
        store,
        store.paths.final_selection,
        {
            "candidate": dataclasses.asdict(selected_candidate),
            "metrics": metrics_dict(selected_metrics),
            "max_error_rate": config.acceptance.max_error_rate,
            "kl_screening_identity": kl_screening_identity,
            "pilot_evaluation_identity": pilot_evaluation_identity,
            "full_validation_identity": full_validation_identity,
        },
        artifact_type="final_selection",
        profile=selection_profile,
        private=False,
    )
    judgments = _load_baseline_judgments(store, chat_template_hash)
    report = {
        "base_model_id": config.model.id,
        "base_revision": config.model.revision,
        "baseline_counts": judgment_counts(judgments),
        "baseline_parser": parser_statistics(trajectories),
        "candidate_search": {
            "layers": sorted({item.layer for item in bundle.candidates}),
            "activation_screening_count": len(ranking),
            "full_validation_count": len(full_validation_candidates),
        },
        "selected_candidate": selected_metadata,
    }
    _write_json_artifact(
        store,
        store.paths.full_validation_report,
        report,
        artifact_type="full_validation_report",
        profile=selection_profile,
    )


def _balanced_subset(values: Sequence[LabeledTrajectory], per_class: int) -> list[LabeledTrajectory]:
    selected: list[LabeledTrajectory] = []
    for label in ("REFUSAL", "NON_REFUSAL"):
        cohort = sorted((item for item in values if item.label == label), key=lambda item: item.prompt_id)
        selected.extend(cohort[: min(per_class, len(cohort))])
    return selected


def _candidate_failure_summary(metrics: Sequence[CandidateMetrics], results_path: Path) -> str:
    rejection_counts = Counter(reason for item in metrics for reason in item.rejection_reasons)
    rejection_summary = ", ".join(f"{reason}={count}" for reason, count in sorted(rejection_counts.items()))
    best_removal = max((item.removal_success_rate for item in metrics), default=0.0)
    best_induction = max((item.activation_addition_induction_rate or 0.0 for item in metrics), default=0.0)
    return (
        f"best removal={best_removal:.2%}, best addition induction={best_induction:.2%}; "
        f"rejection counts: {rejection_summary}; results: {results_path}"
    )


def _pilot_ranking_key(metrics: CandidateMetrics) -> tuple[float, ...]:
    return (
        metrics.removal_success_rate,
        metrics.activation_addition_induction_rate or 0.0,
        metrics.non_refusal_retention_rate,
        -metrics.uncertain_rate,
        -(metrics.mean_kl if metrics.mean_kl is not None else math.inf),
        -(metrics.completion_ce.loss_delta if metrics.completion_ce.loss_delta is not None else math.inf),
    )


def _select_kl_screened_candidates(
    candidates: Sequence[DirectionCandidate],
    mean_kl_by_candidate: Mapping[str, float | None],
    *,
    maximum: float,
    keep: int,
) -> list[DirectionCandidate]:
    return [
        candidate
        for candidate in candidates
        if mean_kl_by_candidate[candidate.candidate_id] is not None
        and math.isfinite(cast(float, mean_kl_by_candidate[candidate.candidate_id]))
        and cast(float, mean_kl_by_candidate[candidate.candidate_id]) <= maximum
    ][:keep]


def _candidate_kl_screen_identity(
    config: ProjectConfig,
    bundle: CandidateBundle,
    candidates: Sequence[DirectionCandidate],
    prompts: Sequence[PromptRecord],
    artifact_profile: ArtifactProfile,
) -> str:
    return object_sha256(
        {
            "profile": artifact_profile.as_dict(),
            "target_generation_config_hash": config.target_generation_config_hash,
            "candidates": [
                {
                    "metadata": dataclasses.asdict(candidate),
                    "direction_sha256": tensor_sha256(bundle.direction(candidate)),
                }
                for candidate in candidates
            ],
            "prompts": [dataclasses.asdict(prompt) for prompt in prompts],
            "implementation_hash": _implementation_hash(
                _next_token_logits,
                mean_next_token_kl,
                WeightEditPlan,
                type(adapter_for_config(config)),
                IntervenedModelRuntime,
            ),
        }
    )


def _validate_candidate_kl_row(
    row: Mapping[str, Any],
    candidate: DirectionCandidate,
) -> float | None:
    if set(row) != {"candidate_id", "mean_kl"} or row["candidate_id"] != candidate.candidate_id:
        raise ArtifactError("candidate KL screening checkpoint identity is invalid")
    mean_kl = row["mean_kl"]
    if mean_kl is None:
        return None
    if isinstance(mean_kl, bool) or not isinstance(mean_kl, int | float) or not math.isfinite(mean_kl):
        raise ArtifactError("candidate KL screening checkpoint value is invalid")
    return float(mean_kl)


def _screen_candidates_by_kl(
    config: ProjectConfig,
    store: ArtifactStore,
    bundle: CandidateBundle,
    candidates: Sequence[DirectionCandidate],
    prompts: Sequence[PromptRecord],
    *,
    artifact_profile: ArtifactProfile,
) -> tuple[list[DirectionCandidate], dict[str, float | None], str]:
    if not candidates or not prompts:
        raise PipelineError("candidate KL screening requires candidates and non-refusal validation prompts")
    identity = _candidate_kl_screen_identity(config, bundle, candidates, prompts, artifact_profile)
    checkpoint_root = store.paths.directions / ".candidate-evaluation-checkpoints" / "kl_screening" / identity
    with PrivateCheckpoint(
        checkpoint_root,
        identity=identity,
        prompt_keys=[candidate.candidate_id for candidate in candidates],
    ) as checkpoint:
        entries = checkpoint.load()
        mean_kl_by_candidate = {
            candidates[entry.ordinal].candidate_id: _validate_candidate_kl_row(
                entry.payload,
                candidates[entry.ordinal],
            )
            for entry in entries
        }
        if len(entries) < len(candidates):
            with IntervenedModelRuntime(config) as runtime:
                _validate_activation_chat_profile(artifact_profile, runtime.chat_template_hash)
                base_logits = [
                    _next_token_logits(runtime, prompt)
                    for prompt in _progress(
                        prompts,
                        desc="KL screening: baseline logits",
                        unit="prompt",
                    )
                ]
                progress = tqdm(
                    total=len(candidates),
                    initial=len(entries),
                    desc="KL screening: candidate directions",
                    unit="candidate",
                    dynamic_ncols=True,
                    disable=None,
                )
                try:
                    completed = {entry.ordinal for entry in entries}
                    for ordinal, candidate in enumerate(candidates):
                        if ordinal in completed:
                            continue
                        direction = bundle.direction(candidate)
                        plan = runtime.adapter.build_weight_edit_plan(runtime.model, direction)
                        with plan.temporary(runtime.model):
                            intervention_logits = [_next_token_logits(runtime, prompt) for prompt in prompts]
                        measured_kl = mean_next_token_kl(base_logits, intervention_logits)
                        mean_kl = measured_kl if math.isfinite(measured_kl) else None
                        checkpoint.write(
                            ordinal,
                            candidate.candidate_id,
                            {"candidate_id": candidate.candidate_id, "mean_kl": mean_kl},
                        )
                        mean_kl_by_candidate[candidate.candidate_id] = mean_kl
                        progress.update()
                finally:
                    progress.close()
        rows = [dict(entry.payload) for entry in checkpoint.require_complete()]
    store.write_jsonl(
        store.paths.directions / "kl_screening_results.jsonl",
        rows,
        artifact_type="kl_screening_results",
        profile=artifact_profile,
        private=False,
    )
    selected = _select_kl_screened_candidates(
        candidates,
        mean_kl_by_candidate,
        maximum=config.search.max_screening_kl,
        keep=config.search.activation_screening_keep,
    )
    if not selected:
        best = min((value for value in mean_kl_by_candidate.values() if value is not None), default=math.inf)
        raise PipelineError(
            "no direction candidate passed next-token KL screening; "
            f"best mean KL={best:.6g}, maximum={config.search.max_screening_kl:.6g}; "
            f"results: {store.paths.directions / 'kl_screening_results.jsonl'}"
        )
    return selected, mean_kl_by_candidate, identity


def _candidate_phase_identity(
    config: ProjectConfig,
    bundle: CandidateBundle,
    candidates: Sequence[DirectionCandidate],
    labeled: Sequence[LabeledTrajectory],
    prompt_records: Mapping[str, PromptRecord],
    baseline_trajectories: Mapping[str, TargetTrajectory],
    baseline_prompt_plan: Sequence[PromptRecord],
    artifact_profile: ArtifactProfile,
    evaluation_phase: str,
    screened_mean_kl: Mapping[str, float | None] | None,
) -> str:
    return object_sha256(
        {
            "phase": evaluation_phase,
            "profile": artifact_profile.as_dict(),
            "target_generation_config_hash": config.target_generation_config_hash,
            "candidates": [
                {
                    "metadata": dataclasses.asdict(candidate),
                    "direction_sha256": tensor_sha256(bundle.direction(candidate)),
                }
                for candidate in candidates
            ],
            "labeled": [dataclasses.asdict(item) for item in labeled],
            "prompts": [dataclasses.asdict(prompt_records[item.prompt_id]) for item in labeled],
            "baseline_trajectory_hashes": {
                item.prompt_id: baseline_trajectories[item.prompt_id].trajectory_hash for item in labeled
            },
            "baseline_batch_plan": [
                [dataclasses.asdict(prompt) for prompt in batch]
                for batch in _fixed_batches(baseline_prompt_plan, config.target_generation.batch_size)
            ],
            "completion_ce_trajectory_hashes": [
                baseline_trajectories[item.prompt_id].trajectory_hash for item in labeled if item.label == "NON_REFUSAL"
            ],
            "activation_addition_beta": config.acceptance.activation_addition_beta,
            "screened_mean_kl": dict(screened_mean_kl) if screened_mean_kl is not None else None,
            "implementation_hash": _implementation_hash(
                _candidate_phase_identity,
                evaluate_behavior,
                mean_next_token_kl,
                completed_non_refusal_completion_inputs,
                compute_ce_loss,
                ce_evaluation_from_losses,
                CEEvaluation,
                WeightEditPlan,
                TargetTrajectoryGenerator,
                type(adapter_for_config(config)),
                IntervenedModelRuntime,
                BaseModelRuntime,
            ),
        }
    )


def _candidate_trajectory_key(candidate_id: str, kind: str, prompt_id: str) -> str:
    return object_sha256(
        {
            "candidate_id": candidate_id,
            "kind": kind,
            "prompt_id": prompt_id,
        }
    )


def _validate_candidate_trajectory_row(
    row: Mapping[str, Any],
    *,
    candidate: DirectionCandidate,
    kind: Literal["removal", "addition"],
    prompt: PromptRecord,
    config: ProjectConfig,
    generation_profile_hash: str,
) -> TargetTrajectory:
    if set(row) != {"candidate_id", "kind", "trajectory"}:
        raise ArtifactError("candidate trajectory checkpoint row is invalid")
    if row["candidate_id"] != candidate.candidate_id or row["kind"] != kind:
        raise ArtifactError("candidate trajectory checkpoint identity is invalid")
    trajectory_value = row["trajectory"]
    if not isinstance(trajectory_value, dict):
        raise ArtifactError("candidate trajectory checkpoint payload is invalid")
    trajectory = TargetTrajectory.from_dict(trajectory_value)
    _validate_checkpointed_trajectory(trajectory, prompt, config, generation_profile_hash)
    return trajectory


def _validate_candidate_quality_row(
    row: Mapping[str, Any],
    candidate: DirectionCandidate,
) -> tuple[float | None, CEEvaluation]:
    if set(row) != {"candidate_id", "mean_kl", "completion_ce"}:
        raise ArtifactError("candidate quality checkpoint row is invalid")
    if row["candidate_id"] != candidate.candidate_id:
        raise ArtifactError("candidate quality checkpoint identity is invalid")
    mean_kl = row["mean_kl"]
    completion_ce = row["completion_ce"]
    if mean_kl is not None and (
        isinstance(mean_kl, bool) or not isinstance(mean_kl, int | float) or not math.isfinite(mean_kl)
    ):
        raise ArtifactError("candidate quality checkpoint values are invalid")
    if not isinstance(completion_ce, dict):
        raise ArtifactError("candidate quality checkpoint values are invalid")
    return (
        float(mean_kl) if mean_kl is not None else None,
        CEEvaluation.from_dict(completion_ce),
    )


def _candidate_reference_ce_identity(
    config: ProjectConfig,
    bundle: CandidateBundle,
    candidates: Sequence[DirectionCandidate],
    artifact_profile: ArtifactProfile,
    evaluation_phase: str,
) -> str:
    return object_sha256(
        {
            "phase": evaluation_phase,
            "profile": artifact_profile.as_dict(),
            "reference_files": [
                {
                    "path": str(Path(path).resolve()),
                    "content_sha256": file_sha256(Path(path).resolve()),
                }
                for path in config.data.reference_files
            ],
            "max_text_tokens": config.data.max_text_tokens,
            "candidates": [
                {
                    "metadata": dataclasses.asdict(candidate),
                    "direction_sha256": tensor_sha256(bundle.direction(candidate)),
                }
                for candidate in candidates
            ],
            "implementation_hash": _implementation_hash(
                _candidate_reference_ce_identity,
                ingest_texts,
                raw_text_ce_inputs,
                compute_ce_loss,
                ce_evaluation_from_losses,
                CEEvaluation,
                WeightEditPlan,
                type(adapter_for_config(config)),
                IntervenedModelRuntime,
            ),
        }
    )


def _validate_candidate_reference_ce_row(
    row: Mapping[str, Any],
    candidate: DirectionCandidate,
) -> CEEvaluation:
    if set(row) != {"candidate_id", "reference_ce"} or row["candidate_id"] != candidate.candidate_id:
        raise ArtifactError("candidate reference CE checkpoint row is invalid")
    reference_ce = row["reference_ce"]
    if not isinstance(reference_ce, dict):
        raise ArtifactError("candidate reference CE checkpoint value is invalid")
    result = CEEvaluation.from_dict(reference_ce)
    if result.source != "reference_files":
        raise ArtifactError("candidate reference CE checkpoint source is invalid")
    return result


def _evaluate_candidate_reference_ce(
    config: ProjectConfig,
    store: ArtifactStore,
    bundle: CandidateBundle,
    candidates: Sequence[DirectionCandidate],
    *,
    artifact_profile: ArtifactProfile,
    evaluation_phase: Literal["pilot_evaluation", "full_validation"],
) -> None:
    results_path = store.paths.directions / f"{evaluation_phase}_reference_ce_results.jsonl"
    identity = _candidate_reference_ce_identity(
        config,
        bundle,
        candidates,
        artifact_profile,
        evaluation_phase,
    )
    diagnostic_profile = dataclasses.replace(artifact_profile, reference_ce_hash=identity)
    if not config.data.reference_files:
        store.write_jsonl(
            results_path,
            (),
            artifact_type=f"{evaluation_phase}_reference_ce_results",
            profile=diagnostic_profile,
            private=False,
        )
        return
    checkpoint_directory = (
        store.paths.directions / ".candidate-evaluation-checkpoints" / evaluation_phase / "reference_ce" / identity
    )
    with PrivateCheckpoint(
        checkpoint_directory,
        identity=identity,
        prompt_keys=[candidate.candidate_id for candidate in candidates],
    ) as checkpoint:
        entries = checkpoint.load()
        results = {
            candidates[entry.ordinal].candidate_id: _validate_candidate_reference_ce_row(
                entry.payload,
                candidates[entry.ordinal],
            )
            for entry in entries
        }
        if len(entries) < len(candidates):
            with IntervenedModelRuntime(config) as runtime:
                _validate_activation_chat_profile(artifact_profile, runtime.chat_template_hash)
                inputs = _reference_ce_inputs(config, runtime)
                baseline = compute_ce_loss(
                    runtime,
                    inputs,
                    desc=f"{evaluation_phase}: baseline reference CE",
                )
                progress = tqdm(
                    total=len(candidates),
                    initial=len(entries),
                    desc=f"{evaluation_phase}: reference CE",
                    unit="candidate",
                    dynamic_ncols=True,
                    disable=None,
                )
                try:
                    for ordinal, candidate in enumerate(candidates):
                        if candidate.candidate_id in results:
                            continue
                        direction = bundle.direction(candidate)
                        plan = runtime.adapter.build_weight_edit_plan(runtime.model, direction)
                        try:
                            with plan.temporary(runtime.model):
                                intervention = compute_ce_loss(
                                    runtime,
                                    inputs,
                                    desc=f"{evaluation_phase}: intervention reference CE",
                                    leave=False,
                                )
                        except NonFiniteMetricError:
                            intervention = None
                        evaluation = ce_evaluation_from_losses(
                            baseline,
                            intervention,
                            source="reference_files",
                            input_count=len(inputs),
                        )
                        checkpoint.write(
                            ordinal,
                            candidate.candidate_id,
                            {
                                "candidate_id": candidate.candidate_id,
                                "reference_ce": evaluation.as_dict(),
                            },
                        )
                        results[candidate.candidate_id] = evaluation
                        progress.update()
                finally:
                    progress.close()
        rows = [dict(entry.payload) for entry in checkpoint.require_complete()]
    store.write_jsonl(
        results_path,
        rows,
        artifact_type=f"{evaluation_phase}_reference_ce_results",
        profile=diagnostic_profile,
        private=False,
    )


def _candidate_addition_error_rate(
    candidate_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> float | None:
    additions = [row for row in rows if row["candidate_id"] == candidate_id and row["kind"] == "addition"]
    if not additions:
        return None
    errors = sum(row["judgment"]["status"] == "ERROR" for row in additions)
    return errors / len(additions)


def _evaluate_candidates_phase(
    config: ProjectConfig,
    store: ArtifactStore,
    bundle: CandidateBundle,
    candidates: Sequence[DirectionCandidate],
    labeled: Sequence[LabeledTrajectory],
    prompt_records: Mapping[str, PromptRecord],
    baseline_trajectories: Mapping[str, TargetTrajectory],
    baseline_prompt_plan: Sequence[PromptRecord],
    *,
    artifact_profile: ArtifactProfile,
    results_profile: ArtifactProfile,
    evaluation_phase: Literal["pilot_evaluation", "full_validation"],
    screened_mean_kl: Mapping[str, float | None] | None,
) -> tuple[list[CandidateMetrics], str]:
    phase_desc = "Pilot evaluation" if evaluation_phase == "pilot_evaluation" else "Full validation"
    labels = {item.prompt_id: item.label for item in labeled}
    prompts = [prompt_records[item.prompt_id] for item in labeled]
    non_refusal_prompts = [prompt_records[item.prompt_id] for item in labeled if item.label == "NON_REFUSAL"]
    baseline_prompt_ids = [prompt.prompt_id for prompt in baseline_prompt_plan]
    if len(set(baseline_prompt_ids)) != len(baseline_prompt_ids) or set(baseline_prompt_ids) != set(
        baseline_trajectories
    ):
        raise ArtifactError("candidate generation baseline batch plan is invalid")
    baseline_batches = tuple(
        tuple(batch) for batch in _fixed_batches(baseline_prompt_plan, config.target_generation.batch_size)
    )
    baseline_batch_index = {
        prompt.prompt_id: batch_index for batch_index, batch in enumerate(baseline_batches) for prompt in batch
    }
    phase_identity = _candidate_phase_identity(
        config,
        bundle,
        candidates,
        labeled,
        prompt_records,
        baseline_trajectories,
        baseline_prompt_plan,
        artifact_profile,
        evaluation_phase,
        screened_mean_kl,
    )
    store.write_json(
        store.paths.directions / f"{evaluation_phase}_attempt.json",
        {"evaluation_identity": phase_identity},
        private=False,
    )
    checkpoint_root = store.paths.directions / ".candidate-evaluation-checkpoints" / evaluation_phase / phase_identity
    generation_plan: list[tuple[DirectionCandidate, Literal["removal", "addition"], PromptRecord]] = []
    for candidate in candidates:
        generation_plan.extend((candidate, "removal", prompt) for prompt in prompts)
        generation_plan.extend((candidate, "addition", prompt) for prompt in non_refusal_prompts)
    generation_keys = [
        _candidate_trajectory_key(candidate.candidate_id, kind, prompt.prompt_id)
        for candidate, kind, prompt in generation_plan
    ]
    generation_profile_hash = next(iter(baseline_trajectories.values())).generation_config_hash
    quality: dict[str, tuple[float | None, CEEvaluation]] = {}
    with (
        PrivateCheckpoint(
            checkpoint_root / "trajectories",
            identity=phase_identity,
            prompt_keys=generation_keys,
        ) as generation_checkpoint,
        PrivateCheckpoint(
            checkpoint_root / "quality",
            identity=phase_identity,
            prompt_keys=[candidate.candidate_id for candidate in candidates],
        ) as quality_checkpoint,
    ):
        generation_entries = generation_checkpoint.load()
        trajectory_by_ordinal: dict[int, TargetTrajectory] = {}
        for entry in generation_entries:
            candidate, kind, prompt = generation_plan[entry.ordinal]
            trajectory_by_ordinal[entry.ordinal] = _validate_candidate_trajectory_row(
                entry.payload,
                candidate=candidate,
                kind=kind,
                prompt=prompt,
                config=config,
                generation_profile_hash=generation_profile_hash,
            )
        quality_entries = quality_checkpoint.load()
        for entry in quality_entries:
            candidate = candidates[entry.ordinal]
            mean_kl, completion_ce = _validate_candidate_quality_row(entry.payload, candidate)
            quality[candidate.candidate_id] = (mean_kl, completion_ce)
        generation_progress = tqdm(
            total=len(generation_plan),
            initial=len(generation_entries),
            desc=f"{phase_desc}: generating trajectories",
            unit="trajectory",
            dynamic_ncols=True,
            disable=None,
        )
        generation_error_count = sum(
            trajectory.parser_status == "ERROR" for trajectory in trajectory_by_ordinal.values()
        )
        generation_progress.set_postfix(errors=generation_error_count)
        quality_progress = tqdm(
            total=len(candidates),
            initial=len(quality_entries),
            desc=f"{phase_desc}: computing quality",
            unit="candidate",
            dynamic_ncols=True,
            disable=None,
        )
        try:
            if len(generation_entries) < len(generation_plan) or len(quality_entries) < len(candidates):
                with IntervenedModelRuntime(config) as runtime:
                    _validate_activation_chat_profile(artifact_profile, runtime.chat_template_hash)

                    def generate_batches(ordinals: Sequence[int]) -> None:
                        nonlocal generation_error_count
                        target_ordinal_by_prompt_id: dict[str, int] = {}
                        for ordinal in ordinals:
                            prompt_id = generation_plan[ordinal][2].prompt_id
                            if prompt_id in target_ordinal_by_prompt_id:
                                raise InvariantError("candidate generation plan contains a duplicate prompt")
                            target_ordinal_by_prompt_id[prompt_id] = ordinal
                        try:
                            batch_indices = sorted(
                                {baseline_batch_index[prompt_id] for prompt_id in target_ordinal_by_prompt_id}
                            )
                        except KeyError as error:
                            raise ArtifactError("candidate prompt is missing from the baseline batch plan") from error
                        for batch_index in batch_indices:
                            batch_prompts = baseline_batches[batch_index]
                            batch_ordinals = [
                                target_ordinal_by_prompt_id[prompt.prompt_id]
                                for prompt in batch_prompts
                                if prompt.prompt_id in target_ordinal_by_prompt_id
                            ]
                            if all(ordinal in trajectory_by_ordinal for ordinal in batch_ordinals):
                                continue
                            trajectories = _generate_target_batch(
                                runtime,
                                batch_prompts,
                                config,
                                generation_profile_hash,
                            )
                            target_trajectories = [
                                (target_ordinal_by_prompt_id[prompt.prompt_id], trajectory)
                                for prompt, trajectory in zip(batch_prompts, trajectories, strict=True)
                                if prompt.prompt_id in target_ordinal_by_prompt_id
                            ]
                            for ordinal, trajectory in target_trajectories:
                                checkpointed = trajectory_by_ordinal.get(ordinal)
                                if checkpointed is not None and trajectory != checkpointed:
                                    raise ArtifactError(
                                        "regenerated candidate target batch does not match its checkpoint"
                                    )
                            for ordinal, trajectory in target_trajectories:
                                if ordinal in trajectory_by_ordinal:
                                    continue
                                planned_candidate, kind, _ = generation_plan[ordinal]
                                generation_checkpoint.write(
                                    ordinal,
                                    generation_keys[ordinal],
                                    {
                                        "candidate_id": planned_candidate.candidate_id,
                                        "kind": kind,
                                        "trajectory": trajectory.as_dict(),
                                    },
                                )
                                trajectory_by_ordinal[ordinal] = trajectory
                                if trajectory.parser_status == "ERROR":
                                    generation_error_count += 1
                                    generation_progress.set_postfix(errors=generation_error_count)
                                generation_progress.update()

                    missing_quality = len(quality_entries) < len(candidates)
                    if missing_quality:
                        base_logits = (
                            [
                                _next_token_logits(runtime, prompt)
                                for prompt in _progress(
                                    non_refusal_prompts,
                                    desc=f"{phase_desc}: baseline logits",
                                    unit="prompt",
                                )
                            ]
                            if screened_mean_kl is None
                            else []
                        )
                        ce_inputs = _completion_ce_inputs(
                            runtime,
                            [baseline_trajectories[item.prompt_id] for item in labeled],
                            labels,
                        )
                        base_ce = compute_ce_loss(
                            runtime,
                            ce_inputs,
                            desc=f"{phase_desc}: baseline CE loss",
                        )
                    for candidate_index, candidate in enumerate(candidates):
                        candidate_ordinals = [
                            index
                            for index, (planned_candidate, _, _) in enumerate(generation_plan)
                            if planned_candidate.candidate_id == candidate.candidate_id
                        ]
                        missing_ordinals = [
                            ordinal for ordinal in candidate_ordinals if ordinal not in trajectory_by_ordinal
                        ]
                        needs_quality = candidate.candidate_id not in quality
                        if not missing_ordinals and not needs_quality:
                            continue
                        direction = bundle.direction(candidate)
                        plan = runtime.adapter.build_weight_edit_plan(runtime.model, direction)
                        removal_ordinals = [
                            ordinal for ordinal in candidate_ordinals if generation_plan[ordinal][1] == "removal"
                        ]
                        with plan.temporary(runtime.model):
                            generate_batches(removal_ordinals)
                            if needs_quality:
                                intervention_logits = (
                                    [
                                        _next_token_logits(runtime, prompt)
                                        for prompt in _progress(
                                            non_refusal_prompts,
                                            desc=f"{phase_desc}: intervention logits",
                                            unit="prompt",
                                            leave=False,
                                        )
                                    ]
                                    if screened_mean_kl is None
                                    else []
                                )
                                try:
                                    intervention_ce = compute_ce_loss(
                                        runtime,
                                        ce_inputs,
                                        desc=f"{phase_desc}: intervention CE loss",
                                        leave=False,
                                    )
                                except NonFiniteMetricError:
                                    intervention_ce = None
                        addition_ordinals = [
                            ordinal for ordinal in candidate_ordinals if generation_plan[ordinal][1] == "addition"
                        ]
                        if any(ordinal not in trajectory_by_ordinal for ordinal in addition_ordinals):
                            beta = config.acceptance.activation_addition_beta
                            block_name = _transformer_block_name(runtime, candidate.layer)
                            with plan.activation_addition(runtime.model, block_name, beta * candidate.norm):
                                generate_batches(addition_ordinals)
                        if needs_quality:
                            measured_kl = (
                                mean_next_token_kl(base_logits, intervention_logits)
                                if screened_mean_kl is None
                                else screened_mean_kl[candidate.candidate_id]
                            )
                            mean_kl = measured_kl if measured_kl is not None and math.isfinite(measured_kl) else None
                            completion_ce = ce_evaluation_from_losses(
                                base_ce,
                                intervention_ce,
                                source="baseline_non_refusal_completions",
                                input_count=len(ce_inputs),
                            )
                            quality_checkpoint.write(
                                candidate_index,
                                candidate.candidate_id,
                                {
                                    "candidate_id": candidate.candidate_id,
                                    "mean_kl": mean_kl,
                                    "completion_ce": completion_ce.as_dict(),
                                },
                            )
                            quality[candidate.candidate_id] = (mean_kl, completion_ce)
                            quality_progress.update()
            trajectory_entries = generation_checkpoint.require_complete()
            quality_checkpoint.require_complete()
        finally:
            generation_progress.close()
            quality_progress.close()
    trajectory_rows = [dict(entry.payload) for entry in trajectory_entries]
    generation_error_rows = [
        {
            "candidate_id": row["candidate_id"],
            "kind": row["kind"],
            "prompt_id": row["trajectory"]["prompt_id"],
            "error_code": row["trajectory"]["error_code"],
            "error_detail": row["trajectory"].get("error_detail"),
        }
        for row in trajectory_rows
        if row["trajectory"]["parser_status"] == "ERROR"
    ]
    generation_errors_path = store.paths.directions / f"{evaluation_phase}_generation_errors.private.jsonl"
    _write_error_diagnostics(
        store,
        phase=f"{evaluation_phase} intervention generation",
        total=len(trajectory_rows),
        rows=generation_error_rows,
        path=generation_errors_path,
        artifact_type=f"{evaluation_phase}_generation_errors",
        profile=artifact_profile,
    )
    trajectories_path = (
        store.paths.pilot_evaluation_trajectories
        if evaluation_phase == "pilot_evaluation"
        else store.paths.full_validation_trajectories
    )
    store.write_jsonl(
        trajectories_path,
        trajectory_rows,
        artifact_type=f"{evaluation_phase}_trajectories",
        profile=artifact_profile,
        private=True,
    )
    judgment_identity = object_sha256(
        {
            "phase_identity": phase_identity,
            "trajectory_hashes": [row["trajectory"]["trajectory_hash"] for row in trajectory_rows],
            "judge_profile_hash": artifact_profile.judge_profile_hash,
            "judge_validation_hash": artifact_profile.judge_validation_hash,
            "judge_implementation_hash": _implementation_hash(
                TrajectoryJudge,
                EnumTrieConstrainedDecoder,
                type(adapter_for_config(config)),
            ),
            "evaluation_implementation_hash": _implementation_hash(
                evaluate_export,
                evaluate_behavior,
                mean_next_token_kl,
                completed_non_refusal_completion_inputs,
                compute_ce_loss,
                ce_evaluation_from_losses,
                CEEvaluation,
            ),
        }
    )
    judgment_keys = [
        object_sha256(
            {
                "candidate_id": row["candidate_id"],
                "kind": row["kind"],
                "trajectory_hash": row["trajectory"]["trajectory_hash"],
            }
        )
        for row in trajectory_rows
    ]
    with PrivateCheckpoint(
        checkpoint_root / "judgments",
        identity=judgment_identity,
        prompt_keys=judgment_keys,
    ) as judgment_checkpoint:
        judgment_entries = judgment_checkpoint.load()
        judgment_by_ordinal: dict[int, dict[str, Any]] = {}
        for entry in judgment_entries:
            row = entry.payload
            trajectory_row = trajectory_rows[entry.ordinal]
            if set(row) != {"candidate_id", "kind", "prompt_id", "trajectory_hash", "judgment"}:
                raise ArtifactError("candidate judgment checkpoint row is invalid")
            if row["candidate_id"] != trajectory_row["candidate_id"] or row["kind"] != trajectory_row["kind"]:
                raise ArtifactError("candidate judgment checkpoint identity is invalid")
            trajectory = TargetTrajectory.from_dict(trajectory_row["trajectory"])
            result_value = row["judgment"]
            if not isinstance(result_value, dict):
                raise ArtifactError("candidate judgment checkpoint payload is invalid")
            _validate_checkpointed_judgment(JudgeResult.from_dict(result_value), trajectory)
            judgment_by_ordinal[entry.ordinal] = dict(row)
        if len(judgment_entries) < len(trajectory_rows):
            with BaseModelRuntime(config) as runtime:
                _require_judge_validation(store, runtime)
                _validate_activation_chat_profile(artifact_profile, runtime.chat_template_hash)
                judge = TrajectoryJudge(runtime.adapter, runtime.model, runtime.processor)
                progress = tqdm(
                    total=len(trajectory_rows),
                    initial=len(judgment_entries),
                    desc=f"{phase_desc}: judging trajectories",
                    unit="trajectory",
                    dynamic_ncols=True,
                    disable=None,
                )
                judgment_error_count = sum(row["judgment"]["status"] == "ERROR" for row in judgment_by_ordinal.values())
                progress.set_postfix(errors=judgment_error_count)
                try:
                    for ordinal, row in enumerate(trajectory_rows):
                        if ordinal in judgment_by_ordinal:
                            continue
                        trajectory = TargetTrajectory.from_dict(row["trajectory"])
                        result = judge.classify(trajectory)
                        _validate_checkpointed_judgment(result, trajectory)
                        judgment_row = {
                            "candidate_id": row["candidate_id"],
                            "kind": row["kind"],
                            "prompt_id": trajectory.prompt_id,
                            "trajectory_hash": trajectory.trajectory_hash,
                            "judgment": result.as_dict(),
                        }
                        judgment_checkpoint.write(ordinal, judgment_keys[ordinal], judgment_row)
                        judgment_by_ordinal[ordinal] = judgment_row
                        if result.status == "ERROR":
                            judgment_error_count += 1
                            progress.set_postfix(errors=judgment_error_count)
                        progress.update()
                finally:
                    progress.close()
                    del judge
        judgment_rows = [dict(entry.payload) for entry in judgment_checkpoint.require_complete()]
    judgment_error_rows = [
        {
            "candidate_id": row["candidate_id"],
            "kind": row["kind"],
            "prompt_id": row["prompt_id"],
            "trajectory_hash": row["trajectory_hash"],
            "error_code": row["judgment"]["error_code"],
        }
        for row in judgment_rows
        if row["judgment"]["status"] == "ERROR"
    ]
    judgment_errors_path = store.paths.directions / f"{evaluation_phase}_judgment_errors.private.jsonl"
    _write_error_diagnostics(
        store,
        phase=f"{evaluation_phase} judging",
        total=len(judgment_rows),
        rows=judgment_error_rows,
        path=judgment_errors_path,
        artifact_type=f"{evaluation_phase}_judgment_errors",
        profile=artifact_profile,
    )
    judgments_path = (
        store.paths.pilot_evaluation_judgments
        if evaluation_phase == "pilot_evaluation"
        else store.paths.full_validation_judgments
    )
    store.write_jsonl(
        judgments_path,
        judgment_rows,
        artifact_type=f"{evaluation_phase}_judgments",
        profile=artifact_profile,
        private=False,
    )
    metrics: list[CandidateMetrics] = []
    for candidate in candidates:
        removal_rows = [
            row for row in trajectory_rows if row["candidate_id"] == candidate.candidate_id and row["kind"] == "removal"
        ]
        candidate_judgments = {
            row["trajectory_hash"]: JudgeResult.from_dict(row["judgment"])
            for row in judgment_rows
            if row["candidate_id"] == candidate.candidate_id and row["kind"] == "removal"
        }
        induction = _addition_induction_rate(candidate.candidate_id, judgment_rows, labels)
        addition_error_rate = _candidate_addition_error_rate(candidate.candidate_id, judgment_rows)
        mean_kl, completion_ce = quality[candidate.candidate_id]
        candidate_metrics = evaluate_behavior(
            candidate_id=candidate.candidate_id,
            baseline_labels=labels,
            baseline_trajectories=baseline_trajectories,
            trajectories=[TargetTrajectory.from_dict(row["trajectory"]) for row in removal_rows],
            judgments=candidate_judgments,
            mean_kl=mean_kl,
            completion_ce=completion_ce,
            acceptance=config.acceptance,
            activation_addition_induction_rate=induction,
            activation_addition_error_rate=addition_error_rate,
            require_activation_addition=True,
        )
        if evaluation_phase == "pilot_evaluation":
            candidate_metrics = apply_pilot_filters(candidate_metrics)
        metrics.append(candidate_metrics)
    results_path = (
        store.paths.pilot_evaluation_results
        if evaluation_phase == "pilot_evaluation"
        else store.paths.full_validation_results
    )
    store.write_jsonl(
        results_path,
        metrics,
        artifact_type=f"{evaluation_phase}_results",
        profile=results_profile,
        private=False,
    )
    _evaluate_candidate_reference_ce(
        config,
        store,
        bundle,
        candidates,
        artifact_profile=artifact_profile,
        evaluation_phase=evaluation_phase,
    )
    return metrics, phase_identity


def _transformer_block_name(runtime: IntervenedModelRuntime, layer: int) -> str:
    model = runtime.model
    block = runtime.adapter.transformer_blocks(model)[layer]
    return _module_name(model, block)


def _next_token_logits(
    runtime: BaseModelRuntime | IntervenedModelRuntime,
    prompt: PromptRecord,
) -> torch.Tensor:
    rendered = runtime.adapter.render_target_chat(
        runtime.processor,
        target_messages(prompt.original_prompt, runtime.config.target_generation.system_prompt),
        config=runtime.config.target_generation,
        prefill_thinking=True,
    )
    inputs = _move_inputs(dict(rendered), runtime.adapter.input_device(runtime.model))
    with torch.inference_mode():
        output = runtime.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
    return output.logits[0, -1].detach().float().cpu()


def _completion_ce_inputs(
    runtime: BaseModelRuntime | IntervenedModelRuntime,
    trajectories: Sequence[TargetTrajectory],
    labels: Mapping[str, str],
) -> tuple[CEInput, ...]:
    return completed_non_refusal_completion_inputs(runtime, trajectories, labels)


def _reference_ce_inputs(
    config: ProjectConfig,
    runtime: BaseModelRuntime | IntervenedModelRuntime,
) -> tuple[CEInput, ...]:
    tokenizer = runtime.processor.tokenizer
    texts = ingest_texts(
        config.data.reference_files,
        token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=True)),
        max_text_tokens=config.data.max_text_tokens,
    )
    if not texts:
        raise PipelineError("configured reference files produced no usable texts")
    return raw_text_ce_inputs(runtime, texts)


def _module_name(model: torch.nn.Module, target: torch.nn.Module) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name
    raise InvariantError("module is not registered in the model")


def _addition_induction_rate(
    candidate_id: str,
    rows: Sequence[Mapping[str, Any]],
    baseline_labels: Mapping[str, str],
) -> float | None:
    addition = [row for row in rows if row["candidate_id"] == candidate_id and row["kind"] == "addition"]
    if not addition:
        return None
    denominator = sum(label == "NON_REFUSAL" for label in baseline_labels.values())
    induced = sum(row["judgment"]["status"] == "OK" and row["judgment"]["label"] == "REFUSAL" for row in addition)
    return induced / denominator if denominator else 0.0


def export_model(config_path: str) -> None:
    from self_judged_refusal_direction.exporting import (
        complete_persisted_deferred_reload,
        export_edited_model,
        load_deferred_reload,
        load_export_manifest,
        write_export_manifest,
    )

    config, store = _load(config_path)
    profile = _selection_dependent_profile(store)
    validation_hash = profile.judge_validation_hash
    if validation_hash is None:
        raise ArtifactError("activation profile has no judge validation hash")
    store.validate(
        store.paths.selected_direction,
        artifact_type="selected_direction",
        expected_profile=profile,
    )
    direction, direction_metadata = load_direction(store.paths.selected_direction)
    selection = _read_json_artifact(
        store,
        store.paths.final_selection,
        artifact_type="final_selection",
        profile=profile,
    )
    _validate_direction_selection(direction_metadata, selection)
    _validate_selection_error_policy(selection, config, store)
    final_directory = store.paths.exported_model
    manifest_path = final_directory / "edit_manifest.json"
    staging_directory = store.paths.root / ".exported_model.staging"
    backup_directory = store.paths.root / ".exported_model.previous"
    reload_directory = store.paths.evaluation / ".export-reload"
    test_started = store.paths.evaluation / "test_evaluation_started.json"
    test_outputs = (
        store.paths.test_report,
        store.paths.quality_metrics,
        store.paths.evaluation / "test_baseline_trajectories.private.jsonl",
        store.paths.evaluation / "test_baseline_judgments.jsonl",
        store.paths.evaluation / "test_export_trajectories.private.jsonl",
        store.paths.evaluation / "test_export_judgments.jsonl",
    )
    test_checkpoint_parent = store.paths.evaluation / ".test-evaluation-checkpoints"
    test_consumed = (
        test_started.exists()
        or any(path.exists() or store.metadata_path(path).exists() for path in test_outputs)
        or (test_checkpoint_parent.is_dir() and any(path.is_file() for path in test_checkpoint_parent.rglob("*")))
    )
    for managed_path in (final_directory, staging_directory, backup_directory, reload_directory):
        if managed_path.is_symlink():
            raise ArtifactError(f"managed export path must not be a symlink: {managed_path}")
    if backup_directory.exists() and not final_directory.exists() and not staging_directory.exists():
        backup_directory.rename(final_directory)
        ArtifactStore._fsync_directory(final_directory.parent)
    if manifest_path.is_file():
        committed_manifest = load_export_manifest(final_directory)
        try:
            _validate_export_manifest(
                config,
                committed_manifest,
                selection,
                direction,
                validation_hash,
            )
        except ArtifactError:
            if test_consumed:
                raise PipelineError("the selected candidate changed after independent test evaluation began") from None
            if backup_directory.exists():
                raise ArtifactError("an interrupted export replacement has ambiguous committed outputs") from None
        else:
            _remove_managed_directory(staging_directory, missing_ok=True)
            _remove_managed_directory(reload_directory, missing_ok=True)
            _remove_managed_directory(backup_directory, missing_ok=True)
            return
    elif final_directory.exists():
        raise ArtifactError("export directory exists without a committed manifest")
    staged_manifest: dict[str, Any] | None = None
    if staging_directory.exists():
        staged_manifest_path = staging_directory / "edit_manifest.json"
        try:
            if not staged_manifest_path.is_file():
                raise ArtifactError("staged export has no committed manifest")
            staged_manifest = load_export_manifest(staging_directory)
            _validate_export_manifest(
                config,
                staged_manifest,
                selection,
                direction,
                validation_hash,
                require_reload=False,
            )
            if "fresh_reload" not in staged_manifest:
                load_deferred_reload(reload_directory)
        except ArtifactError, InvariantError:
            _remove_managed_directory(staging_directory)
            _remove_managed_directory(reload_directory, missing_ok=True)
            staged_manifest = None
    elif reload_directory.exists():
        _remove_managed_directory(reload_directory)
    if staged_manifest is not None:
        if "fresh_reload" not in staged_manifest:
            report = complete_persisted_deferred_reload(reload_directory)
            staged_manifest["fresh_reload"] = report.as_dict()
            write_export_manifest(staging_directory, staged_manifest)
        _validate_export_manifest(
            config,
            staged_manifest,
            selection,
            direction,
            validation_hash,
        )
        _remove_managed_directory(reload_directory, missing_ok=True)
        _promote_export(staging_directory, final_directory, backup_directory)
        return
    validation = _load_labeled_for_probe(config, store, profile)
    prompt = validation[0]
    export_result: Any
    with IntervenedModelRuntime(config, direction=direction, install_temporary=False) as runtime:
        _validate_activation_chat_profile(profile, runtime.chat_template_hash)
        plan = runtime.weight_edit_plan
        probe = runtime.adapter.render_target_chat(
            runtime.processor,
            target_messages(prompt.original_prompt, config.target_generation.system_prompt),
            config=config.target_generation,
            prefill_thinking=True,
        )
        probe = _move_inputs(dict(probe), runtime.adapter.input_device(runtime.model))
        export_result = export_edited_model(
            runtime.model,
            runtime.processor,
            runtime.adapter,
            plan,
            config,
            probe,
            judge_validation_hash=validation_hash,
            output_dir=staging_directory,
            full_validation_metrics=selection["metrics"],
            defer_reload=True,
            deferred_reload_directory=reload_directory,
            direction_layer=int(direction_metadata["candidate"]["layer"]),
        )
    report = complete_persisted_deferred_reload(reload_directory)
    completed_manifest = dict(export_result.manifest)
    completed_manifest["fresh_reload"] = report.as_dict()
    write_export_manifest(staging_directory, completed_manifest)
    _validate_export_manifest(
        config,
        completed_manifest,
        selection,
        direction,
        validation_hash,
    )
    _remove_managed_directory(reload_directory)
    _promote_export(staging_directory, final_directory, backup_directory)


def _remove_managed_directory(path: Path, *, missing_ok: bool = False) -> None:
    if path.is_symlink():
        raise ArtifactError(f"managed path must not be a symlink: {path}")
    if not path.exists():
        if missing_ok:
            return
        raise ArtifactError(f"managed directory does not exist: {path}")
    if path.is_symlink() or not path.is_dir():
        raise ArtifactError(f"managed path is not a directory: {path}")
    shutil.rmtree(path)
    ArtifactStore._fsync_directory(path.parent)


def _promote_export(staging: Path, final: Path, backup: Path) -> None:
    if staging.is_symlink() or not staging.is_dir() or final.is_symlink() or backup.is_symlink():
        raise ArtifactError("export replacement paths are invalid")
    if backup.exists() and final.exists():
        raise ArtifactError("export replacement has both committed and backup directories")
    if final.exists():
        final.rename(backup)
        ArtifactStore._fsync_directory(final.parent)
    try:
        staging.rename(final)
        ArtifactStore._fsync_directory(final.parent)
    except OSError:
        if backup.exists() and not final.exists():
            backup.rename(final)
            ArtifactStore._fsync_directory(final.parent)
        raise
    _remove_managed_directory(backup, missing_ok=True)


def _validate_direction_selection(
    direction_metadata: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    candidate = direction_metadata.get("candidate")
    full_validation_metrics = direction_metadata.get("full_validation_metrics")
    selected_candidate = selection.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or not isinstance(full_validation_metrics, Mapping)
        or not isinstance(selected_candidate, Mapping)
    ):
        raise ArtifactError("selected direction metadata is incomplete")
    if object_sha256(candidate) != object_sha256(selected_candidate):
        raise ArtifactError("selected direction and final selection candidate metadata differ")
    if object_sha256(full_validation_metrics) != object_sha256(selection.get("metrics")):
        raise ArtifactError("selected direction and final selection metrics differ")


def _validate_selection_error_policy(
    selection: Mapping[str, Any],
    config: ProjectConfig,
    store: ArtifactStore,
) -> None:
    if selection.get("max_error_rate") != config.acceptance.max_error_rate:
        raise PipelineError("final selection must be recomputed for acceptance.max_error_rate")
    for phase, field in (
        ("pilot_evaluation", "pilot_evaluation_identity"),
        ("full_validation", "full_validation_identity"),
    ):
        identity = _require_sha256(selection.get(field), f"final selection {field}")
        attempt = _read_json(store.paths.directions / f"{phase}_attempt.json")
        if attempt != {"evaluation_identity": identity}:
            raise ArtifactError(f"final selection does not match the latest {phase} attempt")
    metrics = selection.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ArtifactError("final selection metrics are missing")
    for name in ("error_rate", "activation_addition_error_rate"):
        value = metrics.get(name)
        if value is None and name == "activation_addition_error_rate":
            raise ArtifactError("final selection activation_addition_error_rate is missing")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ArtifactError(f"final selection {name} is invalid")
        if value > config.acceptance.max_error_rate:
            raise PipelineError(f"final selection {name} exceeds acceptance.max_error_rate")


def _load_labeled_for_probe(
    config: ProjectConfig,
    store: ArtifactStore,
    activation_profile: ArtifactProfile,
) -> list[PromptRecord]:
    with BaseModelRuntime(config) as runtime:
        _require_judge_validation(store, runtime)
        _validate_activation_chat_profile(activation_profile, runtime.chat_template_hash)
        labeled = _load_labeled(store, runtime.chat_template_hash, "validation")
    prompts = {item.prompt_id: item for item in _load_prompt_records(store, config)}
    return [prompts[item.prompt_id] for item in labeled]


def _error_rate_allowed(count: int, total: int, config: ProjectConfig) -> bool:
    return total > 0 and 0 <= count <= total and count / total <= config.acceptance.max_error_rate


def _load_test_base_quality(
    directory: Path,
    *,
    identity: str,
    prompt_ids: Sequence[str],
) -> tuple[list[torch.Tensor], CELoss, tuple[CEInput, ...]] | None:
    metadata_path = directory / "quality.private.json"
    tensor_path = directory / "logits.private.safetensors"
    if not metadata_path.exists():
        return None
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or directory.stat().st_mode & 0o077
        or metadata_path.is_symlink()
        or metadata_path.stat().st_mode & 0o077
        or tensor_path.is_symlink()
        or not tensor_path.is_file()
        or tensor_path.stat().st_mode & 0o077
    ):
        raise ArtifactError("test base-quality checkpoint permissions are invalid")
    value = _read_json(metadata_path)
    required = {
        "identity",
        "prompt_ids",
        "base_ce",
        "ce_inputs",
        "tensor_sha256",
        "content_sha256",
    }
    if set(value) != required:
        raise ArtifactError("test base-quality checkpoint fields are invalid")
    content_sha256 = value.pop("content_sha256")
    if _require_sha256(content_sha256, "test base-quality content hash") != object_sha256(value):
        raise ArtifactError("test base-quality checkpoint content hash does not match")
    if value["identity"] != identity or value["prompt_ids"] != list(prompt_ids):
        raise ArtifactError("test base-quality checkpoint identity does not match")
    if _require_sha256(value["tensor_sha256"], "test base-quality tensor hash") != file_sha256(tensor_path):
        raise ArtifactError("test base-quality checkpoint tensor hash does not match")
    base_ce = value["base_ce"]
    ce_input_rows = value["ce_inputs"]
    if (
        not isinstance(base_ce, dict)
        or set(base_ce) != {"total_loss", "token_count"}
        or isinstance(base_ce["total_loss"], bool)
        or not isinstance(base_ce["total_loss"], int | float)
        or isinstance(base_ce["token_count"], bool)
        or not isinstance(base_ce["token_count"], int)
        or not isinstance(ce_input_rows, list)
        or not ce_input_rows
    ):
        raise ArtifactError("test base-quality checkpoint values are invalid")
    try:
        base_ce_loss = CELoss(total_loss=float(base_ce["total_loss"]), token_count=base_ce["token_count"])
        ce_inputs = tuple(
            CEInput(input_ids=tuple(row["input_ids"]), target_start=row["target_start"])
            for row in ce_input_rows
            if isinstance(row, dict) and set(row) == {"input_ids", "target_start"}
        )
    except (KeyError, TypeError, ValueError, InvariantError) as error:
        raise ArtifactError("test base-quality CE checkpoint values are invalid") from error
    if (
        len(ce_inputs) != len(ce_input_rows)
        or sum(item.target_token_count for item in ce_inputs) != base_ce_loss.token_count
    ):
        raise ArtifactError("test base-quality CE checkpoint token counts differ")
    try:
        tensors = load_file(tensor_path, device="cpu")
    except (OSError, RuntimeError, safetensors.SafetensorError) as error:
        raise ArtifactError("test base-quality checkpoint tensor is invalid") from error
    if set(tensors) != {"base_logits"}:
        raise ArtifactError("test base-quality checkpoint tensor fields are invalid")
    base_logits = tensors["base_logits"]
    if (
        base_logits.ndim != 2
        or base_logits.shape[0] != len(prompt_ids)
        or not base_logits.dtype.is_floating_point
        or not bool(torch.isfinite(base_logits).all())
    ):
        raise ArtifactError("test base-quality checkpoint logits are invalid")
    return list(base_logits.unbind(0)), base_ce_loss, ce_inputs


def _save_test_base_quality(
    directory: Path,
    *,
    identity: str,
    prompt_ids: Sequence[str],
    base_logits: Sequence[torch.Tensor],
    base_ce: CELoss,
    ce_inputs: Sequence[CEInput],
) -> None:
    if (
        len(base_logits) != len(prompt_ids)
        or not ce_inputs
        or sum(item.target_token_count for item in ce_inputs) != base_ce.token_count
    ):
        raise InvariantError("test base-quality checkpoint values are invalid")
    if directory.is_symlink():
        raise ArtifactError(f"test base-quality checkpoint must not be a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    if base_logits:
        shapes = {tuple(logits.shape) for logits in base_logits}
        if len(shapes) != 1 or any(logits.ndim != 1 for logits in base_logits):
            raise InvariantError("test base-quality logits have inconsistent shapes")
        stored_logits = torch.stack(
            [logits.detach().to(device="cpu", dtype=torch.float32).contiguous() for logits in base_logits]
        )
    else:
        stored_logits = torch.empty((0, 0), dtype=torch.float32)
    if not bool(torch.isfinite(stored_logits).all()):
        raise InvariantError("test base-quality logits are not finite")
    tensor_path = directory / "logits.private.safetensors"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
        save_file({"base_logits": stored_logits}, temporary)
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(tensor_path)
        temporary = None
        ArtifactStore._fsync_directory(directory)
    except (OSError, TypeError, ValueError, RuntimeError, safetensors.SafetensorError) as error:
        raise ArtifactError("test base-quality checkpoint tensor could not be written") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    body = {
        "identity": identity,
        "prompt_ids": list(prompt_ids),
        "base_ce": {"total_loss": base_ce.total_loss, "token_count": base_ce.token_count},
        "ce_inputs": [dataclasses.asdict(item) for item in ce_inputs],
        "tensor_sha256": file_sha256(tensor_path),
    }
    store_value = {**body, "content_sha256": object_sha256(body)}
    ArtifactStore.write_json(directory / "quality.private.json", store_value, private=True)


def _test_reference_ce_identity(
    config: ProjectConfig,
    profile: ArtifactProfile,
    export_manifest_hash: str,
) -> str:
    return object_sha256(
        {
            "profile": profile.as_dict(),
            "export_manifest_hash": export_manifest_hash,
            "reference_files": [
                {
                    "path": str(Path(path).resolve()),
                    "content_sha256": file_sha256(Path(path).resolve()),
                }
                for path in config.data.reference_files
            ],
            "max_text_tokens": config.data.max_text_tokens,
            "implementation_hash": _implementation_hash(
                _test_reference_ce_identity,
                ingest_texts,
                raw_text_ce_inputs,
                compute_ce_loss,
                ce_evaluation_from_losses,
                CEEvaluation,
                BaseModelRuntime,
                IntervenedModelRuntime,
                type(adapter_for_config(config)),
            ),
        }
    )


def _evaluate_test_reference_ce(
    config: ProjectConfig,
    store: ArtifactStore,
    profile: ArtifactProfile,
) -> None:
    from self_judged_refusal_direction.exporting import load_export_manifest

    results_path = store.paths.evaluation / "test_reference_ce_results.jsonl"
    manifest = load_export_manifest(store.paths.exported_model)
    identity = _test_reference_ce_identity(
        config,
        profile,
        _require_sha256(manifest.get("manifest_sha256"), "export manifest hash"),
    )
    diagnostic_profile = dataclasses.replace(profile, reference_ce_hash=identity)
    if not config.data.reference_files:
        store.write_jsonl(
            results_path,
            (),
            artifact_type="test_reference_ce_results",
            profile=diagnostic_profile,
            private=False,
        )
        return
    checkpoint_directory = store.paths.evaluation / ".test-reference-ce-checkpoints" / identity
    with PrivateCheckpoint(
        checkpoint_directory,
        identity=identity,
        prompt_keys=("exported_model",),
    ) as checkpoint:
        entries = checkpoint.load()
        if not entries:
            with BaseModelRuntime(config) as runtime:
                _require_judge_validation(store, runtime)
                _validate_activation_chat_profile(profile, runtime.chat_template_hash)
                base_fingerprints = runtime.adapter.processor_fingerprints(runtime.processor)
                inputs = _reference_ce_inputs(config, runtime)
                baseline = compute_ce_loss(runtime, inputs, desc="Test baseline reference CE")
            export_config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, id=str(store.paths.exported_model)),
            )
            with IntervenedModelRuntime(export_config, install_temporary=False) as runtime:
                if runtime.adapter.processor_fingerprints(runtime.processor) != base_fingerprints:
                    raise ArtifactError("exported processor differs during reference CE evaluation")
                try:
                    intervention = compute_ce_loss(runtime, inputs, desc="Test export reference CE")
                except NonFiniteMetricError:
                    intervention = None
            evaluation = ce_evaluation_from_losses(
                baseline,
                intervention,
                source="reference_files",
                input_count=len(inputs),
            )
            checkpoint.write(
                0,
                "exported_model",
                {
                    "candidate_id": "exported_model",
                    "reference_ce": evaluation.as_dict(),
                },
            )
        rows = [dict(entry.payload) for entry in checkpoint.require_complete()]
        if (
            len(rows) != 1
            or set(rows[0]) != {"candidate_id", "reference_ce"}
            or rows[0]["candidate_id"] != "exported_model"
            or not isinstance(rows[0]["reference_ce"], dict)
        ):
            raise ArtifactError("test reference CE checkpoint is invalid")
        reference_ce = CEEvaluation.from_dict(rows[0]["reference_ce"])
        if reference_ce.source != "reference_files":
            raise ArtifactError("test reference CE checkpoint source is invalid")
    store.write_jsonl(
        results_path,
        rows,
        artifact_type="test_reference_ce_results",
        profile=diagnostic_profile,
        private=False,
    )


def _trajectory_error_rows(trajectories: Sequence[TargetTrajectory]) -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": trajectory.prompt_id,
            "trajectory_hash": trajectory.trajectory_hash,
            "error_code": trajectory.error_code,
            "error_detail": trajectory.error_detail,
            "generation_truncated": trajectory.generation_truncated,
            "generated_token_count": len(trajectory.raw_generated_token_ids),
        }
        for trajectory in trajectories
        if trajectory.parser_status == "ERROR"
    ]


def _judgment_error_rows(
    judgments: Sequence[JudgeResult],
    trajectories: Sequence[TargetTrajectory],
) -> list[dict[str, Any]]:
    prompt_by_hash = {trajectory.trajectory_hash: trajectory.prompt_id for trajectory in trajectories}
    return [
        {
            "prompt_id": prompt_by_hash[result.trajectory_hash],
            "trajectory_hash": result.trajectory_hash,
            "error_code": result.error_code,
        }
        for result in judgments
        if result.status == "ERROR"
    ]


def _validate_completed_test_error_rates(report: Mapping[str, Any], config: ProjectConfig) -> None:
    for name, section_name, error_name in (
        ("test baseline generation", "baseline_parser", "parser_error_count"),
        ("test export generation", "export_parser", "parser_error_count"),
        ("test baseline judging", "baseline_counts", "ERROR"),
        ("test export judging", "export_counts", "ERROR"),
    ):
        section = report.get(section_name)
        if not isinstance(section, Mapping):
            raise ArtifactError(f"completed test report has no {section_name}")
        count = section.get("count")
        if error_name == "parser_error_count":
            successes = section.get("parser_success_count")
            if type(count) is not int or type(successes) is not int:
                raise ArtifactError(f"completed test report has invalid {section_name}")
            errors = count - successes
        else:
            errors = section.get(error_name)
            if type(count) is not int:
                count = sum(
                    value
                    for key, value in section.items()
                    if key in {"REFUSAL", "NON_REFUSAL", "UNCERTAIN", "ERROR"} and type(value) is int
                )
        if type(errors) is not int or type(count) is not int:
            raise ArtifactError(f"completed test report has invalid {section_name}")
        _check_error_rate(errors, count, config, name)


def evaluate_export(config_path: str) -> None:
    from self_judged_refusal_direction.exporting import load_export_manifest, write_export_manifest

    config, store = _load(config_path)
    validation_hash = _require_judge_validation(store)
    test_baseline_path = store.paths.evaluation / "test_baseline_trajectories.private.jsonl"
    test_baseline_judgments_path = store.paths.evaluation / "test_baseline_judgments.jsonl"
    test_export_path = store.paths.evaluation / "test_export_trajectories.private.jsonl"
    test_export_judgments_path = store.paths.evaluation / "test_export_judgments.jsonl"
    evaluation_started_path = store.paths.evaluation / "test_evaluation_started.json"
    manifest = load_export_manifest(store.paths.exported_model)
    selection_profile = _selection_dependent_profile(store)
    profile = _derived_profile(
        store,
        selection_profile,
        stage="test_evaluation",
        acceptance_policy_hash=selection_profile.acceptance_policy_hash,
    )
    selection = _read_json_artifact(
        store,
        store.paths.final_selection,
        artifact_type="final_selection",
        profile=selection_profile,
    )
    store.validate(
        store.paths.selected_direction,
        artifact_type="selected_direction",
        expected_profile=selection_profile,
    )
    direction, direction_metadata = load_direction(store.paths.selected_direction)
    _validate_direction_selection(direction_metadata, selection)
    _validate_selection_error_policy(selection, config, store)
    _validate_export_manifest(config, manifest, selection, direction, validation_hash)
    initial_manifest = dict(manifest)
    initial_manifest.pop("manifest_sha256")
    completed_test_metrics = initial_manifest.pop("test_metrics", None)
    completed_test_evaluation = initial_manifest.pop("test_evaluation", None)
    if (completed_test_metrics is None) != (completed_test_evaluation is None):
        raise ArtifactError("export manifest has an incomplete test evaluation")
    initial_manifest_hash = object_sha256(initial_manifest)
    _, baseline_profile, baseline_directory = _read_baseline_generation_manifest(store)
    if profile.baseline_generation_hash != baseline_profile.baseline_generation_hash:
        raise ArtifactError("test prompts and selected direction use different baseline generations")
    test_prompt_rows = store.read_jsonl(
        baseline_directory / "test_prompts.private.jsonl",
        artifact_type="test_prompts",
        expected_profile=_profile(store, stage="baseline_generation"),
    )
    prompts = [PromptRecord.from_dict(row) for row in test_prompt_rows]
    if not prompts or any(prompt.split != "test" for prompt in prompts):
        raise PipelineError("independent test artifact must contain only test prompts")
    evaluation_identity = object_sha256(
        {
            "initial_export_manifest_sha256": initial_manifest_hash,
            "profile": profile.as_dict(),
            "selection_sha256": object_sha256(
                {
                    "candidate": selection.get("candidate"),
                    "metrics": selection.get("metrics"),
                }
            ),
            "direction_sha256": tensor_sha256(direction),
            "test_prompts": [dataclasses.asdict(prompt) for prompt in prompts],
            "target_implementation_hash": _implementation_hash(
                TargetTrajectoryGenerator,
                type(adapter_for_config(config)),
                BaseModelRuntime,
                IntervenedModelRuntime,
            ),
            "judge_implementation_hash": _implementation_hash(
                TrajectoryJudge,
                EnumTrieConstrainedDecoder,
                type(adapter_for_config(config)),
                BaseModelRuntime,
            ),
            "evaluation_implementation_hash": _implementation_hash(
                evaluate_export,
                evaluate_behavior,
                mean_next_token_kl,
                completed_non_refusal_completion_inputs,
                compute_ce_loss,
                ce_evaluation_from_losses,
                CEEvaluation,
            ),
        }
    )
    started_value = {
        "evaluation_identity": evaluation_identity,
        "initial_export_manifest_sha256": initial_manifest_hash,
    }
    checkpoint_parent = store.paths.evaluation / ".test-evaluation-checkpoints"
    checkpoint_root = checkpoint_parent / evaluation_identity
    base_quality_directory = checkpoint_root / "base_quality"
    export_quality_directory = checkpoint_root / "export_quality"
    if evaluation_started_path.exists():
        observed_started = _read_json_artifact(
            store,
            evaluation_started_path,
            artifact_type="test_evaluation_started",
            profile=profile,
        )
        if observed_started != started_value:
            raise ArtifactError("independent test evaluation identity changed after consumption")
    else:
        existing_outputs = (
            store.paths.test_report,
            store.paths.quality_metrics,
            test_baseline_path,
            test_baseline_judgments_path,
            test_export_path,
            test_export_judgments_path,
            store.paths.evaluation / "test_baseline_generation_errors.private.jsonl",
            store.paths.evaluation / "test_baseline_judgment_errors.private.jsonl",
            store.paths.evaluation / "test_export_generation_errors.private.jsonl",
            store.paths.evaluation / "test_export_judgment_errors.private.jsonl",
        )
        has_checkpoint = checkpoint_parent.is_dir() and any(path.is_file() for path in checkpoint_parent.rglob("*"))
        if has_checkpoint or any(path.exists() or store.metadata_path(path).exists() for path in existing_outputs):
            raise ArtifactError("independent test outputs exist without a consumption record")
        _write_json_artifact(
            store,
            evaluation_started_path,
            started_value,
            artifact_type="test_evaluation_started",
            profile=profile,
        )
    if completed_test_metrics is not None:
        if not isinstance(completed_test_evaluation, Mapping):
            raise ArtifactError("export manifest test evaluation is invalid")
        report = _read_json_artifact(
            store,
            store.paths.test_report,
            artifact_type="test_report",
            profile=profile,
        )
        if report.get("test_metrics") != completed_test_metrics:
            raise ArtifactError("completed test report differs from the export manifest")
        quality = _read_json_artifact(
            store,
            store.paths.quality_metrics,
            artifact_type="quality_metrics",
            profile=profile,
        )
        if quality.get("test_metrics") != completed_test_metrics:
            raise ArtifactError("completed quality metrics differ from the export manifest")
        _validate_completed_test_error_rates(report, config)
        baseline_counts = report["baseline_counts"]
        export_counts = report["export_counts"]
        if not isinstance(baseline_counts, Mapping) or not isinstance(export_counts, Mapping):
            raise ArtifactError("completed test report judgment counts are invalid")
        artifact_specs = {
            "test_report": (store.paths.test_report, "test_report", None),
            "quality_metrics": (store.paths.quality_metrics, "quality_metrics", None),
            "baseline_trajectories": (
                test_baseline_path,
                "test_baseline_trajectories",
                report["baseline_parser"]["count"],
            ),
            "baseline_judgments": (
                test_baseline_judgments_path,
                "test_baseline_judgments",
                sum(value for value in baseline_counts.values() if type(value) is int),
            ),
            "export_trajectories": (
                test_export_path,
                "test_export_trajectories",
                report["export_parser"]["count"],
            ),
            "export_judgments": (
                test_export_judgments_path,
                "test_export_judgments",
                sum(value for value in export_counts.values() if type(value) is int),
            ),
        }
        committed_artifacts: dict[str, dict[str, Any]] = {}
        for name, (path, artifact_type, count) in artifact_specs.items():
            metadata = store.validate(path, artifact_type=artifact_type, expected_profile=profile)
            if count is not None and metadata.record_count != count:
                raise ArtifactError("completed test evaluation artifact count differs from its report")
            committed_artifacts[name] = {
                "content_sha256": metadata.content_sha256,
                **({"record_count": count} if count is not None else {}),
            }
        evaluation_report_path = store.paths.exported_model / "evaluation_report.json"
        if _read_json(evaluation_report_path) != report:
            raise ArtifactError("exported evaluation report differs from the run artifact")
        expected_commit = {
            "evaluation_identity": evaluation_identity,
            "artifacts": committed_artifacts,
            "evaluation_report_sha256": file_sha256(evaluation_report_path),
        }
        if completed_test_evaluation != expected_commit:
            raise ArtifactError("completed test evaluation artifacts differ from the export manifest")
        _remove_managed_directory(base_quality_directory, missing_ok=True)
        _discard_checkpoint(export_quality_directory)
        _evaluate_test_reference_ce(config, store, profile)
        return
    baseline_judgments: list[JudgeResult] = []
    base_logits: list[torch.Tensor] = []
    base_ce: CELoss | None = None
    ce_inputs: tuple[CEInput, ...] = ()
    base_quality_identity = ""
    with BaseModelRuntime(config) as runtime:
        _require_judge_validation(store, runtime)
        _validate_activation_chat_profile(profile, runtime.chat_template_hash)
        fingerprints = runtime.adapter.processor_fingerprints(runtime.processor)
        if runtime.chat_template_hash != initial_manifest["chat_template_hash"]:
            raise ArtifactError("export manifest chat template hash differs from the pinned base processor")
        if any(initial_manifest.get(name) != value for name, value in fingerprints.items()):
            raise ArtifactError("export manifest processor fingerprints differ from the pinned base processor")
        base_checkpoint_checksum = runtime.checkpoint_checksum
        baseline_generation_identity = object_sha256(
            {
                "evaluation_identity": evaluation_identity,
                "stage": "baseline_generation",
                "checkpoint_checksum": base_checkpoint_checksum,
                "chat_template_hash": runtime.chat_template_hash,
                "generation_profile_hash": _target_generation_profile_hash(config, runtime),
                "processor_fingerprints": fingerprints,
                "implementation_hash": _implementation_hash(
                    TargetTrajectoryGenerator,
                    type(runtime.adapter),
                ),
            }
        )
        baseline_trajectories = _checkpointed_target_trajectories(
            runtime,
            config,
            prompts,
            directory=checkpoint_root / "baseline_trajectories",
            identity=baseline_generation_identity,
            desc="Generating test baseline",
        )
        baseline_generation_errors = _trajectory_error_rows(baseline_trajectories)
        chat_template_hash = runtime.chat_template_hash
        if _error_rate_allowed(len(baseline_generation_errors), len(baseline_trajectories), config):
            baseline_judgment_identity = object_sha256(
                {
                    "evaluation_identity": evaluation_identity,
                    "stage": "baseline_judging",
                    "checkpoint_checksum": base_checkpoint_checksum,
                    "trajectory_hashes": [item.trajectory_hash for item in baseline_trajectories],
                    "judge_validation_hash": validation_hash,
                    "implementation_hash": _implementation_hash(
                        TrajectoryJudge,
                        EnumTrieConstrainedDecoder,
                        type(runtime.adapter),
                    ),
                }
            )
            baseline_judgments = _checkpointed_judgments(
                runtime,
                baseline_trajectories,
                directory=checkpoint_root / "baseline_judgments",
                identity=baseline_judgment_identity,
                desc="Judging test baseline",
            )
            baseline_judgment_errors = _judgment_error_rows(baseline_judgments, baseline_trajectories)
            if _error_rate_allowed(len(baseline_judgment_errors), len(baseline_judgments), config):
                baseline_by_hash = {item.trajectory_hash: item for item in baseline_judgments}
                baseline_labels: dict[str, str] = {}
                for trajectory in baseline_trajectories:
                    result = baseline_by_hash[trajectory.trajectory_hash]
                    if result.status == "OK" and result.label in {"REFUSAL", "NON_REFUSAL"}:
                        baseline_labels[trajectory.prompt_id] = result.label
                non_refusal_prompts = [
                    prompt for prompt in prompts if baseline_labels.get(prompt.prompt_id) == "NON_REFUSAL"
                ]
                non_refusal_ids = [prompt.prompt_id for prompt in non_refusal_prompts]
                base_quality_identity = object_sha256(
                    {
                        "evaluation_identity": evaluation_identity,
                        "stage": "base_quality",
                        "checkpoint_checksum": base_checkpoint_checksum,
                        "prompt_ids": non_refusal_ids,
                        "trajectory_hashes": [
                            trajectory.trajectory_hash
                            for trajectory in baseline_trajectories
                            if trajectory.prompt_id in baseline_labels
                            and baseline_labels[trajectory.prompt_id] == "NON_REFUSAL"
                        ],
                        "implementation_hash": _implementation_hash(
                            completed_non_refusal_completion_inputs,
                            compute_ce_loss,
                        ),
                    }
                )
                saved_quality = _load_test_base_quality(
                    base_quality_directory,
                    identity=base_quality_identity,
                    prompt_ids=non_refusal_ids,
                )
                if saved_quality is None:
                    base_logits = [
                        _next_token_logits(runtime, prompt)
                        for prompt in _progress(
                            non_refusal_prompts,
                            desc="Computing test baseline logits",
                            unit="prompt",
                        )
                    ]
                    ce_inputs = _completion_ce_inputs(
                        runtime,
                        baseline_trajectories,
                        baseline_labels,
                    )
                    base_ce = compute_ce_loss(runtime, ce_inputs, desc="Computing test baseline CE loss")
                    _save_test_base_quality(
                        base_quality_directory,
                        identity=base_quality_identity,
                        prompt_ids=non_refusal_ids,
                        base_logits=base_logits,
                        base_ce=base_ce,
                        ce_inputs=ce_inputs,
                    )
                else:
                    base_logits, base_ce, ce_inputs = saved_quality
    _record_error_diagnostics(
        config,
        store,
        phase="test baseline generation",
        total=len(baseline_trajectories),
        rows=baseline_generation_errors,
        path=store.paths.evaluation / "test_baseline_generation_errors.private.jsonl",
        artifact_type="test_baseline_generation_errors",
        profile=profile,
    )
    store.write_jsonl(
        test_baseline_path,
        baseline_trajectories,
        artifact_type="test_baseline_trajectories",
        profile=profile,
        private=True,
    )
    baseline_judgment_errors = _judgment_error_rows(baseline_judgments, baseline_trajectories)
    _record_error_diagnostics(
        config,
        store,
        phase="test baseline judging",
        total=len(baseline_judgments),
        rows=baseline_judgment_errors,
        path=store.paths.evaluation / "test_baseline_judgment_errors.private.jsonl",
        artifact_type="test_baseline_judgment_errors",
        profile=profile,
    )
    store.write_jsonl(
        test_baseline_judgments_path,
        baseline_judgments,
        artifact_type="test_baseline_judgments",
        profile=profile,
        private=False,
    )
    export_config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, id=str(store.paths.exported_model)),
    )
    with IntervenedModelRuntime(export_config, install_temporary=False) as runtime:
        if runtime.chat_template_hash != chat_template_hash:
            raise InvariantError("exported processor chat template differs from the pinned base processor")
        fingerprints = runtime.adapter.processor_fingerprints(runtime.processor)
        if any(initial_manifest.get(name) != value for name, value in fingerprints.items()):
            raise InvariantError("exported processor fingerprint differs from the export manifest")
        export_generation_identity = object_sha256(
            {
                "evaluation_identity": evaluation_identity,
                "stage": "export_generation",
                "checkpoint_checksum": runtime.checkpoint_checksum,
                "chat_template_hash": runtime.chat_template_hash,
                "generation_profile_hash": _target_generation_profile_hash(export_config, runtime),
                "processor_fingerprints": fingerprints,
                "implementation_hash": _implementation_hash(
                    TargetTrajectoryGenerator,
                    type(runtime.adapter),
                ),
            }
        )
        export_trajectories = _checkpointed_target_trajectories(
            runtime,
            export_config,
            prompts,
            directory=checkpoint_root / "export_trajectories",
            identity=export_generation_identity,
            desc="Generating test export",
        )
        export_generation_error_rows = _trajectory_error_rows(export_trajectories)
        if _error_rate_allowed(len(export_generation_error_rows), len(export_trajectories), config):
            export_quality_identity = object_sha256(
                {
                    "evaluation_identity": evaluation_identity,
                    "stage": "export_quality",
                    "checkpoint_checksum": runtime.checkpoint_checksum,
                    "base_quality_identity": base_quality_identity,
                }
            )
            with PrivateCheckpoint(
                export_quality_directory,
                identity=export_quality_identity,
                prompt_keys=("quality",),
            ) as quality_checkpoint:
                quality_entries = quality_checkpoint.load()
                if quality_entries:
                    quality_row = quality_entries[0].payload
                    if set(quality_row) != {"mean_kl", "completion_ce"}:
                        raise ArtifactError("test export-quality checkpoint fields are invalid")
                    mean_kl = quality_row["mean_kl"]
                    ce_value = quality_row["completion_ce"]
                    if (
                        mean_kl is not None
                        and (
                            isinstance(mean_kl, bool)
                            or not isinstance(mean_kl, int | float)
                            or not math.isfinite(mean_kl)
                        )
                    ) or not isinstance(ce_value, dict):
                        raise ArtifactError("test export-quality checkpoint values are invalid")
                    mean_kl = float(mean_kl) if mean_kl is not None else None
                    completion_ce = CEEvaluation.from_dict(ce_value)
                    if completion_ce.source != "baseline_non_refusal_completions":
                        raise ArtifactError("test export-quality checkpoint CE source is invalid")
                else:
                    if base_ce is None or not ce_inputs:
                        raise InvariantError("test baseline quality is unavailable")
                    export_logits = [
                        _next_token_logits(runtime, prompt)
                        for prompt in _progress(
                            non_refusal_prompts,
                            desc="Computing test export logits",
                            unit="prompt",
                        )
                    ]
                    try:
                        export_ce = compute_ce_loss(runtime, ce_inputs, desc="Computing test export CE loss")
                    except NonFiniteMetricError:
                        export_ce = None
                    completion_ce = ce_evaluation_from_losses(
                        base_ce,
                        export_ce,
                        source="baseline_non_refusal_completions",
                        input_count=len(ce_inputs),
                    )
                    measured_kl = mean_next_token_kl(base_logits, export_logits) if base_logits else 0.0
                    mean_kl = measured_kl if math.isfinite(measured_kl) else None
                    quality_checkpoint.write(
                        0,
                        "quality",
                        {"mean_kl": mean_kl, "completion_ce": completion_ce.as_dict()},
                    )
    _record_error_diagnostics(
        config,
        store,
        phase="test export generation",
        total=len(export_trajectories),
        rows=export_generation_error_rows,
        path=store.paths.evaluation / "test_export_generation_errors.private.jsonl",
        artifact_type="test_export_generation_errors",
        profile=profile,
    )
    store.write_jsonl(
        test_export_path,
        export_trajectories,
        artifact_type="test_export_trajectories",
        profile=profile,
        private=True,
    )
    with BaseModelRuntime(config) as runtime:
        _require_judge_validation(store, runtime)
        if runtime.checkpoint_checksum != base_checkpoint_checksum:
            raise InvariantError("test judges did not use the same immutable base checkpoint")
        export_judgment_identity = object_sha256(
            {
                "evaluation_identity": evaluation_identity,
                "stage": "export_judging",
                "checkpoint_checksum": runtime.checkpoint_checksum,
                "trajectory_hashes": [item.trajectory_hash for item in export_trajectories],
                "judge_validation_hash": validation_hash,
                "implementation_hash": _implementation_hash(
                    TrajectoryJudge,
                    EnumTrieConstrainedDecoder,
                    type(runtime.adapter),
                ),
            }
        )
        export_judgments = _checkpointed_judgments(
            runtime,
            export_trajectories,
            directory=checkpoint_root / "export_judgments",
            identity=export_judgment_identity,
            desc="Judging test export",
        )
    export_judgment_error_rows = _judgment_error_rows(export_judgments, export_trajectories)
    _record_error_diagnostics(
        config,
        store,
        phase="test export judging",
        total=len(export_judgments),
        rows=export_judgment_error_rows,
        path=store.paths.evaluation / "test_export_judgment_errors.private.jsonl",
        artifact_type="test_export_judgment_errors",
        profile=profile,
    )
    store.write_jsonl(
        test_export_judgments_path,
        export_judgments,
        artifact_type="test_export_judgments",
        profile=profile,
        private=False,
    )
    export_judgments_by_hash = {item.trajectory_hash: item for item in export_judgments}
    metrics = evaluate_behavior(
        candidate_id="exported_model",
        baseline_labels=baseline_labels,
        baseline_trajectories={item.prompt_id: item for item in baseline_trajectories},
        trajectories=[item for item in export_trajectories if item.prompt_id in baseline_labels],
        judgments=export_judgments_by_hash,
        mean_kl=mean_kl,
        completion_ce=completion_ce,
        acceptance=config.acceptance,
    )
    full_validation_report = _read_json_artifact(
        store,
        store.paths.full_validation_report,
        artifact_type="full_validation_report",
        profile=selection_profile,
    )
    test_metrics = metrics_dict(metrics)
    test_report = {
        "base_model_id": config.model.id,
        "base_revision": config.model.revision,
        "base_checkpoint_checksum": base_checkpoint_checksum,
        "baseline_counts": judgment_counts(baseline_judgments),
        "baseline_parser": parser_statistics(baseline_trajectories),
        "export_counts": judgment_counts(export_judgments),
        "export_parser": parser_statistics(export_trajectories),
        "test_metrics": test_metrics,
    }
    _write_json_artifact(
        store,
        store.paths.test_report,
        test_report,
        artifact_type="test_report",
        profile=profile,
    )
    _write_json_artifact(
        store,
        store.paths.quality_metrics,
        {
            "full_validation_metrics": full_validation_report["selected_candidate"]["full_validation_metrics"],
            "test_metrics": test_metrics,
        },
        artifact_type="quality_metrics",
        profile=profile,
    )
    store.write_json(store.paths.exported_model / "evaluation_report.json", test_report, private=False)
    final_manifest = dict(manifest)
    final_manifest["test_metrics"] = test_metrics
    final_manifest["test_evaluation"] = {
        "evaluation_identity": evaluation_identity,
        "artifacts": {
            "test_report": {"content_sha256": file_sha256(store.paths.test_report)},
            "quality_metrics": {"content_sha256": file_sha256(store.paths.quality_metrics)},
            "baseline_trajectories": {
                "content_sha256": file_sha256(test_baseline_path),
                "record_count": len(baseline_trajectories),
            },
            "baseline_judgments": {
                "content_sha256": file_sha256(test_baseline_judgments_path),
                "record_count": len(baseline_judgments),
            },
            "export_trajectories": {
                "content_sha256": file_sha256(test_export_path),
                "record_count": len(export_trajectories),
            },
            "export_judgments": {
                "content_sha256": file_sha256(test_export_judgments_path),
                "record_count": len(export_judgments),
            },
        },
        "evaluation_report_sha256": file_sha256(store.paths.exported_model / "evaluation_report.json"),
    }
    write_export_manifest(store.paths.exported_model, final_manifest)
    _evaluate_test_reference_ce(config, store, profile)
    for name in (
        "baseline_trajectories",
        "baseline_judgments",
        "export_trajectories",
        "export_judgments",
    ):
        _discard_checkpoint(checkpoint_root / name)
    _remove_managed_directory(base_quality_directory, missing_ok=True)
    _discard_checkpoint(export_quality_directory)


def _validate_export_manifest(
    config: ProjectConfig,
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    direction: torch.Tensor,
    judge_validation_hash_value: str,
    *,
    require_reload: bool = True,
) -> None:
    from self_judged_refusal_direction.exporting import export_implementation_hash

    expected = {
        "base_model_id": config.model.id,
        "base_revision": config.model.revision,
        "export_config_hash": config.stage_config_hash("export"),
        "target_generation_config_hash": config.target_generation_config_hash,
        "judge_profile_hash": current_judge_profile_hash(),
        "judge_validation_hash": judge_validation_hash_value,
        "export_implementation_hash": export_implementation_hash(),
    }
    mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
    if mismatches:
        raise ArtifactError(f"export manifest profile mismatch: {mismatches}")
    if object_sha256(manifest.get("full_validation_metrics")) != object_sha256(selection.get("metrics")):
        raise ArtifactError("export manifest full validation metrics differ from final selection")
    selected_candidate = selection.get("candidate")
    if not isinstance(selected_candidate, Mapping):
        raise ArtifactError("final selection candidate metadata is missing")
    source_fields = {"direction_source_layer": selected_candidate.get("layer")}
    if any(manifest.get(name) != value for name, value in source_fields.items()):
        raise ArtifactError("export manifest direction source differs from final selection")
    unit_direction = direction.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    norm = torch.linalg.vector_norm(unit_direction)
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise ArtifactError("selected direction has invalid norm")
    if manifest.get("direction_sha256") != tensor_sha256(unit_direction / norm):
        raise ArtifactError("export manifest direction hash differs from selected direction")
    equivalence = manifest.get("temporary_permanent_equivalence")
    if not isinstance(equivalence, Mapping) or equivalence.get("passed") is not True:
        raise ArtifactError("export manifest has no passing temporary/permanent equivalence")
    if manifest.get("untouched_parameters_verified") is not True:
        raise ArtifactError("export manifest has no untouched-parameter verification")
    if not require_reload:
        return
    reload_report = manifest.get("fresh_reload")
    required_reload = {
        "status": "OK",
        "tied_weights_preserved": True,
        "probe_logits_match": True,
        "processor_reload_verified": True,
        "target_trajectory_required": True,
        "target_trajectory_passed": True,
    }
    if not isinstance(reload_report, Mapping) or any(
        reload_report.get(name) != value for name, value in required_reload.items()
    ):
        raise ArtifactError("export manifest fresh reload verification is incomplete")
    if reload_report.get("parameter_shapes_hash") != manifest.get("parameter_shapes_hash") or reload_report.get(
        "parameter_count"
    ) != manifest.get("parameter_count"):
        raise ArtifactError("export manifest topology differs from fresh reload verification")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"JSON artifact is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def run_pipeline(config_path: str) -> None:
    stages = (
        ("inspect model", inspect_model),
        ("validate judge", _ensure_judge_validation),
        ("generate baselines", generate_baseline_trajectories),
        ("judge baselines", judge_baseline_trajectories),
        ("collect activations", collect_activations),
        ("build candidates", build_direction_candidates),
        ("evaluate candidates", evaluate_candidates),
        ("export model", export_model),
        ("evaluate export", evaluate_export),
    )
    progress = tqdm(
        stages,
        desc="Pipeline",
        unit="stage",
        dynamic_ncols=True,
        disable=None,
    )
    for name, stage in progress:
        progress.set_postfix_str(name)
        stage(config_path)
