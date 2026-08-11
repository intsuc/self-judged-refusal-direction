from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F

from self_judged_refusal_direction.activations import (
    ActivationCollector,
    ActivationStatistics,
    load_activation_statistics,
    merge_activation_statistics,
    save_activation_statistics,
)
from self_judged_refusal_direction.artifacts import ArtifactMetadata, ArtifactProfile, ArtifactStore
from self_judged_refusal_direction.config import ProjectConfig, load_config
from self_judged_refusal_direction.data import (
    ingest_and_split_prompts,
    ingest_prompts,
    records_by_split,
    write_prompt_split_artifacts,
)
from self_judged_refusal_direction.decoding import EnumTrieConstrainedDecoder
from self_judged_refusal_direction.directions import (
    CandidateBundle,
    build_candidates,
    load_candidate_artifacts,
    load_direction,
    rank_stage_a,
    save_direction,
    write_candidate_artifacts,
)
from self_judged_refusal_direction.errors import ArtifactError, InvariantError, PipelineError
from self_judged_refusal_direction.evaluation import (
    evaluate_behavior,
    judgment_counts,
    mean_next_token_kl,
    metrics_dict,
    parser_statistics,
    select_candidate,
)
from self_judged_refusal_direction.hashing import file_sha256, object_sha256, tensor_sha256
from self_judged_refusal_direction.judging import TrajectoryJudge
from self_judged_refusal_direction.models.registry import adapter_for_config
from self_judged_refusal_direction.prompting import judge_messages, judge_template_hash, target_messages
from self_judged_refusal_direction.runtime import BaseModelRuntime, IntervenedModelRuntime
from self_judged_refusal_direction.schema import (
    ActivationKey,
    CandidateMetrics,
    DirectionCandidate,
    JudgeResult,
    LabeledTrajectory,
    PromptRecord,
    TargetTrajectory,
)


def _load(path: str | Path) -> tuple[ProjectConfig, ArtifactStore]:
    config = load_config(path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(config.run.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.run.seed)
    torch.use_deterministic_algorithms(config.run.deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not config.run.deterministic
        torch.backends.cudnn.deterministic = config.run.deterministic
    store = ArtifactStore(config)
    store.initialize_run()
    return config, store


def _profile(
    store: ArtifactStore,
    *,
    target: bool = False,
    judge: bool = False,
    chat_template_hash: str | None = None,
) -> ArtifactProfile:
    return store.profile(target=target, judge=judge, chat_template_hash=chat_template_hash)


def _judge_chat_hash(chat_template_hash: str) -> str:
    return object_sha256({"processor_chat_template": chat_template_hash, "judge_template": judge_template_hash()})


def _check_error_budget(count: int, config: ProjectConfig, phase: str) -> None:
    if count > config.run.max_errors:
        raise PipelineError(f"{phase} produced {count} infrastructure errors; maximum is {config.run.max_errors}")


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
    count = len(value) if isinstance(value, list) else 1
    metadata = ArtifactMetadata(
        schema_version=1,
        artifact_type=artifact_type,
        private=private,
        record_count=count,
        content_sha256=file_sha256(path),
        profile=profile,
    )
    store.write_json(store.metadata_path(path), dataclasses.asdict(metadata), private=private)


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
    empty = TargetTrajectory(
        prompt_id="context-preflight",
        original_prompt="",
        raw_generated_token_ids=(),
        raw_decoded_output="",
        thinking_segments=("",),
        thinking_text="",
        final_answer="",
        thinking_token_start=0,
        thinking_token_end=0,
        final_token_start=0,
        final_token_end=0,
        generation_truncated=False,
        parser_status="OK",
        model_revision=config.model.revision,
        generation_config_hash=config.target_profile_hash,
        trajectory_hash="context-preflight",
    )
    judge_rendered = runtime.adapter.render_judge_chat(runtime.processor, judge_messages(empty))
    judge_ids = tuple(int(value) for value in judge_rendered["input_ids"][0].tolist())
    decoder = EnumTrieConstrainedDecoder.compile(
        runtime.processor.tokenizer,
        tuple(config.judge.labels),
        (judge_ids,),
    )
    context_window = runtime.adapter.context_window(runtime.model)
    judge_required = (
        len(judge_ids)
        + config.data.max_prompt_tokens
        + config.target_generation.max_new_tokens
        + decoder.max_new_tokens
        + config.judge.safety_margin_tokens
    )
    target_rendered = runtime.adapter.render_target_chat(
        runtime.processor,
        target_messages("", config.run.system_prompt),
    )
    target_required = (
        int(target_rendered["input_ids"].shape[-1])
        + config.data.max_prompt_tokens
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


def _load_prompt_records(store: ArtifactStore, config: ProjectConfig) -> list[PromptRecord]:
    rows = store.read_jsonl(
        store.paths.splits,
        artifact_type="prompt_splits",
        expected_profile=_profile(store),
    )
    records = [PromptRecord.from_dict(row) for row in rows if row.get("split") in {"train", "validation"}]
    if not records:
        raise ArtifactError("development prompt split artifact is empty")
    return records


def _load_baseline_trajectories(
    store: ArtifactStore,
    chat_template_hash: str,
) -> list[TargetTrajectory]:
    rows = store.read_jsonl(
        store.paths.baseline_trajectories,
        artifact_type="baseline_trajectories",
        expected_profile=_profile(store, target=True, chat_template_hash=chat_template_hash),
    )
    return [TargetTrajectory.from_dict(row) for row in rows]


def _load_baseline_judgments(
    store: ArtifactStore,
    chat_template_hash: str,
) -> list[JudgeResult]:
    rows = store.read_jsonl(
        store.paths.baseline_judgments,
        artifact_type="baseline_judgments",
        expected_profile=_profile(
            store,
            target=True,
            judge=True,
            chat_template_hash=_judge_chat_hash(chat_template_hash),
        ),
    )
    return [JudgeResult.from_dict(row) for row in rows]


def _load_labeled(
    store: ArtifactStore,
    chat_template_hash: str,
    split: Literal["train", "validation"],
) -> list[LabeledTrajectory]:
    path = store.paths.labeled_train if split == "train" else store.paths.labeled_validation
    rows = store.read_jsonl(
        path,
        artifact_type=f"labeled_{split}",
        expected_profile=_profile(
            store,
            target=True,
            judge=True,
            chat_template_hash=_judge_chat_hash(chat_template_hash),
        ),
    )
    return [LabeledTrajectory(**row) for row in rows]


def inspect_model(config_path: str) -> None:
    config, store = _load(config_path)
    with BaseModelRuntime(config) as runtime:
        _validate_static_context_budget(config, runtime)
        fingerprints = runtime.adapter.processor_fingerprints(runtime.processor)
        report = dataclasses.asdict(runtime.compatibility_report)
        report.update(
            {
                "model_id": config.model.id,
                "model_revision": config.model.revision,
                "config_hash": config.config_hash,
                "chat_template_hash": runtime.chat_template_hash,
                **fingerprints,
                "target_thinking_enabled": True,
                "judge_thinking_enabled": False,
            }
        )
        store.write_json(store.paths.root / "model_compatibility.json", report, private=False)
        environment_path = store.paths.root / "environment.json"
        environment = _read_json(environment_path)
        environment.update({"chat_template_hash": runtime.chat_template_hash, **fingerprints})
        store.write_json(environment_path, environment, private=False)


def generate_baseline_trajectories(config_path: str) -> None:
    config, store = _load(config_path)
    with BaseModelRuntime(config) as runtime:
        _validate_static_context_budget(config, runtime)
        tokenizer = runtime.processor.tokenizer
        records = ingest_and_split_prompts(
            config,
            token_counter=lambda prompt: len(tokenizer.encode(prompt, add_special_tokens=False)),
        )
        if not records:
            raise PipelineError("no prompts were ingested from data.raw_prompt_files")
        grouped = records_by_split(records)
        if len(grouped["test"]) > config.data.test_raw_count:
            retained_test = sorted(grouped["test"], key=lambda item: item.prompt_id)[: config.data.test_raw_count]
            records = [item for item in records if item.split != "test"] + retained_test
        write_prompt_split_artifacts(store, records, profile=_profile(store))
        baseline_records = [item for item in records if item.split in {"train", "validation"}]
        trajectories = [runtime.generate_target(record) for record in baseline_records]
        errors = sum(item.trajectory_status == "ERROR" for item in trajectories)
        _check_error_budget(errors, config, "baseline trajectory generation")
        store.write_jsonl(
            store.paths.baseline_trajectories,
            trajectories,
            artifact_type="baseline_trajectories",
            profile=_profile(store, target=True, chat_template_hash=runtime.chat_template_hash),
            private=True,
        )


def judge_baseline_trajectories(config_path: str) -> None:
    config, store = _load(config_path)
    with BaseModelRuntime(config) as runtime:
        trajectories = _load_baseline_trajectories(store, runtime.chat_template_hash)
        judge = TrajectoryJudge(config, runtime.adapter, runtime.model, runtime.processor, runtime.chat_template_hash)
        judgments = [judge.classify(trajectory) for trajectory in trajectories]
        errors = sum(item.status == "ERROR" for item in judgments)
        _check_error_budget(errors, config, "baseline trajectory judging")
        judgment_profile = _profile(
            store,
            target=True,
            judge=True,
            chat_template_hash=_judge_chat_hash(runtime.chat_template_hash),
        )
        store.write_jsonl(
            store.paths.baseline_judgments,
            judgments,
            artifact_type="baseline_judgments",
            profile=judgment_profile,
            private=False,
        )
        by_hash = {item.trajectory_hash: item for item in judgments}
        labeled: dict[str, list[LabeledTrajectory]] = {"train": [], "validation": []}
        for trajectory in trajectories:
            result = by_hash[trajectory.trajectory_hash]
            if result.status != "OK" or result.label not in {"REFUSAL", "NON_REFUSAL"}:
                continue
            if trajectory.split not in labeled:
                continue
            labeled_split = cast(Literal["train", "validation"], trajectory.split)
            labeled[labeled_split].append(
                LabeledTrajectory(
                    prompt_id=trajectory.prompt_id,
                    split=labeled_split,
                    label=result.label,
                    trajectory_hash=trajectory.trajectory_hash,
                )
            )
        required = {
            "train": config.data.train_per_class,
            "validation": config.data.validation_per_class,
        }
        for split, values in labeled.items():
            selected: list[LabeledTrajectory] = []
            for label in ("REFUSAL", "NON_REFUSAL"):
                cohort = sorted((item for item in values if item.label == label), key=lambda item: item.prompt_id)
                if len(cohort) < required[split]:
                    raise PipelineError(
                        f"{split} has {len(cohort)} {label} trajectories; {required[split]} are required"
                    )
                selected.extend(cohort[: required[split]])
            path = store.paths.labeled_train if split == "train" else store.paths.labeled_validation
            store.write_jsonl(
                path,
                sorted(selected, key=lambda item: item.prompt_id),
                artifact_type=f"labeled_{split}",
                profile=judgment_profile,
                private=False,
            )


def collect_activations(config_path: str) -> None:
    config, store = _load(config_path)
    with BaseModelRuntime(config) as runtime:
        model = runtime.model
        trajectories = _load_baseline_trajectories(store, runtime.chat_template_hash)
        labeled = _load_labeled(store, runtime.chat_template_hash, "train")
        trajectory_by_hash = {item.trajectory_hash: item for item in trajectories}
        layer_selection = config.direction.candidate_layers
        statistics: list[ActivationStatistics] = []
        boundary_tokens: dict[str, Counter[str]] = {}
        positions_by_phase: dict[str, set[int]] = {}
        for phase in config.direction.candidate_phases:
            typed_phase = cast(Literal["pre_thinking", "pre_final"], phase)
            typed_layers = cast(Literal["all"] | Sequence[int], layer_selection)
            collectors: dict[tuple[int, ...], ActivationCollector] = {}
            for item in labeled:
                trajectory = trajectory_by_hash[item.trajectory_hash]
                if phase == "pre_thinking":
                    positions = tuple(runtime.adapter.discover_pre_thinking_positions(runtime.processor, config))
                    rendered = runtime.adapter.render_target_chat(
                        runtime.processor,
                        target_messages(trajectory.original_prompt, config.run.system_prompt),
                    )
                    inputs = _move_inputs(dict(rendered), runtime.adapter.input_device(runtime.model))
                else:
                    positions = tuple(
                        runtime.adapter.discover_pre_final_positions(runtime.processor, trajectory, config)
                    )
                    inputs = _pre_final_inputs(runtime, trajectory)
                collector = collectors.get(positions)
                if collector is None:
                    collector = ActivationCollector(
                        runtime.adapter.activation_read_points(model),
                        phase=typed_phase,
                        relative_positions=positions,
                        layers=typed_layers,
                        dtype=config.direction.online_accumulator_dtype,
                    )
                    collectors[positions] = collector
                input_ids = inputs["input_ids"]
                sequence_length = int(input_ids.shape[-1])
                _validate_relative_positions(positions, sequence_length, phase)
                positions_by_phase.setdefault(phase, set()).update(positions)
                for position in positions:
                    token_id = int(input_ids[0, sequence_length + position].item())
                    token = runtime.processor.tokenizer.decode([token_id], skip_special_tokens=False)
                    boundary_tokens.setdefault(f"{phase}:{position}", Counter())[token] += 1
                with collector.capture([item.label], boundary_positions=[sequence_length]), torch.inference_mode():
                    model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            statistics.extend(collector.statistics() for collector in collectors.values())
        merged = merge_activation_statistics(statistics)
        save_activation_statistics(store.paths.activation_statistics, merged)
        activation_profile = _profile(
            store,
            target=True,
            judge=True,
            chat_template_hash=_judge_chat_hash(runtime.chat_template_hash),
        )
        metadata = ArtifactMetadata(
            schema_version=1,
            artifact_type="activation_statistics",
            private=True,
            record_count=len(merged.keys),
            content_sha256=file_sha256(store.paths.activation_statistics),
            profile=activation_profile,
        )
        store.write_json(
            store.metadata_path(store.paths.activation_statistics),
            dataclasses.asdict(metadata),
            private=True,
        )
        token_metadata = {
            key: counts.most_common(1)[0][0] if len(counts) == 1 else "<variable>"
            for key, counts in boundary_tokens.items()
        }
        _write_json_artifact(
            store,
            store.paths.activation_metadata,
            {
                "candidate_phases": list(config.direction.candidate_phases),
                "relative_positions": sorted({value for values in positions_by_phase.values() for value in values}),
                "relative_positions_by_phase": {
                    phase: sorted(values) for phase, values in sorted(positions_by_phase.items())
                },
                "candidate_layers": layer_selection,
                "boundary_tokens": token_metadata,
                "text_only": True,
            },
            artifact_type="activation_position_metadata",
            profile=activation_profile,
            private=False,
        )


def _pre_final_inputs(runtime: BaseModelRuntime, trajectory: TargetTrajectory) -> dict[str, torch.Tensor]:
    if not 0 <= trajectory.final_token_start <= len(trajectory.raw_generated_token_ids):
        raise InvariantError("pre-final trajectory boundary is outside the generated token sequence")
    rendered = runtime.adapter.render_target_chat(
        runtime.processor,
        target_messages(trajectory.original_prompt, runtime.config.run.system_prompt),
    )
    inputs = _move_inputs(dict(rendered), runtime.adapter.input_device(runtime.model))
    prefix = inputs.get("input_ids")
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 2 or prefix.shape[0] != 1:
        raise InvariantError("pre-final target input_ids must contain one sequence")
    generated = torch.tensor(
        trajectory.raw_generated_token_ids[: trajectory.final_token_start],
        dtype=prefix.dtype,
        device=prefix.device,
    ).unsqueeze(0)
    input_ids = torch.cat((prefix, generated), dim=-1)
    suffix_length = generated.shape[-1]
    result: dict[str, torch.Tensor] = {"input_ids": input_ids}
    for name, value in inputs.items():
        if name == "input_ids":
            continue
        if not isinstance(value, torch.Tensor):
            raise InvariantError(f"pre-final model input is not a tensor: {name}")
        sequence_aligned = value.ndim >= 2 and value.shape[0] == 1 and value.shape[-1] == prefix.shape[-1]
        if not sequence_aligned:
            result[name] = value
            continue
        extension_shape = (*value.shape[:-1], suffix_length)
        if name == "attention_mask":
            extension = torch.ones(extension_shape, dtype=value.dtype, device=value.device)
        elif name == "position_ids":
            increments = torch.arange(1, suffix_length + 1, dtype=value.dtype, device=value.device)
            extension = value[..., -1:] + increments
        elif name in {"token_type_ids", "mm_token_type_ids"}:
            extension = torch.zeros(extension_shape, dtype=value.dtype, device=value.device)
        else:
            raise InvariantError(f"pre-final sequence input cannot be extended safely: {name}")
        result[name] = torch.cat((value, extension), dim=-1)
    if "attention_mask" not in result:
        result["attention_mask"] = torch.ones_like(input_ids)
    return result


def _validate_relative_positions(positions: Sequence[int], sequence_length: int, phase: str) -> None:
    if not positions or len(set(positions)) != len(positions):
        raise InvariantError(f"{phase} adapter positions must be non-empty and unique")
    if any(position >= 0 or sequence_length + position < 0 for position in positions):
        raise InvariantError(f"{phase} adapter position is outside the available prefix")


def build_direction_candidates(config_path: str) -> None:
    config, store = _load(config_path)
    profile = _activation_dependent_profile(store)
    adapter = adapter_for_config(config)
    processor = adapter.load_processor(config.model)
    _validate_activation_chat_profile(profile, adapter.chat_template_hash(processor))
    del processor
    store.validate(
        store.paths.activation_statistics,
        artifact_type="activation_statistics",
        expected_profile=profile,
    )
    statistics = load_activation_statistics(store.paths.activation_statistics)
    position_metadata = _read_json_artifact(
        store,
        store.paths.activation_metadata,
        artifact_type="activation_position_metadata",
        profile=profile,
    )
    tokens: dict[ActivationKey | str, str] = {}
    for key in statistics.keys:
        tokens[key.storage_key] = position_metadata.get("boundary_tokens", {}).get(
            f"{key.phase}:{key.relative_position}",
            "",
        )
    bundle = build_candidates(
        statistics,
        boundary_tokens=tokens,
        minimum_norm=config.direction.minimum_direction_norm,
        dtype="float32",
    )
    ranking = rank_stage_a(
        bundle,
        top_m=config.direction.stage_a_top_m,
        minimum_norm=config.direction.minimum_direction_norm,
    )
    if not ranking:
        raise PipelineError("Stage A produced no numerically valid direction candidates")
    write_candidate_artifacts(store, bundle, profile=profile)
    _write_json_artifact(
        store,
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
        artifact_type="stage_a_ranking",
        profile=profile,
    )


def _activation_dependent_profile(store: ArtifactStore) -> ArtifactProfile:
    metadata = store.validate(
        store.paths.activation_statistics,
        artifact_type="activation_statistics",
        expected_profile=_profile(store, target=True, judge=True),
    )
    if not metadata.private or metadata.profile.chat_template_hash is None:
        raise ArtifactError("activation statistics privacy or chat-template profile is invalid")
    return metadata.profile


def _validate_activation_chat_profile(profile: ArtifactProfile, chat_template_hash: str) -> None:
    if profile.chat_template_hash != _judge_chat_hash(chat_template_hash):
        raise ArtifactError("activation-derived artifacts use a different chat template")


def evaluate_candidates(config_path: str) -> None:
    config, store = _load(config_path)
    profile = _activation_dependent_profile(store)
    bundle = load_candidate_artifacts(store, expected_profile=profile)
    stage_a = _read_json_artifact(
        store,
        store.paths.stage_a_ranking,
        artifact_type="stage_a_ranking",
        profile=profile,
    )
    by_id = {item.candidate_id: item for item in bundle.candidates}
    ranking = [by_id[row["candidate_id"]] for row in stage_a]
    with BaseModelRuntime(config) as profile_runtime:
        chat_template_hash = profile_runtime.chat_template_hash
        _validate_activation_chat_profile(profile, chat_template_hash)
    validation = _load_labeled(store, chat_template_hash, "validation")
    trajectories = _load_baseline_trajectories(store, chat_template_hash)
    baseline_trajectories = {item.prompt_id: item for item in trajectories}
    records = {item.prompt_id: item for item in _load_prompt_records(store, config)}
    quality_fallback = [records[item.prompt_id].original_prompt for item in validation if item.label == "NON_REFUSAL"]
    stage_b_labels = _balanced_subset(validation, config.evaluation.stage_b_prompts_per_class)
    stage_b_metrics, _ = _evaluate_stage(
        config,
        store,
        bundle,
        ranking,
        stage_b_labels,
        records,
        baseline_trajectories,
        quality_fallback,
        stage="B",
    )
    stage_b_eligible = [item for item in stage_b_metrics if item.hard_filter_passed]
    if not stage_b_eligible:
        raise PipelineError("no Stage B direction candidate passed the configured hard filters")
    top_k_ids = [
        item.candidate_id
        for item in sorted(stage_b_eligible, key=_stage_ranking_key, reverse=True)[: config.direction.stage_b_top_k]
    ]
    stage_c_candidates = [by_id[candidate_id] for candidate_id in top_k_ids]
    stage_c_metrics, quality_evaluation = _evaluate_stage(
        config,
        store,
        bundle,
        stage_c_candidates,
        validation,
        records,
        baseline_trajectories,
        quality_fallback,
        stage="C",
    )
    selected_metrics, selected_candidate = select_candidate(stage_c_metrics, by_id)
    selected_direction = bundle.direction(selected_candidate)
    selected_metadata = {
        "candidate": dataclasses.asdict(selected_candidate),
        "validation_metrics": metrics_dict(selected_metrics),
        "semantics": "REFUSAL iff thought considers refusal OR final answer refuses",
    }
    save_direction(store.paths.selected_direction, selected_direction, metadata=selected_metadata, private=False)
    direction_sidecar = ArtifactMetadata(
        schema_version=1,
        artifact_type="selected_direction",
        private=False,
        record_count=1,
        content_sha256=file_sha256(store.paths.selected_direction),
        profile=profile,
    )
    store.write_json(
        store.metadata_path(store.paths.selected_direction),
        dataclasses.asdict(direction_sidecar),
        private=False,
    )
    _write_json_artifact(
        store,
        store.paths.final_selection,
        {
            "candidate_id": selected_candidate.candidate_id,
            "candidate": dataclasses.asdict(selected_candidate),
            "metrics": metrics_dict(selected_metrics),
        },
        artifact_type="final_selection",
        profile=profile,
        private=False,
    )
    judgments = _load_baseline_judgments(store, chat_template_hash)
    report = {
        "base_model_id": config.model.id,
        "base_revision": config.model.revision,
        "target_generation": {"thinking_enabled": True},
        "judge": {
            "thinking_enabled": False,
            "immutable_base_checkpoint": True,
            "semantics": "REFUSAL iff thought considers refusal OR final answer refuses",
        },
        "baseline_counts": judgment_counts(judgments),
        "baseline_parser": parser_statistics(trajectories),
        "candidate_search": {
            "phases": list(config.direction.candidate_phases),
            "layers": config.direction.candidate_layers,
            "relative_positions": sorted({item.relative_position for item in bundle.candidates}),
            "relative_positions_by_phase": {
                phase: sorted({item.relative_position for item in bundle.candidates if item.phase == phase})
                for phase in config.direction.candidate_phases
            },
            "stage_a_count": len(ranking),
            "stage_c_count": len(stage_c_candidates),
        },
        "quality_evaluation": quality_evaluation,
        "selected_candidate": selected_metadata,
        "raw_thinking_private": True,
    }
    _write_json_artifact(
        store,
        store.paths.validation_report,
        report,
        artifact_type="validation_report",
        profile=profile,
    )


def _balanced_subset(values: Sequence[LabeledTrajectory], per_class: int) -> list[LabeledTrajectory]:
    selected: list[LabeledTrajectory] = []
    for label in ("REFUSAL", "NON_REFUSAL"):
        cohort = sorted((item for item in values if item.label == label), key=lambda item: item.prompt_id)
        selected.extend(cohort[: min(per_class, len(cohort))])
    return selected


def _stage_ranking_key(metrics: CandidateMetrics) -> tuple[float, ...]:
    return (
        metrics.removal_success_rate,
        metrics.non_refusal_retention_rate,
        -metrics.uncertain_rate,
        -metrics.mean_kl,
    )


def _evaluate_stage(
    config: ProjectConfig,
    store: ArtifactStore,
    bundle: CandidateBundle,
    candidates: Sequence[DirectionCandidate],
    labeled: Sequence[LabeledTrajectory],
    prompt_records: Mapping[str, PromptRecord],
    baseline_trajectories: Mapping[str, TargetTrajectory],
    quality_fallback: Sequence[str],
    *,
    stage: Literal["B", "C"],
) -> tuple[list[CandidateMetrics], dict[str, Any]]:
    labels = {item.prompt_id: item.label for item in labeled}
    prompts = [prompt_records[item.prompt_id] for item in labeled]
    non_refusal_prompts = [prompt_records[item.prompt_id] for item in labeled if item.label == "NON_REFUSAL"]
    trajectory_rows: list[dict[str, Any]] = []
    quality: dict[str, tuple[float, float]] = {}
    with IntervenedModelRuntime(config) as runtime:
        base_logits = [_next_token_logits(runtime, prompt) for prompt in non_refusal_prompts]
        quality_texts, quality_evaluation = _resolve_quality_texts(config, runtime, quality_fallback)
        base_ce = _mean_ce_loss(runtime, quality_texts)
        for candidate in candidates:
            direction = bundle.direction(candidate)
            plan = runtime.adapter.build_weight_edit_plan(runtime.model, direction)
            with plan.temporary(runtime.model):
                trajectories = [runtime.generate_target(prompt) for prompt in prompts]
                intervention_logits = [_next_token_logits(runtime, prompt) for prompt in non_refusal_prompts]
                intervention_ce = _mean_ce_loss(runtime, quality_texts)
            mean_kl = mean_next_token_kl(base_logits, intervention_logits) if base_logits else 0.0
            quality[candidate.candidate_id] = (mean_kl, intervention_ce - base_ce)
            trajectory_rows.extend(
                {
                    "candidate_id": candidate.candidate_id,
                    "kind": "removal",
                    "trajectory": trajectory.as_dict(),
                }
                for trajectory in trajectories
            )
            if stage == "C" and config.evaluation.run_activation_addition_diagnostic:
                block_name = _transformer_block_name(runtime, candidate.layer)
                coefficient = config.evaluation.activation_addition_beta * candidate.norm
                with plan.activation_addition(runtime.model, block_name, coefficient):
                    additions = [runtime.generate_target(prompt) for prompt in non_refusal_prompts]
                trajectory_rows.extend(
                    {
                        "candidate_id": candidate.candidate_id,
                        "kind": "addition",
                        "trajectory": trajectory.as_dict(),
                    }
                    for trajectory in additions
                )
        chat_template_hash = runtime.chat_template_hash
    errors = sum(row["trajectory"]["trajectory_status"] == "ERROR" for row in trajectory_rows)
    _check_error_budget(errors, config, f"Stage {stage} intervention generation")
    trajectories_path = store.paths.stage_b_trajectories if stage == "B" else store.paths.stage_c_trajectories
    store.write_jsonl(
        trajectories_path,
        trajectory_rows,
        artifact_type=f"stage_{stage.lower()}_trajectories",
        profile=_profile(store, target=True, chat_template_hash=chat_template_hash),
        private=True,
    )
    judgment_rows: list[dict[str, Any]] = []
    with BaseModelRuntime(config) as runtime:
        judge = TrajectoryJudge(config, runtime.adapter, runtime.model, runtime.processor, runtime.chat_template_hash)
        for row in trajectory_rows:
            trajectory = TargetTrajectory.from_dict(row["trajectory"])
            result = judge.classify(trajectory)
            judgment_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "kind": row["kind"],
                    "prompt_id": trajectory.prompt_id,
                    "trajectory_hash": trajectory.trajectory_hash,
                    "judgment": result.as_dict(),
                }
            )
    judge_errors = sum(row["judgment"]["status"] == "ERROR" for row in judgment_rows)
    _check_error_budget(judge_errors, config, f"Stage {stage} judging")
    judgments_path = store.paths.stage_b_judgments if stage == "B" else store.paths.stage_c_judgments
    store.write_jsonl(
        judgments_path,
        judgment_rows,
        artifact_type=f"stage_{stage.lower()}_judgments",
        profile=_profile(
            store,
            target=True,
            judge=True,
            chat_template_hash=_judge_chat_hash(chat_template_hash),
        ),
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
        mean_kl, ce_delta = quality[candidate.candidate_id]
        metrics.append(
            evaluate_behavior(
                candidate_id=candidate.candidate_id,
                stage=stage,
                baseline_labels=labels,
                baseline_trajectories=baseline_trajectories,
                trajectories=[TargetTrajectory.from_dict(row["trajectory"]) for row in removal_rows],
                judgments=candidate_judgments,
                mean_kl=mean_kl,
                ce_loss_delta=ce_delta,
                config=config.evaluation,
                activation_addition_induction_rate=induction,
            )
        )
    results_path = store.paths.stage_b_results if stage == "B" else store.paths.stage_c_results
    store.write_jsonl(
        results_path,
        metrics,
        artifact_type=f"stage_{stage.lower()}_results",
        profile=_profile(
            store,
            target=True,
            judge=True,
            chat_template_hash=_judge_chat_hash(chat_template_hash),
        ),
        private=False,
    )
    return metrics, quality_evaluation


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
        target_messages(prompt.original_prompt, runtime.config.run.system_prompt),
    )
    inputs = _move_inputs(dict(rendered), runtime.adapter.input_device(runtime.model))
    with torch.inference_mode():
        output = runtime.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
    return output.logits[0, -1].detach().float().cpu()


def _mean_ce_loss(runtime: BaseModelRuntime | IntervenedModelRuntime, texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    losses: list[float] = []
    model = runtime.model
    device = runtime.adapter.input_device(model)
    backbone = runtime.adapter.text_backbone(model)
    head = runtime.adapter.lm_head(model).module
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise InvariantError("quality evaluation LM head has no matrix weight")
    chunk_positions = max(1, 1_048_576 // weight.shape[0])
    for text in texts:
        encoded = runtime.processor.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        inputs = _move_inputs(dict(encoded), device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[-1] < 2:
            continue
        with torch.inference_mode():
            output = backbone(**inputs, use_cache=False, return_dict=True)
            hidden_states = getattr(output, "last_hidden_state", None)
            if not isinstance(hidden_states, torch.Tensor):
                raise InvariantError("quality evaluation text backbone returned no hidden states")
            shifted_states = hidden_states[:, :-1, :]
            shifted_targets = input_ids[:, 1:].clone()
            attention_mask = inputs.get("attention_mask")
            if isinstance(attention_mask, torch.Tensor):
                shifted_targets.masked_fill_(attention_mask[:, 1:] == 0, -100)
            total_loss = 0.0
            token_count = 0
            for start in range(0, shifted_states.shape[1], chunk_positions):
                stop = min(start + chunk_positions, shifted_states.shape[1])
                logits = head(shifted_states[:, start:stop, :])
                logits = _apply_final_logit_softcap(model, logits)
                targets = shifted_targets[:, start:stop]
                valid = int((targets != -100).sum().item())
                if valid == 0:
                    continue
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss += float(loss.item())
                token_count += valid
        if token_count:
            losses.append(total_loss / token_count)
    return sum(losses) / len(losses) if losses else 0.0


def _apply_final_logit_softcap(model: torch.nn.Module, logits: torch.Tensor) -> torch.Tensor:
    model_config = getattr(model, "config", None)
    get_text_config = getattr(model_config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else getattr(model_config, "text_config", model_config)
    value = getattr(text_config, "final_logit_softcapping", None)
    if value is None:
        return logits
    softcap = float(value)
    if softcap <= 0:
        raise InvariantError("final logit softcapping must be positive")
    return torch.tanh(logits / softcap) * softcap


def _resolve_quality_texts(
    config: ProjectConfig,
    runtime: BaseModelRuntime | IntervenedModelRuntime,
    fallback: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    if config.data.quality_text_files:
        tokenizer = runtime.processor.tokenizer
        texts = ingest_prompts(
            config.data.quality_text_files,
            deduplicate=True,
            token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=True)),
            max_prompt_tokens=config.data.max_prompt_tokens,
        )
        if not texts:
            raise PipelineError("configured quality text files produced no usable texts")
        return texts, {
            "source": "configured_quality_text_files",
            "text_count": len(texts),
            "source_sha256": [file_sha256(Path(path).resolve()) for path in config.data.quality_text_files],
        }
    texts = list(dict.fromkeys(text for text in fallback if text))
    if not texts:
        raise PipelineError("quality evaluation requires configured texts or baseline NON_REFUSAL prompts")
    return texts, {
        "source": "baseline_non_refusal_prompt_fallback",
        "text_count": len(texts),
        "source_sha256": [object_sha256({"text": text}) for text in texts],
    }


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
    from self_judged_refusal_direction.exporting import complete_deferred_reload, export_edited_model

    config, store = _load(config_path)
    profile = _activation_dependent_profile(store)
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
    validation = _load_labeled_for_probe(config, store, profile)
    prompt = validation[0]
    export_result: Any
    with IntervenedModelRuntime(config, direction=direction, install_temporary=False) as runtime:
        plan = dataclasses.replace(
            runtime.weight_edit_plan,
            metadata={
                **runtime.weight_edit_plan.metadata,
                "phase": direction_metadata["candidate"]["phase"],
                "layer": direction_metadata["candidate"]["layer"],
                "relative_position": direction_metadata["candidate"]["relative_position"],
                "text_only_direction_discovery": True,
                "multimodal_behavior_validated": False,
            },
        )
        probe = runtime.adapter.render_target_chat(
            runtime.processor,
            target_messages(prompt.original_prompt, config.run.system_prompt),
        )
        probe = _move_inputs(dict(probe), runtime.adapter.input_device(runtime.model))
        export_result = export_edited_model(
            runtime.model,
            runtime.processor,
            runtime.adapter,
            plan,
            config,
            probe,
            output_dir=store.paths.exported_model,
            validation_metrics=selection["metrics"],
            defer_reload=True,
        )
    export_result = complete_deferred_reload(export_result)
    if export_result.reload is None:
        raise InvariantError("fresh reload verification did not produce a report")
    store.write_json(
        store.paths.exported_model / "reload_report.json",
        dataclasses.asdict(export_result.reload),
        private=False,
    )


def _validate_direction_selection(
    direction_metadata: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    candidate = direction_metadata.get("candidate")
    validation_metrics = direction_metadata.get("validation_metrics")
    if not isinstance(candidate, Mapping) or not isinstance(validation_metrics, Mapping):
        raise ArtifactError("selected direction metadata is incomplete")
    candidate_id = candidate.get("candidate_id")
    if candidate_id != selection.get("candidate_id"):
        raise ArtifactError("selected direction and final selection candidate IDs differ")
    if object_sha256(candidate) != object_sha256(selection.get("candidate")):
        raise ArtifactError("selected direction and final selection candidate metadata differ")
    if object_sha256(validation_metrics) != object_sha256(selection.get("metrics")):
        raise ArtifactError("selected direction and final selection metrics differ")


def _load_labeled_for_probe(
    config: ProjectConfig,
    store: ArtifactStore,
    activation_profile: ArtifactProfile,
) -> list[PromptRecord]:
    with BaseModelRuntime(config) as runtime:
        _validate_activation_chat_profile(activation_profile, runtime.chat_template_hash)
        labeled = _load_labeled(store, runtime.chat_template_hash, "validation")
    prompts = {item.prompt_id: item for item in _load_prompt_records(store, config)}
    return [prompts[item.prompt_id] for item in labeled]


def evaluate_export(config_path: str) -> None:
    from self_judged_refusal_direction.exporting import write_export_manifest

    config, store = _load(config_path)
    manifest_path = store.paths.exported_model / "edit_manifest.json"
    test_baseline_path = store.paths.evaluation / "test_baseline_trajectories.private.jsonl"
    test_baseline_judgments_path = store.paths.evaluation / "test_baseline_judgments.jsonl"
    test_export_path = store.paths.evaluation / "test_export_trajectories.private.jsonl"
    test_export_judgments_path = store.paths.evaluation / "test_export_judgments.jsonl"
    evaluation_started_path = store.paths.evaluation / "test_evaluation_started.json"
    consumed_paths = (
        evaluation_started_path,
        store.paths.test_report,
        store.metadata_path(store.paths.test_report),
        store.paths.quality_metrics,
        store.metadata_path(store.paths.quality_metrics),
        test_baseline_path,
        test_baseline_judgments_path,
        test_export_path,
        test_export_judgments_path,
    )
    if any(path.exists() for path in consumed_paths):
        raise PipelineError("independent test split has already been consumed; use a new run directory")
    store.validate(
        manifest_path,
        artifact_type="edit_manifest",
        expected_profile=_profile(store, target=True, judge=True),
    )
    manifest = _read_json(manifest_path)
    profile = _activation_dependent_profile(store)
    selection = _read_json_artifact(
        store,
        store.paths.final_selection,
        artifact_type="final_selection",
        profile=profile,
    )
    store.validate(
        store.paths.selected_direction,
        artifact_type="selected_direction",
        expected_profile=profile,
    )
    direction, direction_metadata = load_direction(store.paths.selected_direction)
    _validate_direction_selection(direction_metadata, selection)
    _validate_export_manifest(config, manifest, selection, direction)
    initial_manifest_hash = object_sha256(manifest)
    with BaseModelRuntime(config) as runtime:
        _validate_activation_chat_profile(profile, runtime.chat_template_hash)
        if manifest.get("adapter") != runtime.adapter.__class__.__name__:
            raise ArtifactError("export manifest adapter differs from the configured architecture adapter")
        fingerprints = runtime.adapter.processor_fingerprints(runtime.processor)
        if runtime.chat_template_hash != manifest["chat_template_hash"]:
            raise ArtifactError("export manifest chat template hash differs from the pinned base processor")
        if any(manifest.get(name) != value for name, value in fingerprints.items()):
            raise ArtifactError("export manifest processor fingerprints differ from the pinned base processor")
        _write_json_artifact(
            store,
            evaluation_started_path,
            {
                "config_hash": config.config_hash,
                "initial_export_manifest_sha256": initial_manifest_hash,
            },
            artifact_type="test_evaluation_started",
            profile=profile,
        )
        raw_test_rows = store.read_jsonl(
            store.paths.raw_test,
            artifact_type="raw_test_prompts",
            expected_profile=_profile(store),
        )
        prompts = [PromptRecord.from_dict(row) for row in raw_test_rows]
        if not prompts or any(prompt.split != "test" for prompt in prompts):
            raise PipelineError("independent test artifact must contain only test prompts")
        baseline_trajectories = [runtime.generate_target(prompt) for prompt in prompts]
        judge = TrajectoryJudge(config, runtime.adapter, runtime.model, runtime.processor, runtime.chat_template_hash)
        baseline_judgments = [judge.classify(trajectory) for trajectory in baseline_trajectories]
        chat_template_hash = runtime.chat_template_hash
        base_checkpoint_checksum = runtime.checkpoint_checksum
        baseline_by_hash = {item.trajectory_hash: item for item in baseline_judgments}
        baseline_labels: dict[str, str] = {}
        for trajectory in baseline_trajectories:
            result = baseline_by_hash[trajectory.trajectory_hash]
            if result.status == "OK" and result.label in {"REFUSAL", "NON_REFUSAL"}:
                baseline_labels[trajectory.prompt_id] = result.label
        non_refusal_prompts = [prompt for prompt in prompts if baseline_labels.get(prompt.prompt_id) == "NON_REFUSAL"]
        base_logits = [_next_token_logits(runtime, prompt) for prompt in non_refusal_prompts]
        quality_texts, quality_evaluation = _resolve_quality_texts(
            config,
            runtime,
            [prompt.original_prompt for prompt in non_refusal_prompts],
        )
        base_ce = _mean_ce_loss(runtime, quality_texts)
    baseline_errors = sum(item.status == "ERROR" for item in baseline_judgments)
    _check_error_budget(baseline_errors, config, "test baseline judging")
    store.write_jsonl(
        test_baseline_path,
        baseline_trajectories,
        artifact_type="test_baseline_trajectories",
        profile=_profile(store, target=True, chat_template_hash=chat_template_hash),
        private=True,
    )
    store.write_jsonl(
        test_baseline_judgments_path,
        baseline_judgments,
        artifact_type="test_baseline_judgments",
        profile=_profile(
            store,
            target=True,
            judge=True,
            chat_template_hash=_judge_chat_hash(chat_template_hash),
        ),
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
        if any(manifest.get(name) != value for name, value in fingerprints.items()):
            raise InvariantError("exported processor fingerprint differs from the export manifest")
        export_trajectories = [runtime.generate_target(prompt) for prompt in prompts]
        export_logits = [_next_token_logits(runtime, prompt) for prompt in non_refusal_prompts]
        export_ce = _mean_ce_loss(runtime, quality_texts)
    export_generation_errors = sum(item.trajectory_status == "ERROR" for item in export_trajectories)
    _check_error_budget(export_generation_errors, config, "test export trajectory generation")
    with BaseModelRuntime(config) as runtime:
        if runtime.checkpoint_checksum != base_checkpoint_checksum:
            raise InvariantError("test judges did not use the same immutable base checkpoint")
        judge = TrajectoryJudge(config, runtime.adapter, runtime.model, runtime.processor, runtime.chat_template_hash)
        export_judgments = [judge.classify(trajectory) for trajectory in export_trajectories]
    export_judge_errors = sum(item.status == "ERROR" for item in export_judgments)
    _check_error_budget(export_judge_errors, config, "test export trajectory judging")
    export_judgments_by_hash = {item.trajectory_hash: item for item in export_judgments}
    mean_kl = mean_next_token_kl(base_logits, export_logits) if base_logits else 0.0
    metrics = evaluate_behavior(
        candidate_id="exported_model",
        stage="C",
        baseline_labels=baseline_labels,
        baseline_trajectories={item.prompt_id: item for item in baseline_trajectories},
        trajectories=[item for item in export_trajectories if item.prompt_id in baseline_labels],
        judgments=export_judgments_by_hash,
        mean_kl=mean_kl,
        ce_loss_delta=export_ce - base_ce,
        config=config.evaluation,
    )
    validation_report = _read_json_artifact(
        store,
        store.paths.validation_report,
        artifact_type="validation_report",
        profile=profile,
    )
    test_metrics = metrics_dict(metrics)
    final_manifest = dict(manifest)
    final_manifest["test_metrics"] = test_metrics
    manifest_hash = object_sha256(final_manifest)
    test_report = {
        "base_model_id": config.model.id,
        "base_revision": config.model.revision,
        "target_generation_thinking_enabled_all_phases": True,
        "judge_thinking_enabled_all_phases": False,
        "judge_uses_immutable_base_checkpoint": True,
        "base_checkpoint_checksum": base_checkpoint_checksum,
        "judge_semantics": "REFUSAL iff thought considers refusal OR final answer refuses",
        "baseline_counts": judgment_counts(baseline_judgments),
        "baseline_parser": parser_statistics(baseline_trajectories),
        "export_parser": parser_statistics(export_trajectories),
        "candidate_search": validation_report["candidate_search"],
        "selected_candidate": validation_report["selected_candidate"],
        "test_metrics": test_metrics,
        "quality_evaluation": quality_evaluation,
        "export_manifest_sha256": manifest_hash,
        "intervention_uncertain_count": test_metrics["uncertain_count"],
        "judge_or_parser_error_count": test_metrics["error_count"],
        "activation_addition_diagnostic": validation_report["selected_candidate"]["validation_metrics"].get(
            "activation_addition_induction_rate"
        ),
        "edited_parameters": manifest["edited_parameter_names"],
        "projection_rules": manifest["projection_rules"],
        "temporary_permanent_equivalence": manifest["temporary_permanent_equivalence"],
        "fresh_reload": manifest["fresh_reload"],
        "evaluation_scope": {
            "direction_discovery": "text-only",
            "text_behavior_validated": True,
            "multimodal_loader_preserved": True,
            "multimodal_refusal_behavior_validated": False,
        },
        "privacy": {
            "raw_thinking_artifacts": "private run directory only",
            "raw_thinking_in_export": False,
            "automatic_publication": False,
        },
        "known_limitations": [
            "rank-1 direction only",
            "dense Gemma 4 only",
            "MoE and PLE export fail closed",
            "multimodal refusal removal was not behaviorally evaluated",
        ],
    }
    store.write_jsonl(
        test_export_path,
        (
            {"export_manifest_hash": manifest_hash, "trajectory": trajectory.as_dict()}
            for trajectory in export_trajectories
        ),
        artifact_type="test_export_trajectories",
        profile=_profile(store, target=True, chat_template_hash=chat_template_hash),
        private=True,
    )
    store.write_jsonl(
        test_export_judgments_path,
        ({"export_manifest_hash": manifest_hash, "judgment": judgment.as_dict()} for judgment in export_judgments),
        artifact_type="test_export_judgments",
        profile=_profile(
            store,
            target=True,
            judge=True,
            chat_template_hash=_judge_chat_hash(chat_template_hash),
        ),
        private=False,
    )
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
            "quality_evaluation": quality_evaluation,
            "validation_metrics": validation_report["selected_candidate"]["validation_metrics"],
            "test_metrics": test_metrics,
        },
        artifact_type="quality_metrics",
        profile=profile,
    )
    store.write_json(store.paths.exported_model / "evaluation_report.json", test_report, private=False)
    write_export_manifest(store.paths.exported_model, final_manifest)


def _validate_export_manifest(
    config: ProjectConfig,
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    direction: torch.Tensor,
) -> None:
    expected = {
        "schema_version": 1,
        "base_model_id": config.model.id,
        "base_revision": config.model.revision,
        "config_hash": config.config_hash,
        "target_profile_hash": config.target_profile_hash,
        "judge_profile_hash": config.judge_profile_hash,
    }
    mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
    if mismatches:
        raise ArtifactError(f"export manifest profile mismatch: {mismatches}")
    if manifest.get("target_generation") != {"thinking_enabled": True}:
        raise ArtifactError("export manifest target generation profile is invalid")
    judge_profile = manifest.get("judge_profile")
    if not isinstance(judge_profile, Mapping) or judge_profile.get("thinking_enabled") is not False:
        raise ArtifactError("export manifest judge profile is invalid")
    if object_sha256(manifest.get("validation_metrics")) != object_sha256(selection.get("metrics")):
        raise ArtifactError("export manifest validation metrics differ from final selection")
    selected_candidate = selection.get("candidate")
    if not isinstance(selected_candidate, Mapping):
        raise ArtifactError("final selection candidate metadata is missing")
    source_fields = {
        "direction_source_phase": selected_candidate.get("phase"),
        "direction_source_layer": selected_candidate.get("layer"),
        "direction_source_relative_position": selected_candidate.get("relative_position"),
    }
    if any(manifest.get(name) != value for name, value in source_fields.items()):
        raise ArtifactError("export manifest direction source differs from final selection")
    if manifest.get("test_metrics"):
        raise PipelineError("export manifest already contains independent test metrics")
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
    if manifest.get("privacy") != {"raw_thinking_included": False, "push_to_hub": False}:
        raise ArtifactError("export manifest privacy profile is invalid")
    reload_report = manifest.get("fresh_reload")
    required_reload = {
        "status": "OK",
        "tied_weights_preserved": True,
        "probe_logits_match": True,
        "processor_reload_verified": True,
        "target_trajectory_required": True,
        "target_thinking_enabled": True,
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
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def run_pipeline(config_path: str) -> None:
    inspect_model(config_path)
    generate_baseline_trajectories(config_path)
    judge_baseline_trajectories(config_path)
    collect_activations(config_path)
    build_direction_candidates(config_path)
    evaluate_candidates(config_path)
    export_model(config_path)
    evaluate_export(config_path)
