from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F

from self_judged_refusal_direction.activations import (
    ActivationCollector,
    load_activation_statistics,
    save_activation_statistics,
)
from self_judged_refusal_direction.artifacts import ArtifactMetadata, ArtifactProfile, ArtifactStore
from self_judged_refusal_direction.config import ProjectConfig, load_config
from self_judged_refusal_direction.data import (
    ingest_and_split_prompts,
    ingest_texts,
    records_by_split,
    write_prompt_split_artifacts,
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
from self_judged_refusal_direction.prompting import (
    judge_messages,
    target_messages,
)
from self_judged_refusal_direction.prompting import (
    judge_template_hash as current_judge_template_hash,
)
from self_judged_refusal_direction.runtime import BaseModelRuntime, IntervenedModelRuntime
from self_judged_refusal_direction.schema import (
    CandidateMetrics,
    DirectionCandidate,
    JudgeLabel,
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
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    store = ArtifactStore(config)
    store.initialize_run()
    return config, store


def _profile(
    store: ArtifactStore,
    *,
    target: bool = False,
    chat_template_hash: str | None = None,
    judge_template_hash: str | None = None,
) -> ArtifactProfile:
    return store.profile(
        target=target,
        chat_template_hash=chat_template_hash,
        judge_template_hash=judge_template_hash,
    )


def _check_error_budget(count: int, config: ProjectConfig, phase: str) -> None:
    maximum = config.run.max_infrastructure_errors
    if count > maximum:
        raise PipelineError(f"{phase} produced {count} infrastructure errors; maximum is {maximum}")


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
    empty = TargetTrajectory(
        prompt_id="context-preflight",
        original_prompt="",
        raw_generated_token_ids=(),
        raw_decoded_output="",
        thinking_text="",
        final_answer="",
        thinking_token_start=0,
        thinking_token_end=0,
        final_token_start=0,
        final_token_end=0,
        generation_truncated=False,
        parser_status="OK",
        model_revision=config.model.revision,
        generation_config_hash=config.target_generation_config_hash,
        trajectory_hash="context-preflight",
    )
    judge_rendered = runtime.adapter.render_judge_chat(runtime.processor, judge_messages(empty))
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
            chat_template_hash=chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
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
            chat_template_hash=chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
        ),
    )
    return [LabeledTrajectory(**row) for row in rows]


def inspect_model(config_path: str) -> None:
    config, store = _load(config_path)
    with BaseModelRuntime(config) as runtime:
        _validate_static_context_budget(config, runtime)
        fingerprints = runtime.adapter.processor_fingerprints(runtime.processor)
        report = dataclasses.asdict(runtime.compatibility_report)
        if not report["errors"]:
            del report["errors"]
        report.update(
            {
                "model_id": config.model.id,
                "model_revision": config.model.revision,
                "config_hash": config.config_hash,
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
    with BaseModelRuntime(config) as runtime:
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
        write_prompt_split_artifacts(store, records, profile=_profile(store))
        baseline_records = [item for item in records if item.split in {"train", "validation"}]
        trajectories = [runtime.generate_target(record) for record in baseline_records]
        errors = sum(item.parser_status == "ERROR" for item in trajectories)
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
        judge = TrajectoryJudge(runtime.adapter, runtime.model, runtime.processor)
        judgments = [judge.classify(trajectory) for trajectory in trajectories]
        errors = sum(item.status == "ERROR" for item in judgments)
        _check_error_budget(errors, config, "baseline trajectory judging")
        judgment_profile = _profile(
            store,
            target=True,
            chat_template_hash=runtime.chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
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
        trajectories = _load_baseline_trajectories(store, runtime.chat_template_hash)
        labeled = _load_labeled(store, runtime.chat_template_hash, "train")
        trajectory_by_hash = {item.trajectory_hash: item for item in trajectories}
        statistics = _collect_activation_statistics(config, runtime, labeled, trajectory_by_hash)
        save_activation_statistics(store.paths.activation_statistics, statistics)
        activation_profile = _profile(
            store,
            target=True,
            chat_template_hash=runtime.chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
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
    for item in labeled:
        trajectory = trajectory_by_hash[item.trajectory_hash]
        rendered = runtime.adapter.render_target_chat(
            runtime.processor,
            target_messages(trajectory.original_prompt, config.target_generation.system_prompt),
            config=config.target_generation,
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
    bundle = build_candidates(statistics, dtype="float32")
    ranking = rank_activation_screening(bundle, keep=config.search.activation_screening_keep)
    if not ranking:
        raise PipelineError("activation screening produced no numerically valid direction candidates")
    write_candidate_artifacts(store, bundle, profile=profile)
    _write_json_artifact(
        store,
        store.paths.activation_screening_ranking,
        [candidate.candidate_id for candidate in ranking],
        artifact_type="activation_screening_ranking",
        profile=profile,
    )


def _activation_dependent_profile(store: ArtifactStore) -> ArtifactProfile:
    metadata = store.validate(
        store.paths.activation_statistics,
        artifact_type="activation_statistics",
        expected_profile=_profile(store, target=True),
    )
    if not metadata.private or metadata.profile.chat_template_hash is None:
        raise ArtifactError("activation statistics privacy or chat-template profile is invalid")
    return metadata.profile


def _validate_activation_chat_profile(profile: ArtifactProfile, chat_template_hash: str) -> None:
    if profile.chat_template_hash != chat_template_hash or profile.judge_template_hash != current_judge_template_hash():
        raise ArtifactError("activation-derived artifacts use a different chat template")


def evaluate_candidates(config_path: str) -> None:
    config, store = _load(config_path)
    profile = _activation_dependent_profile(store)
    bundle = load_candidate_artifacts(store, expected_profile=profile)
    screening = _read_json_artifact(
        store,
        store.paths.activation_screening_ranking,
        artifact_type="activation_screening_ranking",
        profile=profile,
    )
    if not isinstance(screening, list) or any(not isinstance(candidate_id, str) for candidate_id in screening):
        raise ArtifactError("activation screening ranking is invalid")
    by_id = {item.candidate_id: item for item in bundle.candidates}
    try:
        ranking = [by_id[candidate_id] for candidate_id in screening]
    except KeyError as error:
        raise ArtifactError("activation screening ranking references an unknown candidate") from error
    with BaseModelRuntime(config) as profile_runtime:
        chat_template_hash = profile_runtime.chat_template_hash
        _validate_activation_chat_profile(profile, chat_template_hash)
    validation = _load_labeled(store, chat_template_hash, "validation")
    trajectories = _load_baseline_trajectories(store, chat_template_hash)
    baseline_trajectories = {item.prompt_id: item for item in trajectories}
    records = {item.prompt_id: item for item in _load_prompt_records(store, config)}
    reference_fallback = [records[item.prompt_id].original_prompt for item in validation if item.label == "NON_REFUSAL"]
    pilot_labels = _balanced_subset(validation, config.search.pilot_prompts_per_class)
    pilot_metrics, _ = _evaluate_candidates_phase(
        config,
        store,
        bundle,
        ranking,
        pilot_labels,
        records,
        baseline_trajectories,
        reference_fallback,
        evaluation_phase="pilot_evaluation",
    )
    pilot_eligible = [item for item in pilot_metrics if item.hard_filter_passed]
    if not pilot_eligible:
        raise PipelineError("no pilot evaluation candidate passed the configured hard filters")
    full_validation_ids = [
        item.candidate_id
        for item in sorted(pilot_eligible, key=_pilot_ranking_key, reverse=True)[: config.search.pilot_evaluation_keep]
    ]
    full_validation_candidates = [by_id[candidate_id] for candidate_id in full_validation_ids]
    full_validation_metrics, reference_corpus = _evaluate_candidates_phase(
        config,
        store,
        bundle,
        full_validation_candidates,
        validation,
        records,
        baseline_trajectories,
        reference_fallback,
        evaluation_phase="full_validation",
    )
    selected_metrics, selected_candidate = select_candidate(full_validation_metrics, by_id)
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
        profile=profile,
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
        },
        artifact_type="final_selection",
        profile=profile,
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
        "reference_corpus": reference_corpus,
        "selected_candidate": selected_metadata,
    }
    _write_json_artifact(
        store,
        store.paths.full_validation_report,
        report,
        artifact_type="full_validation_report",
        profile=profile,
    )


def _balanced_subset(values: Sequence[LabeledTrajectory], per_class: int) -> list[LabeledTrajectory]:
    selected: list[LabeledTrajectory] = []
    for label in ("REFUSAL", "NON_REFUSAL"):
        cohort = sorted((item for item in values if item.label == label), key=lambda item: item.prompt_id)
        selected.extend(cohort[: min(per_class, len(cohort))])
    return selected


def _pilot_ranking_key(metrics: CandidateMetrics) -> tuple[float, ...]:
    return (
        metrics.removal_success_rate,
        metrics.non_refusal_retention_rate,
        -metrics.uncertain_rate,
        -metrics.mean_kl,
    )


def _evaluate_candidates_phase(
    config: ProjectConfig,
    store: ArtifactStore,
    bundle: CandidateBundle,
    candidates: Sequence[DirectionCandidate],
    labeled: Sequence[LabeledTrajectory],
    prompt_records: Mapping[str, PromptRecord],
    baseline_trajectories: Mapping[str, TargetTrajectory],
    reference_fallback: Sequence[str],
    *,
    evaluation_phase: Literal["pilot_evaluation", "full_validation"],
) -> tuple[list[CandidateMetrics], dict[str, Any]]:
    labels = {item.prompt_id: item.label for item in labeled}
    prompts = [prompt_records[item.prompt_id] for item in labeled]
    non_refusal_prompts = [prompt_records[item.prompt_id] for item in labeled if item.label == "NON_REFUSAL"]
    trajectory_rows: list[dict[str, Any]] = []
    quality: dict[str, tuple[float, float]] = {}
    with IntervenedModelRuntime(config) as runtime:
        base_logits = [_next_token_logits(runtime, prompt) for prompt in non_refusal_prompts]
        reference_texts, reference_corpus = _resolve_reference_texts(config, runtime, reference_fallback)
        base_ce = _mean_ce_loss(runtime, reference_texts)
        for candidate in candidates:
            direction = bundle.direction(candidate)
            plan = runtime.adapter.build_weight_edit_plan(runtime.model, direction)
            with plan.temporary(runtime.model):
                trajectories = [runtime.generate_target(prompt) for prompt in prompts]
                intervention_logits = [_next_token_logits(runtime, prompt) for prompt in non_refusal_prompts]
                intervention_ce = _mean_ce_loss(runtime, reference_texts)
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
            beta = config.acceptance.activation_addition_beta
            if evaluation_phase == "full_validation" and beta is not None:
                block_name = _transformer_block_name(runtime, candidate.layer)
                coefficient = beta * candidate.norm
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
    errors = sum(row["trajectory"]["parser_status"] == "ERROR" for row in trajectory_rows)
    _check_error_budget(errors, config, f"{evaluation_phase} intervention generation")
    trajectories_path = (
        store.paths.pilot_evaluation_trajectories
        if evaluation_phase == "pilot_evaluation"
        else store.paths.full_validation_trajectories
    )
    store.write_jsonl(
        trajectories_path,
        trajectory_rows,
        artifact_type=f"{evaluation_phase}_trajectories",
        profile=_profile(
            store,
            target=True,
            chat_template_hash=chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
        ),
        private=True,
    )
    judgment_rows: list[dict[str, Any]] = []
    with BaseModelRuntime(config) as runtime:
        judge = TrajectoryJudge(runtime.adapter, runtime.model, runtime.processor)
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
        del judge
    judge_errors = sum(row["judgment"]["status"] == "ERROR" for row in judgment_rows)
    _check_error_budget(judge_errors, config, f"{evaluation_phase} judging")
    judgments_path = (
        store.paths.pilot_evaluation_judgments
        if evaluation_phase == "pilot_evaluation"
        else store.paths.full_validation_judgments
    )
    store.write_jsonl(
        judgments_path,
        judgment_rows,
        artifact_type=f"{evaluation_phase}_judgments",
        profile=_profile(
            store,
            target=True,
            chat_template_hash=chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
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
                baseline_labels=labels,
                baseline_trajectories=baseline_trajectories,
                trajectories=[TargetTrajectory.from_dict(row["trajectory"]) for row in removal_rows],
                judgments=candidate_judgments,
                mean_kl=mean_kl,
                ce_loss_delta=ce_delta,
                acceptance=config.acceptance,
                activation_addition_induction_rate=induction,
            )
        )
    results_path = (
        store.paths.pilot_evaluation_results
        if evaluation_phase == "pilot_evaluation"
        else store.paths.full_validation_results
    )
    store.write_jsonl(
        results_path,
        metrics,
        artifact_type=f"{evaluation_phase}_results",
        profile=_profile(
            store,
            target=True,
            chat_template_hash=chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
        ),
        private=False,
    )
    return metrics, reference_corpus


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
        raise InvariantError("CE-loss evaluation LM head has no matrix weight")
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
                raise InvariantError("CE-loss evaluation text backbone returned no hidden states")
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


def _resolve_reference_texts(
    config: ProjectConfig,
    runtime: BaseModelRuntime | IntervenedModelRuntime,
    fallback: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    if config.data.reference_files:
        tokenizer = runtime.processor.tokenizer
        texts = ingest_texts(
            config.data.reference_files,
            token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=True)),
            max_text_tokens=config.data.max_text_tokens,
        )
        if not texts:
            raise PipelineError("configured reference files produced no usable texts")
        return texts, {
            "source": "reference_files",
            "text_count": len(texts),
            "source_sha256": [file_sha256(Path(path).resolve()) for path in config.data.reference_files],
        }
    texts = list(dict.fromkeys(text for text in fallback if text))
    if not texts:
        raise PipelineError("CE-loss evaluation requires reference files or baseline NON_REFUSAL prompts")
    return texts, {
        "source": "baseline_non_refusal_prompts",
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
        plan = runtime.weight_edit_plan
        probe = runtime.adapter.render_target_chat(
            runtime.processor,
            target_messages(prompt.original_prompt, config.target_generation.system_prompt),
            config=config.target_generation,
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
            full_validation_metrics=selection["metrics"],
            defer_reload=True,
            direction_layer=int(direction_metadata["candidate"]["layer"]),
        )
    export_result = complete_deferred_reload(export_result)
    if export_result.reload is None:
        raise InvariantError("fresh reload verification did not produce a report")


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
        expected_profile=_profile(
            store,
            target=True,
            judge_template_hash=current_judge_template_hash(),
        ),
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
        fingerprints = runtime.adapter.processor_fingerprints(runtime.processor)
        if runtime.chat_template_hash != manifest["chat_template_hash"]:
            raise ArtifactError("export manifest chat template hash differs from the pinned base processor")
        if any(manifest.get(name) != value for name, value in fingerprints.items()):
            raise ArtifactError("export manifest processor fingerprints differ from the pinned base processor")
        _write_json_artifact(
            store,
            evaluation_started_path,
            {
                "initial_export_manifest_sha256": initial_manifest_hash,
            },
            artifact_type="test_evaluation_started",
            profile=profile,
        )
        test_prompt_rows = store.read_jsonl(
            store.paths.test_prompts,
            artifact_type="test_prompts",
            expected_profile=_profile(store),
        )
        prompts = [PromptRecord.from_dict(row) for row in test_prompt_rows]
        if not prompts or any(prompt.split != "test" for prompt in prompts):
            raise PipelineError("independent test artifact must contain only test prompts")
        baseline_trajectories = [runtime.generate_target(prompt) for prompt in prompts]
        judge = TrajectoryJudge(runtime.adapter, runtime.model, runtime.processor)
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
        reference_texts, reference_corpus = _resolve_reference_texts(
            config,
            runtime,
            [prompt.original_prompt for prompt in non_refusal_prompts],
        )
        base_ce = _mean_ce_loss(runtime, reference_texts)
        del judge
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
            chat_template_hash=chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
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
        export_ce = _mean_ce_loss(runtime, reference_texts)
    export_generation_errors = sum(item.parser_status == "ERROR" for item in export_trajectories)
    _check_error_budget(export_generation_errors, config, "test export trajectory generation")
    with BaseModelRuntime(config) as runtime:
        if runtime.checkpoint_checksum != base_checkpoint_checksum:
            raise InvariantError("test judges did not use the same immutable base checkpoint")
        judge = TrajectoryJudge(runtime.adapter, runtime.model, runtime.processor)
        export_judgments = [judge.classify(trajectory) for trajectory in export_trajectories]
        del judge
    export_judge_errors = sum(item.status == "ERROR" for item in export_judgments)
    _check_error_budget(export_judge_errors, config, "test export trajectory judging")
    export_judgments_by_hash = {item.trajectory_hash: item for item in export_judgments}
    mean_kl = mean_next_token_kl(base_logits, export_logits) if base_logits else 0.0
    metrics = evaluate_behavior(
        candidate_id="exported_model",
        baseline_labels=baseline_labels,
        baseline_trajectories={item.prompt_id: item for item in baseline_trajectories},
        trajectories=[item for item in export_trajectories if item.prompt_id in baseline_labels],
        judgments=export_judgments_by_hash,
        mean_kl=mean_kl,
        ce_loss_delta=export_ce - base_ce,
        acceptance=config.acceptance,
    )
    full_validation_report = _read_json_artifact(
        store,
        store.paths.full_validation_report,
        artifact_type="full_validation_report",
        profile=profile,
    )
    test_metrics = metrics_dict(metrics)
    final_manifest = dict(manifest)
    final_manifest["test_metrics"] = test_metrics
    manifest_hash = object_sha256(final_manifest)
    test_report = {
        "base_model_id": config.model.id,
        "base_revision": config.model.revision,
        "base_checkpoint_checksum": base_checkpoint_checksum,
        "baseline_counts": judgment_counts(baseline_judgments),
        "baseline_parser": parser_statistics(baseline_trajectories),
        "export_parser": parser_statistics(export_trajectories),
        "test_metrics": test_metrics,
        "reference_corpus": reference_corpus,
        "export_manifest_sha256": manifest_hash,
    }
    store.write_jsonl(
        test_export_path,
        (
            {"export_manifest_hash": manifest_hash, "trajectory": trajectory.as_dict()}
            for trajectory in export_trajectories
        ),
        artifact_type="test_export_trajectories",
        profile=_profile(
            store,
            target=True,
            chat_template_hash=chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
        ),
        private=True,
    )
    store.write_jsonl(
        test_export_judgments_path,
        ({"export_manifest_hash": manifest_hash, "judgment": judgment.as_dict()} for judgment in export_judgments),
        artifact_type="test_export_judgments",
        profile=_profile(
            store,
            target=True,
            chat_template_hash=chat_template_hash,
            judge_template_hash=current_judge_template_hash(),
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
            "reference_corpus": reference_corpus,
            "full_validation_metrics": full_validation_report["selected_candidate"]["full_validation_metrics"],
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
        "base_model_id": config.model.id,
        "base_revision": config.model.revision,
        "config_hash": config.config_hash,
        "target_generation_config_hash": config.target_generation_config_hash,
        "judge_template_hash": current_judge_template_hash(),
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
    if "test_metrics" in manifest:
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
