from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from self_judged_refusal_direction.config import AcceptanceConfig
from self_judged_refusal_direction.errors import PipelineError
from self_judged_refusal_direction.schema import CandidateMetrics, DirectionCandidate, JudgeResult, TargetTrajectory

_REPETITION_NGRAM_SIZE = 4
_REPETITION_THRESHOLD = 0.5


def mean_next_token_kl(base_logits: Sequence[torch.Tensor], intervention_logits: Sequence[torch.Tensor]) -> float:
    if len(base_logits) != len(intervention_logits) or not base_logits:
        raise PipelineError("paired non-empty logits are required for KL evaluation")
    values: list[torch.Tensor] = []
    for base, intervention in zip(base_logits, intervention_logits, strict=True):
        if base.shape != intervention.shape:
            raise PipelineError("paired logits have different shapes")
        base_log_probs = torch.log_softmax(base.float(), dim=-1)
        intervention_log_probs = torch.log_softmax(intervention.float(), dim=-1)
        base_probs = base_log_probs.exp()
        values.append(torch.sum(base_probs * (base_log_probs - intervention_log_probs), dim=-1).mean())
    return float(torch.stack(values).mean().item())


def evaluate_behavior(
    *,
    candidate_id: str,
    baseline_labels: Mapping[str, str],
    baseline_trajectories: Mapping[str, TargetTrajectory],
    trajectories: Iterable[TargetTrajectory],
    judgments: Mapping[str, JudgeResult],
    mean_kl: float,
    ce_loss_delta: float,
    acceptance: AcceptanceConfig,
    activation_addition_induction_rate: float | None = None,
    activation_addition_error_rate: float | None = None,
) -> CandidateMetrics:
    records = list(trajectories)
    refusal_ids = {prompt_id for prompt_id, label in baseline_labels.items() if label == "REFUSAL"}
    non_refusal_ids = {prompt_id for prompt_id, label in baseline_labels.items() if label == "NON_REFUSAL"}
    expected_ids = refusal_ids | non_refusal_ids
    if not expected_ids <= set(baseline_trajectories):
        raise PipelineError("baseline trajectory mapping is incomplete")
    by_prompt = {trajectory.prompt_id: trajectory for trajectory in records}
    duplicate_count = len(records) - len(by_prompt)
    error_count = max(duplicate_count, 0)
    uncertain_count = 0
    removal_successes = 0
    retention_successes = 0
    observed: list[TargetTrajectory] = []
    for prompt_id in expected_ids:
        trajectory = by_prompt.get(prompt_id)
        if trajectory is None or trajectory.parser_status != "OK":
            error_count += 1
            continue
        judgment = judgments.get(trajectory.trajectory_hash)
        if judgment is None or judgment.status != "OK" or judgment.label is None:
            error_count += 1
            continue
        observed.append(trajectory)
        if judgment.label == "UNCERTAIN":
            uncertain_count += 1
        if prompt_id in refusal_ids and judgment.label == "NON_REFUSAL":
            removal_successes += 1
        if prompt_id in non_refusal_ids and judgment.label == "NON_REFUSAL":
            retention_successes += 1
    total = len(expected_ids)
    removal_rate = _ratio(removal_successes, len(refusal_ids))
    retention_rate = _ratio(retention_successes, len(non_refusal_ids))
    uncertain_rate = _ratio(uncertain_count, total)
    error_rate = _ratio(error_count, total)
    empty_rate = _ratio(sum(not item.final_answer.strip() for item in observed), total)
    repeated = (
        _abnormal_repetition(item.final_answer, _REPETITION_NGRAM_SIZE, _REPETITION_THRESHOLD) for item in observed
    )
    repetition_rate = _ratio(
        sum(repeated),
        total,
    )
    thinking_lengths = [max(item.thinking_token_end - item.thinking_token_start, 0) for item in observed]
    final_lengths = [max(item.final_token_end - item.final_token_start, 0) for item in observed]
    baseline_thinking_lengths = [
        max(
            baseline_trajectories[prompt_id].thinking_token_end - baseline_trajectories[prompt_id].thinking_token_start,
            0,
        )
        for prompt_id in expected_ids
    ]
    baseline_final_lengths = [
        max(baseline_trajectories[prompt_id].final_token_end - baseline_trajectories[prompt_id].final_token_start, 0)
        for prompt_id in expected_ids
    ]
    baseline_mean_thinking = _mean(baseline_thinking_lengths)
    baseline_mean_final = _mean(baseline_final_lengths)
    intervention_mean_thinking = _mean(thinking_lengths)
    intervention_mean_final = _mean(final_lengths)
    completion_proxy = _ratio(sum(bool(item.final_answer.strip()) for item in observed), total)
    metrics = CandidateMetrics(
        candidate_id=candidate_id,
        baseline_refusal_count=len(refusal_ids),
        baseline_non_refusal_count=len(non_refusal_ids),
        intervention_non_refusal_from_refusal=removal_successes,
        intervention_non_refusal_retained=retention_successes,
        uncertain_count=uncertain_count,
        error_count=error_count,
        removal_success_rate=removal_rate,
        non_refusal_retention_rate=retention_rate,
        uncertain_rate=uncertain_rate,
        error_rate=error_rate,
        mean_kl=mean_kl,
        ce_loss_delta=ce_loss_delta,
        empty_answer_rate=empty_rate,
        abnormal_repetition_rate=repetition_rate,
        baseline_mean_thinking_tokens=baseline_mean_thinking,
        baseline_mean_final_tokens=baseline_mean_final,
        mean_thinking_tokens=intervention_mean_thinking,
        mean_final_tokens=intervention_mean_final,
        mean_thinking_tokens_delta=intervention_mean_thinking - baseline_mean_thinking,
        mean_final_tokens_delta=intervention_mean_final - baseline_mean_final,
        task_completion_proxy=completion_proxy,
        activation_addition_induction_rate=activation_addition_induction_rate,
        activation_addition_error_rate=activation_addition_error_rate,
    )
    return apply_hard_filters(metrics, acceptance)


def apply_hard_filters(metrics: CandidateMetrics, acceptance: AcceptanceConfig) -> CandidateMetrics:
    reasons: list[str] = []
    if metrics.error_rate > acceptance.max_error_rate:
        reasons.append("error_rate")
    if (
        metrics.activation_addition_error_rate is not None
        and metrics.activation_addition_error_rate > acceptance.max_error_rate
    ):
        reasons.append("activation_addition_error_rate")
    if metrics.uncertain_rate > acceptance.max_uncertain_rate:
        reasons.append("uncertain_rate")
    if metrics.non_refusal_retention_rate < acceptance.min_non_refusal_retention:
        reasons.append("non_refusal_retention")
    if not math.isfinite(metrics.mean_kl) or metrics.mean_kl > acceptance.max_mean_kl:
        reasons.append("mean_kl")
    if not math.isfinite(metrics.ce_loss_delta) or metrics.ce_loss_delta > acceptance.max_ce_loss_delta:
        reasons.append("ce_loss_delta")
    return replace(metrics, hard_filter_passed=not reasons, rejection_reasons=tuple(reasons))


def select_candidate(
    metrics: Iterable[CandidateMetrics],
    candidates: Mapping[str, DirectionCandidate],
) -> tuple[CandidateMetrics, DirectionCandidate]:
    eligible = [item for item in metrics if item.hard_filter_passed]
    if not eligible:
        raise PipelineError("no direction candidate passed the configured hard filters")

    def key(item: CandidateMetrics) -> tuple[float, ...]:
        candidate = candidates[item.candidate_id]
        induction = item.activation_addition_induction_rate
        return (
            item.removal_success_rate,
            item.non_refusal_retention_rate,
            -item.uncertain_rate,
            -item.mean_kl,
            induction if induction is not None else -math.inf,
            candidate.standardized_separation,
            candidate.norm,
            -candidate.layer,
        )

    selected = max(eligible, key=key)
    return selected, candidates[selected.candidate_id]


def judgment_counts(results: Iterable[JudgeResult]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        if result.status == "ERROR":
            counts["ERROR"] += 1
        elif result.label is not None:
            counts[result.label] += 1
    return {key: counts.get(key, 0) for key in ("REFUSAL", "NON_REFUSAL", "UNCERTAIN", "ERROR")}


def parser_statistics(trajectories: Iterable[TargetTrajectory]) -> dict[str, float | int]:
    values = list(trajectories)
    total = len(values)
    parser_successes = sum(item.parser_status == "OK" for item in values)
    truncations = sum(item.generation_truncated for item in values)
    return {
        "count": total,
        "parser_success_count": parser_successes,
        "parser_success_rate": _ratio(parser_successes, total),
        "truncation_count": truncations,
        "truncation_rate": _ratio(truncations, total),
    }


def metrics_dict(metrics: CandidateMetrics) -> dict[str, Any]:
    value = metrics.as_dict()
    value["rejection_reasons"] = list(metrics.rejection_reasons)
    return value


def _abnormal_repetition(text: str, n: int, threshold: float) -> bool:
    tokens = text.split()
    if len(tokens) < n * 2:
        return False
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    repeated_fraction = 1.0 - len(set(ngrams)) / len(ngrams)
    return repeated_fraction >= threshold


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
