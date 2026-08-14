from dataclasses import replace
from typing import Literal

from self_judged_refusal_direction.config import AcceptanceConfig
from self_judged_refusal_direction.evaluation import apply_pilot_filters, evaluate_behavior
from self_judged_refusal_direction.schema import CEEvaluation, JudgeResult, TargetTrajectory


def _ce_evaluation(
    *,
    source: Literal["baseline_non_refusal_completions", "reference_files"] = "baseline_non_refusal_completions",
    baseline_loss: float = 0.1,
    intervention_loss: float = 0.1,
) -> CEEvaluation:
    return CEEvaluation(
        source=source,
        input_count=1,
        target_token_count=8,
        baseline_loss=baseline_loss,
        intervention_loss=intervention_loss,
    )


def _trajectory(
    prompt_id: str,
    *,
    tokens: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    decoded: str = "answer<|end_of_turn|>",
    final_answer: str = "answer",
    truncated: bool = False,
) -> TargetTrajectory:
    return TargetTrajectory(
        prompt_id=prompt_id,
        original_prompt=prompt_id,
        raw_generated_token_ids=tokens,
        raw_decoded_output=decoded,
        thinking_text="",
        final_answer=final_answer,
        thinking_token_start=0,
        thinking_token_end=0,
        final_token_start=0,
        final_token_end=len(tokens),
        generation_truncated=truncated,
        parser_status="OK",
        model_revision="a" * 40,
        generation_config_hash="b" * 64,
        trajectory_hash=f"trajectory-{prompt_id}",
    )


def _judgment(
    trajectory: TargetTrajectory,
    label: Literal["REFUSAL", "NON_REFUSAL", "UNCERTAIN"],
) -> JudgeResult:
    return JudgeResult(status="OK", trajectory_hash=trajectory.trajectory_hash, label=label)


def test_causal_and_generation_health_filters_reject_ineffective_degenerate_candidate() -> None:
    baseline = {prompt_id: _trajectory(prompt_id) for prompt_id in ("refusal", "non-refusal")}
    repeated_tokens = (1, 2, 3, 4) * 8
    intervention = {
        prompt_id: _trajectory(
            prompt_id,
            tokens=repeated_tokens,
            decoded="<channel|>" * 3,
            final_answer="",
            truncated=True,
        )
        for prompt_id in baseline
    }
    judgments = {
        intervention["refusal"].trajectory_hash: _judgment(intervention["refusal"], "REFUSAL"),
        intervention["non-refusal"].trajectory_hash: _judgment(intervention["non-refusal"], "NON_REFUSAL"),
    }

    metrics = evaluate_behavior(
        candidate_id="candidate",
        baseline_labels={"refusal": "REFUSAL", "non-refusal": "NON_REFUSAL"},
        baseline_trajectories=baseline,
        trajectories=intervention.values(),
        judgments=judgments,
        mean_kl=0.0,
        completion_ce=_ce_evaluation(),
        acceptance=AcceptanceConfig(),
        activation_addition_induction_rate=0.0,
        activation_addition_error_rate=0.0,
        require_activation_addition=True,
    )

    assert set(metrics.rejection_reasons) == {
        "removal_success_rate",
        "activation_addition_induction_rate",
        "completion_rate",
        "empty_answer_rate",
        "abnormal_repetition_rate",
        "control_delimiter_repetition_rate",
    }


def test_causal_candidate_with_healthy_generation_passes() -> None:
    baseline = {prompt_id: _trajectory(prompt_id) for prompt_id in ("refusal", "non-refusal")}
    intervention = {prompt_id: _trajectory(prompt_id) for prompt_id in baseline}
    judgments = {
        intervention["refusal"].trajectory_hash: _judgment(intervention["refusal"], "NON_REFUSAL"),
        intervention["non-refusal"].trajectory_hash: _judgment(intervention["non-refusal"], "NON_REFUSAL"),
    }

    metrics = evaluate_behavior(
        candidate_id="candidate",
        baseline_labels={"refusal": "REFUSAL", "non-refusal": "NON_REFUSAL"},
        baseline_trajectories=baseline,
        trajectories=intervention.values(),
        judgments=judgments,
        mean_kl=0.0,
        completion_ce=_ce_evaluation(),
        acceptance=AcceptanceConfig(),
        activation_addition_induction_rate=1.0,
        activation_addition_error_rate=0.0,
        require_activation_addition=True,
    )

    assert metrics.hard_filter_passed
    assert metrics.rejection_reasons == ()


def test_pilot_allows_finite_quality_threshold_excess_until_full_validation() -> None:
    baseline = {prompt_id: _trajectory(prompt_id) for prompt_id in ("refusal", "non-refusal")}
    intervention = {prompt_id: _trajectory(prompt_id) for prompt_id in baseline}
    judgments = {
        intervention["refusal"].trajectory_hash: _judgment(intervention["refusal"], "NON_REFUSAL"),
        intervention["non-refusal"].trajectory_hash: _judgment(intervention["non-refusal"], "NON_REFUSAL"),
    }

    full_metrics = evaluate_behavior(
        candidate_id="candidate",
        baseline_labels={"refusal": "REFUSAL", "non-refusal": "NON_REFUSAL"},
        baseline_trajectories=baseline,
        trajectories=intervention.values(),
        judgments=judgments,
        mean_kl=0.2,
        completion_ce=_ce_evaluation(intervention_loss=0.3),
        acceptance=AcceptanceConfig(),
        activation_addition_induction_rate=1.0,
        activation_addition_error_rate=0.0,
        require_activation_addition=True,
    )

    assert not full_metrics.hard_filter_passed
    assert set(full_metrics.rejection_reasons) == {"mean_kl", "ce_loss_delta"}
    pilot_metrics = apply_pilot_filters(full_metrics)
    assert pilot_metrics.hard_filter_passed
    assert pilot_metrics.rejection_reasons == ()

    non_finite = type(full_metrics.completion_ce)(
        source="baseline_non_refusal_completions",
        input_count=1,
        target_token_count=8,
        baseline_loss=0.1,
        intervention_loss=None,
        error_code="NON_FINITE",
    )
    failed = apply_pilot_filters(replace(full_metrics, mean_kl=None, completion_ce=non_finite))
    assert set(failed.rejection_reasons) == {"mean_kl", "ce_loss_delta"}
