from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal, cast

from self_judged_refusal_direction.errors import ArtifactError, InvariantError


class JudgeLabel(StrEnum):
    REFUSAL = "REFUSAL"
    NON_REFUSAL = "NON_REFUSAL"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class JudgeInput:
    original_prompt: str
    trajectory: str
    generation_truncated: bool
    input_hash: str


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    original_prompt: str
    group_id: str
    split: Literal["train", "validation", "test"]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PromptRecord:
        return cls(**value)


@dataclass(frozen=True)
class TargetTrajectory:
    prompt_id: str
    original_prompt: str
    raw_generated_token_ids: tuple[int, ...]
    raw_decoded_output: str
    thinking_text: str
    final_answer: str
    thinking_token_start: int
    thinking_token_end: int
    final_token_start: int
    final_token_end: int
    generation_truncated: bool
    parser_status: Literal["OK", "ERROR"]
    model_revision: str
    generation_config_hash: str
    trajectory_hash: str
    split: Literal["train", "validation", "test"] | None = None
    seed: int = 42
    error_code: str | None = None
    error_detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TargetTrajectory:
        data = dict(value)
        data["raw_generated_token_ids"] = tuple(data["raw_generated_token_ids"])
        return cls(**data)


@dataclass(frozen=True)
class JudgeResult:
    status: Literal["OK", "ERROR"]
    trajectory_hash: str
    label: Literal["REFUSAL", "NON_REFUSAL", "UNCERTAIN"] | None = None
    raw_output: str | None = None
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JudgeResult:
        return cls(**value)


@dataclass(frozen=True)
class JudgeValidationCase:
    case_id: str
    original_prompt: str
    trajectory: str
    generation_truncated: bool
    expected_label: JudgeLabel

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "case_id": self.case_id,
            "original_prompt": self.original_prompt,
            "trajectory": self.trajectory,
            "generation_truncated": self.generation_truncated,
            "expected_label": self.expected_label.value,
        }


@dataclass(frozen=True)
class JudgeValidationResult:
    case_id: str
    expected_label: JudgeLabel
    status: Literal["OK", "ERROR"]
    actual_label: JudgeLabel | None = None
    error_code: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {
            "case_id": self.case_id,
            "expected_label": self.expected_label.value,
            "status": self.status,
        }
        if self.actual_label is not None:
            result["actual_label"] = self.actual_label.value
        if self.error_code is not None:
            result["error_code"] = self.error_code
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JudgeValidationResult:
        if not isinstance(value, dict):
            raise ArtifactError("judge validation result must be an object")
        status = value.get("status")
        required = {"case_id", "expected_label", "status"}
        if status == "OK":
            required.add("actual_label")
        elif status == "ERROR":
            required.add("error_code")
        else:
            raise ArtifactError("judge validation result has an invalid status")
        if set(value) != required:
            raise ArtifactError("judge validation result has invalid fields")
        case_id = value["case_id"]
        error_code = value.get("error_code")
        if type(case_id) is not str or not case_id:
            raise ArtifactError("judge validation result has an invalid case_id")
        if error_code is not None and (type(error_code) is not str or not error_code):
            raise ArtifactError("judge validation result has an invalid error_code")
        try:
            expected_label = JudgeLabel(value["expected_label"])
            actual_label = JudgeLabel(value["actual_label"]) if "actual_label" in value else None
        except (TypeError, ValueError) as error:
            raise ArtifactError("judge validation result has an invalid label") from error
        return cls(
            case_id=case_id,
            expected_label=expected_label,
            status=status,
            actual_label=actual_label,
            error_code=error_code,
        )


@dataclass(frozen=True)
class LabeledTrajectory:
    prompt_id: str
    label: Literal["REFUSAL", "NON_REFUSAL"]
    trajectory_hash: str


@dataclass(frozen=True)
class ActivationKey:
    layer: int

    @property
    def storage_key(self) -> str:
        return str(self.layer)

    @classmethod
    def parse(cls, value: str) -> ActivationKey:
        return cls(layer=int(value))


@dataclass(frozen=True)
class DirectionCandidate:
    candidate_id: str
    layer: int
    norm: float
    refusal_count: int
    non_refusal_count: int
    standardized_separation: float
    refusal_projected_mean: float
    non_refusal_projected_mean: float
    refusal_projected_variance_diagonal: float
    non_refusal_projected_variance_diagonal: float
    finite: bool


@dataclass(frozen=True)
class CEEvaluation:
    source: Literal["baseline_non_refusal_completions", "reference_files"]
    input_count: int
    target_token_count: int
    baseline_loss: float
    intervention_loss: float | None
    error_code: Literal["NON_FINITE"] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.input_count, bool) or not isinstance(self.input_count, int) or self.input_count <= 0:
            raise InvariantError("CE input count must be positive")
        if (
            isinstance(self.target_token_count, bool)
            or not isinstance(self.target_token_count, int)
            or self.target_token_count <= 0
        ):
            raise InvariantError("CE target token count must be positive")
        if not math.isfinite(self.baseline_loss) or self.baseline_loss < 0:
            raise InvariantError("baseline CE loss must be finite and non-negative")
        if self.intervention_loss is None:
            if self.error_code != "NON_FINITE":
                raise InvariantError("missing intervention CE loss requires a non-finite error")
        elif not math.isfinite(self.intervention_loss) or self.intervention_loss < 0 or self.error_code is not None:
            raise InvariantError("intervention CE loss or error is invalid")

    @property
    def loss_delta(self) -> float | None:
        return self.intervention_loss - self.baseline_loss if self.intervention_loss is not None else None

    def as_dict(self) -> dict[str, str | int | float]:
        value: dict[str, str | int | float] = {
            "source": self.source,
            "input_count": self.input_count,
            "target_token_count": self.target_token_count,
            "baseline_loss": self.baseline_loss,
        }
        if self.intervention_loss is None:
            value["error_code"] = "NON_FINITE"
        else:
            value["intervention_loss"] = self.intervention_loss
            value["loss_delta"] = self.intervention_loss - self.baseline_loss
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CEEvaluation:
        common = {"source", "input_count", "target_token_count", "baseline_loss"}
        if not isinstance(value, dict):
            raise ArtifactError("CE evaluation has invalid fields")
        fields = set(value)
        success = fields == common | {"intervention_loss", "loss_delta"}
        failure = fields == common | {"error_code"}
        if not success and not failure:
            raise ArtifactError("CE evaluation has invalid fields")
        source = value["source"]
        input_count = value["input_count"]
        target_token_count = value["target_token_count"]
        baseline_loss = value["baseline_loss"]
        intervention_loss = value.get("intervention_loss")
        loss_delta = value.get("loss_delta")
        error_code = value.get("error_code")
        if (
            source not in {"baseline_non_refusal_completions", "reference_files"}
            or isinstance(input_count, bool)
            or not isinstance(input_count, int)
            or isinstance(target_token_count, bool)
            or not isinstance(target_token_count, int)
            or isinstance(baseline_loss, bool)
            or not isinstance(baseline_loss, int | float)
            or (
                success
                and (
                    isinstance(intervention_loss, bool)
                    or not isinstance(intervention_loss, int | float)
                    or isinstance(loss_delta, bool)
                    or not isinstance(loss_delta, int | float)
                )
            )
            or (failure and error_code != "NON_FINITE")
        ):
            raise ArtifactError("CE evaluation has invalid values")
        try:
            result = cls(
                source=source,
                input_count=input_count,
                target_token_count=target_token_count,
                baseline_loss=float(baseline_loss),
                intervention_loss=float(intervention_loss) if intervention_loss is not None else None,
                error_code=error_code,
            )
        except InvariantError as error:
            raise ArtifactError("CE evaluation has invalid values") from error
        if success and not math.isclose(
            cast(float, result.loss_delta),
            float(cast(int | float, loss_delta)),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ArtifactError("CE evaluation loss delta does not match its losses")
        return result


@dataclass(frozen=True)
class CandidateMetrics:
    candidate_id: str
    baseline_refusal_count: int
    baseline_non_refusal_count: int
    intervention_non_refusal_from_refusal: int
    intervention_non_refusal_retained: int
    uncertain_count: int
    error_count: int
    removal_success_rate: float
    non_refusal_retention_rate: float
    uncertain_rate: float
    error_rate: float
    mean_kl: float | None
    completion_ce: CEEvaluation
    baseline_completion_rate: float
    completion_rate: float
    completion_rate_delta: float
    baseline_empty_answer_rate: float
    empty_answer_rate: float
    empty_answer_rate_delta: float
    baseline_abnormal_repetition_rate: float
    abnormal_repetition_rate: float
    abnormal_repetition_rate_delta: float
    baseline_control_delimiter_repetition_rate: float
    control_delimiter_repetition_rate: float
    control_delimiter_repetition_rate_delta: float
    baseline_mean_thinking_tokens: float
    baseline_mean_final_tokens: float
    mean_thinking_tokens: float
    mean_final_tokens: float
    mean_thinking_tokens_delta: float
    mean_final_tokens_delta: float
    activation_addition_induction_rate: float | None = None
    activation_addition_error_rate: float | None = None
    hard_filter_passed: bool = False
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.completion_ce.source != "baseline_non_refusal_completions":
            raise InvariantError("acceptance CE must use baseline NON_REFUSAL completions")

    def as_dict(self) -> dict[str, Any]:
        value = {key: item for key, item in asdict(self).items() if item is not None}
        value["mean_kl"] = self.mean_kl
        value["completion_ce"] = self.completion_ce.as_dict()
        return value


@dataclass(frozen=True)
class CompatibilityReport:
    adapter: str
    model_class: str
    architecture: tuple[str, ...]
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    parameter_count: int
    parameter_shapes_hash: str
    compatible: bool
    errors: tuple[str, ...] = ()
    topology: dict[str, Any] = field(default_factory=dict)
