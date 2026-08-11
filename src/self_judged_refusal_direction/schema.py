from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

from self_judged_refusal_direction.errors import ArtifactError


class JudgeLabel(StrEnum):
    REFUSAL = "REFUSAL"
    NON_REFUSAL = "NON_REFUSAL"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class JudgeInput:
    original_prompt: str
    thinking_text: str
    final_answer: str
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
    thinking_text: str
    final_answer: str
    generation_truncated: bool
    expected_label: JudgeLabel

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "case_id": self.case_id,
            "original_prompt": self.original_prompt,
            "thinking_text": self.thinking_text,
            "final_answer": self.final_answer,
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
    mean_kl: float
    ce_loss_delta: float
    empty_answer_rate: float
    abnormal_repetition_rate: float
    baseline_mean_thinking_tokens: float
    baseline_mean_final_tokens: float
    mean_thinking_tokens: float
    mean_final_tokens: float
    mean_thinking_tokens_delta: float
    mean_final_tokens_delta: float
    task_completion_proxy: float
    activation_addition_induction_rate: float | None = None
    activation_addition_error_rate: float | None = None
    hard_filter_passed: bool = False
    rejection_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


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
