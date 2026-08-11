from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal


class TrajectoryStatus(StrEnum):
    OK = "OK"
    ERROR = "ERROR"


class JudgeStatus(StrEnum):
    OK = "OK"
    ERROR = "ERROR"


class JudgeLabel(StrEnum):
    REFUSAL = "REFUSAL"
    NON_REFUSAL = "NON_REFUSAL"
    UNCERTAIN = "UNCERTAIN"


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
    thinking_segments: tuple[str, ...]
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
    trajectory_status: Literal["OK", "ERROR"] = "OK"
    split: Literal["train", "validation", "test"] | None = None
    seed: int = 42
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TargetTrajectory:
        data = dict(value)
        data["raw_generated_token_ids"] = tuple(data["raw_generated_token_ids"])
        data["thinking_segments"] = tuple(data["thinking_segments"])
        return cls(**data)


@dataclass(frozen=True)
class JudgeResult:
    status: Literal["OK", "ERROR"]
    label: Literal["REFUSAL", "NON_REFUSAL", "UNCERTAIN"] | None
    raw_output: str | None
    label_logprobs: dict[str, float] | None
    calibrated_margin: float | None
    trajectory_hash: str
    judge_profile_hash: str
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JudgeResult:
        return cls(**value)


@dataclass(frozen=True)
class LabeledTrajectory:
    prompt_id: str
    split: Literal["train", "validation"]
    label: Literal["REFUSAL", "NON_REFUSAL"]
    trajectory_hash: str


@dataclass(frozen=True)
class ActivationKey:
    phase: Literal["pre_thinking", "pre_final"]
    layer: int
    relative_position: int

    @property
    def storage_key(self) -> str:
        return f"{self.phase}:{self.layer}:{self.relative_position}"

    @classmethod
    def parse(cls, value: str) -> ActivationKey:
        phase, layer, position = value.split(":", maxsplit=2)
        if phase not in {"pre_thinking", "pre_final"}:
            raise ValueError(f"unsupported activation phase: {phase}")
        return cls(phase=phase, layer=int(layer), relative_position=int(position))


@dataclass(frozen=True)
class DirectionCandidate:
    candidate_id: str
    phase: Literal["pre_thinking", "pre_final"]
    layer: int
    relative_position: int
    direction_index: int
    norm: float
    refusal_count: int
    non_refusal_count: int
    standardized_separation: float
    refusal_projected_mean: float
    non_refusal_projected_mean: float
    refusal_projected_variance_diagonal: float
    non_refusal_projected_variance_diagonal: float
    boundary_token: str
    finite: bool


@dataclass(frozen=True)
class CandidateMetrics:
    candidate_id: str
    stage: Literal["B", "C"]
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
    hard_filter_passed: bool = False
    rejection_reasons: tuple[str, ...] = ()


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
    supports_thinking_parse: bool
    supports_pre_final_activation: bool
    supports_dense_export: bool
    supports_moe_export: bool
    supports_ple_export: bool
    supports_multimodal_projection_export: bool
    supports_tied_weight_export: bool
    compatible: bool
    errors: tuple[str, ...] = ()
    topology: dict[str, Any] = field(default_factory=dict)
