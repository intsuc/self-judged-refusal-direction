from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from self_judged_refusal_direction.errors import ConfigurationError
from self_judged_refusal_direction.hashing import object_sha256

LABELS = ("REFUSAL", "NON_REFUSAL", "UNCERTAIN")
PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    output_dir: str = "runs/gemma4_31b_it"
    deterministic: bool = True
    max_errors: int = 0
    system_prompt: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    id: str = "google/gemma-4-31B-it"
    revision: str = ""
    adapter: str = "gemma4"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    attention_implementation: str = "sdpa"
    low_cpu_mem_usage: bool = True


@dataclass(frozen=True)
class TargetGenerationConfig:
    thinking_enabled: bool = True
    do_sample: bool = False
    num_beams: int = 1
    max_new_tokens: int = 4096
    use_cache: bool = True


@dataclass(frozen=True)
class JudgeConfig:
    backend: str = "enum_trie"
    thinking_enabled: bool = False
    labels: tuple[str, ...] = LABELS
    exact_output_only: bool = True
    do_sample: bool = False
    num_beams: int = 1
    force_eos_at_terminal: bool = True
    score_allowed_labels: bool = True
    calibrated_margin_threshold: float | None = None
    require_full_trajectory_in_context: bool = True
    infrastructure_error_policy: str = "fail_closed"
    safety_margin_tokens: int = 32


@dataclass(frozen=True)
class DataConfig:
    raw_prompt_files: tuple[str, ...] = ()
    quality_text_files: tuple[str, ...] = ()
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    train_per_class: int = 128
    validation_per_class: int = 64
    test_raw_count: int = 256
    max_prompt_tokens: int = 8192
    deduplicate: bool = True
    split_before_labeling: bool = True
    approximate_duplicate_threshold: float = 0.9


@dataclass(frozen=True)
class DirectionConfig:
    candidate_phases: tuple[str, ...] = ("pre_thinking",)
    enable_pre_final_fallback: bool = False
    max_boundary_positions: int = 8
    candidate_layers: str | tuple[int, ...] = "all"
    online_accumulator_dtype: str = "float64"
    stage_a_top_m: int = 32
    stage_b_top_k: int = 5
    minimum_direction_norm: float = 1e-8


@dataclass(frozen=True)
class EvaluationConfig:
    stage_b_prompts_per_class: int = 16
    max_uncertain_rate: float = 0.05
    max_error_rate: float = 0.0
    min_non_refusal_retention: float = 0.95
    max_mean_kl: float = 0.10
    max_ce_loss_delta: float = 0.10
    run_activation_addition_diagnostic: bool = True
    activation_addition_beta: float = 1.0
    repetition_ngram_size: int = 4
    repetition_threshold: float = 0.5


@dataclass(frozen=True)
class ExportConfig:
    safe_serialization: bool = True
    max_shard_size: str = "5GB"
    edit_compute_dtype: str = "float32"
    edit_chunk_rows: int = 4096
    include_processor: bool = True
    include_raw_thinking: bool = False
    push_to_hub: bool = False
    verify_fresh_process: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    target_generation: TargetGenerationConfig = field(default_factory=TargetGenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @property
    def config_hash(self) -> str:
        return object_sha256(self)

    @property
    def target_profile_hash(self) -> str:
        return object_sha256(
            {
                "model_id": self.model.id,
                "revision": self.model.revision,
                "system_prompt": self.run.system_prompt,
                "generation": self.target_generation,
                "seed": self.run.seed,
            }
        )

    @property
    def judge_profile_hash(self) -> str:
        return object_sha256(
            {
                "model_id": self.model.id,
                "revision": self.model.revision,
                "judge": self.judge,
            }
        )

    def validate(self) -> None:
        errors: list[str] = []
        if not isinstance(self.run.seed, int) or isinstance(self.run.seed, bool):
            errors.append("run.seed must be an integer")
        if not isinstance(self.run.output_dir, str) or not self.run.output_dir.strip():
            errors.append("run.output_dir must be a non-empty string")
        if self.run.deterministic is not True:
            errors.append("run.deterministic must be true")
        if not _non_negative_integer(self.run.max_errors):
            errors.append("run.max_errors must be a non-negative integer")
        if self.run.system_prompt is not None and not isinstance(self.run.system_prompt, str):
            errors.append("run.system_prompt must be a string or null")
        if not isinstance(self.model.id, str) or not self.model.id.strip():
            errors.append("model.id must be a non-empty string")
        if not isinstance(self.model.adapter, str) or not self.model.adapter.strip():
            errors.append("model.adapter must be a non-empty string")
        if not isinstance(self.model.revision, str) or not PINNED_REVISION.fullmatch(self.model.revision):
            errors.append("model.revision must be a 40-character lowercase commit SHA")
        if self.model.trust_remote_code is not False:
            errors.append("model.trust_remote_code must be false")
        if self.target_generation.thinking_enabled is not True:
            errors.append("target_generation.thinking_enabled must be true")
        if self.judge.thinking_enabled is not False:
            errors.append("judge.thinking_enabled must be false")
        if tuple(self.judge.labels) != LABELS:
            errors.append(f"judge.labels must equal {list(LABELS)} in that order")
        if self.judge.backend != "enum_trie":
            errors.append("judge.backend must be enum_trie")
        if self.judge.exact_output_only is not True or self.judge.force_eos_at_terminal is not True:
            errors.append("judge exact output and terminal EOS constraints must be enabled")
        if (
            self.judge.do_sample is not False
            or not _positive_integer(self.judge.num_beams)
            or self.judge.num_beams != 1
        ):
            errors.append("judge must use greedy decoding with one beam")
        if self.judge.require_full_trajectory_in_context is not True:
            errors.append("judge.require_full_trajectory_in_context must be true")
        if self.judge.infrastructure_error_policy != "fail_closed":
            errors.append("judge.infrastructure_error_policy must be fail_closed")
        if not _non_negative_integer(self.judge.safety_margin_tokens):
            errors.append("judge.safety_margin_tokens must be a non-negative integer")
        margin_threshold = self.judge.calibrated_margin_threshold
        if margin_threshold is not None and (not _finite_number(margin_threshold) or margin_threshold < 0):
            errors.append("judge.calibrated_margin_threshold must be non-negative or null")
        if not isinstance(self.judge.score_allowed_labels, bool):
            errors.append("judge.score_allowed_labels must be a boolean")
        elif margin_threshold is not None and not self.judge.score_allowed_labels:
            errors.append("judge.calibrated_margin_threshold requires score_allowed_labels")
        if not _positive_integer(self.target_generation.num_beams) or self.target_generation.num_beams != 1:
            errors.append("target generation must use one beam")
        if self.target_generation.do_sample is not False:
            errors.append("the initial pipeline requires deterministic target generation")
        if self.target_generation.use_cache is not True:
            errors.append("target generation must use the cache")
        if self.export.safe_serialization is not True:
            errors.append("export.safe_serialization must be true")
        if self.export.include_processor is not True:
            errors.append("export.include_processor must be true")
        if self.export.include_raw_thinking is not False:
            errors.append("export.include_raw_thinking must be false")
        if self.export.push_to_hub is not False:
            errors.append("export.push_to_hub must be false")
        if self.export.verify_fresh_process is not True:
            errors.append("export.verify_fresh_process must be true")
        if self.export.edit_compute_dtype != "float32":
            errors.append("export.edit_compute_dtype must be float32")
        if not _positive_integer(self.export.edit_chunk_rows):
            errors.append("export.edit_chunk_rows must be a positive integer")
        if not isinstance(self.export.max_shard_size, str) or not self.export.max_shard_size.strip():
            errors.append("export.max_shard_size must be a non-empty string")
        if not _positive_integer(self.target_generation.max_new_tokens):
            errors.append("target_generation.max_new_tokens must be a positive integer")
        if not _positive_integer(self.data.max_prompt_tokens):
            errors.append("data.max_prompt_tokens must be a positive integer")
        split_fractions = (self.data.train_fraction, self.data.validation_fraction, self.data.test_fraction)
        if not all(_finite_number(value) for value in split_fractions):
            errors.append("data split fractions must be finite numbers")
        else:
            if abs(sum(split_fractions) - 1.0) > 1e-9:
                errors.append("data split fractions must sum to 1")
            if any(value <= 0 for value in split_fractions):
                errors.append("data split fractions must be positive")
        if not all(
            isinstance(paths, tuple) and all(isinstance(path, str) for path in paths)
            for paths in (self.data.raw_prompt_files, self.data.quality_text_files)
        ):
            errors.append("data prompt file collections must contain strings")
        elif any(
            Path(path).suffix.casefold() != ".txt"
            for paths in (self.data.raw_prompt_files, self.data.quality_text_files)
            for path in paths
        ):
            errors.append("data prompt files must use the .txt extension")
        if self.data.deduplicate is not True:
            errors.append("data.deduplicate must be true")
        if self.data.split_before_labeling is not True:
            errors.append("data.split_before_labeling must be true")
        for name, value in (
            ("data.train_per_class", self.data.train_per_class),
            ("data.validation_per_class", self.data.validation_per_class),
            ("data.test_raw_count", self.data.test_raw_count),
        ):
            if not _positive_integer(value):
                errors.append(f"{name} must be a positive integer")
        threshold = self.data.approximate_duplicate_threshold
        if not _finite_number(threshold) or not 0 <= threshold <= 1:
            errors.append("data.approximate_duplicate_threshold must be between zero and one")
        phases = self.direction.candidate_phases
        if not isinstance(phases, tuple) or not phases:
            errors.append("direction.candidate_phases must be a non-empty tuple")
        elif not set(phases) <= {"pre_thinking", "pre_final"}:
            errors.append("direction.candidate_phases contains an unsupported phase")
        if self.direction.enable_pre_final_fallback is not False:
            errors.append("direction.enable_pre_final_fallback must be false")
        layers = self.direction.candidate_layers
        if layers != "all" and (
            not isinstance(layers, tuple) or not layers or any(not _non_negative_integer(layer) for layer in layers)
        ):
            errors.append("direction.candidate_layers must be all or a non-empty tuple of non-negative integers")
        if self.direction.online_accumulator_dtype not in {"float32", "float64", "torch.float32", "torch.float64"}:
            errors.append("direction.online_accumulator_dtype must be float32 or float64")
        if not _positive_integer(self.direction.max_boundary_positions):
            errors.append("direction.max_boundary_positions must be a positive integer")
        if not _positive_integer(self.direction.stage_a_top_m):
            errors.append("direction.stage_a_top_m must be a positive integer")
        if not _positive_integer(self.direction.stage_b_top_k):
            errors.append("direction.stage_b_top_k must be a positive integer")
        if (
            _positive_integer(self.direction.stage_a_top_m)
            and _positive_integer(self.direction.stage_b_top_k)
            and self.direction.stage_a_top_m < self.direction.stage_b_top_k
        ):
            errors.append("direction.stage_a_top_m must be at least direction.stage_b_top_k")
        minimum_norm = self.direction.minimum_direction_norm
        if not _finite_number(minimum_norm) or minimum_norm <= 0:
            errors.append("direction.minimum_direction_norm must be positive")
        if not _positive_integer(self.evaluation.stage_b_prompts_per_class):
            errors.append("evaluation.stage_b_prompts_per_class must be a positive integer")
        for name, value in (
            ("evaluation.max_uncertain_rate", self.evaluation.max_uncertain_rate),
            ("evaluation.max_error_rate", self.evaluation.max_error_rate),
            ("evaluation.min_non_refusal_retention", self.evaluation.min_non_refusal_retention),
            ("evaluation.repetition_threshold", self.evaluation.repetition_threshold),
        ):
            if not _finite_number(value) or not 0 <= value <= 1:
                errors.append(f"{name} must be between zero and one")
        for name, value in (
            ("evaluation.max_mean_kl", self.evaluation.max_mean_kl),
            ("evaluation.max_ce_loss_delta", self.evaluation.max_ce_loss_delta),
        ):
            if not _finite_number(value) or value < 0:
                errors.append(f"{name} must be non-negative")
        addition_beta = self.evaluation.activation_addition_beta
        if not _finite_number(addition_beta) or addition_beta <= 0:
            errors.append("evaluation.activation_addition_beta must be positive")
        if not _positive_integer(self.evaluation.repetition_ngram_size):
            errors.append("evaluation.repetition_ngram_size must be a positive integer")
        if errors:
            raise ConfigurationError("; ".join(errors))

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ProjectConfig:
        allowed = {field.name for field in dataclasses.fields(cls)}
        extras = set(raw) - allowed
        if extras:
            raise ConfigurationError(f"unknown top-level config keys: {sorted(extras, key=str)}")

        def section(name: str, section_type: type[Any]) -> Any:
            raw_section = raw.get(name, {})
            if not isinstance(raw_section, dict):
                raise ConfigurationError(f"{name} config must be a mapping")
            values = dict(raw_section)
            fields = {item.name for item in dataclasses.fields(section_type)}
            unexpected = set(values) - fields
            if unexpected:
                raise ConfigurationError(f"unknown {name} config keys: {sorted(unexpected, key=str)}")
            for key in ("labels", "raw_prompt_files", "quality_text_files", "candidate_phases"):
                if key in values and isinstance(values[key], list):
                    values[key] = tuple(values[key])
            if name == "direction" and isinstance(values.get("candidate_layers"), list):
                values["candidate_layers"] = tuple(values["candidate_layers"])
            try:
                return section_type(**values)
            except TypeError as error:
                raise ConfigurationError(f"invalid {name} config: {error}") from error

        config = cls(
            run=section("run", RunConfig),
            model=section("model", ModelConfig),
            target_generation=section("target_generation", TargetGenerationConfig),
            judge=section("judge", JudgeConfig),
            data=section("data", DataConfig),
            direction=section("direction", DirectionConfig),
            evaluation=section("evaluation", EvaluationConfig),
            export=section("export", ExportConfig),
        )
        try:
            config.validate()
        except (AttributeError, TypeError, ValueError) as error:
            raise ConfigurationError("configuration values have invalid types") from error
        return config


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    try:
        with config_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigurationError(f"failed to read configuration: {config_path}") from error
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping")
    config = ProjectConfig.from_mapping(raw)
    return config


def resolved_config_mapping(config: ProjectConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)
