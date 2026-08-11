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
    output_dir: str | None = None
    max_infrastructure_errors: int = 0


@dataclass(frozen=True)
class ModelConfig:
    id: str | None = None
    revision: str = ""
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attention_implementation: str = "sdpa"


@dataclass(frozen=True)
class TargetGenerationConfig:
    system_prompt: str | None = None
    thinking_enabled: bool = True
    max_new_tokens: int = 4096
    do_sample: bool | None = False
    num_beams: int | None = 1
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    typical_p: float | None = None
    repetition_penalty: float | None = None


@dataclass(frozen=True)
class DataConfig:
    prompt_files: tuple[str, ...] = ()
    reference_files: tuple[str, ...] = ()
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    train_per_class: int = 128
    validation_per_class: int = 64
    max_test_prompts: int = 256
    max_text_tokens: int = 8192
    template_similarity_threshold: float = 0.9

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.validation_fraction


@dataclass(frozen=True)
class SearchConfig:
    layers: str | tuple[int, ...] = "all"
    accumulator_dtype: str = "float64"
    activation_screening_keep: int = 32
    pilot_evaluation_keep: int = 5
    pilot_prompts_per_class: int = 16


@dataclass(frozen=True)
class AcceptanceConfig:
    max_uncertain_rate: float = 0.05
    max_error_rate: float = 0.0
    min_non_refusal_retention: float = 0.95
    max_mean_kl: float = 0.10
    max_ce_loss_delta: float = 0.10
    activation_addition_beta: float | None = 1.0


@dataclass(frozen=True)
class ExportConfig:
    max_shard_size: str = "5GB"
    edit_chunk_rows: int = 4096


@dataclass(frozen=True)
class ProjectConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    target_generation: TargetGenerationConfig = field(default_factory=TargetGenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @property
    def config_hash(self) -> str:
        return object_sha256(
            {
                "seed": self.run.seed,
                "model": {
                    "id": self.model.id,
                    "revision": self.model.revision,
                    "dtype": self.model.dtype,
                    "device_map": self.model.device_map,
                    "attention_implementation": self.model.attention_implementation,
                },
                "target_generation": self.target_generation,
                "data": self.data,
                "search": self.search,
                "acceptance": self.acceptance,
            }
        )

    @property
    def target_generation_config_hash(self) -> str:
        return object_sha256(
            {
                "model": {
                    "id": self.model.id,
                    "revision": self.model.revision,
                    "dtype": self.model.dtype,
                    "device_map": self.model.device_map,
                    "attention_implementation": self.model.attention_implementation,
                },
                "generation": self.target_generation,
                "seed": self.run.seed,
            }
        )

    def validate(self) -> None:
        errors: list[str] = []
        if not isinstance(self.run.seed, int) or isinstance(self.run.seed, bool):
            errors.append("run.seed must be an integer")
        if not isinstance(self.run.output_dir, str) or not self.run.output_dir.strip():
            errors.append("run.output_dir must be a non-empty string")
        if not _non_negative_integer(self.run.max_infrastructure_errors):
            errors.append("run.max_infrastructure_errors must be a non-negative integer")
        for name, value in (
            ("model.id", self.model.id),
            ("model.dtype", self.model.dtype),
            ("model.device_map", self.model.device_map),
            ("model.attention_implementation", self.model.attention_implementation),
        ):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name} must be a non-empty string")
        if not isinstance(self.model.revision, str) or not PINNED_REVISION.fullmatch(self.model.revision):
            errors.append("model.revision must be a 40-character lowercase commit SHA")
        generation = self.target_generation
        if generation.system_prompt is not None and not isinstance(generation.system_prompt, str):
            errors.append("target_generation.system_prompt must be a string or null")
        if not isinstance(generation.thinking_enabled, bool):
            errors.append("target_generation.thinking_enabled must be a boolean")
        if not _positive_integer(generation.max_new_tokens):
            errors.append("target_generation.max_new_tokens must be a positive integer")
        if generation.do_sample is not None and not isinstance(generation.do_sample, bool):
            errors.append("target_generation.do_sample must be a boolean or null")
        if generation.num_beams is not None and not _positive_integer(generation.num_beams):
            errors.append("target_generation.num_beams must be a positive integer or null")
        if generation.temperature is not None and (
            not _finite_number(generation.temperature) or generation.temperature <= 0
        ):
            errors.append("target_generation.temperature must be positive or null")
        for name, value, lower_inclusive in (
            ("target_generation.top_p", generation.top_p, False),
            ("target_generation.min_p", generation.min_p, True),
            ("target_generation.typical_p", generation.typical_p, False),
        ):
            if value is not None and (
                not _finite_number(value) or value > 1 or value < 0 or (not lower_inclusive and value == 0)
            ):
                errors.append(f"{name} must be between zero and one or null")
        if generation.top_k is not None and not _non_negative_integer(generation.top_k):
            errors.append("target_generation.top_k must be a non-negative integer or null")
        if generation.repetition_penalty is not None and (
            not _finite_number(generation.repetition_penalty) or generation.repetition_penalty <= 0
        ):
            errors.append("target_generation.repetition_penalty must be positive or null")
        sampling_values = (
            generation.temperature,
            generation.top_p,
            generation.top_k,
            generation.min_p,
            generation.typical_p,
        )
        if any(value is not None for value in sampling_values) and generation.do_sample is False:
            errors.append("target generation sampling parameters require do_sample=true or null")
        if not all(
            isinstance(paths, tuple) and all(isinstance(path, str) for path in paths)
            for paths in (self.data.prompt_files, self.data.reference_files)
        ):
            errors.append("data file collections must contain strings")
        elif any(
            Path(path).suffix.casefold() != ".txt"
            for paths in (self.data.prompt_files, self.data.reference_files)
            for path in paths
        ):
            errors.append("data files must use the .txt extension")
        if not _positive_integer(self.data.max_text_tokens):
            errors.append("data.max_text_tokens must be a positive integer")
        if not _finite_number(self.data.train_fraction) or not _finite_number(self.data.validation_fraction):
            errors.append("data train and validation fractions must be finite numbers")
        else:
            if self.data.train_fraction <= 0 or self.data.validation_fraction <= 0 or self.data.test_fraction <= 0:
                errors.append("data train, validation, and derived test fractions must be positive")
        for name, value in (
            ("data.train_per_class", self.data.train_per_class),
            ("data.validation_per_class", self.data.validation_per_class),
            ("data.max_test_prompts", self.data.max_test_prompts),
        ):
            if not _positive_integer(value):
                errors.append(f"{name} must be a positive integer")
        threshold = self.data.template_similarity_threshold
        if not _finite_number(threshold) or not 0 <= threshold <= 1:
            errors.append("data.template_similarity_threshold must be between zero and one")
        layers = self.search.layers
        if layers != "all" and (
            not isinstance(layers, tuple)
            or not layers
            or any(not _non_negative_integer(layer) for layer in layers)
            or len(set(layers)) != len(layers)
        ):
            errors.append("search.layers must be all or a non-empty tuple of unique non-negative integers")
        if self.search.accumulator_dtype not in {"float32", "float64"}:
            errors.append("search.accumulator_dtype must be float32 or float64")
        for name, value in (
            ("search.activation_screening_keep", self.search.activation_screening_keep),
            ("search.pilot_evaluation_keep", self.search.pilot_evaluation_keep),
            ("search.pilot_prompts_per_class", self.search.pilot_prompts_per_class),
        ):
            if not _positive_integer(value):
                errors.append(f"{name} must be a positive integer")
        if (
            _positive_integer(self.search.activation_screening_keep)
            and _positive_integer(self.search.pilot_evaluation_keep)
            and self.search.pilot_evaluation_keep > self.search.activation_screening_keep
        ):
            errors.append("search.pilot_evaluation_keep must not exceed search.activation_screening_keep")
        if (
            _positive_integer(self.search.pilot_prompts_per_class)
            and _positive_integer(self.data.validation_per_class)
            and self.search.pilot_prompts_per_class > self.data.validation_per_class
        ):
            errors.append("search.pilot_prompts_per_class must not exceed data.validation_per_class")
        for name, value in (
            ("acceptance.max_uncertain_rate", self.acceptance.max_uncertain_rate),
            ("acceptance.max_error_rate", self.acceptance.max_error_rate),
            ("acceptance.min_non_refusal_retention", self.acceptance.min_non_refusal_retention),
        ):
            if not _finite_number(value) or not 0 <= value <= 1:
                errors.append(f"{name} must be between zero and one")
        for name, value in (
            ("acceptance.max_mean_kl", self.acceptance.max_mean_kl),
            ("acceptance.max_ce_loss_delta", self.acceptance.max_ce_loss_delta),
        ):
            if not _finite_number(value) or value < 0:
                errors.append(f"{name} must be non-negative")
        addition_beta = self.acceptance.activation_addition_beta
        if addition_beta is not None and (not _finite_number(addition_beta) or addition_beta <= 0):
            errors.append("acceptance.activation_addition_beta must be positive or null")
        if not isinstance(self.export.max_shard_size, str) or not self.export.max_shard_size.strip():
            errors.append("export.max_shard_size must be a non-empty string")
        if not _positive_integer(self.export.edit_chunk_rows):
            errors.append("export.edit_chunk_rows must be a positive integer")
        if errors:
            raise ConfigurationError("; ".join(errors))

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ProjectConfig:
        allowed = {item.name for item in dataclasses.fields(cls)}
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
            for key in ("prompt_files", "reference_files"):
                if key in values and isinstance(values[key], list):
                    values[key] = tuple(values[key])
            if name == "search" and isinstance(values.get("layers"), list):
                values["layers"] = tuple(values["layers"])
            try:
                return section_type(**values)
            except TypeError as error:
                raise ConfigurationError(f"invalid {name} config: {error}") from error

        config = cls(
            run=section("run", RunConfig),
            model=section("model", ModelConfig),
            target_generation=section("target_generation", TargetGenerationConfig),
            data=section("data", DataConfig),
            search=section("search", SearchConfig),
            acceptance=section("acceptance", AcceptanceConfig),
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
    return ProjectConfig.from_mapping(raw)


def resolved_config_mapping(config: ProjectConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)
