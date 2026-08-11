from dataclasses import fields, replace
from pathlib import Path

import pytest

from self_judged_refusal_direction.config import (
    AcceptanceConfig,
    DataConfig,
    ExportConfig,
    ModelConfig,
    ProjectConfig,
    RunConfig,
    SearchConfig,
    TargetGenerationConfig,
    load_config,
)
from self_judged_refusal_direction.errors import ConfigurationError

REVISION = "a" * 40
MODEL_ID = "model/id"
OUTPUT_DIR = "runs/test"


def valid_config() -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(output_dir=OUTPUT_DIR),
        model=ModelConfig(id=MODEL_ID, revision=REVISION),
    )


def test_required_path_and_model_identity_have_no_defaults() -> None:
    with pytest.raises(ConfigurationError, match=r"run\.output_dir"):
        ProjectConfig().validate()
    with pytest.raises(ConfigurationError, match=r"model\.id"):
        ProjectConfig(run=RunConfig(output_dir=OUTPUT_DIR), model=ModelConfig(revision=REVISION)).validate()

    valid_config().validate()


def test_schema_contains_only_supported_sections_and_fields() -> None:
    assert tuple(item.name for item in fields(ProjectConfig)) == (
        "run",
        "model",
        "target_generation",
        "data",
        "search",
        "acceptance",
        "export",
    )
    assert tuple(item.name for item in fields(RunConfig)) == (
        "seed",
        "output_dir",
        "max_infrastructure_errors",
    )
    assert tuple(item.name for item in fields(ModelConfig)) == (
        "id",
        "revision",
        "dtype",
        "device_map",
        "attention_implementation",
    )
    assert tuple(item.name for item in fields(TargetGenerationConfig)) == (
        "system_prompt",
        "thinking_enabled",
        "max_new_tokens",
        "do_sample",
        "num_beams",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "repetition_penalty",
    )
    assert tuple(item.name for item in fields(DataConfig)) == (
        "prompt_files",
        "reference_files",
        "train_fraction",
        "validation_fraction",
        "train_per_class",
        "validation_per_class",
        "max_test_prompts",
        "max_text_tokens",
        "template_similarity_threshold",
    )
    assert tuple(item.name for item in fields(SearchConfig)) == (
        "layers",
        "accumulator_dtype",
        "activation_screening_keep",
        "pilot_evaluation_keep",
        "pilot_prompts_per_class",
    )
    assert tuple(item.name for item in fields(AcceptanceConfig)) == (
        "max_uncertain_rate",
        "max_error_rate",
        "min_non_refusal_retention",
        "max_mean_kl",
        "max_ce_loss_delta",
        "activation_addition_beta",
    )
    assert tuple(item.name for item in fields(ExportConfig)) == ("max_shard_size", "edit_chunk_rows")


def test_reference_config_is_valid_and_derives_test_fraction() -> None:
    path = Path(__file__).parents[2] / "configs" / "gemma4_31b_it.yaml"
    config = load_config(path)

    assert config.data.test_fraction == pytest.approx(0.2)
    assert config.target_generation.system_prompt is None


@pytest.mark.parametrize(
    "raw",
    [
        {"judge": {}},
        {"direction": {}},
        {"evaluation": {}},
        {"model": {"revision": REVISION}, "run": {"max_errors": 0}},
        {"model": {"revision": REVISION}, "run": {"system_prompt": None}},
        {"model": {"revision": REVISION}, "model_config": {}},
        {"model": {"revision": REVISION, "adapter": "gemma4"}},
        {"model": {"revision": REVISION}, "data": {"test_fraction": 0.2}},
        {"model": {"revision": REVISION}, "export": {"safe_serialization": True}},
    ],
)
def test_legacy_keys_are_rejected(raw: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError, match="unknown"):
        ProjectConfig.from_mapping(raw)


def test_structured_prompt_files_are_rejected() -> None:
    config = valid_config()

    with pytest.raises(ConfigurationError, match=r"\.txt extension"):
        replace(config, data=replace(config.data, prompt_files=("prompts.jsonl",))).validate()


def test_sampling_parameters_require_sampling() -> None:
    config = valid_config()
    sampling = replace(
        config.target_generation,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.1,
        typical_p=1.0,
    )
    replace(config, target_generation=sampling).validate()
    replace(config, target_generation=replace(sampling, do_sample=None)).validate()

    with pytest.raises(ConfigurationError, match="require do_sample=true or null"):
        replace(config, target_generation=replace(sampling, do_sample=False)).validate()


def test_thinking_mode_is_user_selectable() -> None:
    config = valid_config()

    replace(config, target_generation=replace(config.target_generation, thinking_enabled=False)).validate()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (replace(valid_config(), run=replace(valid_config().run, max_infrastructure_errors=-1)), "max_infrastructure"),
        (replace(valid_config(), model=replace(valid_config().model, revision="main")), "commit SHA"),
        (
            replace(
                valid_config(),
                target_generation=replace(valid_config().target_generation, thinking_enabled="yes"),
            ),
            "thinking_enabled",
        ),
        (
            replace(valid_config(), target_generation=replace(valid_config().target_generation, num_beams=0)),
            "num_beams",
        ),
        (
            replace(valid_config(), target_generation=replace(valid_config().target_generation, temperature=0.0)),
            "temperature",
        ),
        (
            replace(
                valid_config(),
                target_generation=replace(valid_config().target_generation, do_sample=True, top_p=0.0),
            ),
            "top_p",
        ),
        (
            replace(
                valid_config(),
                target_generation=replace(valid_config().target_generation, repetition_penalty=0.0),
            ),
            "repetition_penalty",
        ),
        (
            replace(valid_config(), data=replace(valid_config().data, train_fraction=0.8, validation_fraction=0.2)),
            "derived test fractions",
        ),
        (
            replace(
                valid_config(),
                data=replace(valid_config().data, template_similarity_threshold=float("nan")),
            ),
            "template_similarity_threshold",
        ),
        (replace(valid_config(), search=replace(valid_config().search, layers=(-1,))), "search.layers"),
        (replace(valid_config(), search=replace(valid_config().search, layers=(1, 1))), "search.layers"),
        (
            replace(valid_config(), search=replace(valid_config().search, accumulator_dtype="torch.float64")),
            "accumulator_dtype",
        ),
        (
            replace(
                valid_config(),
                search=replace(valid_config().search, activation_screening_keep=4, pilot_evaluation_keep=5),
            ),
            "pilot_evaluation_keep",
        ),
        (
            replace(
                valid_config(),
                search=replace(valid_config().search, pilot_prompts_per_class=65),
            ),
            "pilot_prompts_per_class",
        ),
        (
            replace(
                valid_config(),
                acceptance=replace(valid_config().acceptance, max_uncertain_rate=1.1),
            ),
            "max_uncertain_rate",
        ),
        (
            replace(valid_config(), acceptance=replace(valid_config().acceptance, max_mean_kl=-0.1)),
            "max_mean_kl",
        ),
        (
            replace(valid_config(), acceptance=replace(valid_config().acceptance, activation_addition_beta=0.0)),
            "activation_addition_beta",
        ),
        (replace(valid_config(), export=replace(valid_config().export, edit_chunk_rows=0)), "edit_chunk_rows"),
    ],
)
def test_invalid_values_and_cross_section_constraints_fail_fast(
    config: ProjectConfig,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        config.validate()


def test_experiment_hash_excludes_storage_controls_and_tracks_model_execution() -> None:
    config = valid_config()
    operational = replace(
        config,
        run=replace(config.run, output_dir="another-run", max_infrastructure_errors=7),
        export=replace(config.export, max_shard_size="1GB", edit_chunk_rows=128),
    )

    assert operational.config_hash == config.config_hash
    assert operational.target_generation_config_hash == config.target_generation_config_hash
    for model in (
        replace(config.model, dtype="float16"),
        replace(config.model, device_map="cuda:0"),
        replace(config.model, attention_implementation="eager"),
    ):
        changed = replace(config, model=model)
        assert changed.config_hash != config.config_hash
        assert changed.target_generation_config_hash != config.target_generation_config_hash
    assert replace(config, run=replace(config.run, seed=43)).config_hash != config.config_hash
    assert replace(config, acceptance=replace(config.acceptance, max_mean_kl=0.2)).config_hash != config.config_hash


def test_target_generation_hash_tracks_raw_generation_config() -> None:
    config = valid_config()
    changed = replace(
        config,
        target_generation=replace(config.target_generation, system_prompt="system", repetition_penalty=1.1),
    )

    assert changed.target_generation_config_hash != config.target_generation_config_hash
    assert changed.config_hash != config.config_hash
