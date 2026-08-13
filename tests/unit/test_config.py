from dataclasses import replace
from pathlib import Path

import pytest

from self_judged_refusal_direction.config import ModelConfig, ProjectConfig, RunConfig, load_config
from self_judged_refusal_direction.errors import ConfigurationError

REVISION = "a" * 40
MODEL_ID = "model/id"
OUTPUT_DIR = "runs/test"


def valid_config() -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(output_dir=OUTPUT_DIR),
        model=ModelConfig(id=MODEL_ID, revision=REVISION),
    )


def test_reference_config_loads() -> None:
    path = Path(__file__).parents[2] / "configs" / "gemma4_31b_it.yaml"
    load_config(path)


def test_unknown_config_keys_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown"):
        ProjectConfig.from_mapping({"unknown": {}})
    with pytest.raises(ConfigurationError, match="unknown"):
        ProjectConfig.from_mapping({"run": {"unknown": 0}})


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


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (ProjectConfig(), r"run\.output_dir"),
        (
            ProjectConfig(run=RunConfig(output_dir=OUTPUT_DIR), model=ModelConfig(revision=REVISION)),
            r"model\.id",
        ),
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
            replace(valid_config(), target_generation=replace(valid_config().target_generation, batch_size=0)),
            "batch_size",
        ),
        (
            replace(
                valid_config(),
                target_generation=replace(valid_config().target_generation, batch_size=2, do_sample=True),
            ),
            "effective greedy",
        ),
        (
            replace(
                valid_config(),
                target_generation=replace(valid_config().target_generation, batch_size=2, num_beams=2),
            ),
            "num_beams=1",
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
            replace(valid_config(), acceptance=replace(valid_config().acceptance, min_removal_success_rate=-0.1)),
            "min_removal_success_rate",
        ),
        (
            replace(
                valid_config(),
                acceptance=replace(valid_config().acceptance, min_activation_addition_induction_rate=1.1),
            ),
            "min_activation_addition_induction_rate",
        ),
        (
            replace(
                valid_config(),
                acceptance=replace(valid_config().acceptance, max_generation_failure_rate_increase=-0.1),
            ),
            "max_generation_failure_rate_increase",
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


def test_experiment_hash_boundary() -> None:
    config = valid_config()
    operational = replace(
        config,
        run=replace(config.run, output_dir="another-run"),
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
    error_rate_changed = replace(config, acceptance=replace(config.acceptance, max_error_rate=0.2))
    assert error_rate_changed.config_hash == config.config_hash
    assert error_rate_changed.target_generation_config_hash == config.target_generation_config_hash
    acceptance_changed = replace(config, acceptance=replace(config.acceptance, max_mean_kl=0.2))
    assert acceptance_changed.config_hash != config.config_hash
    assert acceptance_changed.target_generation_config_hash == config.target_generation_config_hash
    generation_changed = replace(
        config,
        target_generation=replace(config.target_generation, batch_size=2),
    )
    assert generation_changed.target_generation_config_hash != config.target_generation_config_hash
    assert generation_changed.config_hash != config.config_hash
