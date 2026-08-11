from dataclasses import replace

import pytest

from self_judged_refusal_direction.config import JudgeConfig, ModelConfig, ProjectConfig, TargetGenerationConfig
from self_judged_refusal_direction.errors import ConfigurationError

REVISION = "a" * 40


def valid_config() -> ProjectConfig:
    return ProjectConfig(model=ModelConfig(revision=REVISION))


def test_mode_invariants_are_not_configurable() -> None:
    config = valid_config()
    config.validate()
    with pytest.raises(ConfigurationError, match=r"target_generation\.thinking_enabled"):
        replace(config, target_generation=TargetGenerationConfig(thinking_enabled=False)).validate()
    with pytest.raises(ConfigurationError, match=r"judge\.thinking_enabled"):
        replace(config, judge=JudgeConfig(thinking_enabled=True)).validate()
    with pytest.raises(ConfigurationError, match="use the cache"):
        replace(config, target_generation=replace(config.target_generation, use_cache=False)).validate()


def test_revision_and_publication_invariants_fail_closed() -> None:
    config = valid_config()
    with pytest.raises(ConfigurationError, match="commit SHA"):
        replace(config, model=replace(config.model, revision="main")).validate()
    with pytest.raises(ConfigurationError, match="include_raw_thinking"):
        replace(config, export=replace(config.export, include_raw_thinking=True)).validate()
    with pytest.raises(ConfigurationError, match="push_to_hub"):
        replace(config, export=replace(config.export, push_to_hub=True)).validate()
    with pytest.raises(ConfigurationError, match="safe_serialization"):
        replace(config, export=replace(config.export, safe_serialization=False)).validate()
    with pytest.raises(ConfigurationError, match="verify_fresh_process"):
        replace(config, export=replace(config.export, verify_fresh_process=False)).validate()


def test_profile_hash_separates_target_and_judge_modes() -> None:
    config = valid_config()
    assert config.target_profile_hash != config.judge_profile_hash


def test_full_context_and_fail_closed_policies_are_not_configurable() -> None:
    config = valid_config()
    with pytest.raises(ConfigurationError, match="require_full_trajectory_in_context"):
        replace(config, judge=replace(config.judge, require_full_trajectory_in_context=False)).validate()
    with pytest.raises(ConfigurationError, match="infrastructure_error_policy"):
        replace(config, judge=replace(config.judge, infrastructure_error_policy="continue")).validate()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (replace(valid_config(), data=replace(valid_config().data, train_per_class=0)), "train_per_class"),
        (
            replace(valid_config(), data=replace(valid_config().data, approximate_duplicate_threshold=float("nan"))),
            "approximate_duplicate_threshold",
        ),
        (replace(valid_config(), direction=replace(valid_config().direction, candidate_phases=())), "candidate_phases"),
        (
            replace(valid_config(), direction=replace(valid_config().direction, candidate_layers=(-1,))),
            "candidate_layers",
        ),
        (replace(valid_config(), direction=replace(valid_config().direction, stage_b_top_k=0)), "stage_b_top_k"),
        (replace(valid_config(), judge=replace(valid_config().judge, safety_margin_tokens=-1)), "safety_margin_tokens"),
        (
            replace(
                valid_config(),
                judge=replace(
                    valid_config().judge,
                    score_allowed_labels=False,
                    calibrated_margin_threshold=0.1,
                ),
            ),
            "requires score_allowed_labels",
        ),
    ],
)
def test_invalid_counts_thresholds_and_candidate_space_fail_fast(
    config: ProjectConfig,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        config.validate()
