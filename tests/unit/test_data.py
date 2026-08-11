import pytest

from self_judged_refusal_direction.config import DataConfig
from self_judged_refusal_direction.data import prepare_prompt_records, split_prompt_groups
from self_judged_refusal_direction.errors import ConfigurationError


def test_prepare_requires_splitting_before_labeling() -> None:
    config = DataConfig(split_before_labeling=False)

    with pytest.raises(ConfigurationError, match="split_before_labeling"):
        prepare_prompt_records(config, seed=7)


def test_template_family_groups_do_not_cross_splits() -> None:
    prompts = (
        "family-a-one",
        "family-a-two",
        "family-b-one",
        "family-b-two",
        "family-c",
        "family-d",
        "family-e",
        "family-f",
    )
    group_ids = {
        "family-a-one": "a",
        "family-a-two": "a",
        "family-b-one": "b",
        "family-b-two": "b",
        "family-c": "c",
        "family-d": "d",
        "family-e": "e",
        "family-f": "f",
    }

    records = split_prompt_groups(
        prompts,
        group_ids,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=11,
    )

    splits_by_group = {
        group_id: {record.split for record in records if record.group_id == group_id}
        for group_id in set(group_ids.values())
    }
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert {record.original_prompt for record in records} == set(prompts)
