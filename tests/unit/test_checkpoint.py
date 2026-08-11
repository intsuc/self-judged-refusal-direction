import sqlite3
from dataclasses import asdict
from typing import Any, cast

import pytest

from self_judged_refusal_direction.checkpoint import PrivateCheckpoint
from self_judged_refusal_direction.config import ModelConfig, ProjectConfig, TargetGenerationConfig
from self_judged_refusal_direction.errors import ArtifactError
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.pipeline import _checkpointed_target_trajectories
from self_judged_refusal_direction.schema import PromptRecord, TargetTrajectory


class TargetRuntime:
    def __init__(self, config: ProjectConfig, fail_prompt_id: str | None = None):
        self.config = config
        self.fail_prompt_id = fail_prompt_id
        self.calls: list[str] = []
        self.batches: list[tuple[str, ...]] = []

    def generate_targets(self, prompts: list[PromptRecord]) -> list[TargetTrajectory]:
        self.batches.append(tuple(prompt.prompt_id for prompt in prompts))
        return [self.generate_target(prompt) for prompt in prompts]

    def generate_target(self, prompt: PromptRecord) -> TargetTrajectory:
        self.calls.append(prompt.prompt_id)
        if prompt.prompt_id == self.fail_prompt_id:
            raise RuntimeError("generation stopped")
        values: dict[str, Any] = {
            "prompt_id": prompt.prompt_id,
            "original_prompt": prompt.original_prompt,
            "raw_generated_token_ids": (1, 2),
            "raw_decoded_output": "answer",
            "thinking_text": "",
            "final_answer": "answer",
            "thinking_token_start": 0,
            "thinking_token_end": 0,
            "final_token_start": 0,
            "final_token_end": 1,
            "generation_truncated": False,
            "parser_status": "OK",
            "model_revision": self.config.model.revision,
            "generation_config_hash": "generation-profile",
            "split": prompt.split,
            "seed": self.config.run.seed,
            "error_code": None,
            "error_detail": None,
        }
        return TargetTrajectory(trajectory_hash=object_sha256(values), **values)


def test_checkpoint_resumes_committed_rows_after_interruption(tmp_path) -> None:
    directory = tmp_path / "private-checkpoint"
    prompt_keys = ("prompt-a", "prompt-b", "prompt-c")
    with PrivateCheckpoint(directory, identity="generation", prompt_keys=prompt_keys) as checkpoint:
        checkpoint.write(0, "prompt-a", {"prompt_id": "prompt-a", "tokens": [1, 2]})
        with pytest.raises(ArtifactError, match="incomplete"):
            checkpoint.require_complete()

    with PrivateCheckpoint(directory, identity="generation", prompt_keys=prompt_keys) as checkpoint:
        assert [(entry.ordinal, entry.prompt_key) for entry in checkpoint.load()] == [(0, "prompt-a")]
        checkpoint.write(1, "prompt-b", {"prompt_id": "prompt-b", "tokens": [3]})
        checkpoint.write(2, "prompt-c", {"prompt_id": "prompt-c", "tokens": []})
        assert [entry.prompt_key for entry in checkpoint.require_complete()] == list(prompt_keys)

    assert directory.stat().st_mode & 0o777 == 0o700
    assert (directory / "checkpoint.sqlite3").stat().st_mode & 0o777 == 0o600


def test_checkpoint_rejects_identity_mismatch_without_resetting_rows(tmp_path) -> None:
    directory = tmp_path / "private-checkpoint"
    with PrivateCheckpoint(directory, identity="generation-a", prompt_keys=("prompt",)) as checkpoint:
        checkpoint.write(0, "prompt", {"value": "retained"})

    with pytest.raises(ArtifactError, match="identity does not match"):
        PrivateCheckpoint(directory, identity="generation-b", prompt_keys=("prompt",))

    with PrivateCheckpoint(directory, identity="generation-a", prompt_keys=("prompt",)) as checkpoint:
        assert checkpoint.require_complete()[0].payload == {"value": "retained"}


def test_checkpoint_rejects_corrupted_payload(tmp_path) -> None:
    directory = tmp_path / "private-checkpoint"
    with PrivateCheckpoint(directory, identity="generation", prompt_keys=("prompt",)) as checkpoint:
        checkpoint.write(0, "prompt", {"value": "original"})

    path = directory / "checkpoint.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE checkpoint_rows SET payload = ? WHERE ordinal = 0", (b'{"value":"changed"}',))

    with (
        PrivateCheckpoint(directory, identity="generation", prompt_keys=("prompt",)) as checkpoint,
        pytest.raises(ArtifactError, match="payload hash does not match"),
    ):
        checkpoint.load()


def test_checkpoint_rejects_missing_identity_without_rebinding_rows(tmp_path) -> None:
    directory = tmp_path / "private-checkpoint"
    with PrivateCheckpoint(directory, identity="original", prompt_keys=("prompt",)) as checkpoint:
        checkpoint.write(0, "prompt", {"value": "original"})

    path = directory / "checkpoint.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM checkpoint_identity")

    with pytest.raises(ArtifactError, match="invalid identity metadata"):
        PrivateCheckpoint(directory, identity="replacement", prompt_keys=("prompt",))


def test_target_generation_resumes_after_fatal_interruption(tmp_path, monkeypatch) -> None:
    config = ProjectConfig(model=ModelConfig(id="model", revision="a" * 40))
    prompts = [
        PromptRecord(
            prompt_id=f"prompt-{index}", original_prompt=f"request {index}", group_id=str(index), split="train"
        )
        for index in range(3)
    ]
    monkeypatch.setattr(
        "self_judged_refusal_direction.pipeline._target_generation_profile_hash",
        lambda config, runtime: "generation-profile",
    )
    directory = tmp_path / "generation"
    interrupted = TargetRuntime(config, fail_prompt_id="prompt-1")

    with pytest.raises(RuntimeError, match="generation stopped"):
        _checkpointed_target_trajectories(
            cast(Any, interrupted),
            config,
            prompts,
            directory=directory,
            identity="generation",
            desc="test",
        )

    resumed = TargetRuntime(config)
    trajectories = _checkpointed_target_trajectories(
        cast(Any, resumed),
        config,
        prompts,
        directory=directory,
        identity="generation",
        desc="test",
    )

    assert interrupted.calls == ["prompt-0", "prompt-1"]
    assert resumed.calls == ["prompt-1", "prompt-2"]
    assert [asdict(item)["prompt_id"] for item in trajectories] == [item.prompt_id for item in prompts]


def test_target_generation_resumes_with_the_original_batch_plan(tmp_path, monkeypatch) -> None:
    config = ProjectConfig(
        model=ModelConfig(id="model", revision="a" * 40),
        target_generation=TargetGenerationConfig(batch_size=2),
    )
    prompts = [
        PromptRecord(
            prompt_id=f"prompt-{index}", original_prompt=f"request {index}", group_id=str(index), split="train"
        )
        for index in range(3)
    ]
    monkeypatch.setattr(
        "self_judged_refusal_direction.pipeline._target_generation_profile_hash",
        lambda config, runtime: "generation-profile",
    )
    directory = tmp_path / "generation"
    existing = TargetRuntime(config).generate_target(prompts[0])
    with PrivateCheckpoint(
        directory,
        identity="generation",
        prompt_keys=[prompt.prompt_id for prompt in prompts],
    ) as checkpoint:
        checkpoint.write(0, prompts[0].prompt_id, existing.as_dict())

    resumed = TargetRuntime(config)
    trajectories = _checkpointed_target_trajectories(
        cast(Any, resumed),
        config,
        prompts,
        directory=directory,
        identity="generation",
        desc="test",
    )

    assert resumed.batches == [("prompt-0", "prompt-1"), ("prompt-2",)]
    assert [item.prompt_id for item in trajectories] == [item.prompt_id for item in prompts]
