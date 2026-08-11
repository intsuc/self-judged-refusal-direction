import json
from dataclasses import replace

import pytest

from self_judged_refusal_direction.artifacts import ArtifactProfile, ArtifactStore
from self_judged_refusal_direction.config import ModelConfig, ProjectConfig, RunConfig
from self_judged_refusal_direction.errors import ArtifactError
from self_judged_refusal_direction.pipeline import _judge_chat_hash, _validate_activation_chat_profile


def test_artifact_reuse_requires_matching_profile_and_content(tmp_path) -> None:
    config = ProjectConfig(run=RunConfig(output_dir=str(tmp_path)), model=ModelConfig(revision="a" * 40))
    store = ArtifactStore(config)
    path = tmp_path / "rows.jsonl"
    profile = store.profile(target=True, chat_template_hash="chat-a")
    store.write_jsonl(path, [{"value": 1}], artifact_type="rows", profile=profile, private=True)
    assert list(store.read_jsonl(path, artifact_type="rows", expected_profile=profile)) == [{"value": 1}]
    mismatch = replace(profile, chat_template_hash="chat-b")
    with pytest.raises(ArtifactError, match="profile mismatch"):
        list(store.read_jsonl(path, artifact_type="rows", expected_profile=mismatch))
    path.write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="content hash"):
        list(store.read_jsonl(path, artifact_type="rows", expected_profile=profile))


def test_private_artifact_schema_and_permissions_fail_closed(tmp_path) -> None:
    config = ProjectConfig(run=RunConfig(output_dir=str(tmp_path)), model=ModelConfig(revision="a" * 40))
    store = ArtifactStore(config)
    path = tmp_path / "trajectories.private.jsonl"
    profile = store.profile(target=True)
    store.write_jsonl(path, [{"thinking": "private"}], artifact_type="trajectories", profile=profile, private=True)
    metadata_path = store.metadata_path(path)

    assert path.stat().st_mode & 0o077 == 0
    assert metadata_path.stat().st_mode & 0o077 == 0

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["private"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ArtifactError, match="not marked private"):
        list(store.read_jsonl(path, artifact_type="trajectories", expected_profile=profile))

    metadata["private"] = True
    metadata["schema_version"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ArtifactError, match="schema version"):
        list(store.read_jsonl(path, artifact_type="trajectories", expected_profile=profile))


def test_environment_extensions_survive_only_for_the_same_run_identity(tmp_path) -> None:
    config = ProjectConfig(run=RunConfig(output_dir=str(tmp_path)), model=ModelConfig(revision="a" * 40))
    store = ArtifactStore(config)
    store.initialize_run()
    path = store.paths.root / "environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["processor_sha256"] = "processor"
    store.write_json(path, environment, private=False)

    store.initialize_run()
    assert json.loads(path.read_text(encoding="utf-8"))["processor_sha256"] == "processor"

    changed = ProjectConfig(run=config.run, model=ModelConfig(revision="b" * 40))
    ArtifactStore(changed).initialize_run()
    assert "processor_sha256" not in json.loads(path.read_text(encoding="utf-8"))


def test_activation_artifacts_require_the_current_chat_template() -> None:
    profile = ArtifactProfile(
        model_id="model",
        model_revision="a" * 40,
        config_hash="config",
        chat_template_hash=_judge_chat_hash("chat-a"),
    )

    _validate_activation_chat_profile(profile, "chat-a")
    with pytest.raises(ArtifactError, match="different chat template"):
        _validate_activation_chat_profile(profile, "chat-b")
