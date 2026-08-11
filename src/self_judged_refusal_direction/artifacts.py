from __future__ import annotations

import json
import os
import platform
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import accelerate
import numpy
import safetensors
import torch
import transformers
import yaml

from self_judged_refusal_direction.config import ProjectConfig, resolved_config_mapping
from self_judged_refusal_direction.errors import ArtifactError
from self_judged_refusal_direction.hashing import canonical_json_bytes, file_sha256


@dataclass(frozen=True)
class ArtifactProfile:
    model_id: str
    model_revision: str
    config_hash: str
    target_generation_config_hash: str | None = None
    chat_template_hash: str | None = None
    judge_template_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_type: str
    private: bool
    content_sha256: str
    profile: ArtifactProfile
    record_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "private": self.private,
            "content_sha256": self.content_sha256,
            "profile": self.profile.as_dict(),
        }
        if self.record_count is not None:
            value["record_count"] = self.record_count
        return value


class ArtifactPaths:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.data = self.root / "data"
        self.activations = self.root / "activations"
        self.directions = self.root / "directions"
        self.evaluation = self.root / "evaluation"
        self.exported_model = self.root / "exported_model"

    def create(self) -> None:
        for path in (self.root, self.data, self.activations, self.directions, self.evaluation):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def splits(self) -> Path:
        return self.data / "splits.jsonl"

    @property
    def test_prompts(self) -> Path:
        return self.data / "test_prompts.jsonl"

    @property
    def baseline_trajectories(self) -> Path:
        return self.data / "baseline_trajectories.private.jsonl"

    @property
    def baseline_judgments(self) -> Path:
        return self.data / "baseline_judgments.jsonl"

    @property
    def labeled_train(self) -> Path:
        return self.data / "labeled_train.jsonl"

    @property
    def labeled_validation(self) -> Path:
        return self.data / "labeled_validation.jsonl"

    @property
    def activation_statistics(self) -> Path:
        return self.activations / "statistics.safetensors"

    @property
    def candidates(self) -> Path:
        return self.directions / "candidates.safetensors"

    @property
    def candidate_metadata(self) -> Path:
        return self.directions / "candidate_metadata.jsonl"

    @property
    def activation_screening_ranking(self) -> Path:
        return self.directions / "activation_screening_ranking.json"

    @property
    def pilot_evaluation_results(self) -> Path:
        return self.directions / "pilot_evaluation_results.jsonl"

    @property
    def full_validation_results(self) -> Path:
        return self.directions / "full_validation_results.jsonl"

    @property
    def pilot_evaluation_trajectories(self) -> Path:
        return self.directions / "pilot_evaluation_trajectories.private.jsonl"

    @property
    def pilot_evaluation_judgments(self) -> Path:
        return self.directions / "pilot_evaluation_judgments.jsonl"

    @property
    def full_validation_trajectories(self) -> Path:
        return self.directions / "full_validation_trajectories.private.jsonl"

    @property
    def full_validation_judgments(self) -> Path:
        return self.directions / "full_validation_judgments.jsonl"

    @property
    def final_selection(self) -> Path:
        return self.directions / "final_selection.json"

    @property
    def selected_direction(self) -> Path:
        return self.directions / "selected_direction.safetensors"

    @property
    def full_validation_report(self) -> Path:
        return self.evaluation / "full_validation_report.json"

    @property
    def test_report(self) -> Path:
        return self.evaluation / "test_report.json"

    @property
    def quality_metrics(self) -> Path:
        return self.evaluation / "quality_metrics.json"


class ArtifactStore:
    def __init__(self, config: ProjectConfig):
        if config.run.output_dir is None:
            raise ArtifactError("run output directory is required")
        self.config = config
        self.paths = ArtifactPaths(config.run.output_dir)
        self.paths.create()

    def profile(
        self,
        *,
        target: bool = False,
        chat_template_hash: str | None = None,
        judge_template_hash: str | None = None,
    ) -> ArtifactProfile:
        if self.config.model.id is None:
            raise ArtifactError("model ID is required")
        return ArtifactProfile(
            model_id=self.config.model.id,
            model_revision=self.config.model.revision,
            config_hash=self.config.config_hash,
            target_generation_config_hash=self.config.target_generation_config_hash if target else None,
            chat_template_hash=chat_template_hash,
            judge_template_hash=judge_template_hash,
        )

    def initialize_run(self) -> None:
        environment_path = self.paths.root / "environment.json"
        run_outputs = (
            self.paths.root / "resolved_config.yaml",
            self.paths.root / "model_compatibility.json",
            self.paths.data,
            self.paths.activations,
            self.paths.directions,
            self.paths.evaluation,
            self.paths.exported_model,
        )
        has_run_outputs = any(
            path.is_file() or (path.is_dir() and any(child.is_file() for child in path.rglob("*")))
            for path in run_outputs
        )
        if not environment_path.exists() and has_run_outputs:
            raise ArtifactError("existing run environment is missing")
        environment = {
            "model_id": self.config.model.id,
            "model_revision": self.config.model.revision,
            "config_hash": self.config.config_hash,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "safetensors": safetensors.__version__,
            "numpy": numpy.__version__,
            "dtype": self.config.model.dtype,
            "attention_implementation": self.config.model.attention_implementation,
            "device_map": self.config.model.device_map,
        }
        if environment_path.exists():
            try:
                existing = json.loads(environment_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ArtifactError("existing run environment is invalid") from error
            if not isinstance(existing, dict) or any(existing.get(key) != value for key, value in environment.items()):
                raise ArtifactError("existing run environment does not match")
            environment = {**existing, **environment}
        self.write_yaml(self.paths.root / "resolved_config.yaml", resolved_config_mapping(self.config), private=False)
        self.write_json(environment_path, environment, private=False)

    def write_jsonl(
        self,
        path: str | Path,
        records: Iterable[Any],
        *,
        artifact_type: str,
        profile: ArtifactProfile,
        private: bool,
    ) -> ArtifactMetadata:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for record in records:
                as_dict = getattr(record, "as_dict", None)
                value = as_dict() if callable(as_dict) else asdict(record) if is_dataclass(record) else record
                stream.write(canonical_json_bytes(value) + b"\n")
                count += 1
        os.chmod(temporary, 0o600 if private else 0o644)
        temporary.replace(target)
        metadata = ArtifactMetadata(
            artifact_type=artifact_type,
            private=private,
            record_count=count,
            content_sha256=file_sha256(target),
            profile=profile,
        )
        self.write_json(self.metadata_path(target), metadata.as_dict(), private=private)
        return metadata

    def read_jsonl(
        self,
        path: str | Path,
        *,
        artifact_type: str,
        expected_profile: ArtifactProfile,
    ) -> Iterator[dict[str, Any]]:
        target = Path(path)
        metadata = self.validate(target, artifact_type=artifact_type, expected_profile=expected_profile)
        if metadata.record_count is None:
            raise ArtifactError(f"JSONL artifact record count is missing for {target}")
        seen = 0
        with target.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ArtifactError(f"invalid JSON at {target}:{line_number}") from error
                if not isinstance(value, dict):
                    raise ArtifactError(f"artifact row must be an object at {target}:{line_number}")
                seen += 1
                yield value
        if seen != metadata.record_count:
            raise ArtifactError(f"record count mismatch for {target}")

    def validate(
        self,
        path: str | Path,
        *,
        artifact_type: str,
        expected_profile: ArtifactProfile,
    ) -> ArtifactMetadata:
        target = Path(path)
        if not target.is_file():
            raise ArtifactError(f"required artifact does not exist: {target}")
        metadata_path = self.metadata_path(target)
        if not metadata_path.is_file():
            raise ArtifactError(f"artifact metadata does not exist: {metadata_path}")
        try:
            with metadata_path.open(encoding="utf-8") as stream:
                raw = json.load(stream)
            if not isinstance(raw, dict) or not isinstance(raw.get("profile"), dict):
                raise TypeError
            raw["profile"] = ArtifactProfile(**raw["profile"])
            metadata = ArtifactMetadata(**raw)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"invalid artifact metadata: {metadata_path}") from error
        if not isinstance(metadata.private, bool):
            raise ArtifactError(f"artifact privacy flag is invalid for {target}")
        if metadata.record_count is not None and (
            not isinstance(metadata.record_count, int)
            or isinstance(metadata.record_count, bool)
            or metadata.record_count < 0
        ):
            raise ArtifactError(f"artifact record count is invalid for {target}")
        if ".private." in target.name and not metadata.private:
            raise ArtifactError(f"private artifact is not marked private: {target}")
        if metadata.private:
            if target.stat().st_mode & 0o077:
                raise ArtifactError(f"private artifact permissions are too broad: {target}")
            if metadata_path.stat().st_mode & 0o077:
                raise ArtifactError(f"private artifact metadata permissions are too broad: {metadata_path}")
        if metadata.artifact_type != artifact_type:
            raise ArtifactError(f"artifact type mismatch for {target}")
        if metadata.content_sha256 != file_sha256(target):
            raise ArtifactError(f"artifact content hash mismatch for {target}")
        expected = expected_profile.as_dict()
        actual = metadata.profile.as_dict()
        mismatches = [key for key, value in expected.items() if actual.get(key) != value]
        if mismatches:
            raise ArtifactError(f"artifact profile mismatch for {target}: {mismatches}")
        return metadata

    @staticmethod
    def metadata_path(path: str | Path) -> Path:
        target = Path(path)
        return target.with_name(f"{target.name}.meta.json")

    @staticmethod
    def write_json(path: str | Path, value: Any, *, private: bool) -> None:
        ArtifactStore._atomic_write(Path(path), canonical_json_bytes(value) + b"\n", private)

    @staticmethod
    def write_yaml(path: str | Path, value: Mapping[str, Any], *, private: bool) -> None:
        content = yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=True).encode()
        ArtifactStore._atomic_write(Path(path), content, private)

    @staticmethod
    def _atomic_write(path: Path, content: bytes, private: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.chmod(temporary, 0o600 if private else 0o644)
        temporary.replace(path)
