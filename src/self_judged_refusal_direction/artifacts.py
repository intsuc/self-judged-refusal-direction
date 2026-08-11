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
from self_judged_refusal_direction.hashing import canonical_json_bytes, file_sha256, object_sha256


@dataclass(frozen=True)
class ArtifactProfile:
    model_id: str
    model_revision: str
    config_hash: str
    target_profile_hash: str | None = None
    judge_profile_hash: str | None = None
    chat_template_hash: str | None = None


@dataclass(frozen=True)
class ArtifactMetadata:
    schema_version: int
    artifact_type: str
    private: bool
    record_count: int
    content_sha256: str
    profile: ArtifactProfile


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
    def raw_test(self) -> Path:
        return self.data / "raw_test.jsonl"

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
    def activation_means(self) -> Path:
        return self.activations / "means.pt"

    @property
    def activation_variances(self) -> Path:
        return self.activations / "variances.pt"

    @property
    def activation_metadata(self) -> Path:
        return self.activations / "position_metadata.json"

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
    def stage_a_ranking(self) -> Path:
        return self.directions / "stage_a_ranking.json"

    @property
    def stage_b_results(self) -> Path:
        return self.directions / "stage_b_results.jsonl"

    @property
    def stage_c_results(self) -> Path:
        return self.directions / "stage_c_results.jsonl"

    @property
    def stage_b_trajectories(self) -> Path:
        return self.directions / "stage_b_trajectories.private.jsonl"

    @property
    def stage_b_judgments(self) -> Path:
        return self.directions / "stage_b_judgments.jsonl"

    @property
    def stage_c_trajectories(self) -> Path:
        return self.directions / "stage_c_trajectories.private.jsonl"

    @property
    def stage_c_judgments(self) -> Path:
        return self.directions / "stage_c_judgments.jsonl"

    @property
    def final_selection(self) -> Path:
        return self.directions / "final_selection.json"

    @property
    def selected_direction(self) -> Path:
        return self.directions / "selected_direction.safetensors"

    @property
    def validation_report(self) -> Path:
        return self.evaluation / "validation_report.json"

    @property
    def test_report(self) -> Path:
        return self.evaluation / "test_report.json"

    @property
    def quality_metrics(self) -> Path:
        return self.evaluation / "quality_metrics.json"


class ArtifactStore:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.paths = ArtifactPaths(config.run.output_dir)
        self.paths.create()

    def profile(
        self,
        *,
        target: bool = False,
        judge: bool = False,
        chat_template_hash: str | None = None,
    ) -> ArtifactProfile:
        return ArtifactProfile(
            model_id=self.config.model.id,
            model_revision=self.config.model.revision,
            config_hash=self.config.config_hash,
            target_profile_hash=self.config.target_profile_hash if target else None,
            judge_profile_hash=self.config.judge_profile_hash if judge else None,
            chat_template_hash=chat_template_hash,
        )

    def initialize_run(self) -> None:
        self.write_yaml(self.paths.root / "resolved_config.yaml", resolved_config_mapping(self.config), private=False)
        environment_path = self.paths.root / "environment.json"
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
        try:
            existing = json.loads(environment_path.read_text(encoding="utf-8"))
        except OSError, UnicodeError, json.JSONDecodeError:
            existing = None
        identity = tuple(environment)
        if isinstance(existing, dict) and all(existing.get(key) == environment[key] for key in identity):
            environment = {**existing, **environment}
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
                value = asdict(record) if is_dataclass(record) else record
                stream.write(canonical_json_bytes(value) + b"\n")
                count += 1
        os.chmod(temporary, 0o600 if private else 0o644)
        temporary.replace(target)
        metadata = ArtifactMetadata(
            schema_version=1,
            artifact_type=artifact_type,
            private=private,
            record_count=count,
            content_sha256=file_sha256(target),
            profile=profile,
        )
        self.write_json(self.metadata_path(target), asdict(metadata), private=private)
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
        if (
            not isinstance(metadata.schema_version, int)
            or isinstance(metadata.schema_version, bool)
            or metadata.schema_version != 1
        ):
            raise ArtifactError(f"unsupported artifact schema version for {target}")
        if not isinstance(metadata.private, bool):
            raise ArtifactError(f"artifact privacy flag is invalid for {target}")
        if (
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
        expected = asdict(expected_profile)
        actual = asdict(metadata.profile)
        mismatches = [key for key, value in expected.items() if value is not None and actual.get(key) != value]
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


def artifact_key_hash(profile: ArtifactProfile, artifact_type: str) -> str:
    return object_sha256({"profile": profile, "artifact_type": artifact_type})
