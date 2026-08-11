from __future__ import annotations

import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from self_judged_refusal_direction.artifacts import ArtifactProfile, ArtifactStore
from self_judged_refusal_direction.config import DataConfig, ProjectConfig
from self_judged_refusal_direction.errors import ArtifactError, ConfigurationError, InvariantError
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.schema import PromptRecord

SplitName = Literal["train", "validation", "test"]

_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_COMPACT_SPACE = re.compile(r"\s+")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_UUID = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.IGNORECASE)
_HEX = re.compile(r"\b(?:0x)?[0-9a-f]{12,}\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,:/-]\d+)*|\.\d+)(?!\w)")
_QUOTED = re.compile(r"(?s)([\"'`])(?:\\.|(?!\1).){2,}\1")
_TOKEN = re.compile(r"<[^>]+>|[\w]+|[^\w\s]", re.UNICODE)


def normalize_prompt(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("prompt must be a string")
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    normalized = "\n".join(_HORIZONTAL_SPACE.sub(" ", line).rstrip() for line in normalized.split("\n"))
    return _BLANK_LINES.sub("\n\n", normalized).strip()


def prompt_deduplication_key(prompt: str) -> str:
    return object_sha256({"prompt": _COMPACT_SPACE.sub(" ", normalize_prompt(prompt))})


def deduplicate_prompts(prompts: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        normalized = normalize_prompt(prompt)
        if not normalized:
            continue
        key = prompt_deduplication_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def template_family_signature(prompt: str) -> str:
    value = normalize_prompt(prompt).casefold()
    value = _URL.sub("<url>", value)
    value = _EMAIL.sub("<email>", value)
    value = _UUID.sub("<id>", value)
    value = _HEX.sub("<id>", value)
    value = _QUOTED.sub("<quoted>", value)
    value = _NUMBER.sub("<number>", value)
    return " ".join(_TOKEN.findall(value))


def _signature_features(signature: str) -> frozenset[str]:
    tokens = signature.split()
    if len(tokens) < 3:
        return frozenset(tokens)
    return frozenset("\u241f".join(tokens[index : index + 3]) for index in range(len(tokens) - 2))


def template_family_similarity(left: str, right: str) -> float:
    left_signature = template_family_signature(left)
    right_signature = template_family_signature(right)
    if left_signature == right_signature:
        return 1.0
    left_features = _signature_features(left_signature)
    right_features = _signature_features(right_signature)
    if not left_features and not right_features:
        return 1.0
    union = left_features | right_features
    return len(left_features & right_features) / len(union) if union else 0.0


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def assign_template_family_groups(prompts: Sequence[str], threshold: float = 0.9) -> dict[str, str]:
    if not 0 <= threshold <= 1:
        raise ConfigurationError("template similarity threshold must be between zero and one")
    normalized = deduplicate_prompts(prompts)
    signatures = [template_family_signature(prompt) for prompt in normalized]
    features = [_signature_features(signature) for signature in signatures]
    disjoint = _DisjointSet(len(normalized))
    exact: dict[str, int] = {}
    for index, signature in enumerate(signatures):
        previous = exact.setdefault(signature, index)
        disjoint.union(index, previous)
    pair_count = len(normalized) * (len(normalized) - 1) // 2
    pairs = combinations(range(len(normalized)), 2)
    for left, right in tqdm(
        pairs,
        total=pair_count,
        desc="Grouping prompts",
        unit="pair",
        dynamic_ncols=True,
        disable=None,
    ):
        if disjoint.find(left) == disjoint.find(right):
            continue
        smaller = min(len(features[left]), len(features[right]))
        larger = max(len(features[left]), len(features[right]))
        if larger and smaller / larger < threshold:
            continue
        union = features[left] | features[right]
        similarity = len(features[left] & features[right]) / len(union) if union else 1.0
        if similarity >= threshold:
            disjoint.union(left, right)
    members: dict[int, list[str]] = defaultdict(list)
    for index, prompt in enumerate(normalized):
        members[disjoint.find(index)].append(prompt_deduplication_key(prompt))
    group_ids = {root: object_sha256({"template_family": sorted(member_keys)}) for root, member_keys in members.items()}
    return {prompt: group_ids[disjoint.find(index)] for index, prompt in enumerate(normalized)}


def _read_text_file(path: Path) -> Iterator[str]:
    if path.suffix.casefold() != ".txt":
        raise ArtifactError(f"text files must use the .txt extension: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield line
    except (OSError, UnicodeError) as error:
        raise ArtifactError(f"failed to read text file: {path}") from error


def ingest_texts(
    paths: Iterable[str | Path],
    *,
    token_counter: Callable[[str], int] | None = None,
    max_text_tokens: int | None = None,
) -> list[str]:
    texts: list[str] = []
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise ArtifactError(f"text file does not exist: {path}")
        texts.extend(_read_text_file(path))
    normalized = deduplicate_prompts(texts)
    normalized = [text for text in normalized if text]
    if token_counter is not None and max_text_tokens is not None:
        normalized = [
            text
            for text in tqdm(
                normalized,
                desc="Filtering input text",
                unit="text",
                dynamic_ncols=True,
                disable=None,
            )
            if token_counter(text) <= max_text_tokens
        ]
    return normalized


def _validate_fractions(fractions: Mapping[SplitName, float]) -> None:
    if set(fractions) != {"train", "validation", "test"}:
        raise ConfigurationError("split fractions must define train, validation, and test")
    if any(value <= 0 for value in fractions.values()):
        raise ConfigurationError("split fractions must be positive")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ConfigurationError("split fractions must sum to one")


def split_prompt_groups(
    prompts: Sequence[str],
    group_ids: Mapping[str, str],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[PromptRecord]:
    normalized = deduplicate_prompts(prompts)
    fractions: dict[SplitName, float] = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    _validate_fractions(fractions)
    missing = [prompt for prompt in normalized if prompt not in group_ids]
    if missing:
        raise InvariantError("every prompt must have a template-family group")
    grouped: dict[str, list[str]] = defaultdict(list)
    for prompt in normalized:
        grouped[group_ids[prompt]].append(prompt)
    groups = sorted(grouped.items())
    random.Random(seed).shuffle(groups)
    target = {name: len(normalized) * fraction for name, fraction in fractions.items()}
    assigned_count = {name: 0 for name in fractions}
    assignment: dict[str, SplitName] = {}
    names: tuple[SplitName, ...] = ("train", "validation", "test")
    for group_id, members in groups:
        split = max(
            names,
            key=lambda name: (
                (target[name] - assigned_count[name]) / target[name],
                -assigned_count[name],
                -names.index(name),
            ),
        )
        assignment[group_id] = split
        assigned_count[split] += len(members)
    records = [
        PromptRecord(
            prompt_id=object_sha256({"prompt": prompt}),
            original_prompt=prompt,
            group_id=group_ids[prompt],
            split=assignment[group_ids[prompt]],
        )
        for prompt in normalized
    ]
    for group_id in grouped:
        splits = {record.split for record in records if record.group_id == group_id}
        if len(splits) != 1:
            raise InvariantError("template-family group crossed split boundaries")
    return records


def prepare_prompt_records(
    data_config: DataConfig,
    *,
    seed: int,
    token_counter: Callable[[str], int] | None = None,
) -> list[PromptRecord]:
    prompts = ingest_texts(
        data_config.prompt_files,
        token_counter=token_counter,
        max_text_tokens=data_config.max_text_tokens,
    )
    if not prompts:
        raise InvariantError("no prompts remain after ingestion and normalization")
    groups = assign_template_family_groups(prompts, data_config.template_similarity_threshold)
    return split_prompt_groups(
        prompts,
        groups,
        train_fraction=data_config.train_fraction,
        validation_fraction=data_config.validation_fraction,
        test_fraction=data_config.test_fraction,
        seed=seed,
    )


def ingest_and_split_prompts(
    config: ProjectConfig,
    *,
    token_counter: Callable[[str], int] | None = None,
) -> list[PromptRecord]:
    return prepare_prompt_records(config.data, seed=config.run.seed, token_counter=token_counter)


def records_by_split(records: Iterable[PromptRecord]) -> dict[SplitName, list[PromptRecord]]:
    result: dict[SplitName, list[PromptRecord]] = {"train": [], "validation": [], "test": []}
    for record in records:
        result[record.split].append(record)
    return result


def write_prompt_split_artifacts(
    store: ArtifactStore,
    records: Sequence[PromptRecord],
    *,
    profile: ArtifactProfile,
) -> None:
    store.write_jsonl(
        store.paths.splits,
        records,
        artifact_type="prompt_splits",
        profile=profile,
        private=False,
    )
    store.write_jsonl(
        store.paths.test_prompts,
        (record for record in records if record.split == "test"),
        artifact_type="test_prompts",
        profile=profile,
        private=False,
    )
