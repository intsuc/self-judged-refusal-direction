from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from self_judged_refusal_direction.errors import ConfigurationError, InvariantError
from self_judged_refusal_direction.hashing import object_sha256


@dataclass
class _TrieNode:
    children: dict[int, _TrieNode]
    label: str | None = None

    def __init__(self) -> None:
        self.children = {}
        self.label = None


class EnumTrieConstrainedDecoder:
    def __init__(
        self,
        label_token_ids: dict[str, tuple[int, ...]],
        eos_token_id: int,
        generation_start_lengths: tuple[int, ...],
    ):
        if not label_token_ids:
            raise ConfigurationError("at least one judge label is required")
        if any(not token_ids for token_ids in label_token_ids.values()):
            raise ConfigurationError("judge label token sequences must be non-empty")
        self.label_token_ids = dict(label_token_ids)
        self.eos_token_id = int(eos_token_id)
        self.generation_start_lengths = generation_start_lengths
        self.root = _TrieNode()
        for label, token_ids in self.label_token_ids.items():
            node = self.root
            for token_id in token_ids:
                if node.label is not None:
                    raise ConfigurationError("a judge label token sequence cannot prefix another label")
                node = node.children.setdefault(int(token_id), _TrieNode())
            if node.children:
                raise ConfigurationError("a judge label token sequence cannot prefix another label")
            if node.label is not None:
                raise ConfigurationError("judge labels must have distinct token sequences")
            node.label = label

    @property
    def max_new_tokens(self) -> int:
        return max(len(value) for value in self.label_token_ids.values()) + 1

    @property
    def signature_hash(self) -> str:
        return object_sha256(
            {
                "labels": self.label_token_ids,
                "eos_token_id": self.eos_token_id,
                "max_new_tokens": self.max_new_tokens,
            }
        )

    def allowed_tokens(self, batch_id: int, input_ids: torch.Tensor | list[int]) -> list[int]:
        values = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
        try:
            start = self.generation_start_lengths[batch_id]
        except IndexError as error:
            raise InvariantError(f"missing generation start length for batch {batch_id}") from error
        if len(values) < start:
            raise InvariantError("generated sequence is shorter than its prompt")
        suffix = values[start:]
        node = self.root
        for token_id in suffix:
            if node.label is not None:
                if token_id == self.eos_token_id and token_id == suffix[-1]:
                    return []
                raise InvariantError("tokens appeared after a terminal judge label")
            try:
                node = node.children[int(token_id)]
            except KeyError as error:
                raise InvariantError("generated judge suffix left the finite label language") from error
        if node.label is not None:
            return [self.eos_token_id]
        allowed = sorted(node.children)
        if not allowed:
            raise InvariantError("judge trie reached a non-terminal dead end")
        return allowed

    def prefix_allowed_tokens_fn(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        return self.allowed_tokens(batch_id, input_ids)

    def parse_suffix(self, suffix: torch.Tensor | list[int]) -> tuple[str, str]:
        values = suffix.tolist() if isinstance(suffix, torch.Tensor) else list(suffix)
        if not values or values[-1] != self.eos_token_id:
            raise InvariantError("judge output did not terminate with EOS")
        label_ids = tuple(values[:-1])
        matches = [label for label, token_ids in self.label_token_ids.items() if token_ids == label_ids]
        if len(matches) != 1:
            raise InvariantError("judge output does not exactly match one constrained label")
        return matches[0], matches[0]

    @classmethod
    def compile(
        cls,
        tokenizer: Any,
        labels: tuple[str, ...],
        prompt_token_ids: tuple[tuple[int, ...], ...],
        padded_generation_start_lengths: tuple[int, ...] | None = None,
    ) -> EnumTrieConstrainedDecoder:
        if not prompt_token_ids:
            raise ConfigurationError("judge prompt token IDs are required")
        per_prompt = [_continuation_token_ids(tokenizer, prompt, labels) for prompt in prompt_token_ids]
        first = per_prompt[0]
        if any(item != first for item in per_prompt[1:]):
            raise ConfigurationError("judge labels tokenize differently across batch prompts")
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, list | tuple):
            if not eos_token_id:
                raise ConfigurationError("tokenizer has no EOS token")
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            raise ConfigurationError("tokenizer has no EOS token")
        starts = padded_generation_start_lengths or tuple(len(prompt) for prompt in prompt_token_ids)
        return cls(first, int(eos_token_id), starts)


def _continuation_token_ids(
    tokenizer: Any,
    prompt_token_ids: tuple[int, ...],
    labels: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    result: dict[str, tuple[int, ...]] = {}
    for label in labels:
        combined = tuple(tokenizer.encode(prompt_text + label, add_special_tokens=False))
        if combined[: len(prompt_token_ids)] == prompt_token_ids:
            token_ids = combined[len(prompt_token_ids) :]
        else:
            token_ids = tuple(tokenizer.encode(label, add_special_tokens=False))
        if tokenizer.decode(token_ids, skip_special_tokens=False) != label:
            raise ConfigurationError(f"judge label does not round-trip exactly through tokenizer: {label}")
        result[label] = token_ids
    return result
