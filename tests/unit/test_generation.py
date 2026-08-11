from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import GenerationConfig
from transformers.generation import GenerationMixin

from self_judged_refusal_direction.config import (
    DataConfig,
    ModelConfig,
    ProjectConfig,
    RunConfig,
    TargetGenerationConfig,
)
from self_judged_refusal_direction.errors import InvariantError, TargetParseError, TargetParseErrorCode
from self_judged_refusal_direction.generation import TargetTrajectoryGenerator, resolved_generation_kwargs
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.models.base import ParsedTargetOutput
from self_judged_refusal_direction.schema import PromptRecord


class TokenizerSpy:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


class ProcessorSpy:
    def __init__(self) -> None:
        self.tokenizer = TokenizerSpy()

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        del kwargs
        return " ".join(str(token) for token in tokens)


class BrokenDecoderProcessor(ProcessorSpy):
    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        del tokens, kwargs
        raise ValueError("decoder failure")


class ModelSpy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(_get_generation_parameters=lambda: {})
        self.generation_config = GenerationConfig(
            do_sample=True,
            num_beams=1,
            temperature=0.7,
            top_p=0.9,
            top_k=32,
            min_p=0.05,
            typical_p=1.0,
            repetition_penalty=1.0,
            eos_token_id=[3, 4],
            pad_token_id=0,
        )
        self.options: dict[str, Any] | None = None

    def _prepare_generation_config(
        self,
        generation_config: GenerationConfig | None,
        **kwargs: Any,
    ) -> tuple[GenerationConfig, dict[str, Any]]:
        return GenerationMixin._prepare_generation_config(cast(Any, self), generation_config, **kwargs)

    def generate(self, **options: Any) -> torch.Tensor:
        self.options = options
        prefix = options["input_ids"][0].tolist()
        return torch.tensor([[*prefix, 9, 3]], dtype=torch.long)


class GreedyBatchModelSpy(ModelSpy):
    def __init__(self) -> None:
        super().__init__()
        self.generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            eos_token_id=[3, 4],
            pad_token_id=0,
        )

    def generate(self, **options: Any) -> torch.Tensor:
        self.options = options
        prefixes = options["input_ids"].tolist()
        return torch.tensor(
            [
                [*prefixes[0], 9, 3, 0],
                [*prefixes[1], 8, 7, 4],
            ],
            dtype=torch.long,
        )


class InvalidTailBatchModelSpy(GreedyBatchModelSpy):
    def generate(self, **options: Any) -> torch.Tensor:
        output = super().generate(**options)
        output[0, -1] = 99
        return output


class AdapterSpy:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.generation_config: TargetGenerationConfig | None = None
        self.parse_thinking_enabled: bool | None = None
        self.parse_prefix: tuple[int, ...] | None = None

    def render_target_chat(
        self,
        processor: Any,
        messages: list[dict[str, str]],
        config: TargetGenerationConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del processor, kwargs
        self.messages = messages
        self.generation_config = config
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def input_device(self, model: Any) -> torch.device:
        del model
        return torch.device("cpu")

    def context_window(self, model: Any) -> int:
        del model
        return 128

    def parse_target_trajectory(
        self,
        processor: Any,
        generated_ids: tuple[int, ...],
        *,
        prefix_ids: tuple[int, ...],
        thinking_enabled: bool,
    ) -> ParsedTargetOutput:
        del processor
        tokens = tuple(int(token) for token in generated_ids)
        self.parse_thinking_enabled = thinking_enabled
        self.parse_prefix = tuple(prefix_ids)
        return ParsedTargetOutput(
            raw_generated_token_ids=tokens,
            raw_decoded_output="answer<turn|>",
            thinking_text="",
            final_answer="answer",
            thinking_token_start=0,
            thinking_token_end=0,
            final_token_start=0,
            final_token_end=1,
            terminal_found=True,
        )


class BatchAdapterSpy(AdapterSpy):
    def __init__(self) -> None:
        super().__init__()
        self.conversations: list[list[dict[str, str]]] | None = None
        self.parse_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def render_target_chat_batch(
        self,
        processor: Any,
        conversations: list[list[dict[str, str]]],
        config: TargetGenerationConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del processor, config, kwargs
        self.conversations = conversations
        return {
            "input_ids": torch.tensor([[0, 0, 11, 12], [21, 22, 23, 24]], dtype=torch.long),
            "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
        }

    def parse_target_trajectory(
        self,
        processor: Any,
        generated_ids: tuple[int, ...],
        *,
        prefix_ids: tuple[int, ...],
        thinking_enabled: bool,
    ) -> ParsedTargetOutput:
        self.parse_calls.append((tuple(generated_ids), tuple(prefix_ids)))
        return super().parse_target_trajectory(
            processor,
            generated_ids,
            prefix_ids=prefix_ids,
            thinking_enabled=thinking_enabled,
        )


class FailingAdapterSpy(AdapterSpy):
    def parse_target_trajectory(
        self,
        processor: Any,
        generated_ids: tuple[int, ...],
        *,
        prefix_ids: tuple[int, ...],
        thinking_enabled: bool,
    ) -> ParsedTargetOutput:
        del processor, generated_ids, prefix_ids, thinking_enabled
        raise TargetParseError(TargetParseErrorCode.TERMINAL_MISSING, "private parser detail")


class RuntimeFailingAdapterSpy(AdapterSpy):
    def parse_target_trajectory(
        self,
        processor: Any,
        generated_ids: tuple[int, ...],
        *,
        prefix_ids: tuple[int, ...],
        thinking_enabled: bool,
    ) -> ParsedTargetOutput:
        del processor, generated_ids, prefix_ids, thinking_enabled
        raise RuntimeError("parser runtime failed")


class RuntimeSpy:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.adapter = AdapterSpy()
        self.model = ModelSpy()
        self.processor = ProcessorSpy()

    def load(self) -> RuntimeSpy:
        return self


def test_generation_uses_target_profile_and_hashes_resolved_model_defaults() -> None:
    generation = TargetGenerationConfig(
        system_prompt="target system",
        thinking_enabled=False,
        max_new_tokens=24,
        do_sample=None,
        num_beams=None,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=None,
        typical_p=None,
        repetition_penalty=None,
    )
    config = ProjectConfig(
        run=RunConfig(seed=17),
        model=ModelConfig(revision="a" * 40),
        target_generation=generation,
        data=DataConfig(max_text_tokens=32),
    )
    runtime = RuntimeSpy(config)

    trajectory = TargetTrajectoryGenerator(cast(Any, runtime)).generate("request", prompt_id="prompt")

    assert runtime.adapter.messages == [
        {"role": "system", "content": "target system"},
        {"role": "user", "content": "request"},
    ]
    assert runtime.adapter.parse_thinking_enabled is False
    assert runtime.adapter.parse_prefix == (11, 12)
    assert runtime.model.options is not None
    assert runtime.model.options["max_new_tokens"] == 24
    assert runtime.model.options["use_cache"] is True
    assert runtime.model.options["temperature"] == 1.0
    assert runtime.model.options["top_p"] == 0.95
    assert runtime.model.options["top_k"] == 64
    for omitted in ("do_sample", "num_beams", "min_p", "typical_p", "repetition_penalty"):
        assert omitted not in runtime.model.options
    effective = resolved_generation_kwargs(runtime.model, generation)
    assert {
        name: effective[name]
        for name in (
            "max_new_tokens",
            "use_cache",
            "do_sample",
            "num_beams",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "typical_p",
            "repetition_penalty",
            "eos_token_id",
            "pad_token_id",
        )
    } == {
        "max_new_tokens": 24,
        "use_cache": True,
        "do_sample": True,
        "num_beams": 1,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0.05,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "eos_token_id": [3, 4],
        "pad_token_id": 0,
    }
    assert trajectory.generation_config_hash == object_sha256(
        {
            "system_prompt": "target system",
            "thinking_enabled": False,
            "batch_size": 1,
            "generate_kwargs": effective,
        }
    )
    assert trajectory.parser_status == "OK"


def test_greedy_batch_generation_restores_prefixes_and_removes_only_post_eos_padding() -> None:
    generation = TargetGenerationConfig(
        system_prompt="target system",
        thinking_enabled=False,
        max_new_tokens=24,
        batch_size=2,
        do_sample=None,
        num_beams=None,
    )
    config = ProjectConfig(
        run=RunConfig(seed=17),
        model=ModelConfig(revision="a" * 40),
        target_generation=generation,
        data=DataConfig(max_text_tokens=32),
    )
    runtime = RuntimeSpy(config)
    runtime.adapter = BatchAdapterSpy()
    runtime.model = GreedyBatchModelSpy()
    prompts = [
        PromptRecord(prompt_id="first", original_prompt="one", group_id="a", split="train"),
        PromptRecord(prompt_id="second", original_prompt="two", group_id="b", split="validation"),
    ]

    trajectories = TargetTrajectoryGenerator(cast(Any, runtime)).generate_batch(prompts)

    assert runtime.adapter.conversations == [
        [
            {"role": "system", "content": "target system"},
            {"role": "user", "content": "one"},
        ],
        [
            {"role": "system", "content": "target system"},
            {"role": "user", "content": "two"},
        ],
    ]
    assert runtime.adapter.parse_calls == [
        ((9, 3), (11, 12)),
        ((8, 7, 4), (21, 22, 23, 24)),
    ]
    assert [trajectory.prompt_id for trajectory in trajectories] == ["first", "second"]
    assert [trajectory.split for trajectory in trajectories] == ["train", "validation"]
    assert [trajectory.raw_generated_token_ids for trajectory in trajectories] == [(9, 3), (8, 7, 4)]
    effective = resolved_generation_kwargs(runtime.model, generation)
    assert {trajectory.generation_config_hash for trajectory in trajectories} == {
        object_sha256(
            {
                "system_prompt": "target system",
                "thinking_enabled": False,
                "batch_size": 2,
                "generate_kwargs": effective,
            }
        )
    }


def test_batch_profile_rejects_inherited_non_greedy_mode_for_single_remainder() -> None:
    config = ProjectConfig(
        model=ModelConfig(revision="a" * 40),
        target_generation=TargetGenerationConfig(batch_size=2, do_sample=None, num_beams=None),
    )
    runtime = RuntimeSpy(config)
    runtime.model = GreedyBatchModelSpy()
    runtime.model.generation_config.penalty_alpha = 0.6
    runtime.model.generation_config.top_k = 4
    prompt = PromptRecord(prompt_id="only", original_prompt="one", group_id="a", split="train")

    with pytest.raises(InvariantError, match="requires greedy search"):
        TargetTrajectoryGenerator(cast(Any, runtime)).generate_batch([prompt])


def test_batch_generation_rejects_non_padding_after_first_eos() -> None:
    config = ProjectConfig(
        model=ModelConfig(revision="a" * 40),
        target_generation=TargetGenerationConfig(batch_size=2, max_new_tokens=24),
    )
    runtime = RuntimeSpy(config)
    runtime.adapter = BatchAdapterSpy()
    runtime.model = InvalidTailBatchModelSpy()
    prompts = [
        PromptRecord(prompt_id="first", original_prompt="one", group_id="a", split="train"),
        PromptRecord(prompt_id="second", original_prompt="two", group_id="b", split="train"),
    ]

    with pytest.raises(InvariantError, match="continued after its first terminal"):
        TargetTrajectoryGenerator(cast(Any, runtime)).generate_batch(prompts)


def test_generation_preserves_structured_parser_diagnostics() -> None:
    config = ProjectConfig(
        run=RunConfig(seed=17),
        model=ModelConfig(revision="a" * 40),
        target_generation=TargetGenerationConfig(thinking_enabled=False, max_new_tokens=24),
        data=DataConfig(max_text_tokens=32),
    )
    runtime = RuntimeSpy(config)
    runtime.adapter = FailingAdapterSpy()
    runtime.processor = BrokenDecoderProcessor()

    trajectory = TargetTrajectoryGenerator(cast(Any, runtime)).generate("request", prompt_id="prompt")

    assert trajectory.parser_status == "ERROR"
    assert trajectory.error_code == TargetParseErrorCode.TERMINAL_MISSING.value
    assert trajectory.error_detail == "private parser detail"
    assert trajectory.raw_generated_token_ids == (9, 3)
    assert trajectory.raw_decoded_output == ""
    assert trajectory.thinking_text == ""
    assert trajectory.final_answer == ""

    runtime.adapter = RuntimeFailingAdapterSpy()
    with pytest.raises(RuntimeError, match="parser runtime failed"):
        TargetTrajectoryGenerator(cast(Any, runtime)).generate("request", prompt_id="prompt")


def test_resolved_generation_kwargs_rejects_explicit_sampling_controls_for_effective_greedy() -> None:
    model = ModelSpy()
    model.generation_config = GenerationConfig(eos_token_id=[3, 4], pad_token_id=0, length_penalty=2.0)
    config = TargetGenerationConfig(do_sample=None, temperature=1.0)

    inherited = resolved_generation_kwargs(model, TargetGenerationConfig(do_sample=None, num_beams=None))
    assert inherited["do_sample"] is False
    assert inherited["length_penalty"] == 2.0
    with pytest.raises(InvariantError, match="require sampling-enabled"):
        resolved_generation_kwargs(model, config)
