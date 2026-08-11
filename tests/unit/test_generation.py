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
from self_judged_refusal_direction.errors import InvariantError
from self_judged_refusal_direction.generation import (
    TargetTrajectoryGenerator,
    generation_kwargs,
    resolved_generation_kwargs,
)
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.models.base import ParsedTargetOutput


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
        data=DataConfig(max_prompt_tokens=32),
    )
    runtime = RuntimeSpy(config)

    trajectory = TargetTrajectoryGenerator(cast(Any, runtime)).generate("request", prompt_id="prompt")

    assert runtime.adapter.messages == [
        {"role": "system", "content": "target system"},
        {"role": "user", "content": "request"},
    ]
    assert runtime.adapter.generation_config is generation
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
            "generate_kwargs": effective,
        }
    )
    assert trajectory.parser_status == "OK"


def test_resolved_generation_kwargs_rejects_explicit_sampling_controls_for_effective_greedy() -> None:
    model = ModelSpy()
    model.generation_config = GenerationConfig(eos_token_id=[3, 4], pad_token_id=0, length_penalty=2.0)
    config = TargetGenerationConfig(do_sample=None, temperature=1.0)

    inherited = resolved_generation_kwargs(model, TargetGenerationConfig(do_sample=None, num_beams=None))
    assert inherited["do_sample"] is False
    assert inherited["num_beams"] == 1
    assert inherited["temperature"] == 1.0
    assert inherited["top_p"] == 1.0
    assert inherited["top_k"] == 50
    assert inherited["typical_p"] == 1.0
    assert inherited["repetition_penalty"] == 1.0
    assert inherited["length_penalty"] == 2.0
    assert generation_kwargs(config)["temperature"] == 1.0
    with pytest.raises(InvariantError, match="require sampling-enabled"):
        resolved_generation_kwargs(model, config)
