from __future__ import annotations

import gc
import weakref
from dataclasses import replace
from typing import Any

import pytest
import torch

from self_judged_refusal_direction.judging import TrajectoryJudge
from self_judged_refusal_direction.models.gemma4 import Gemma4Adapter
from self_judged_refusal_direction.schema import TargetTrajectory


class CharacterTokenizer:
    eos_token_id = 3
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids if token_id > self.eos_token_id)


class ProcessorSpy:
    def __init__(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.options: dict[str, Any] | None = None

    def apply_chat_template(self, messages: list[dict[str, str]], **options: Any) -> dict[str, torch.Tensor]:
        del messages
        self.options = options
        return {
            "input_ids": torch.tensor([[ord("P")]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }


class BrokenProcessor(ProcessorSpy):
    def apply_chat_template(self, messages: list[dict[str, str]], **options: Any) -> dict[str, torch.Tensor]:
        del messages, options
        raise TypeError("broken processor")


class ToyJudgeAdapter(Gemma4Adapter):
    def __init__(self, context: int):
        self.context = context

    def context_window(self, model: Any) -> int:
        del model
        return self.context

    def input_device(self, model: Any) -> torch.device:
        del model
        return torch.device("cpu")


class ToyJudgeModel:
    def __init__(self, tokenizer: CharacterTokenizer, label: str):
        self.tokenizer = tokenizer
        self.label = label
        self.generate_calls = 0

    def generate(self, input_ids: torch.Tensor, **options: Any) -> torch.Tensor:
        self.generate_calls += 1
        prefix_allowed_tokens_fn = options["prefix_allowed_tokens_fn"]
        sequence = input_ids[0].tolist()
        for token_id in self.tokenizer.encode(self.label):
            assert token_id in prefix_allowed_tokens_fn(0, torch.tensor(sequence))
            sequence.append(token_id)
        assert self.tokenizer.eos_token_id in prefix_allowed_tokens_fn(0, torch.tensor(sequence))
        sequence.append(self.tokenizer.eos_token_id)
        return torch.tensor([sequence], dtype=input_ids.dtype)


class FailedJudgeModel(ToyJudgeModel):
    def generate(self, input_ids: torch.Tensor, **options: Any) -> torch.Tensor:
        del input_ids, options
        raise RuntimeError("device failure")


def trajectory() -> TargetTrajectory:
    return TargetTrajectory(
        prompt_id="prompt",
        original_prompt="request",
        raw_generated_token_ids=(1, 2),
        raw_decoded_output="thinking answer",
        thinking_text="thinking",
        final_answer="answer",
        thinking_token_start=0,
        thinking_token_end=1,
        final_token_start=1,
        final_token_end=2,
        generation_truncated=False,
        parser_status="OK",
        model_revision="a" * 40,
        generation_config_hash="generation",
        trajectory_hash="trajectory",
    )


def test_judge_constrains_and_parses_uncertain_label() -> None:
    processor = ProcessorSpy()
    adapter = ToyJudgeAdapter(context=128)
    model = ToyJudgeModel(processor.tokenizer, "UNCERTAIN")
    judge = TrajectoryJudge(adapter, model, processor)

    result = judge.classify(trajectory())

    assert processor.options is not None
    assert processor.options["enable_thinking"] is False
    assert model.generate_calls == 1
    assert result.status == "OK"
    assert result.label == "UNCERTAIN"


def test_context_overflow_is_error_not_uncertain() -> None:
    processor = ProcessorSpy()
    adapter = ToyJudgeAdapter(context=12)
    model = ToyJudgeModel(processor.tokenizer, "UNCERTAIN")
    judge = TrajectoryJudge(adapter, model, processor)

    result = judge.classify(trajectory())

    assert processor.options is not None
    assert processor.options["enable_thinking"] is False
    assert model.generate_calls == 0
    assert result.status == "ERROR"
    assert result.label is None
    assert result.error_code == "ConfigurationError"


def test_adapter_failure_is_returned_as_judge_error() -> None:
    processor = BrokenProcessor()
    adapter = ToyJudgeAdapter(context=128)
    model = ToyJudgeModel(processor.tokenizer, "NON_REFUSAL")
    judge = TrajectoryJudge(adapter, model, processor)

    result = judge.classify(trajectory())

    assert result.status == "ERROR"
    assert result.label is None
    assert result.error_code == "TypeError"

    parser_error = judge.classify(
        replace(
            trajectory(),
            parser_status="ERROR",
            error_code="TARGET_PARSE_INVARIANTERROR",
        )
    )
    missing_code = judge.classify(replace(trajectory(), parser_status="ERROR"))

    assert model.generate_calls == 0
    assert parser_error.status == "ERROR"
    assert parser_error.error_code == "TARGET_PARSE_INVARIANTERROR"
    assert missing_code.status == "ERROR"
    assert missing_code.error_code == "INVALID_TRAJECTORY"


def test_judge_does_not_own_runtime_resources() -> None:
    processor = ProcessorSpy()
    model = ToyJudgeModel(processor.tokenizer, "NON_REFUSAL")
    model_ref = weakref.ref(model)
    processor_ref = weakref.ref(processor)
    judge = TrajectoryJudge(ToyJudgeAdapter(context=128), model, processor)

    del model
    del processor
    gc.collect()

    assert model_ref() is None
    assert processor_ref() is None
    result = judge.classify(trajectory())
    assert result.status == "ERROR"
    assert result.error_code == "InvariantError"


def test_judge_does_not_convert_runtime_failure_to_record_error() -> None:
    processor = ProcessorSpy()
    model = FailedJudgeModel(processor.tokenizer, "NON_REFUSAL")
    judge = TrajectoryJudge(ToyJudgeAdapter(context=128), model, processor)

    with pytest.raises(RuntimeError, match="device failure"):
        judge.classify(trajectory())
