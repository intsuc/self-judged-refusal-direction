from __future__ import annotations

from typing import Any

import pytest

from self_judged_refusal_direction.config import TargetGenerationConfig
from self_judged_refusal_direction.errors import InvariantError, TargetParseError, TargetParseErrorCode
from self_judged_refusal_direction.models.gemma4 import Gemma4Adapter

THINKING_OPEN = "<|channel>thought\n"
THINKING_CLOSE = "<channel|>"
CONTENT_CLOSE = "<turn|>"


class CharacterTokenizer:
    class Backend:
        def to_str(self) -> str:
            return "character-tokenizer"

    backend_tokenizer = Backend()
    pad_token_id = 0
    chat_template = "chat-template"

    def __init__(self) -> None:
        self.response_template: dict[str, Any] = {
            "fields": {
                "thinking": {"open": THINKING_OPEN, "close": THINKING_CLOSE},
                "content": {"close": [CONTENT_CLOSE]},
            }
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


class ProcessorSpy:
    chat_template = "processor-chat-template"

    def __init__(self, parsed: dict[str, str] | None = None) -> None:
        self.tokenizer = CharacterTokenizer()
        self.parsed = parsed or {"role": "assistant", "content": ""}
        self.prefix: list[int] | None = None
        self.chat_messages: Any = None
        self.chat_options: dict[str, Any] | None = None

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids if token_id != self.tokenizer.pad_token_id)

    def parse_response(self, response: list[int], *, prefix: list[int]) -> dict[str, str]:
        del response
        self.prefix = prefix
        return self.parsed

    def apply_chat_template(self, messages: list[dict[str, str]], **options: Any) -> dict[str, Any]:
        self.chat_messages = messages
        self.chat_options = options
        return {"input_ids": [[1]]}

    def to_dict(self) -> dict[str, str]:
        return {"processor_class": type(self).__name__}


def token_ids(value: str) -> tuple[int, ...]:
    return tuple(ord(character) for character in value)


def test_render_target_chat_resolves_config_and_explicit_thinking_modes() -> None:
    adapter = Gemma4Adapter()
    processor = ProcessorSpy()
    messages = [{"role": "user", "content": "request"}]

    adapter.render_target_chat(
        processor,
        messages,
        config=TargetGenerationConfig(thinking_enabled=False),
    )
    assert processor.chat_options is not None
    assert processor.chat_options["enable_thinking"] is False

    adapter.render_target_chat(processor, messages, thinking_enabled=True)
    assert processor.chat_options["enable_thinking"] is True

    with pytest.raises(InvariantError, match="conflicts"):
        adapter.render_target_chat(
            processor,
            messages,
            config=TargetGenerationConfig(thinking_enabled=True),
            thinking_enabled=False,
        )


def test_render_target_chat_batch_requires_left_padding() -> None:
    adapter = Gemma4Adapter()
    processor = ProcessorSpy()
    conversations = [
        [{"role": "user", "content": "short"}],
        [{"role": "user", "content": "longer"}],
    ]

    adapter.render_target_chat_batch(
        processor,
        conversations,
        config=TargetGenerationConfig(thinking_enabled=False),
    )

    assert processor.chat_messages == conversations
    assert processor.chat_options is not None
    assert processor.chat_options["enable_thinking"] is False
    assert processor.chat_options["processor_kwargs"] == {"padding": True, "padding_side": "left"}

    with pytest.raises(InvariantError, match="padding cannot be overridden"):
        adapter.render_target_chat_batch(processor, conversations, processor_kwargs={"padding_side": "right"})


def test_processor_fingerprints_include_chat_and_response_templates() -> None:
    adapter = Gemma4Adapter()
    processor = ProcessorSpy()
    original = adapter.processor_fingerprints(processor)

    processor.chat_template = "changed-processor-chat-template"
    assert adapter.processor_fingerprints(processor)["processor_sha256"] != original["processor_sha256"]

    processor.tokenizer.response_template = {
        "fields": {
            "thinking": {"open": "<thought>", "close": "</thought>"},
            "content": {"close": ["</answer>"]},
        }
    }
    assert adapter.processor_fingerprints(processor)["tokenizer_sha256"] != original["tokenizer_sha256"]


def test_parse_thinking_trajectory_preserves_strict_token_boundaries() -> None:
    adapter = Gemma4Adapter()
    processor = ProcessorSpy({"role": "assistant", "thinking": "reason", "content": "answer"})
    prefix = (7, 8)
    output = token_ids(f"{THINKING_OPEN}reason{THINKING_CLOSE}answer{CONTENT_CLOSE}")

    parsed = adapter.parse_target_trajectory(processor, output, prefix_ids=prefix, thinking_enabled=True)

    thinking_start = len(token_ids(THINKING_OPEN))
    thinking_end = thinking_start + len("reason")
    final_start = thinking_end + len(token_ids(THINKING_CLOSE))
    assert parsed.thinking_token_start == thinking_start
    assert parsed.thinking_token_end == thinking_end
    assert parsed.final_token_start == final_start
    assert parsed.final_token_end == final_start + len("answer")
    assert parsed.terminal_found
    assert processor.prefix == list(prefix)


def test_parse_content_only_trajectory_has_empty_thinking_span() -> None:
    adapter = Gemma4Adapter()
    processor = ProcessorSpy({"role": "assistant", "content": "answer"})
    prefix = (4, 5, 6)
    output = (*token_ids(f"answer{CONTENT_CLOSE}"), 0, 0)

    parsed = adapter.parse_target_trajectory(processor, output, prefix_ids=prefix, thinking_enabled=False)

    assert parsed.thinking_text == ""
    assert parsed.thinking_token_start == 0
    assert parsed.thinking_token_end == 0
    assert parsed.final_answer == "answer"
    assert parsed.final_token_start == 0
    assert parsed.final_token_end == len("answer")
    assert parsed.terminal_found
    assert processor.prefix == list(prefix)


@pytest.mark.parametrize(
    ("output", "error_code"),
    [
        (
            f"{THINKING_OPEN}reason{THINKING_CLOSE}answer{CONTENT_CLOSE}",
            TargetParseErrorCode.THINKING_DELIMITER_IN_CONTENT,
        ),
        (f"answer{THINKING_CLOSE}{CONTENT_CLOSE}", TargetParseErrorCode.THINKING_DELIMITER_IN_CONTENT),
        ("answer", TargetParseErrorCode.TERMINAL_MISSING),
        (f"answer{CONTENT_CLOSE}trailing", TargetParseErrorCode.TRAILING_TOKENS),
    ],
)
def test_content_only_parser_rejects_thinking_delimiters_truncation_and_trailing(
    output: str,
    error_code: TargetParseErrorCode,
) -> None:
    adapter = Gemma4Adapter()
    processor = ProcessorSpy({"role": "assistant", "content": "answer"})

    with pytest.raises(TargetParseError) as raised:
        adapter.parse_target_trajectory(processor, token_ids(output), thinking_enabled=False)
    assert raised.value.code is error_code


def test_content_only_parser_rejects_nonempty_official_thinking() -> None:
    adapter = Gemma4Adapter()
    processor = ProcessorSpy({"role": "assistant", "thinking": "reason", "content": "answer"})

    with pytest.raises(TargetParseError, match="found thinking") as raised:
        adapter.parse_target_trajectory(
            processor,
            token_ids(f"answer{CONTENT_CLOSE}"),
            thinking_enabled=False,
        )
    assert raised.value.code is TargetParseErrorCode.UNEXPECTED_THINKING
