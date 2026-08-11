from __future__ import annotations

from collections.abc import Mapping, Sequence
from subprocess import SubprocessError
from typing import Any, Literal, Protocol, Self

import torch

from self_judged_refusal_direction.config import ProjectConfig, TargetGenerationConfig
from self_judged_refusal_direction.errors import InvariantError, TargetParseError, TargetParseErrorCode
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.models.base import ArchitectureAdapter, ParsedTargetOutput
from self_judged_refusal_direction.schema import PromptRecord, TargetTrajectory


class Runtime(Protocol):
    config: ProjectConfig
    adapter: ArchitectureAdapter

    @property
    def model(self) -> Any: ...

    @property
    def processor(self) -> Any: ...

    def load(self) -> Self: ...


Split = Literal["train", "validation", "test"]
_GENERATION_OPTION_NAMES = (
    "do_sample",
    "num_beams",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "repetition_penalty",
)


def _as_token_ids(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        data = value.detach().to(device="cpu").tolist()
    elif hasattr(value, "tolist") and not isinstance(value, list | tuple):
        data = value.tolist()
    else:
        data = value
    if isinstance(data, list | tuple) and len(data) == 1 and isinstance(data[0], list | tuple):
        data = data[0]
    if not isinstance(data, list | tuple) or any(not isinstance(item, int) for item in data):
        raise InvariantError(f"{name} must contain exactly one token sequence")
    return tuple(data)


def _as_token_rows(value: Any, name: str) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, torch.Tensor):
        data = value.detach().to(device="cpu").tolist()
    elif hasattr(value, "tolist") and not isinstance(value, list | tuple):
        data = value.tolist()
    else:
        data = value
    if not isinstance(data, list | tuple) or any(
        not isinstance(row, list | tuple) or any(not isinstance(item, int) for item in row) for row in data
    ):
        raise InvariantError(f"{name} must contain a batch of token sequences")
    return tuple(tuple(row) for row in data)


def _move_inputs(inputs: Any, device: torch.device | None) -> Any:
    if device is None:
        return inputs
    move = getattr(inputs, "to", None)
    if callable(move):
        return move(device)
    if not isinstance(inputs, Mapping):
        raise InvariantError("rendered target chat is not a model input mapping")
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def _generated_sequence(output: Any) -> Any:
    sequences = getattr(output, "sequences", output)
    if isinstance(sequences, torch.Tensor):
        if sequences.ndim == 2 and sequences.shape[0] == 1:
            return sequences[0]
        if sequences.ndim == 1:
            return sequences
    if isinstance(sequences, list | tuple) and len(sequences) == 1:
        return sequences[0]
    raise InvariantError("target generation must return exactly one sequence")


def _generated_sequences(output: Any, batch_size: int) -> tuple[tuple[int, ...], ...]:
    sequences = _as_token_rows(getattr(output, "sequences", output), "target generation")
    if len(sequences) != batch_size:
        raise InvariantError("target generation returned a different batch size")
    return sequences


def _left_padded_prefixes(
    inputs: Mapping[str, Any],
    processor: Any,
    batch_size: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    input_rows = _as_token_rows(inputs.get("input_ids"), "target input_ids")
    attention_rows = _as_token_rows(inputs.get("attention_mask"), "target attention_mask")
    if len(input_rows) != batch_size or len(attention_rows) != batch_size:
        raise InvariantError("rendered target chat returned a different batch size")
    if not input_rows or not input_rows[0]:
        raise InvariantError("rendered target chat has an empty input sequence")
    width = len(input_rows[0])
    pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    prefixes: list[tuple[int, ...]] = []
    for input_row, attention_row in zip(input_rows, attention_rows, strict=True):
        if len(input_row) != width or len(attention_row) != width:
            raise InvariantError("rendered target chat batch is not rectangular")
        try:
            prefix_start = attention_row.index(1)
        except ValueError as error:
            raise InvariantError("rendered target chat has an empty input sequence") from error
        if attention_row != (0,) * prefix_start + (1,) * (width - prefix_start):
            raise InvariantError("rendered target chat batch is not left padded")
        if prefix_start and (
            not isinstance(pad_token_id, int) or any(token != pad_token_id for token in input_row[:prefix_start])
        ):
            raise InvariantError("rendered target chat has invalid left padding")
        prefixes.append(input_row[prefix_start:])
    return input_rows, tuple(prefixes)


def _terminal_token_ids(value: Any) -> frozenset[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return frozenset((value,))
    if isinstance(value, list | tuple) and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return frozenset(value)
    return frozenset()


def _processor_pad_token_id(processor: Any) -> int:
    value = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvariantError("batched target generation requires a processor pad token")
    return value


def _normalize_generated(
    tokens: tuple[int, ...],
    terminal_ids: frozenset[int],
    pad_token_id: int,
) -> tuple[int, ...]:
    for index, token in enumerate(tokens):
        if token in terminal_ids:
            if any(value != pad_token_id for value in tokens[index + 1 :]):
                raise InvariantError("target generation continued after its first terminal token")
            return tokens[: index + 1]
    return tokens


def _safe_decode(processor: Any, tokens: Sequence[int]) -> str:
    try:
        try:
            value = processor.decode(
                list(tokens),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            value = processor.decode(list(tokens), skip_special_tokens=False)
    except ImportError, MemoryError, OSError, RuntimeError, SubprocessError:
        raise
    except Exception:
        return ""
    return value if isinstance(value, str) else ""


def _terminal_generated(runtime: Runtime, tokens: tuple[int, ...]) -> bool:
    generation_config = getattr(runtime.model, "generation_config", None)
    terminal_ids = _terminal_token_ids(getattr(generation_config, "eos_token_id", None))
    if tokens and tokens[-1] in terminal_ids:
        return True
    pad_token_id = getattr(getattr(runtime.processor, "tokenizer", None), "pad_token_id", None)
    meaningful = list(tokens)
    while meaningful and pad_token_id is not None and meaningful[-1] == pad_token_id:
        meaningful.pop()
    return bool(meaningful and meaningful[-1] in terminal_ids)


def _trajectory_hash(values: dict[str, Any]) -> str:
    return object_sha256(values)


def generation_config_hash(config: TargetGenerationConfig, resolved_kwargs: Mapping[str, Any]) -> str:
    return object_sha256(
        {
            "system_prompt": config.system_prompt,
            "thinking_enabled": config.thinking_enabled,
            "batch_size": config.batch_size,
            "generate_kwargs": resolved_kwargs,
        }
    )


def generation_kwargs(config: TargetGenerationConfig) -> dict[str, Any]:
    values: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "use_cache": True,
    }
    for name in _GENERATION_OPTION_NAMES:
        value = getattr(config, name)
        if value is not None:
            values[name] = value
    return values


def _resolved_generation_configuration(
    model: Any,
    config: TargetGenerationConfig,
) -> tuple[Any, dict[str, Any]]:
    prepare = getattr(model, "_prepare_generation_config", None)
    if not callable(prepare):
        raise InvariantError("model does not support generation configuration resolution")
    try:
        resolved, _ = prepare(None, **generation_kwargs(config))
        to_dict = getattr(resolved, "to_dict", None)
        if not callable(to_dict):
            raise TypeError
        values = to_dict()
    except (AttributeError, TypeError, ValueError) as error:
        raise InvariantError("model generation configuration could not be resolved") from error
    if not isinstance(values, dict):
        raise InvariantError("resolved model generation configuration is invalid")
    values.pop("_from_model_config", None)
    values.pop("transformers_version", None)
    if values.get("do_sample") is not True and any(
        getattr(config, name) is not None for name in ("temperature", "top_p", "top_k", "min_p", "typical_p")
    ):
        raise InvariantError("sampling parameters require sampling-enabled generation")
    return resolved, values


def _generation_mode(resolved: Any) -> str:
    get_generation_mode = getattr(resolved, "get_generation_mode", None)
    if not callable(get_generation_mode):
        raise InvariantError("resolved generation configuration has no generation mode")
    try:
        mode = get_generation_mode()
    except (AttributeError, TypeError, ValueError) as error:
        raise InvariantError("model generation mode could not be resolved") from error
    value = getattr(mode, "value", mode)
    if not isinstance(value, str):
        raise InvariantError("resolved model generation mode is invalid")
    return value


def resolved_generation_kwargs(model: Any, config: TargetGenerationConfig) -> dict[str, Any]:
    _, values = _resolved_generation_configuration(model, config)
    return values


def resolved_batch_generation_kwargs(
    model: Any,
    processor: Any,
    config: TargetGenerationConfig,
) -> dict[str, Any]:
    resolved, values = _resolved_generation_configuration(model, config)
    if config.batch_size == 1:
        return values
    if values.get("do_sample") is not False or _generation_mode(resolved) != "greedy_search":
        raise InvariantError("batched target generation requires greedy search")
    if values.get("num_beams") != 1:
        raise InvariantError("batched target generation requires num_beams=1")
    if values.get("num_return_sequences") != 1:
        raise InvariantError("batched target generation requires num_return_sequences=1")
    processor_pad_token_id = _processor_pad_token_id(processor)
    if values.get("pad_token_id") != processor_pad_token_id:
        raise InvariantError("model and processor pad token ids differ")
    return values


class TargetTrajectoryGenerator:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    def generate(
        self,
        prompt: PromptRecord | str,
        *,
        prompt_id: str | None = None,
        split: str | None = None,
        seed: int | None = None,
    ) -> TargetTrajectory:
        runtime = self.runtime.load()
        config = runtime.config
        generation = config.target_generation
        resolved_kwargs = resolved_generation_kwargs(runtime.model, generation)
        if isinstance(prompt, PromptRecord):
            if prompt_id is not None and prompt_id != prompt.prompt_id:
                raise InvariantError("prompt_id conflicts with PromptRecord")
            if split is not None and split != prompt.split:
                raise InvariantError("split conflicts with PromptRecord")
            original_prompt = prompt.original_prompt
            resolved_prompt_id = prompt.prompt_id
            resolved_split: Split | None = prompt.split
        else:
            original_prompt = prompt
            resolved_prompt_id = prompt_id or object_sha256({"prompt": original_prompt})[:24]
            if split not in (None, "train", "validation", "test"):
                raise InvariantError(f"unsupported split: {split}")
            resolved_split = split
        resolved_seed = config.run.seed if seed is None else seed
        torch.manual_seed(resolved_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved_seed)

        messages: list[dict[str, str]] = []
        if generation.system_prompt is not None:
            messages.append({"role": "system", "content": generation.system_prompt})
        messages.append({"role": "user", "content": original_prompt})
        prompt_token_count = len(runtime.processor.tokenizer.encode(original_prompt, add_special_tokens=False))
        if prompt_token_count > config.data.max_text_tokens:
            raise InvariantError(f"prompt has {prompt_token_count} tokens; maximum is {config.data.max_text_tokens}")
        rendered = runtime.adapter.render_target_chat(runtime.processor, messages, config=generation)
        inputs = _move_inputs(rendered, runtime.adapter.input_device(runtime.model))
        if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
            raise InvariantError("rendered target chat has no input_ids")
        prefix_ids = _as_token_ids(inputs["input_ids"], "target input_ids")
        required_context = len(prefix_ids) + generation.max_new_tokens
        context_window = runtime.adapter.context_window(runtime.model)
        if required_context > context_window:
            raise InvariantError(
                f"target generation requires {required_context} tokens but context window is {context_window}"
            )
        target_generation_kwargs = generation_kwargs(generation)
        generate = getattr(runtime.model, "generate", None)
        if not callable(generate):
            raise InvariantError("model does not support generation")
        with torch.inference_mode():
            output = generate(**dict(inputs), **target_generation_kwargs)
        sequence = _as_token_ids(_generated_sequence(output), "target generation")
        if sequence[: len(prefix_ids)] != prefix_ids:
            raise InvariantError("target generation did not preserve its input prefix")
        generated_ids = sequence[len(prefix_ids) :]
        generation_hash = generation_config_hash(generation, resolved_kwargs)
        return self._trajectory_from_generation(
            prompt_id=resolved_prompt_id,
            original_prompt=original_prompt,
            generated_ids=generated_ids,
            prefix_ids=prefix_ids,
            generation_config_hash=generation_hash,
            split=resolved_split,
            seed=resolved_seed,
        )

    def generate_batch(self, prompts: Sequence[PromptRecord]) -> list[TargetTrajectory]:
        records = tuple(prompts)
        if not records:
            return []
        if any(not isinstance(prompt, PromptRecord) for prompt in records):
            raise InvariantError("target generation batch must contain PromptRecord values")

        runtime = self.runtime.load()
        config = runtime.config
        generation = config.target_generation
        if len(records) > generation.batch_size:
            raise InvariantError("target generation batch exceeds configured batch_size")
        if generation.batch_size == 1:
            return [self.generate(records[0])]
        resolved_kwargs = resolved_batch_generation_kwargs(runtime.model, runtime.processor, generation)
        processor_pad_token_id = _processor_pad_token_id(runtime.processor)
        if len(records) == 1:
            return [self.generate(records[0])]

        resolved_seed = config.run.seed
        torch.manual_seed(resolved_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved_seed)

        conversations: list[list[dict[str, str]]] = []
        for prompt in records:
            messages: list[dict[str, str]] = []
            if generation.system_prompt is not None:
                messages.append({"role": "system", "content": generation.system_prompt})
            messages.append({"role": "user", "content": prompt.original_prompt})
            prompt_token_count = len(
                runtime.processor.tokenizer.encode(prompt.original_prompt, add_special_tokens=False)
            )
            if prompt_token_count > config.data.max_text_tokens:
                raise InvariantError(
                    f"prompt has {prompt_token_count} tokens; maximum is {config.data.max_text_tokens}"
                )
            conversations.append(messages)

        rendered = runtime.adapter.render_target_chat_batch(
            runtime.processor,
            conversations,
            config=generation,
        )
        inputs = _move_inputs(rendered, runtime.adapter.input_device(runtime.model))
        if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
            raise InvariantError("rendered target chat has no input_ids")
        input_rows, prefix_rows = _left_padded_prefixes(inputs, runtime.processor, len(records))
        required_context = len(input_rows[0]) + generation.max_new_tokens
        context_window = runtime.adapter.context_window(runtime.model)
        if required_context > context_window:
            raise InvariantError(
                f"target generation requires {required_context} tokens but context window is {context_window}"
            )

        target_generation_kwargs = generation_kwargs(generation)
        generate = getattr(runtime.model, "generate", None)
        if not callable(generate):
            raise InvariantError("model does not support generation")
        with torch.inference_mode():
            output = generate(**dict(inputs), **target_generation_kwargs)
        sequences = _generated_sequences(output, len(records))
        input_width = len(input_rows[0])
        terminal_ids = _terminal_token_ids(resolved_kwargs.get("eos_token_id"))
        generation_hash = generation_config_hash(generation, resolved_kwargs)
        trajectories: list[TargetTrajectory] = []
        for record, input_row, prefix_ids, sequence in zip(
            records,
            input_rows,
            prefix_rows,
            sequences,
            strict=True,
        ):
            if sequence[:input_width] != input_row:
                raise InvariantError("target generation did not preserve its input prefix")
            generated_ids = _normalize_generated(
                sequence[input_width:],
                terminal_ids,
                processor_pad_token_id,
            )
            trajectories.append(
                self._trajectory_from_generation(
                    prompt_id=record.prompt_id,
                    original_prompt=record.original_prompt,
                    generated_ids=generated_ids,
                    prefix_ids=prefix_ids,
                    generation_config_hash=generation_hash,
                    split=record.split,
                    seed=resolved_seed,
                )
            )
        return trajectories

    def _trajectory_from_generation(
        self,
        *,
        prompt_id: str,
        original_prompt: str,
        generated_ids: tuple[int, ...],
        prefix_ids: tuple[int, ...],
        generation_config_hash: str,
        split: Split | None,
        seed: int,
    ) -> TargetTrajectory:
        try:
            parsed = self.runtime.adapter.parse_target_trajectory(
                self.runtime.processor,
                generated_ids,
                prefix_ids=prefix_ids,
                thinking_enabled=self.runtime.config.target_generation.thinking_enabled,
            )
        except ImportError, MemoryError, OSError, RuntimeError, SubprocessError:
            raise
        except Exception as error:
            return self._parser_error_trajectory(
                prompt_id=prompt_id,
                original_prompt=original_prompt,
                generated_ids=generated_ids,
                generation_config_hash=generation_config_hash,
                generation_truncated=not _terminal_generated(self.runtime, generated_ids),
                split=split,
                seed=seed,
                error=error,
            )
        return self._successful_trajectory(
            prompt_id=prompt_id,
            original_prompt=original_prompt,
            parsed=parsed,
            generation_config_hash=generation_config_hash,
            split=split,
            seed=seed,
        )

    def _successful_trajectory(
        self,
        *,
        prompt_id: str,
        original_prompt: str,
        parsed: ParsedTargetOutput,
        generation_config_hash: str,
        split: Split | None,
        seed: int,
    ) -> TargetTrajectory:
        values = {
            "prompt_id": prompt_id,
            "original_prompt": original_prompt,
            "raw_generated_token_ids": parsed.raw_generated_token_ids,
            "raw_decoded_output": parsed.raw_decoded_output,
            "thinking_text": parsed.thinking_text,
            "final_answer": parsed.final_answer,
            "thinking_token_start": parsed.thinking_token_start,
            "thinking_token_end": parsed.thinking_token_end,
            "final_token_start": parsed.final_token_start,
            "final_token_end": parsed.final_token_end,
            "generation_truncated": not parsed.terminal_found,
            "parser_status": "OK",
            "model_revision": self.runtime.config.model.revision,
            "generation_config_hash": generation_config_hash,
            "split": split,
            "seed": seed,
            "error_code": None,
            "error_detail": None,
        }
        return TargetTrajectory(trajectory_hash=_trajectory_hash(values), **values)

    def _parser_error_trajectory(
        self,
        *,
        prompt_id: str,
        original_prompt: str,
        generated_ids: tuple[int, ...],
        generation_config_hash: str,
        generation_truncated: bool,
        split: Split | None,
        seed: int,
        error: Exception,
    ) -> TargetTrajectory:
        if isinstance(error, TargetParseError):
            error_code = error.code.value
            error_detail = error.detail
        else:
            error_code = TargetParseErrorCode.INTERNAL.value
            error_detail = f"{type(error).__name__}: {error}"
        values = {
            "prompt_id": prompt_id,
            "original_prompt": original_prompt,
            "raw_generated_token_ids": generated_ids,
            "raw_decoded_output": _safe_decode(self.runtime.processor, generated_ids),
            "thinking_text": "",
            "final_answer": "",
            "thinking_token_start": -1,
            "thinking_token_end": -1,
            "final_token_start": -1,
            "final_token_end": -1,
            "generation_truncated": generation_truncated,
            "parser_status": "ERROR",
            "model_revision": self.runtime.config.model.revision,
            "generation_config_hash": generation_config_hash,
            "split": split,
            "seed": seed,
            "error_code": error_code,
            "error_detail": error_detail,
        }
        return TargetTrajectory(trajectory_hash=_trajectory_hash(values), **values)


def generate_target_trajectory(
    runtime: Runtime,
    prompt: PromptRecord | str,
    *,
    prompt_id: str | None = None,
    split: str | None = None,
    seed: int | None = None,
) -> TargetTrajectory:
    return TargetTrajectoryGenerator(runtime).generate(
        prompt,
        prompt_id=prompt_id,
        split=split,
        seed=seed,
    )


__all__ = [
    "TargetTrajectoryGenerator",
    "generate_target_trajectory",
    "generation_config_hash",
    "generation_kwargs",
    "resolved_batch_generation_kwargs",
    "resolved_generation_kwargs",
]
