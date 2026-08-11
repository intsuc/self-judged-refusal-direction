from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, Self

import torch

from self_judged_refusal_direction.config import ProjectConfig, TargetGenerationConfig
from self_judged_refusal_direction.errors import InvariantError
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


def _safe_decode(processor: Any, tokens: Sequence[int]) -> str:
    try:
        value = processor.decode(
            list(tokens),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        value = processor.decode(list(tokens), skip_special_tokens=False)
    return value if isinstance(value, str) else ""


def _terminal_generated(runtime: Runtime, tokens: tuple[int, ...]) -> bool:
    generation_config = getattr(runtime.model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        terminal_ids = {eos_token_id}
    elif isinstance(eos_token_id, list | tuple):
        terminal_ids = {int(value) for value in eos_token_id}
    else:
        terminal_ids = set()
    pad_token_id = getattr(getattr(runtime.processor, "tokenizer", None), "pad_token_id", None)
    meaningful = list(tokens)
    while meaningful and pad_token_id is not None and meaningful[-1] == pad_token_id:
        meaningful.pop()
    return bool(meaningful and meaningful[-1] in terminal_ids)


def _trajectory_hash(values: dict[str, Any]) -> str:
    return object_sha256(values)


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


def resolved_generation_kwargs(model: Any, config: TargetGenerationConfig) -> dict[str, Any]:
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
        if prompt_token_count > config.data.max_prompt_tokens:
            raise InvariantError(f"prompt has {prompt_token_count} tokens; maximum is {config.data.max_prompt_tokens}")
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
        generation_config_hash = object_sha256(
            {
                "system_prompt": generation.system_prompt,
                "thinking_enabled": generation.thinking_enabled,
                "generate_kwargs": resolved_kwargs,
            }
        )

        try:
            parsed = runtime.adapter.parse_target_trajectory(
                runtime.processor,
                generated_ids,
                prefix_ids=prefix_ids,
                thinking_enabled=generation.thinking_enabled,
            )
        except Exception as error:
            return self._parser_error_trajectory(
                prompt_id=resolved_prompt_id,
                original_prompt=original_prompt,
                generated_ids=generated_ids,
                generation_config_hash=generation_config_hash,
                generation_truncated=not _terminal_generated(runtime, generated_ids),
                split=resolved_split,
                seed=resolved_seed,
                error=error,
            )
        return self._successful_trajectory(
            prompt_id=resolved_prompt_id,
            original_prompt=original_prompt,
            parsed=parsed,
            generation_config_hash=generation_config_hash,
            split=resolved_split,
            seed=resolved_seed,
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
        error_code = f"TARGET_PARSE_{type(error).__name__.upper()}"
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
    "generation_kwargs",
    "resolved_generation_kwargs",
]
