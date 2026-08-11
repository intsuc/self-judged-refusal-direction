from __future__ import annotations

import weakref
from typing import Any, Literal, Protocol, cast

import torch

from self_judged_refusal_direction.decoding import EnumTrieConstrainedDecoder
from self_judged_refusal_direction.errors import ConfigurationError, InvariantError
from self_judged_refusal_direction.prompting import judge_messages
from self_judged_refusal_direction.schema import JudgeResult, TargetTrajectory

LABELS = ("REFUSAL", "NON_REFUSAL", "UNCERTAIN")


class JudgeAdapter(Protocol):
    def render_judge_chat(self, processor: Any, messages: list[dict[str, str]]) -> Any: ...

    def context_window(self, model: Any) -> int: ...

    def input_device(self, model: Any) -> torch.device: ...


class TrajectoryJudge:
    def __init__(
        self,
        adapter: JudgeAdapter,
        model: Any,
        processor: Any,
    ):
        self.adapter = adapter
        try:
            self._model_ref: weakref.ReferenceType[Any] = weakref.ref(model)
            self._processor_ref: weakref.ReferenceType[Any] = weakref.ref(processor)
        except TypeError as error:
            raise InvariantError("judge resources must support weak references") from error

    def classify(self, trajectory: TargetTrajectory) -> JudgeResult:
        if trajectory.parser_status != "OK":
            return self._error(trajectory, "INVALID_TRAJECTORY")
        try:
            model, processor = self._resources()
            rendered = self.adapter.render_judge_chat(processor, judge_messages(trajectory))
            rendered = _move_batch(rendered, self.adapter.input_device(model))
            input_ids = rendered["input_ids"]
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise InvariantError("judge classifier requires a single rendered trajectory")
            prompt_ids = tuple(int(value) for value in input_ids[0].tolist())
            decoder = EnumTrieConstrainedDecoder.compile(
                processor.tokenizer,
                LABELS,
                (prompt_ids,),
                (input_ids.shape[-1],),
            )
            self._validate_context(model, input_ids.shape[-1], decoder.max_new_tokens)
            generation_inputs = dict(rendered)
            with torch.inference_mode():
                output = model.generate(
                    **generation_inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=decoder.max_new_tokens,
                    prefix_allowed_tokens_fn=decoder.prefix_allowed_tokens_fn,
                    eos_token_id=decoder.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    use_cache=True,
                )
            sequences = output.sequences if hasattr(output, "sequences") else output
            suffix = sequences[0, input_ids.shape[-1] :]
            label, raw_output = decoder.parse_suffix(suffix)
            if raw_output not in LABELS:
                raise InvariantError("decoded judge output is outside the enum")
            typed_label = cast(Literal["REFUSAL", "NON_REFUSAL", "UNCERTAIN"], label)
            return JudgeResult(
                status="OK",
                label=typed_label,
                raw_output=raw_output,
                trajectory_hash=trajectory.trajectory_hash,
            )
        except Exception as error:
            return self._error(trajectory, type(error).__name__)

    def _resources(self) -> tuple[Any, Any]:
        model = self._model_ref()
        processor = self._processor_ref()
        if model is None or processor is None:
            raise InvariantError("judge resources are no longer available")
        return model, processor

    def _validate_context(self, model: Any, input_tokens: int, output_tokens: int) -> None:
        context = self.adapter.context_window(model)
        total = input_tokens + output_tokens
        if total > context:
            raise ConfigurationError(f"full judge input requires {total} tokens but context window is {context}")

    def _error(self, trajectory: TargetTrajectory, error_code: str) -> JudgeResult:
        return JudgeResult(
            status="ERROR",
            label=None,
            raw_output=None,
            trajectory_hash=trajectory.trajectory_hash,
            error_code=error_code,
        )


def _move_batch(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
