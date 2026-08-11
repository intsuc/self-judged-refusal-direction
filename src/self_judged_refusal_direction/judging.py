from __future__ import annotations

import math
import weakref
from typing import Any, Literal, Protocol, cast

import torch

from self_judged_refusal_direction.config import ProjectConfig
from self_judged_refusal_direction.decoding import EnumTrieConstrainedDecoder
from self_judged_refusal_direction.errors import ConfigurationError, InvariantError
from self_judged_refusal_direction.hashing import object_sha256
from self_judged_refusal_direction.prompting import judge_messages, judge_template_hash
from self_judged_refusal_direction.schema import JudgeResult, TargetTrajectory


class JudgeAdapter(Protocol):
    def render_judge_chat(self, processor: Any, messages: list[dict[str, str]]) -> Any: ...

    def context_window(self, model: Any) -> int: ...

    def input_device(self, model: Any) -> torch.device: ...


class TrajectoryJudge:
    def __init__(
        self,
        config: ProjectConfig,
        adapter: JudgeAdapter,
        model: Any,
        processor: Any,
        chat_template_hash: str,
    ):
        self.config = config
        self.adapter = adapter
        try:
            self._model_ref: weakref.ReferenceType[Any] = weakref.ref(model)
            self._processor_ref: weakref.ReferenceType[Any] = weakref.ref(processor)
        except TypeError as error:
            raise InvariantError("judge resources must support weak references") from error
        self.chat_template_hash = chat_template_hash

    def classify(self, trajectory: TargetTrajectory) -> JudgeResult:
        if trajectory.parser_status != "OK" or trajectory.trajectory_status != "OK":
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
                tuple(self.config.judge.labels),
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
            if raw_output not in self.config.judge.labels:
                raise InvariantError("decoded judge output is outside the configured enum")
            scores = self._score_labels(model, processor, input_ids, rendered.get("attention_mask"), decoder)
            margin = _score_margin(scores)
            threshold = self.config.judge.calibrated_margin_threshold
            if threshold is not None:
                if margin is None:
                    raise InvariantError("calibrated margin is unavailable")
                if margin < threshold:
                    label = "UNCERTAIN"
            typed_label = cast(Literal["REFUSAL", "NON_REFUSAL", "UNCERTAIN"], label)
            profile_hash = self.profile_hash(decoder)
            return JudgeResult(
                status="OK",
                label=typed_label,
                raw_output=raw_output,
                label_logprobs=scores,
                calibrated_margin=margin,
                trajectory_hash=trajectory.trajectory_hash,
                judge_profile_hash=profile_hash,
            )
        except Exception as error:
            return self._error(trajectory, type(error).__name__)

    def profile_hash(self, decoder: EnumTrieConstrainedDecoder) -> str:
        return object_sha256(
            {
                "base": self.config.judge_profile_hash,
                "judge_template": judge_template_hash(),
                "chat_template": self.chat_template_hash,
                "decoder": decoder.signature_hash,
            }
        )

    def _resources(self) -> tuple[Any, Any]:
        model = self._model_ref()
        processor = self._processor_ref()
        if model is None or processor is None:
            raise InvariantError("judge resources are no longer available")
        return model, processor

    def _validate_context(self, model: Any, input_tokens: int, output_tokens: int) -> None:
        context = self.adapter.context_window(model)
        total = input_tokens + output_tokens + self.config.judge.safety_margin_tokens
        if total > context:
            raise ConfigurationError(f"full judge input requires {total} tokens but context window is {context}")

    def _score_labels(
        self,
        model: Any,
        processor: Any,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        decoder: EnumTrieConstrainedDecoder,
    ) -> dict[str, float] | None:
        if not self.config.judge.score_allowed_labels:
            return None
        suffixes = {label: (*token_ids, decoder.eos_token_id) for label, token_ids in decoder.label_token_ids.items()}
        max_suffix = max(len(value) for value in suffixes.values())
        batch_size = len(suffixes)
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = decoder.eos_token_id
        repeated_prompt = prompt_ids.repeat(batch_size, 1)
        suffix_tensor = torch.full(
            (batch_size, max_suffix),
            int(pad_token_id),
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        suffix_mask = torch.zeros_like(suffix_tensor)
        for row, suffix in enumerate(suffixes.values()):
            suffix_tensor[row, : len(suffix)] = torch.tensor(suffix, device=prompt_ids.device, dtype=prompt_ids.dtype)
            suffix_mask[row, : len(suffix)] = 1
        model_input_ids = torch.cat((repeated_prompt, suffix_tensor), dim=1)
        if attention_mask is None:
            prompt_mask = torch.ones_like(repeated_prompt)
        else:
            prompt_mask = attention_mask.repeat(batch_size, 1)
        model_attention = torch.cat((prompt_mask, suffix_mask), dim=1)
        with torch.inference_mode():
            output = model(
                input_ids=model_input_ids,
                attention_mask=model_attention,
                use_cache=False,
                logits_to_keep=max_suffix + 1,
                return_dict=True,
            )
        logits = output.logits.float()
        scores: dict[str, float] = {}
        for row, (label, suffix) in enumerate(suffixes.items()):
            total = 0.0
            for index, token_id in enumerate(suffix):
                total += float(torch.log_softmax(logits[row, index], dim=-1)[token_id].item())
            scores[label] = total
        return scores

    def _error(self, trajectory: TargetTrajectory, error_code: str) -> JudgeResult:
        return JudgeResult(
            status="ERROR",
            label=None,
            raw_output=None,
            label_logprobs=None,
            calibrated_margin=None,
            trajectory_hash=trajectory.trajectory_hash,
            judge_profile_hash=self.config.judge_profile_hash,
            error_code=error_code,
        )


def _move_batch(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _score_margin(scores: dict[str, float] | None) -> float | None:
    if scores is None:
        return None
    finite = sorted((value for value in scores.values() if math.isfinite(value)), reverse=True)
    if len(finite) < 2:
        raise InvariantError("judge label scores do not contain two finite values")
    return finite[0] - finite[1]
