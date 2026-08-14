from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
import torch.nn.functional as F
from tqdm import tqdm

from self_judged_refusal_direction.config import ProjectConfig
from self_judged_refusal_direction.errors import InvariantError, NonFiniteMetricError, PipelineError
from self_judged_refusal_direction.models.base import ArchitectureAdapter
from self_judged_refusal_direction.prompting import target_messages
from self_judged_refusal_direction.schema import CEEvaluation, TargetTrajectory


class CERuntime(Protocol):
    config: ProjectConfig
    adapter: ArchitectureAdapter

    @property
    def model(self) -> Any: ...

    @property
    def processor(self) -> Any: ...


@dataclass(frozen=True)
class CEInput:
    input_ids: tuple[int, ...]
    target_start: int

    def __post_init__(self) -> None:
        if len(self.input_ids) < 2:
            raise InvariantError("CE input must contain at least two tokens")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in self.input_ids):
            raise InvariantError("CE input_ids must contain integers")
        if isinstance(self.target_start, bool) or not isinstance(self.target_start, int):
            raise InvariantError("CE target_start must be an integer")
        if not 1 <= self.target_start < len(self.input_ids):
            raise InvariantError("CE target_start must identify a predictable token")

    @property
    def target_token_count(self) -> int:
        return len(self.input_ids) - self.target_start


@dataclass(frozen=True)
class CELoss:
    total_loss: float
    token_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_loss):
            raise NonFiniteMetricError("CE total loss is not finite")
        if self.total_loss < 0:
            raise InvariantError("CE total loss must be non-negative")
        if self.token_count <= 0:
            raise InvariantError("CE loss must contain at least one target token")

    @property
    def mean_loss(self) -> float:
        return self.total_loss / self.token_count


@dataclass(frozen=True)
class CEComparison:
    baseline: CELoss
    intervention: CELoss

    def __post_init__(self) -> None:
        if self.baseline.token_count != self.intervention.token_count:
            raise InvariantError("baseline and intervention CE token counts differ")

    @property
    def loss_delta(self) -> float:
        return self.intervention.mean_loss - self.baseline.mean_loss

    def as_dict(self) -> dict[str, float | int]:
        return {
            "baseline_loss": self.baseline.mean_loss,
            "intervention_loss": self.intervention.mean_loss,
            "loss_delta": self.loss_delta,
            "target_token_count": self.baseline.token_count,
        }

    def as_evaluation(
        self,
        *,
        source: Literal["baseline_non_refusal_completions", "reference_files"],
        input_count: int,
    ) -> CEEvaluation:
        if source not in {"baseline_non_refusal_completions", "reference_files"}:
            raise InvariantError("CE evaluation source is invalid")
        if isinstance(input_count, bool) or not isinstance(input_count, int) or input_count <= 0:
            raise InvariantError("CE evaluation input count must be positive")
        return CEEvaluation(
            source=source,
            input_count=input_count,
            target_token_count=self.baseline.token_count,
            baseline_loss=self.baseline.mean_loss,
            intervention_loss=self.intervention.mean_loss,
        )

    @staticmethod
    def non_finite_evaluation(
        baseline: CELoss,
        *,
        source: Literal["baseline_non_refusal_completions", "reference_files"],
        input_count: int,
    ) -> CEEvaluation:
        return CEEvaluation(
            source=source,
            input_count=input_count,
            target_token_count=baseline.token_count,
            baseline_loss=baseline.mean_loss,
            intervention_loss=None,
            error_code="NON_FINITE",
        )


def ce_evaluation_from_losses(
    baseline: CELoss,
    intervention: CELoss | None,
    *,
    source: Literal["baseline_non_refusal_completions", "reference_files"],
    input_count: int,
) -> CEEvaluation:
    if intervention is None:
        return CEComparison.non_finite_evaluation(baseline, source=source, input_count=input_count)
    return compare_ce_losses(baseline, intervention).as_evaluation(source=source, input_count=input_count)


def raw_text_ce_inputs(runtime: CERuntime, texts: Iterable[str]) -> tuple[CEInput, ...]:
    tokenizer = runtime.processor.tokenizer
    values: list[CEInput] = []
    for text in texts:
        if not isinstance(text, str) or not text:
            raise InvariantError("CE reference text must be a non-empty string")
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        values.append(CEInput(input_ids=_single_token_ids(encoded, "reference text"), target_start=1))
    if not values:
        raise PipelineError("CE-loss evaluation requires reference inputs")
    return tuple(values)


def completed_non_refusal_completion_inputs(
    runtime: CERuntime,
    trajectories: Iterable[TargetTrajectory],
    baseline_labels: Mapping[str, str],
) -> tuple[CEInput, ...]:
    values: list[CEInput] = []
    seen_prompt_ids: set[str] = set()
    generation = runtime.config.target_generation
    context_window = runtime.adapter.context_window(runtime.model)
    for trajectory in trajectories:
        if trajectory.prompt_id in seen_prompt_ids:
            raise InvariantError("baseline trajectories contain duplicate prompt ids")
        seen_prompt_ids.add(trajectory.prompt_id)
        if (
            baseline_labels.get(trajectory.prompt_id) != "NON_REFUSAL"
            or trajectory.parser_status != "OK"
            or trajectory.generation_truncated
            or not trajectory.final_answer.strip()
            or not trajectory.raw_generated_token_ids
        ):
            continue
        rendered = runtime.adapter.render_target_chat(
            runtime.processor,
            target_messages(trajectory.original_prompt, generation.system_prompt),
            config=generation,
            prefill_thinking=True,
        )
        prefix_ids = _single_token_ids(rendered, "rendered target chat")
        input_ids = prefix_ids + trajectory.raw_generated_token_ids
        if len(input_ids) > context_window:
            raise InvariantError(
                f"CE completion requires {len(input_ids)} tokens but context window is {context_window}"
            )
        values.append(CEInput(input_ids=input_ids, target_start=len(prefix_ids)))
    if not values:
        raise PipelineError("CE-loss evaluation requires completed baseline NON_REFUSAL trajectories")
    return tuple(values)


def compute_ce_loss(
    runtime: CERuntime,
    inputs: Sequence[CEInput],
    *,
    desc: str = "Evaluating CE loss",
    leave: bool = True,
) -> CELoss:
    if not inputs:
        raise PipelineError("CE-loss evaluation requires reference inputs")
    model = runtime.model
    device = runtime.adapter.input_device(model)
    backbone = runtime.adapter.text_backbone(model)
    head = runtime.adapter.lm_head(model).module
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise InvariantError("CE-loss evaluation LM head has no matrix weight")
    chunk_positions = max(1, 1_048_576 // weight.shape[0])
    total_loss = 0.0
    token_count = 0
    for item in tqdm(
        inputs,
        desc=desc,
        unit="input",
        leave=leave,
        dynamic_ncols=True,
        disable=None,
    ):
        input_ids = torch.tensor((item.input_ids,), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = getattr(output, "last_hidden_state", None)
            if not isinstance(hidden_states, torch.Tensor):
                raise InvariantError("CE-loss evaluation text backbone returned no hidden states")
            shifted_states = hidden_states[:, :-1, :]
            shifted_targets = input_ids[:, 1:].clone()
            shifted_targets[:, : item.target_start - 1] = -100
            for start in tqdm(
                range(0, shifted_states.shape[1], chunk_positions),
                desc=f"{desc}: chunks",
                unit="chunk",
                leave=False,
                dynamic_ncols=True,
                disable=None,
            ):
                stop = min(start + chunk_positions, shifted_states.shape[1])
                logits = head(shifted_states[:, start:stop, :])
                logits = _apply_final_logit_softcap(model, logits)
                targets = shifted_targets[:, start:stop]
                valid = int((targets != -100).sum().item())
                if valid == 0:
                    continue
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss += float(loss.item())
                token_count += valid
    return CELoss(total_loss=total_loss, token_count=token_count)


def compare_ce_losses(baseline: CELoss, intervention: CELoss) -> CEComparison:
    return CEComparison(baseline=baseline, intervention=intervention)


def _single_token_ids(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Mapping) or "input_ids" not in value:
        raise InvariantError(f"{name} has no input_ids")
    data = value["input_ids"]
    if isinstance(data, torch.Tensor):
        data = data.detach().to(device="cpu").tolist()
    elif hasattr(data, "tolist") and not isinstance(data, list | tuple):
        data = data.tolist()
    if isinstance(data, list | tuple) and len(data) == 1 and isinstance(data[0], list | tuple):
        data = data[0]
    if not isinstance(data, list | tuple) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in data
    ):
        raise InvariantError(f"{name} must contain exactly one token sequence")
    return tuple(data)


def _apply_final_logit_softcap(model: torch.nn.Module, logits: torch.Tensor) -> torch.Tensor:
    model_config = getattr(model, "config", None)
    get_text_config = getattr(model_config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else getattr(model_config, "text_config", model_config)
    value = getattr(text_config, "final_logit_softcapping", None)
    if value is None:
        return logits
    softcap = float(value)
    if softcap <= 0:
        raise InvariantError("final logit softcapping must be positive")
    return torch.tanh(logits / softcap) * softcap


__all__ = [
    "CEComparison",
    "CEInput",
    "CELoss",
    "compare_ce_losses",
    "completed_non_refusal_completion_inputs",
    "compute_ce_loss",
    "raw_text_ce_inputs",
]
