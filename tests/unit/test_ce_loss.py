from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from self_judged_refusal_direction.ce_loss import (
    CEInput,
    completed_non_refusal_completion_inputs,
    compute_ce_loss,
)
from self_judged_refusal_direction.config import ProjectConfig
from self_judged_refusal_direction.models.base import ModuleTarget
from self_judged_refusal_direction.schema import TargetTrajectory


class ToyBackbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=F.one_hot(input_ids, num_classes=5).float())


class ToyAdapter:
    def __init__(self, backbone: nn.Module, head: nn.Module):
        self.backbone = backbone
        self.head = head

    def input_device(self, _model: nn.Module) -> torch.device:
        return torch.device("cpu")

    def text_backbone(self, _model: nn.Module) -> nn.Module:
        return self.backbone

    def lm_head(self, _model: nn.Module) -> ModuleTarget:
        return ModuleTarget("head", self.head)


def test_ce_loss_uses_only_target_tokens_and_weights_them_equally() -> None:
    backbone = ToyBackbone()
    head = nn.Linear(5, 5, bias=False)
    with torch.no_grad():
        head.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
    runtime = SimpleNamespace(
        model=nn.Module(),
        adapter=ToyAdapter(backbone, head),
    )
    inputs = (
        CEInput(input_ids=(0, 1), target_start=1),
        CEInput(input_ids=(0, 1, 2, 3), target_start=2),
    )

    result = compute_ce_loss(cast(Any, runtime), inputs)
    with torch.no_grad():
        first = F.cross_entropy(head(F.one_hot(torch.tensor([0]), num_classes=5).float()), torch.tensor([1]))
        second = F.cross_entropy(
            head(F.one_hot(torch.tensor([1, 2]), num_classes=5).float()),
            torch.tensor([2, 3]),
            reduction="sum",
        )

    assert result.token_count == 3
    assert result.total_loss == pytest.approx(float(first + second))
    assert result.mean_loss == pytest.approx(float((first + second) / 3))


class CompletionAdapter:
    def context_window(self, _model: object) -> int:
        return 32

    def render_target_chat(self, _processor: object, messages: object, **_kwargs: object) -> dict[str, Any]:
        prompt_length = len(cast(list[dict[str, str]], messages)[-1]["content"])
        return {"input_ids": torch.tensor([[7, prompt_length]])}


def _trajectory(
    prompt_id: str,
    *,
    truncated: bool,
    parser_status: str = "OK",
    final_answer: str = "output",
) -> TargetTrajectory:
    return TargetTrajectory(
        prompt_id=prompt_id,
        original_prompt=prompt_id,
        raw_generated_token_ids=(3, 4),
        raw_decoded_output="output",
        thinking_text="",
        final_answer=final_answer,
        thinking_token_start=0,
        thinking_token_end=0,
        final_token_start=0,
        final_token_end=2,
        generation_truncated=truncated,
        parser_status=cast(Any, parser_status),
        model_revision="revision",
        generation_config_hash="generation",
        trajectory_hash=prompt_id,
    )


def test_completion_inputs_keep_chat_context_and_exclude_incomplete_baselines() -> None:
    runtime = SimpleNamespace(
        config=ProjectConfig(),
        adapter=CompletionAdapter(),
        processor=object(),
        model=object(),
    )
    trajectories = (
        _trajectory("complete", truncated=False),
        _trajectory("truncated", truncated=True),
        _trajectory("refusal", truncated=False),
        _trajectory("parse-error", truncated=False, parser_status="ERROR"),
        _trajectory("empty", truncated=False, final_answer="  "),
    )
    labels = {
        "complete": "NON_REFUSAL",
        "truncated": "NON_REFUSAL",
        "refusal": "REFUSAL",
        "parse-error": "NON_REFUSAL",
        "empty": "NON_REFUSAL",
    }

    inputs = completed_non_refusal_completion_inputs(cast(Any, runtime), trajectories, labels)

    assert inputs == (CEInput(input_ids=(7, 8, 3, 4), target_start=2),)
