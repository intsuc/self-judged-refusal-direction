from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from self_judged_refusal_direction.editing import HookKind, ProjectionKind, WeightEditPlan, WeightEditPlanBuilder


class DiagonalRMSNorm(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.7, 1.3, size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        return (value.float() * scale * self.weight.float()).to(value.dtype)


class ToyModel(nn.Module):
    def __init__(self, tied: bool):
        super().__init__()
        self.embedding = nn.Embedding(17, 6)
        self.multimodal = nn.Linear(4, 6, bias=True)
        self.writer = nn.Linear(6, 6, bias=True)
        self.post_norm = DiagonalRMSNorm(6)
        self.final_norm = DiagonalRMSNorm(6)
        self.lm_head = nn.Linear(6, 17, bias=False)
        self.tied = tied
        if tied:
            self.lm_head.weight = self.embedding.weight

    def tie_weights(self) -> None:
        if self.tied:
            self.lm_head.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor, multimodal: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids) + self.multimodal(multimodal)
        hidden = hidden + self.post_norm(self.writer(hidden))
        return self.lm_head(self.final_norm(hidden))


class ToyAdapter:
    def text_embedding(self, model: ToyModel):
        return SimpleNamespace(name="embedding", module=model.embedding)

    def lm_head(self, model: ToyModel):
        return SimpleNamespace(name="lm_head", module=model.lm_head)

    def residual_writers(self, model: ToyModel):
        return (
            SimpleNamespace(
                name="writer",
                module=model.writer,
                post_norm_name="post_norm",
                post_norm=model.post_norm,
            ),
        )

    def multimodal_projections(self, model: ToyModel):
        return (SimpleNamespace(name="multimodal", module=model.multimodal),)

    def dual_direction(self, post_norm: DiagonalRMSNorm, direction: torch.Tensor) -> torch.Tensor:
        return post_norm.weight.detach().cpu() * direction


@pytest.fixture
def direction() -> torch.Tensor:
    return torch.tensor([0.2, -0.4, 0.1, 0.7, -0.3, 0.5])


def build_plan(model: ToyModel, direction: torch.Tensor) -> WeightEditPlan:
    return (
        WeightEditPlanBuilder(model, direction)
        .add_embedding("embedding")
        .add_lm_head("lm_head")
        .add_multimodal_projection("multimodal")
        .add_residual_writer("writer", effective_scale_name="post_norm.weight")
        .build()
    )


@pytest.mark.parametrize("tied", [False, True])
def test_temporary_and_permanent_logits_match(tied: bool, direction: torch.Tensor) -> None:
    torch.manual_seed(3)
    temporary_model = ToyModel(tied)
    permanent_model = copy.deepcopy(temporary_model)
    plan = build_plan(temporary_model, direction)
    token_ids = torch.tensor([[1, 5, 9], [2, 4, 7]])
    multimodal = torch.randn(2, 3, 4)

    with plan.install_export_equivalent_hooks(temporary_model):
        temporary_logits = temporary_model(token_ids, multimodal)

    plan.apply_in_place(permanent_model, chunk_rows=2)
    permanent_logits = permanent_model(token_ids, multimodal)

    torch.testing.assert_close(temporary_logits, permanent_logits, atol=2e-5, rtol=2e-5)
    assert all(not module._forward_hooks for module in temporary_model.modules())
    assert all(not module._forward_pre_hooks for module in temporary_model.modules())


def test_dual_rmsnorm_projection_is_orthogonal(direction: torch.Tensor) -> None:
    torch.manual_seed(4)
    model = ToyModel(tied=False)
    plan = build_plan(model, direction)
    plan.apply_in_place(model, chunk_rows=2)
    values = torch.randn(5, 6)
    contribution = model.post_norm(model.writer(values))
    unit_direction = plan.direction.to(contribution)

    assert torch.max(torch.abs(contribution @ unit_direction)).item() < 2e-5


def test_chunked_and_unchunked_edits_match(direction: torch.Tensor) -> None:
    torch.manual_seed(5)
    small_chunks = ToyModel(tied=False)
    large_chunks = copy.deepcopy(small_chunks)
    plan = build_plan(small_chunks, direction)

    plan.apply_in_place(small_chunks, chunk_rows=1)
    plan.apply_in_place(large_chunks, chunk_rows=10_000)

    for name, value in small_chunks.state_dict().items():
        torch.testing.assert_close(value, large_chunks.state_dict()[name], atol=2e-7, rtol=2e-6)


def test_tied_embedding_is_one_permanent_operation(direction: torch.Tensor) -> None:
    torch.manual_seed(6)
    model = ToyModel(tied=True)
    original = model.embedding.weight.detach().clone()
    plan = build_plan(model, direction)
    tied_ops = [
        operation for operation in plan.operations if operation.parameter_name in {"embedding.weight", "lm_head.weight"}
    ]

    assert len(tied_ops) == 1
    assert tied_ops[0].projection_kind is ProjectionKind.RIGHT
    assert {(site.module_name, site.hook_kind) for site in tied_ops[0].runtime_sites} == {
        ("embedding", HookKind.OUTPUT),
        ("lm_head", HookKind.INPUT),
    }

    edited = plan.apply_in_place(model, chunk_rows=2)
    unit_direction = plan.direction
    expected = original - (original @ unit_direction).unsqueeze(1) * unit_direction.unsqueeze(0)

    assert edited.count("embedding.weight") == 1
    torch.testing.assert_close(model.embedding.weight, expected, atol=1e-6, rtol=1e-6)
    assert model.embedding.weight is model.lm_head.weight


def test_bias_and_multimodal_projection_are_orthogonal(direction: torch.Tensor) -> None:
    torch.manual_seed(7)
    model = ToyModel(tied=False)
    plan = build_plan(model, direction)
    plan.apply_in_place(model, chunk_rows=2)
    unit_direction = plan.direction
    dual_direction = plan.vector("dual:writer")

    assert torch.abs(model.multimodal.bias @ unit_direction).item() < 1e-6
    assert torch.max(torch.abs(unit_direction @ model.multimodal.weight)).item() < 1e-6
    assert torch.abs(model.writer.bias @ dual_direction).item() < 1e-6
    assert torch.max(torch.abs(dual_direction @ model.writer.weight)).item() < 1e-6


def test_serialization_preserves_operations_and_vectors(direction: torch.Tensor) -> None:
    model = ToyModel(tied=True)
    plan = build_plan(model, direction)
    metadata, tensors = plan.serialize()
    restored = WeightEditPlan.deserialize(metadata, tensors)

    assert restored.to_metadata() == plan.to_metadata()
    assert restored.plan_hash == plan.plan_hash
    for key, value in plan.vectors.items():
        torch.testing.assert_close(restored.vector(key), value)


def test_permanent_edit_uses_fp32_compute(direction: torch.Tensor) -> None:
    model = nn.Module()
    model.embedding = nn.Embedding(9, 6, dtype=torch.bfloat16)
    original = model.embedding.weight.detach().float().clone()
    plan = WeightEditPlanBuilder(model, direction).add_embedding("embedding").build()
    expected = original - torch.sum(original * plan.direction, dim=1, keepdim=True) * plan.direction

    plan.apply_in_place(model, chunk_rows=2)

    torch.testing.assert_close(model.embedding.weight, expected.to(torch.bfloat16), atol=0.0, rtol=0.0)


def test_activation_addition_context_cleans_up(direction: torch.Tensor) -> None:
    model = nn.Sequential(nn.Identity())
    plan = WeightEditPlanBuilder(ToyModel(tied=False), direction).build()
    values = torch.zeros(2, 6)

    with plan.activation_addition(model, "0", coefficient=2.5):
        changed = model(values)

    torch.testing.assert_close(changed, 2.5 * plan.direction.expand_as(values))
    torch.testing.assert_close(model(values), values)
    assert not model[0]._forward_pre_hooks


def test_temporary_hooks_clean_up_after_error(direction: torch.Tensor) -> None:
    model = ToyModel(tied=True)
    plan = build_plan(model, direction)

    with pytest.raises(RuntimeError, match="stop"), plan.temporary(model):
        raise RuntimeError("stop")

    assert all(not module._forward_hooks for module in model.modules())
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_from_adapter_builds_export_equivalent_plan(direction: torch.Tensor) -> None:
    torch.manual_seed(8)
    temporary_model = ToyModel(tied=True)
    permanent_model = copy.deepcopy(temporary_model)
    plan = WeightEditPlan.from_adapter(ToyAdapter(), temporary_model, direction)
    token_ids = torch.tensor([[1, 3, 5]])
    multimodal = torch.randn(1, 3, 4)

    with plan.temporary(temporary_model):
        temporary_logits = temporary_model(token_ids, multimodal)
    plan.apply_in_place(permanent_model, chunk_rows=2)

    torch.testing.assert_close(
        temporary_logits,
        permanent_model(token_ids, multimodal),
        atol=2e-5,
        rtol=2e-5,
    )
    assert plan.operations
