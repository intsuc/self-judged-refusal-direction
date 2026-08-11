import torch

from self_judged_refusal_direction.activations import ActivationMoments, ActivationStatistics
from self_judged_refusal_direction.directions import (
    build_direction_candidates,
    load_candidates,
    load_direction,
    rank_stage_a,
    save_candidates,
    save_direction,
)
from self_judged_refusal_direction.schema import ActivationKey


def moments(mean: list[float], *, count: int = 2, m2: list[float] | None = None) -> ActivationMoments:
    values = torch.tensor(mean, dtype=torch.float64)
    return ActivationMoments(
        count=count,
        mean=values,
        m2=torch.tensor(m2, dtype=torch.float64) if m2 is not None else torch.zeros_like(values),
    )


def test_direction_is_normalized_refusal_minus_non_refusal_mean() -> None:
    key = ActivationKey(phase="pre_thinking", layer=3, relative_position=-1)
    statistics = ActivationStatistics(
        refusal={key: moments([4.0, 2.0], m2=[2.0, 8.0])},
        non_refusal={key: moments([1.0, -2.0], m2=[4.0, 2.0])},
    )

    bundle = build_direction_candidates(statistics, dtype=torch.float64)
    candidate = bundle.candidates[0]

    torch.testing.assert_close(bundle.direction(candidate), torch.tensor([0.6, 0.8], dtype=torch.float64))
    assert candidate.norm == 5.0
    assert candidate.refusal_projected_mean == 4.0
    assert candidate.non_refusal_projected_mean == -1.0
    assert candidate.finite is True


def test_zero_norm_and_nan_candidates_are_not_ranked() -> None:
    valid_key = ActivationKey(phase="pre_thinking", layer=0, relative_position=-1)
    zero_key = ActivationKey(phase="pre_thinking", layer=1, relative_position=-1)
    nan_key = ActivationKey(phase="pre_thinking", layer=2, relative_position=-1)
    statistics = ActivationStatistics(
        refusal={
            valid_key: moments([1.0, 0.0]),
            zero_key: moments([2.0, 2.0]),
            nan_key: moments([float("nan"), 0.0]),
        },
        non_refusal={
            valid_key: moments([0.0, 0.0]),
            zero_key: moments([2.0, 2.0]),
            nan_key: moments([0.0, 0.0]),
        },
    )

    bundle = build_direction_candidates(statistics)
    candidates = {candidate.layer: candidate for candidate in bundle.candidates}

    assert candidates[1].finite is True
    assert candidates[1].norm == 0.0
    assert candidates[2].finite is False
    assert candidates[2].norm == 0.0
    torch.testing.assert_close(bundle.direction(candidates[1]), torch.zeros(2))
    torch.testing.assert_close(bundle.direction(candidates[2]), torch.zeros(2))
    assert rank_stage_a(bundle, top_m=3) == (candidates[0],)


def test_candidate_and_selected_direction_artifacts_round_trip(tmp_path) -> None:
    key = ActivationKey(phase="pre_final", layer=4, relative_position=-2)
    statistics = ActivationStatistics(
        refusal={key: moments([3.0, 4.0])},
        non_refusal={key: moments([0.0, 0.0])},
    )
    bundle = build_direction_candidates(statistics, boundary_tokens={key: "boundary"})
    candidates_path = tmp_path / "candidates.safetensors"
    direction_path = tmp_path / "direction.safetensors"

    save_candidates(candidates_path, bundle)
    restored_bundle = load_candidates(candidates_path)
    save_direction(
        direction_path,
        bundle.directions[0],
        metadata={"candidate_id": bundle.candidates[0].candidate_id},
        private=True,
    )
    restored_direction, restored_metadata = load_direction(direction_path)

    assert restored_bundle.candidates == bundle.candidates
    torch.testing.assert_close(restored_bundle.directions, bundle.directions)
    torch.testing.assert_close(restored_direction, bundle.directions[0])
    assert restored_metadata == {"candidate_id": bundle.candidates[0].candidate_id}
