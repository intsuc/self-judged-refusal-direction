import torch
from torch import nn

from self_judged_refusal_direction.activations import ActivationCollector, OnlineWelford
from self_judged_refusal_direction.schema import ActivationKey


class OffsetBlock(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + 1_000


def test_welford_matches_full_batch_moments_across_updates() -> None:
    values = torch.tensor(
        [
            [1.0, 10.0, -2.0],
            [3.0, 14.0, 4.0],
            [9.0, 18.0, 8.0],
            [11.0, 26.0, 14.0],
        ],
        dtype=torch.float32,
    )
    accumulator = OnlineWelford(dtype=torch.float64)

    accumulator.update(values[:1])
    accumulator.update(values[1:3])
    accumulator.update(values[3:])

    moments = accumulator.snapshot()
    expected = values.to(torch.float64)
    expected_mean = expected.mean(dim=0)
    expected_m2 = (expected - expected_mean).square().sum(dim=0)
    assert moments.count == len(values)
    torch.testing.assert_close(moments.mean, expected_mean)
    torch.testing.assert_close(moments.m2, expected_m2)
    torch.testing.assert_close(moments.variance, expected_m2 / len(values))


def test_collector_reads_last_prefix_token_and_excludes_non_training_labels() -> None:
    block = OffsetBlock()
    collector = ActivationCollector((block,), dtype=torch.float64)
    values = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 3)

    output = collector.collect(
        lambda: block(values),
        ("REFUSAL", "NON_REFUSAL", "UNCERTAIN", "ERROR"),
    )

    torch.testing.assert_close(output, values + 1_000)
    statistics = collector.statistics()
    key = ActivationKey(layer=0)
    refusal = statistics.refusal[key]
    non_refusal = statistics.non_refusal[key]
    assert refusal.count == 1
    assert non_refusal.count == 1
    torch.testing.assert_close(refusal.mean, values[0, -1].to(torch.float64))
    torch.testing.assert_close(non_refusal.mean, values[1, -1].to(torch.float64))
    assert all(moments.count == 1 for moments in statistics.refusal.values())
    assert all(moments.count == 1 for moments in statistics.non_refusal.values())
