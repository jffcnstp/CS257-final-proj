"""Task 6 — cost-score: monotonicity, weight normalization, raw reporting."""

from __future__ import annotations

import math

import pytest

from advisor.cost import COMPONENTS, CostComponents, CostModel


@pytest.fixture()
def model() -> CostModel:
    return CostModel.load()


def _components(**overrides) -> CostComponents:
    base = dict(
        read_runtime=10.0,
        write_overhead=10.0,
        index_storage_bytes=10_000_000.0,
        index_build_time=5.0,
    )
    base.update(overrides)
    return CostComponents(**base)


def test_weights_sum_to_one(model):
    assert math.isclose(model.weight_sum(), 1.0, rel_tol=1e-9)


def test_raw_components_always_reported(model):
    comps = _components()
    result = model.score(comps)
    assert result.raw == comps.as_dict()
    for comp in COMPONENTS:
        assert comp in result.normalized
        assert comp in result.breakdown()


def test_monotonic_in_each_component(model):
    """Increasing any single component must not lower the score."""
    base_score = model.score(_components()).score
    for comp in COMPONENTS:
        worse = model.score(_components(**{comp: _components().as_dict()[comp] * 2 + 1.0})).score
        assert worse >= base_score, f"increasing {comp} lowered the score"


def test_breakdown_exposes_tradeoff(model):
    """Fast reads + slow writes must remain visible in the per-component raw values."""
    fast_reads_slow_writes = model.score(_components(read_runtime=1.0, write_overhead=55.0))
    bd = fast_reads_slow_writes.breakdown()
    assert bd["read_runtime"]["raw"] == 1.0
    assert bd["write_overhead"]["raw"] == 55.0
    # The write contribution should dominate the read contribution here.
    assert bd["write_overhead"]["contribution"] > bd["read_runtime"]["contribution"]


def test_normalization_clamped(model):
    """Values beyond the configured bounds clamp to [0, 1]."""
    huge = model.score(_components(read_runtime=10_000.0))
    assert huge.normalized["read_runtime"] == 1.0
    zero = model.score(_components(read_runtime=0.0))
    assert zero.normalized["read_runtime"] == 0.0
