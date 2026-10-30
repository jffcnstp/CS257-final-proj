"""Task 1 — scenario generator: count, both index-state starts, factors
recorded, and deterministic regeneration. DB-independent."""

from __future__ import annotations

import pytest

from advisor.scenarios import (
    START_ABSENT,
    START_PRESENT,
    ScenarioConfig,
    breakdown,
    generate,
    projected_dataset_size,
)
from advisor.workload import Workload


@pytest.fixture(scope="module")
def workload() -> Workload:
    return Workload.load()


@pytest.fixture(scope="module")
def cfg() -> ScenarioConfig:
    return ScenarioConfig.load()


def _base_weights(workload: Workload, cfg: ScenarioConfig) -> dict[str, dict[str, float]]:
    return {m: {wt.template.name: wt.weight for wt in workload.mix(m)} for m in cfg.base_mixes}


def test_generates_configured_count(workload, cfg):
    scenarios = generate(cfg, _base_weights(workload, cfg))
    expected = (
        len(cfg.base_mixes)
        * (cfg.mix_variants + 1)
        * len(cfg.start_states)
        * len(cfg.scales)
        * cfg.seeds_per_combo
    )
    assert len(scenarios) == expected


def test_both_index_state_starts_present(workload, cfg):
    scenarios = generate(cfg, _base_weights(workload, cfg))
    starts = {s.start_state for s in scenarios}
    assert START_ABSENT in starts and START_PRESENT in starts


def test_varied_factors_recorded(workload, cfg):
    scenarios = generate(cfg, _base_weights(workload, cfg))
    for s in scenarios:
        assert s.mix in cfg.base_mixes
        assert 0 <= s.mix_variant <= cfg.mix_variants
        assert s.start_state in cfg.start_states
        assert s.scale in cfg.scales
        assert isinstance(s.seed, int) and s.seed > 0
        # Jittered variants carry concrete weight overrides; variant 0 does not.
        if s.mix_variant == 0:
            assert s.weight_overrides == {}
        else:
            assert set(s.weight_overrides)  # non-empty


def test_regeneration_is_identical(workload, cfg):
    bw = _base_weights(workload, cfg)
    first = generate(cfg, bw)
    second = generate(cfg, bw)
    assert [s.as_dict() for s in first] == [s.as_dict() for s in second]


def test_projected_dataset_size_reaches_training_target(workload, cfg):
    scenarios = generate(cfg, _base_weights(workload, cfg))
    size = projected_dataset_size(scenarios, n_candidates=6)
    # The whole point of Task 1: a dataset big enough to fine-tune (hundreds).
    assert size >= 200


def test_breakdown_sums_to_total(workload, cfg):
    scenarios = generate(cfg, _base_weights(workload, cfg))
    bd = breakdown(scenarios)
    assert sum(bd["mix"].values()) == len(scenarios)
    assert sum(bd["start_state"].values()) == len(scenarios)
