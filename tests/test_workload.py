"""Task 4 — workload: frequency sums, read/write ratio, runnable SQL (EXPLAIN)."""

from __future__ import annotations

import math
import random

import pytest

from advisor.workload import READ, WRITE, Workload, explain_ok


@pytest.fixture(scope="module")
def workload() -> Workload:
    return Workload.load()


def test_two_named_mixes(workload):
    assert set(workload.mix_names) == {"baseline", "shifted"}


def test_frequencies_normalize_to_one(workload):
    for name in workload.mix_names:
        total = sum(wt.weight for wt in workload.mix(name))
        assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_baseline_is_80_20(workload):
    read_frac, write_frac = workload.read_write_ratio("baseline")
    assert math.isclose(read_frac, 0.80, abs_tol=1e-9)
    assert math.isclose(write_frac, 0.20, abs_tol=1e-9)


def test_shifted_mix_differs_from_baseline(workload):
    assert workload.read_write_ratio("shifted") != workload.read_write_ratio("baseline")


def test_every_template_has_read_or_write_kind(workload):
    for tmpl in workload.templates.values():
        assert tmpl.kind in (READ, WRITE)


def test_generate_instance_fills_all_params(workload):
    rng = random.Random(0)
    for name, tmpl in workload.templates.items():
        params = workload.generate_instance(name, rng)
        assert set(params) == set(tmpl.params), f"template {name} params mismatch"


def test_int_offset_is_derived_from_base(workload):
    rng = random.Random(1)
    params = workload.generate_instance("read_group_orders_by_customer", rng)
    assert params["hi"] == params["lo"] + 1000


@pytest.mark.db
def test_every_template_explains_ok(workload, test_db):
    """Every rendered template must EXPLAIN cleanly against the seeded schema."""
    from advisor import schema_loader

    schema_loader.apply_schema(test_db)
    schema_loader.seed_data(test_db, scale=0.001)
    rng = random.Random(42)
    for name in workload.templates:
        assert explain_ok(workload, name, test_db, rng=rng)
