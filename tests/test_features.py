"""Task 2 — feature extraction: all fields populate with correct types, and
creating an index flips index_exists and changes size/usage fields.

DB-backed: skipped when no PostgreSQL server is reachable.
"""

from __future__ import annotations

import pytest

from advisor import features, schema_loader
from advisor.candidates import Candidate
from advisor.db import connect
from advisor.features import CandidateFeatures, extract
from advisor.workload import Workload


@pytest.fixture(scope="module")
def workload() -> Workload:
    return Workload.load()


def _seed(cfg) -> None:
    schema_loader.apply_schema(cfg)
    schema_loader.seed_data(cfg, scale=0.002)


def _create_index(cfg, cand: Candidate) -> None:
    with connect(cfg) as conn:
        conn.execute(f"CREATE INDEX {cand.index_name} ON {cand.table} ({cand.column})")


@pytest.mark.db
def test_all_fields_populate_with_correct_types(workload, test_db):
    _seed(test_db)
    cand = Candidate("orders", "status")
    feats = extract(workload, "baseline", cand, test_db)

    assert isinstance(feats, CandidateFeatures)
    assert feats.table == "orders" and feats.column == "status"
    assert isinstance(feats.query_frequency, float) and feats.query_frequency > 0
    assert isinstance(feats.est_selectivity, float) and 0.0 <= feats.est_selectivity <= 1.0
    assert isinstance(feats.est_cardinality, float) and feats.est_cardinality >= 0
    assert isinstance(feats.explain_startup_cost, float)
    assert isinstance(feats.explain_total_cost, float)
    assert isinstance(feats.explain_plan_rows, float)
    assert isinstance(feats.index_exists, bool)
    assert isinstance(feats.index_scans_observed, int)
    assert isinstance(feats.index_size_bytes, int)
    assert isinstance(feats.workload_write_rate, float)


@pytest.mark.db
def test_creating_index_flips_exists_and_changes_size(workload, test_db):
    _seed(test_db)
    cand = Candidate("orders", "status")

    before = extract(workload, "baseline", cand, test_db)
    assert before.index_exists is False
    assert before.index_size_bytes == 0
    assert before.index_scans_observed == 0

    _create_index(test_db, cand)
    # ANALYZE so planner/stat views reflect the new index.
    with connect(test_db) as conn:
        conn.execute("ANALYZE orders")

    after = extract(workload, "baseline", cand, test_db)
    assert after.index_exists is True
    assert after.index_size_bytes > 0


@pytest.mark.db
def test_workload_write_rate_is_shared_across_candidates(workload, test_db):
    _seed(test_db)
    rows = features.extract_all(
        workload,
        "baseline",
        [Candidate("orders", "status"), Candidate("customers", "country")],
        test_db,
    )
    assert len({r.workload_write_rate for r in rows}) == 1
    assert abs(rows[0].workload_write_rate - 0.20) < 1e-9
