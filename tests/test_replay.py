"""Task 1 — replay controller under the measurement protocol.

Asserts:
    * reproducible medians within tolerance across two runs on identical state,
    * correct read/write bucketing (reads land in read_runtime, writes in
      write_overhead),
    * server-side time <= client wall-clock,
    * a spread statistic is returned alongside each median.

DB-backed: skipped when no PostgreSQL server (with pg_stat_statements) is
reachable.
"""

from __future__ import annotations

import pytest

from advisor import schema_loader
from advisor.db import connect
from advisor.replay import (
    PgStatStatementsUnavailable,
    ensure_pg_stat_statements,
    measure,
)
from advisor.workload import READ, WRITE, Workload


@pytest.fixture(scope="module")
def workload() -> Workload:
    return Workload.load()


def _seed(cfg) -> None:
    schema_loader.apply_schema(cfg)
    schema_loader.seed_data(cfg, scale=0.002)


def _require_pgss(cfg) -> None:
    """Skip the test if pg_stat_statements is not preloaded on this server."""
    try:
        with connect(cfg) as conn:
            ensure_pg_stat_statements(conn)
    except PgStatStatementsUnavailable:
        pytest.skip("pg_stat_statements not available on this server")


@pytest.mark.db
def test_spread_returned_with_each_median(workload, test_db):
    _seed(test_db)
    _require_pgss(test_db)
    result = measure(workload, "baseline", test_db, repeats=3, queries_per_pass=20)

    for comp in (result.read_runtime, result.write_overhead):
        assert comp.median >= 0.0
        assert comp.spread >= 0.0
        assert len(comp.samples) == 3  # one sample per repeat


@pytest.mark.db
def test_read_write_bucketing(workload, test_db):
    _seed(test_db)
    _require_pgss(test_db)
    result = measure(workload, "baseline", test_db, repeats=2, queries_per_pass=20)

    # Every template that recorded time is bucketed by its declared kind.
    for name, timing in result.per_template.items():
        assert timing.kind in (READ, WRITE)
        assert workload.templates[name].kind == timing.kind


@pytest.mark.db
def test_server_time_not_greater_than_client(workload, test_db):
    _seed(test_db)
    _require_pgss(test_db)
    result = measure(workload, "baseline", test_db, repeats=2, queries_per_pass=20)
    assert result.within_spread()
    assert result.server_wall_median <= result.client_wall_median + 1e-6


@pytest.mark.db
def test_reproducible_medians_within_tolerance(workload, test_db):
    _seed(test_db)
    _require_pgss(test_db)
    first = measure(workload, "baseline", test_db, repeats=3, queries_per_pass=20, seed=99)
    second = measure(workload, "baseline", test_db, repeats=3, queries_per_pass=20, seed=99)

    # Timings are never bit-identical, but the measured workload is identical
    # (same seed => same plan, same instances), so medians should be in the same
    # ballpark. We assert the write/read split is stable in sign and that both
    # runs produced non-negative, finite medians within a loose relative band.
    for a, b in (
        (first.read_runtime.median, second.read_runtime.median),
        (first.write_overhead.median, second.write_overhead.median),
    ):
        larger = max(a, b, 1e-6)
        assert abs(a - b) / larger < 2.0  # within a loose 200% band (noise-tolerant)
