"""Per-candidate feature extraction from PostgreSQL.

Collects every decision-state field for a single candidate index, directly from
the live database. These are model **features** (inputs), distinct from the
measured cost **outcomes** produced by :mod:`advisor.replay`.

Important distinction kept throughout: the ``explain_*`` fields are PostgreSQL's
*estimated* costs (via ``EXPLAIN (FORMAT JSON)``). They are collected only as
features — the very quantities we are testing the models' ability to improve
upon — and are NEVER used as a measured outcome.

Fields collected (the shared decision-state, see docs/experiment-spec.md):
    query_frequency      — candidate's column frequency in the workload mix
    est_selectivity      — estimated fraction of rows the probe query returns
    est_cardinality      — estimated rows in the candidate's table (reltuples)
    explain_startup_cost — EXPLAIN estimated startup cost (feature, not outcome)
    explain_total_cost   — EXPLAIN estimated total cost   (feature, not outcome)
    explain_plan_rows    — EXPLAIN estimated plan rows     (feature, not outcome)
    index_exists         — whether the candidate index currently exists
    index_scans_observed — idx_scan from pg_stat_user_indexes (0 if absent)
    index_size_bytes     — pg_relation_size of the index    (0 if absent)
    workload_write_rate  — write fraction of the mix (shared across candidates)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .candidates import Candidate
from .db import DBConfig, connect
from .workload import WRITE, Workload


@dataclass(frozen=True)
class CandidateFeatures:
    """Decision-state feature row for one (scenario, candidate)."""

    table: str
    column: str
    query_frequency: float
    est_selectivity: float
    est_cardinality: float
    explain_startup_cost: float
    explain_total_cost: float
    explain_plan_rows: float
    index_exists: bool
    index_scans_observed: int
    index_size_bytes: int
    workload_write_rate: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Workload-derived features (no DB round-trip)
# --------------------------------------------------------------------------- #
def column_frequency(workload: Workload, mix: str, table: str, column: str) -> float:
    """Summed normalized frequency of templates that index ``table.column``."""
    freq = 0.0
    for wt in workload.mix(mix):
        for ref in wt.template.indexable:
            if ref["table"] == table and ref["column"] == column:
                freq += wt.weight
                break
    return freq


def workload_write_rate(workload: Workload, mix: str) -> float:
    """Fraction of the mix that is write templates (shared across candidates)."""
    rate = 0.0
    for wt in workload.mix(mix):
        if wt.template.kind == WRITE:
            rate += wt.weight
    return rate


# --------------------------------------------------------------------------- #
# Catalog / stats features (DB round-trips)
# --------------------------------------------------------------------------- #
def index_exists(conn, candidate: Candidate) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = %s
          AND indexname = %s
        """,
        (candidate.table, candidate.index_name),
    ).fetchone()
    return row is not None


def index_usage_and_size(conn, candidate: Candidate) -> tuple[int, int]:
    """Return (index_scans_observed, index_size_bytes); (0, 0) if the index is absent."""
    row = conn.execute(
        """
        SELECT COALESCE(s.idx_scan, 0) AS idx_scan,
               COALESCE(pg_relation_size(s.indexrelid), 0) AS size_bytes
        FROM pg_stat_user_indexes s
        JOIN pg_class c ON c.oid = s.indexrelid
        WHERE c.relname = %s
        """,
        (candidate.index_name,),
    ).fetchone()
    if row is None:
        return 0, 0
    return int(row[0]), int(row[1])


def table_cardinality(conn, table: str) -> float:
    """Estimated live-row count from pg_class.reltuples (planner's view).

    PostgreSQL stores ``reltuples = -1`` for a table that has never been
    ANALYZEd (meaning "unknown"). In that case we fall back to an exact
    ``COUNT(*)`` so the feature is always a sane, non-negative cardinality.
    """
    row = conn.execute(
        "SELECT reltuples FROM pg_class WHERE relname = %s",
        (table,),
    ).fetchone()
    reltuples = float(row[0]) if row and row[0] is not None else -1.0
    if reltuples < 0:
        count_row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return float(count_row[0]) if count_row else 0.0
    return reltuples


def _probe_sql(candidate: Candidate) -> str:
    """A representative single-column probe used to read EXPLAIN estimates.

    A simple equality/range probe on the candidate column is enough to make the
    planner surface a selectivity estimate and a plan cost for that column.
    """
    return (
        f"SELECT * FROM {candidate.table} "
        f"WHERE {candidate.column} = "
        f"(SELECT {candidate.column} FROM {candidate.table} LIMIT 1)"
    )


def explain_estimates(conn, candidate: Candidate) -> tuple[float, float, float]:
    """Return (startup_cost, total_cost, plan_rows) from EXPLAIN (FORMAT JSON).

    These are planner ESTIMATES collected as features only — never a measured
    outcome.
    """
    rows = conn.execute(
        f"EXPLAIN (FORMAT JSON) {_probe_sql(candidate)}"
    ).fetchone()
    plan = rows[0][0]["Plan"]
    return (
        float(plan.get("Startup Cost", 0.0)),
        float(plan.get("Total Cost", 0.0)),
        float(plan.get("Plan Rows", 0.0)),
    )


def estimated_selectivity(conn, candidate: Candidate) -> float:
    """Estimated fraction of the table the probe returns (plan_rows / reltuples)."""
    _startup, _total, plan_rows = explain_estimates(conn, candidate)
    card = table_cardinality(conn, candidate.table)
    if card <= 0:
        return 0.0
    return min(1.0, plan_rows / card)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract(
    workload: Workload,
    mix: str,
    candidate: Candidate,
    cfg: DBConfig | None = None,
    *,
    conn=None,
) -> CandidateFeatures:
    """Collect the full feature row for one candidate under ``mix``.

    Pass an open ``conn`` to reuse a connection (e.g. the scenario dump loops
    over all candidates); otherwise a short-lived connection is opened.
    """
    cfg = cfg or DBConfig.from_env()
    if conn is not None:
        return _extract_with_conn(workload, mix, candidate, conn)
    with connect(cfg) as own_conn:
        return _extract_with_conn(workload, mix, candidate, own_conn)


def _extract_with_conn(
    workload: Workload, mix: str, candidate: Candidate, conn
) -> CandidateFeatures:
    exists = index_exists(conn, candidate)
    scans, size_bytes = index_usage_and_size(conn, candidate)
    startup, total, plan_rows = explain_estimates(conn, candidate)
    card = table_cardinality(conn, candidate.table)
    selectivity = min(1.0, plan_rows / card) if card > 0 else 0.0

    return CandidateFeatures(
        table=candidate.table,
        column=candidate.column,
        query_frequency=column_frequency(workload, mix, candidate.table, candidate.column),
        est_selectivity=selectivity,
        est_cardinality=card,
        explain_startup_cost=startup,
        explain_total_cost=total,
        explain_plan_rows=plan_rows,
        index_exists=exists,
        index_scans_observed=scans,
        index_size_bytes=size_bytes,
        workload_write_rate=workload_write_rate(workload, mix),
    )


def extract_all(
    workload: Workload,
    mix: str,
    candidates: list[Candidate],
    cfg: DBConfig | None = None,
) -> list[CandidateFeatures]:
    """Feature rows for every candidate under ``mix`` using one connection."""
    cfg = cfg or DBConfig.from_env()
    with connect(cfg) as conn:
        return [_extract_with_conn(workload, mix, c, conn) for c in candidates]
