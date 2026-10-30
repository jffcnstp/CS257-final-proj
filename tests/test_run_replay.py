"""Task 4 — end-to-end scenario dump integration test.

Runs a tiny scenario on a disposable DB and asserts a well-formed artifact with
one decision-state per candidate in BOTH views (tabular + Laya).

DB-backed: skipped when no PostgreSQL server (with pg_stat_statements) is
reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_replay  # noqa: E402
from advisor import schema_loader  # noqa: E402
from advisor.candidates import CandidateSet  # noqa: E402
from advisor.db import connect  # noqa: E402
from advisor.replay import (  # noqa: E402
    PgStatStatementsUnavailable,
    ensure_pg_stat_statements,
    measure,
)
from advisor.workload import Workload  # noqa: E402


def _require_pgss(cfg) -> None:
    try:
        with connect(cfg) as conn:
            ensure_pg_stat_statements(conn)
    except PgStatStatementsUnavailable:
        pytest.skip("pg_stat_statements not available on this server")


@pytest.mark.db
def test_end_to_end_scenario_artifact(test_db):
    schema_loader.apply_schema(test_db)
    schema_loader.seed_data(test_db, scale=0.002)
    _require_pgss(test_db)

    workload = Workload.load()
    candidate_set = CandidateSet.load(workload=workload)
    replay = measure(workload, "baseline", test_db, repeats=2, queries_per_pass=15)

    artifact = run_replay.build_artifact(
        workload, candidate_set, "baseline", "test-scn", replay, test_db
    )

    # One decision-state per candidate.
    assert len(artifact["decisions"]) == len(candidate_set.candidates)

    for decision in artifact["decisions"]:
        # Both views present.
        assert "tabular" in decision
        assert "laya" in decision
        # Tabular carries identifiers + features.
        assert set(("scenario_id", "table", "column")).issubset(decision["tabular"])
        # Laya view is a valid {state, questions} payload.
        laya = decision["laya"]
        assert set(laya) == {"state", "questions"}
        assert isinstance(laya["state"], str)
        q = laya["questions"][0]
        assert q["type"] == "choice"
        # Criteria equal the allowed actions.
        assert set(q["criteria"]) == set(decision["allowed_actions"])

    # Measurement block is well-formed with medians + spreads.
    m = artifact["measurement"]
    assert "read_runtime" in m and "median" in m["read_runtime"] and "spread" in m["read_runtime"]
    assert "write_overhead" in m
    assert m["server_le_client"] is True


@pytest.mark.db
def test_run_replay_main_writes_file(test_db, tmp_path):
    schema_loader.apply_schema(test_db)
    schema_loader.seed_data(test_db, scale=0.002)
    _require_pgss(test_db)

    out = tmp_path / "scenario.json"
    rc = run_replay.main(
        ["--mix", "baseline", "--scenario-id", "t", "--out", str(out),
         "--repeats", "2", "--queries-per-pass", "15"]
    )
    assert rc == 0
    assert out.exists()

    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scenario_id"] == "t"
    assert data["mix"] == "baseline"
    assert len(data["decisions"]) > 0
