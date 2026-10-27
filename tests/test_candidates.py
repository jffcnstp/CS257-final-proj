"""Task 5 — candidates: backed by workload clauses; correct action sets."""

from __future__ import annotations

import pytest

from advisor.candidates import (
    ABSENT_ACTIONS,
    PRESENT_ACTIONS,
    CandidateSet,
)
from advisor.workload import Workload


@pytest.fixture(scope="module")
def workload() -> Workload:
    return Workload.load()


@pytest.fixture(scope="module")
def candidates(workload) -> CandidateSet:
    return CandidateSet.load(workload=workload)


def test_candidates_backed_by_workload_clauses(workload):
    # load() with workload= runs validate_against_workload and raises if any
    # candidate lacks a WHERE/JOIN/GROUP BY/ORDER BY clause.
    cset = CandidateSet.load(workload=workload)
    backed = {(t, c) for (t, c, _clause) in workload.indexable_columns()}
    for cand in cset.candidates:
        assert (cand.table, cand.column) in backed


def test_action_set_absent(candidates):
    for cand in candidates.candidates:
        assert cand.allowed_actions(index_exists=False) == ABSENT_ACTIONS


def test_action_set_present(candidates):
    for cand in candidates.candidates:
        assert cand.allowed_actions(index_exists=True) == PRESENT_ACTIONS


def test_single_column_btree(candidates):
    assert candidates.index_type == "btree"
    # Candidate is identified by exactly one (table, column) pair.
    for cand in candidates.candidates:
        assert cand.table and cand.column


def test_unbacked_candidate_rejected(tmp_path, workload):
    bad = tmp_path / "candidates.yaml"
    bad.write_text(
        "index_type: btree\n"
        "indexable_clauses: [where, join, group_by, order_by]\n"
        "actions:\n"
        "  absent: [CREATE, NO-OP]\n"
        "  present: [KEEP, DROP]\n"
        "candidates:\n"
        "  - { table: customers, column: email }\n",  # email is never in a clause
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not justified"):
        CandidateSet.load(bad, workload=workload)
