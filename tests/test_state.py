"""Task 3 — shared decision-state: no drift, criteria == actions, valid payload.

These tests are DB-independent: they build a CandidateFeatures row directly and
exercise the serializer, so they run without a PostgreSQL server.
"""

from __future__ import annotations

from advisor.candidates import ABSENT_ACTIONS, PRESENT_ACTIONS
from advisor.features import CandidateFeatures
from advisor.state import FEATURE_FIELDS, build_state


def _features(*, index_exists: bool) -> CandidateFeatures:
    return CandidateFeatures(
        table="orders",
        column="status",
        query_frequency=0.22,
        est_selectivity=0.2,
        est_cardinality=500_000.0,
        explain_startup_cost=0.0,
        explain_total_cost=1234.5,
        explain_plan_rows=100_000.0,
        index_exists=index_exists,
        index_scans_observed=7 if index_exists else 0,
        index_size_bytes=4096 if index_exists else 0,
        workload_write_rate=0.2,
    )


def test_two_views_derive_from_same_dict_no_drift():
    state = build_state("scn-1", _features(index_exists=False))
    assert state.views_agree()
    # Every feature field + identifier is present in the tabular row.
    row = state.tabular_row()
    for name in FEATURE_FIELDS:
        assert name in row
    for name in ("scenario_id", "table", "column"):
        assert name in row


def test_criteria_match_allowed_actions_when_absent():
    state = build_state("scn-1", _features(index_exists=False))
    q = state.laya_question()
    assert set(q["criteria"]) == set(ABSENT_ACTIONS)
    assert tuple(state.allowed_actions) == ABSENT_ACTIONS


def test_criteria_match_allowed_actions_when_present():
    state = build_state("scn-1", _features(index_exists=True))
    q = state.laya_question()
    assert set(q["criteria"]) == set(PRESENT_ACTIONS)
    assert tuple(state.allowed_actions) == PRESENT_ACTIONS


def test_laya_payload_shape_matches_router_predict():
    """The payload validates against router.predict's expected shape (mocked)."""
    state = build_state("scn-1", _features(index_exists=False))
    payload = state.laya_payload()

    # Shape: {state: str, questions: [choice-question]}.
    assert set(payload) == {"state", "questions"}
    assert isinstance(payload["state"], str)
    assert isinstance(payload["questions"], list) and len(payload["questions"]) == 1
    q = payload["questions"][0]
    assert q["type"] == "choice"
    assert isinstance(q["instructions"], str) and q["instructions"]
    assert isinstance(q["criteria"], dict) and q["criteria"]

    # A mock Router.predict that enforces exactly this shape must accept it.
    class _FakeRouter:
        def predict(self, state, questions):
            assert isinstance(state, str)
            assert isinstance(questions, list)
            for question in questions:
                assert question["type"] in ("choice", "score", "noul")
                assert "instructions" in question
                assert "criteria" in question
            return [{"answer": next(iter(questions[0]["criteria"])), "prob": 1.0}]

    answers = _FakeRouter().predict(payload["state"], payload["questions"])
    assert answers and answers[0]["answer"] in payload["questions"][0]["criteria"]


def test_state_string_is_valid_json_and_matches_tabular():
    import json

    state = build_state("scn-1", _features(index_exists=True))
    decoded = json.loads(state.laya_state_string())
    assert decoded == state.tabular_row()
