"""Shared decision-state serializer — one canonical dict, two views.

The whole point of this module is to guarantee that the tabular models
(scikit-learn / XGBoost) and the Laya typed decision model see the *same*
decision state, serialized two ways **without drift**. Both views are derived
from a single canonical dict (:class:`DecisionState`), so a field can never
appear in one view and silently differ (or be missing) in the other.

Two views:

    * **Tabular view** — a flat dict of numeric/boolean features plus the string
      identifiers, suitable for a pandas row / scikit-learn feature vector.

    * **Laya view** — a ``state`` string (JSON of the same features) plus a
      typed ``choice`` question whose ``criteria`` are EXACTLY the allowed
      actions for the current ``index_exists`` value, matching the
      ``Router().predict(state, questions)`` API documented in the plans.

The allowed-action set comes from the Week-1 candidate rules:
    absent  => {CREATE, NO-OP}
    present => {KEEP,   DROP}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .candidates import (
    ABSENT_ACTIONS,
    CREATE,
    DROP,
    KEEP,
    NO_OP,
    PRESENT_ACTIONS,
)
from .features import CandidateFeatures

# Numeric/boolean feature fields (order is stable for the tabular vector).
FEATURE_FIELDS = (
    "query_frequency",
    "est_selectivity",
    "est_cardinality",
    "explain_startup_cost",
    "explain_total_cost",
    "explain_plan_rows",
    "index_exists",
    "index_scans_observed",
    "index_size_bytes",
    "workload_write_rate",
)

IDENTIFIER_FIELDS = ("scenario_id", "table", "column")

# Human-readable description of each action, used as the Laya choice criteria.
ACTION_DESCRIPTIONS = {
    CREATE: "Create the single-column index on this column.",
    NO_OP:  "Leave the column un-indexed; do not create the index.",
    KEEP:   "Keep the existing index on this column.",
    DROP:   "Drop the existing index on this column.",
}

CHOICE_INSTRUCTIONS = (
    "Given the index decision-state, choose the index-lifecycle action that "
    "minimizes the declared cost objective (read runtime, write overhead, "
    "index storage, and build time)."
)


@dataclass(frozen=True)
class DecisionState:
    """The single canonical decision-state for one (scenario, candidate)."""

    scenario_id: str
    features: CandidateFeatures
    # populated in __post_init__ via object.__setattr__ (frozen dataclass)
    canonical: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical: dict[str, object] = {
            "scenario_id": self.scenario_id,
            "table": self.features.table,
            "column": self.features.column,
        }
        feat = self.features.as_dict()
        for name in FEATURE_FIELDS:
            canonical[name] = feat[name]
        object.__setattr__(self, "canonical", canonical)

    # ------------------------------ actions ------------------------------- #
    @property
    def allowed_actions(self) -> tuple[str, str]:
        """Legal actions for the current index state."""
        return PRESENT_ACTIONS if self.features.index_exists else ABSENT_ACTIONS

    # ---------------------------- tabular view ---------------------------- #
    def tabular_row(self) -> dict[str, object]:
        """Flat feature row for scikit-learn / XGBoost (identifiers included)."""
        # A copy so callers cannot mutate the canonical dict.
        return dict(self.canonical)

    # ----------------------------- Laya view ------------------------------ #
    def laya_state_string(self) -> str:
        """JSON ``state`` string for ``Router().predict`` (same canonical fields)."""
        return json.dumps(self.canonical, sort_keys=True, default=_json_default)

    def laya_question(self) -> dict[str, object]:
        """A typed ``choice`` question whose criteria == the allowed actions."""
        return {
            "type": "choice",
            "instructions": CHOICE_INSTRUCTIONS,
            "criteria": {
                action: ACTION_DESCRIPTIONS[action] for action in self.allowed_actions
            },
        }

    def laya_payload(self) -> dict[str, object]:
        """The full ``{state, questions}`` payload for ``router.predict``."""
        return {
            "state": self.laya_state_string(),
            "questions": [self.laya_question()],
        }

    # --------------------------- drift guarantee -------------------------- #
    def views_agree(self) -> bool:
        """True iff the tabular row and the Laya state encode the same fields.

        This is the structural no-drift guarantee: both views are re-derived
        here from ``canonical`` and compared, so a divergence is impossible to
        introduce silently.
        """
        tabular = self.tabular_row()
        laya = json.loads(self.laya_state_string())
        return tabular == laya


def _json_default(value: object) -> object:
    """Make numpy/decimal-ish values JSON-serializable without importing them."""
    if hasattr(value, "item"):          # numpy scalar
        return value.item()
    return str(value)


def build_state(scenario_id: str, features: CandidateFeatures) -> DecisionState:
    """Convenience constructor for a decision state."""
    return DecisionState(scenario_id=scenario_id, features=features)
