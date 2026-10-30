"""Controlled index-comparison labeler — the reference-policy action labels.

For each (scenario, candidate) the labeler measures the real outcome of every
allowed action under the *declared cost objective* (Week-1 `cost_score.yaml`)
and records the lowest-cost action as the **reference action**. This is a
reference under a stated objective, not objective ground truth.

Method (differential / A/B measurement):
    1. The resulting database state of an action is binary for a single-column
       candidate — the index is either PRESENT (CREATE, KEEP) or ABSENT
       (NO-OP, DROP). We therefore measure each distinct resulting state once
       under the Week-2 measurement protocol and reuse it across the actions
       that share it.
    2. Index DDL is applied and TIMED OUTSIDE the measured read/write window, so
       `index_build_time` never contaminates `write_overhead`. Index storage is
       read from the catalog after the build.
    3. Because the same workload is measured on cloned states differing only in
       the index, shared overhead cancels in the comparison.

Label confidence from the measured spread:
    * clear win  (best score beats the alternative by more than the combined
      measured spread) -> a confident single answer (`expected`).
    * near-tie   (scores within the spread) -> a probability distribution over
      the actions (`gold`), derived from the score gap, rather than a forced
      hard winner.

Both label shapes are carried to the Week-3 export: `expected` for a confident
decision, `gold` for a near-tie — the two shapes Laya's `laya-train` reads.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .candidates import CREATE, DROP, KEEP, NO_OP, Candidate
from .cost import CostComponents, CostModel, CostResult
from .db import DBConfig, connect
from .replay import ReplayResult, measure
from .workload import Workload

# Resulting index state for each action.
_PRESENT_RESULT = {CREATE, KEEP}
_ABSENT_RESULT = {NO_OP, DROP}


@dataclass(frozen=True)
class ActionOutcome:
    """Measured outcome + score for a single action on a candidate."""

    action: str
    components: CostComponents
    score: float
    score_spread: float             # propagated measurement spread on the score
    index_present_after: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "score": self.score,
            "score_spread": self.score_spread,
            "index_present_after": self.index_present_after,
            "raw": self.components.as_dict(),
        }


@dataclass(frozen=True)
class CandidateLabel:
    """The reference label for one (scenario, candidate)."""

    table: str
    column: str
    start_index_exists: bool
    allowed_actions: tuple[str, ...]
    reference_action: str
    confident: bool                 # True => expected; False => gold distribution
    probabilities: dict[str, float] # over allowed_actions; one-hot when confident
    outcomes: dict[str, ActionOutcome]

    def as_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "column": self.column,
            "start_index_exists": self.start_index_exists,
            "allowed_actions": list(self.allowed_actions),
            "reference_action": self.reference_action,
            "confident": self.confident,
            "probabilities": self.probabilities,
            "outcomes": {a: o.as_dict() for a, o in self.outcomes.items()},
        }


# --------------------------------------------------------------------------- #
# Index DDL — applied/timed OUTSIDE the measured window
# --------------------------------------------------------------------------- #
def index_exists(conn, candidate: Candidate) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname='public' "
        "AND tablename=%s AND indexname=%s",
        (candidate.table, candidate.index_name),
    ).fetchone()
    return row is not None


def create_index_timed(conn, candidate: Candidate) -> float:
    """CREATE the candidate index; return build time in seconds (0 if present)."""
    if index_exists(conn, candidate):
        return 0.0
    # Settle any implicit transaction opened by the existence check before
    # switching to autocommit (CREATE INDEX must not run in a transaction block
    # here, and psycopg forbids toggling autocommit mid-transaction).
    conn.commit()
    old_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        start = time.perf_counter()
        conn.execute(
            f"CREATE INDEX {candidate.index_name} "
            f"ON {candidate.table} ({candidate.column})"
        )
        build = time.perf_counter() - start
    finally:
        conn.autocommit = old_autocommit
    return build


def drop_index(conn, candidate: Candidate) -> None:
    conn.commit()
    old_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        conn.execute(f"DROP INDEX IF EXISTS {candidate.index_name}")
    finally:
        conn.autocommit = old_autocommit


def index_size_bytes(conn, candidate: Candidate) -> int:
    row = conn.execute(
        "SELECT COALESCE(pg_relation_size(c.oid), 0) FROM pg_class c "
        "WHERE c.relname = %s",
        (candidate.index_name,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


# --------------------------------------------------------------------------- #
# Scoring a resulting state
# --------------------------------------------------------------------------- #
def _score_result(
    cost_model: CostModel,
    replay: ReplayResult,
    build_time: float,
    storage_bytes: int,
) -> tuple[CostResult, float]:
    """Score a measured resulting state and propagate a spread onto the score."""
    comps = CostComponents(
        read_runtime=replay.read_runtime.median,
        write_overhead=replay.write_overhead.median,
        index_storage_bytes=float(storage_bytes),
        index_build_time=float(build_time),
    )
    result = cost_model.score(comps)
    # Propagate the read/write measurement spread through the (linear) score so a
    # near-tie can be judged against measurement noise, not an arbitrary epsilon.
    b = cost_model.bounds
    w = cost_model.weights
    read_span = float(b["read_runtime"]["max"]) - float(b["read_runtime"]["min"])
    write_span = float(b["write_overhead"]["max"]) - float(b["write_overhead"]["min"])
    score_spread = 0.0
    if read_span > 0:
        score_spread += w["w_r"] * (replay.read_runtime.spread / read_span)
    if write_span > 0:
        score_spread += w["w_w"] * (replay.write_overhead.spread / write_span)
    return result, score_spread


def _measure_state(
    workload: Workload,
    mix: str,
    cfg: DBConfig,
    *,
    repeats: int,
    queries_per_pass: int,
    seed: int,
) -> ReplayResult:
    return measure(
        workload, mix, cfg,
        repeats=repeats, queries_per_pass=queries_per_pass, seed=seed,
    )


# --------------------------------------------------------------------------- #
# Labeling a candidate
# --------------------------------------------------------------------------- #
def label_candidate(
    workload: Workload,
    mix: str,
    candidate: Candidate,
    cfg: DBConfig | None = None,
    *,
    start_index_exists: bool,
    repeats: int = 3,
    queries_per_pass: int = 20,
    seed: int = 257,
) -> CandidateLabel:
    """Measure every allowed action for ``candidate`` and pick the reference.

    The DB at ``cfg`` is left with the index in its ORIGINAL ``start_index_exists``
    state on return, so callers can label candidates independently.
    """
    cfg = cfg or DBConfig.from_env()
    allowed = candidate.allowed_actions(index_exists=start_index_exists)

    # Measure both distinct resulting states (present / absent) at most once.
    present_replay: ReplayResult | None = None
    absent_replay: ReplayResult | None = None
    build_time_present = 0.0
    storage_present = 0

    needs_present = any(a in _PRESENT_RESULT for a in allowed)
    needs_absent = any(a in _ABSENT_RESULT for a in allowed)

    with connect(cfg) as conn:
        original_present = index_exists(conn, candidate)

    # --- resulting state: index PRESENT -------------------------------------
    if needs_present:
        with connect(cfg) as conn:
            build_time_present = create_index_timed(conn, candidate)
            storage_present = index_size_bytes(conn, candidate)
        present_replay = _measure_state(
            workload, mix, cfg,
            repeats=repeats, queries_per_pass=queries_per_pass, seed=seed,
        )

    # --- resulting state: index ABSENT --------------------------------------
    if needs_absent:
        with connect(cfg) as conn:
            drop_index(conn, candidate)
        absent_replay = _measure_state(
            workload, mix, cfg,
            repeats=repeats, queries_per_pass=queries_per_pass, seed=seed,
        )

    # Restore the original index state so candidate labeling is independent.
    with connect(cfg) as conn:
        if original_present and not index_exists(conn, candidate):
            create_index_timed(conn, candidate)
        elif not original_present and index_exists(conn, candidate):
            drop_index(conn, candidate)

    cost_model = CostModel.load()
    outcomes: dict[str, ActionOutcome] = {}
    for action in allowed:
        if action in _PRESENT_RESULT:
            assert present_replay is not None
            # KEEP reuses an already-built index (no new build cost); CREATE
            # pays the build. Both carry the index's storage.
            build = build_time_present if action == CREATE else 0.0
            result, spread = _score_result(
                cost_model, present_replay, build, storage_present
            )
            outcomes[action] = ActionOutcome(
                action=action,
                components=CostComponents(**result.raw),
                score=result.score,
                score_spread=spread,
                index_present_after=True,
            )
        else:  # NO-OP / DROP -> index absent, zero storage + build
            assert absent_replay is not None
            result, spread = _score_result(cost_model, absent_replay, 0.0, 0)
            outcomes[action] = ActionOutcome(
                action=action,
                components=CostComponents(**result.raw),
                score=result.score,
                score_spread=spread,
                index_present_after=False,
            )

    return _decide(candidate, start_index_exists, allowed, outcomes)


def _decide(
    candidate: Candidate,
    start_index_exists: bool,
    allowed: tuple[str, ...],
    outcomes: dict[str, ActionOutcome],
) -> CandidateLabel:
    """Pick the min-cost action; emit confident vs near-tie label from spread."""
    ranked = sorted(allowed, key=lambda a: outcomes[a].score)
    best, second = ranked[0], ranked[1]
    gap = outcomes[second].score - outcomes[best].score
    # Combined measurement spread of the two closest actions.
    combined_spread = outcomes[best].score_spread + outcomes[second].score_spread

    confident = gap > combined_spread
    if confident:
        probs = {a: (1.0 if a == best else 0.0) for a in allowed}
    else:
        probs = _softmin_probabilities(allowed, outcomes, combined_spread)

    return CandidateLabel(
        table=candidate.table,
        column=candidate.column,
        start_index_exists=start_index_exists,
        allowed_actions=tuple(allowed),
        reference_action=best,
        confident=confident,
        probabilities=probs,
        outcomes=outcomes,
    )


def _softmin_probabilities(
    allowed: tuple[str, ...],
    outcomes: dict[str, ActionOutcome],
    spread: float,
) -> dict[str, float]:
    """Teacher probabilities over actions (lower score => higher prob).

    Uses a soft-min with a temperature tied to the measured spread, so two
    near-tied actions get near-equal probability and a clearer gap sharpens the
    distribution. Falls back to a small temperature when spread is ~0.
    """
    temp = spread if spread > 1e-9 else 1e-3
    best_score = min(outcomes[a].score for a in allowed)
    weights = {a: math.exp(-(outcomes[a].score - best_score) / temp) for a in allowed}
    total = sum(weights.values())
    return {a: round(weights[a] / total, 6) for a in allowed}
