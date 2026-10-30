"""Non-learned baseline advisors the learned models must beat.

Two advisors:

    * :class:`NoIndexAdvisor` — the lower-bound policy that never maintains a
      candidate index. It chooses NO-OP when the index is absent and DROP when
      it is present (always ending in the no-index state).

    * :class:`RuleBasedAdvisor` — a transparent heuristic driven by thresholds
      in ``config/baselines.yaml``:
        absent  -> CREATE if query_frequency & selectivity & write-rate pass the
                   thresholds, else NO-OP.
        present -> KEEP if observed index scans are high enough, else DROP.

Both advisors decide from a candidate's decision-state features (the same
:class:`CandidateFeatures` the models consume), so a baseline decision is a pure
function of the recorded state and is therefore reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from . import CONFIG_DIR
from .candidates import CREATE, DROP, KEEP, NO_OP
from .features import CandidateFeatures

BASELINES_YAML = CONFIG_DIR / "baselines.yaml"


@dataclass(frozen=True)
class Decision:
    """A baseline's chosen action for a candidate, with the triggering reason."""

    table: str
    column: str
    action: str
    rule: str                       # human-readable reason, for the demo/report


class NoIndexAdvisor:
    """Never maintains candidate indexes: NO-OP if absent, DROP if present."""

    name = "no_index"

    def decide(self, feats: CandidateFeatures) -> Decision:
        if feats.index_exists:
            return Decision(feats.table, feats.column, DROP, "no-index policy: drop")
        return Decision(feats.table, feats.column, NO_OP, "no-index policy: no-op")

    def decide_all(self, rows: list[CandidateFeatures]) -> list[Decision]:
        return [self.decide(r) for r in rows]


@dataclass(frozen=True)
class RuleThresholds:
    min_query_frequency: float
    max_est_selectivity: float
    max_write_rate: float
    min_index_scans: int

    @classmethod
    def load(cls, path: Path = BASELINES_YAML) -> RuleThresholds:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["rule_based"]
        return cls(
            min_query_frequency=float(raw["min_query_frequency"]),
            max_est_selectivity=float(raw["max_est_selectivity"]),
            max_write_rate=float(raw["max_write_rate"]),
            min_index_scans=int(raw["min_index_scans"]),
        )


class RuleBasedAdvisor:
    """Transparent heuristic advisor driven by configured thresholds."""

    name = "rule_based"

    def __init__(self, thresholds: RuleThresholds | None = None):
        self.t = thresholds or RuleThresholds.load()

    def decide(self, feats: CandidateFeatures) -> Decision:
        t = self.t
        if feats.index_exists:
            if feats.index_scans_observed >= t.min_index_scans:
                return Decision(
                    feats.table, feats.column, KEEP,
                    f"present & index_scans {feats.index_scans_observed} "
                    f">= {t.min_index_scans} -> KEEP",
                )
            return Decision(
                feats.table, feats.column, DROP,
                f"present & index_scans {feats.index_scans_observed} "
                f"< {t.min_index_scans} -> DROP",
            )

        # Absent: decide CREATE vs NO-OP from the three gating conditions.
        freq_ok = feats.query_frequency >= t.min_query_frequency
        sel_ok = feats.est_selectivity <= t.max_est_selectivity
        write_ok = feats.workload_write_rate < t.max_write_rate
        if freq_ok and sel_ok and write_ok:
            return Decision(
                feats.table, feats.column, CREATE,
                f"absent & freq {feats.query_frequency:.3f}>="
                f"{t.min_query_frequency} & sel {feats.est_selectivity:.3f}<="
                f"{t.max_est_selectivity} & write {feats.workload_write_rate:.3f}<"
                f"{t.max_write_rate} -> CREATE",
            )
        reasons = []
        if not freq_ok:
            reasons.append("freq low")
        if not sel_ok:
            reasons.append("not selective")
        if not write_ok:
            reasons.append("write-heavy")
        return Decision(
            feats.table, feats.column, NO_OP,
            f"absent & ({', '.join(reasons)}) -> NO-OP",
        )

    def decide_all(self, rows: list[CandidateFeatures]) -> list[Decision]:
        return [self.decide(r) for r in rows]
