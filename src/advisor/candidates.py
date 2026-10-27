"""Enumerate candidate indexes and the allowed action set per index state.

Rules (config/candidates.yaml):
    * Single-column B-tree indexes only.
    * A column is a candidate only if it appears in a WHERE/JOIN/GROUP BY/
      ORDER BY clause of some workload template.
    * absent  => [CREATE, NO-OP]
      present => [KEEP,   DROP]

The loader cross-checks every declared candidate against the clauses actually
used in config/workload.yaml and refuses a candidate with no backing clause.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from . import CONFIG_DIR
from .workload import Workload

CANDIDATES_YAML = CONFIG_DIR / "candidates.yaml"

# Canonical action names.
CREATE = "CREATE"
NO_OP = "NO-OP"
KEEP = "KEEP"
DROP = "DROP"

ABSENT_ACTIONS = (CREATE, NO_OP)
PRESENT_ACTIONS = (KEEP, DROP)


@dataclass(frozen=True)
class Candidate:
    table: str
    column: str

    @property
    def index_name(self) -> str:
        return f"ix_{self.table}_{self.column}"

    def allowed_actions(self, *, index_exists: bool) -> tuple[str, str]:
        """Return the legal actions given the current index state."""
        return PRESENT_ACTIONS if index_exists else ABSENT_ACTIONS


class CandidateSet:
    """Parsed candidate-index configuration."""

    def __init__(self, candidates: list[Candidate], index_type: str, clauses: list[str]):
        self.candidates = candidates
        self.index_type = index_type
        self.indexable_clauses = clauses

    @classmethod
    def load(
        cls,
        path: Path = CANDIDATES_YAML,
        *,
        workload: Workload | None = None,
    ) -> CandidateSet:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        clauses = list(raw["indexable_clauses"])
        candidates = [Candidate(c["table"], c["column"]) for c in raw["candidates"]]

        # Validate action mapping matches the required semantics.
        actions = raw["actions"]
        if tuple(actions["absent"]) != ABSENT_ACTIONS:
            raise ValueError(f"absent actions must be {ABSENT_ACTIONS}, got {actions['absent']}")
        if tuple(actions["present"]) != PRESENT_ACTIONS:
            raise ValueError(f"present actions must be {PRESENT_ACTIONS}, got {actions['present']}")

        cset = cls(candidates, raw.get("index_type", "btree"), clauses)
        if workload is not None:
            cset.validate_against_workload(workload)
        return cset

    def validate_against_workload(self, workload: Workload) -> None:
        """Ensure every candidate is backed by an indexable clause in the workload."""
        backed = workload.indexable_columns()  # set of (table, column, clause)
        backed_pairs = {(t, c) for (t, c, _clause) in backed}
        allowed_clauses = set(self.indexable_clauses)

        # Every indexable ref in the workload must use a permitted clause.
        for (table, column, clause) in backed:
            if clause not in allowed_clauses:
                raise ValueError(
                    f"workload references {table}.{column} via unsupported clause {clause!r}"
                )

        for cand in self.candidates:
            if (cand.table, cand.column) not in backed_pairs:
                raise ValueError(
                    f"candidate {cand.table}.{cand.column} is not justified by any "
                    f"WHERE/JOIN/GROUP BY/ORDER BY clause in the workload"
                )

    def describe(
        self, *, index_exists: bool = False
    ) -> list[tuple[str, str, bool, tuple[str, str]]]:
        """Return [(index_name, 'table.column', index_exists, allowed_actions)]."""
        return [
            (
                c.index_name,
                f"{c.table}.{c.column}",
                index_exists,
                c.allowed_actions(index_exists=index_exists),
            )
            for c in self.candidates
        ]
