"""Parse config/workload.yaml and generate concrete query instances.

A *template* is a parameterized SQL statement (``%(name)s`` placeholders) tagged
read/write, with per-parameter generators. A *mix* assigns relative frequency
weights to a subset of templates; weights are normalized to fractions.

The loader is DB-independent: it parses YAML, validates structure, normalizes
frequencies, and renders concrete parameter sets deterministically. The only
DB-touching helper, :func:`explain_ok`, is optional and used by DB-backed tests.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from . import CONFIG_DIR
from .db import DBConfig, connect

WORKLOAD_YAML = CONFIG_DIR / "workload.yaml"

READ = "read"
WRITE = "write"


@dataclass(frozen=True)
class Template:
    name: str
    kind: str                      # READ or WRITE
    sql: str
    params: dict                   # param-name -> generator spec
    tables: list[str]
    indexable: list[dict]          # [{table, column, clause}, ...]


@dataclass(frozen=True)
class WeightedTemplate:
    template: Template
    weight: float                  # normalized fraction within its mix


class Workload:
    """Parsed workload: templates + named mixes."""

    def __init__(self, templates: dict[str, Template], mixes: dict[str, dict[str, float]]):
        self.templates = templates
        self._mixes = mixes

    # ----------------------------- construction -----------------------------
    @classmethod
    def load(cls, path: Path = WORKLOAD_YAML) -> Workload:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        templates: dict[str, Template] = {}
        for name, spec in raw["templates"].items():
            kind = spec["kind"]
            if kind not in (READ, WRITE):
                raise ValueError(f"template {name!r}: kind must be 'read' or 'write'")
            templates[name] = Template(
                name=name,
                kind=kind,
                sql=spec["sql"].strip(),
                params=spec.get("params", {}) or {},
                tables=list(spec.get("tables", [])),
                indexable=list(spec.get("indexable", []) or []),
            )

        mixes: dict[str, dict[str, float]] = {}
        for mix_name, weights in raw["mixes"].items():
            unknown = set(weights) - set(templates)
            if unknown:
                raise ValueError(
                    f"mix {mix_name!r} references unknown templates: {sorted(unknown)}"
                )
            mixes[mix_name] = {k: float(v) for k, v in weights.items()}

        return cls(templates, mixes)

    # ------------------------------- queries --------------------------------
    @property
    def mix_names(self) -> list[str]:
        return list(self._mixes)

    def mix(self, name: str) -> list[WeightedTemplate]:
        """Return the mix as weighted templates with weights normalized to sum 1."""
        if name not in self._mixes:
            raise KeyError(f"unknown mix {name!r}; known: {self.mix_names}")
        raw_weights = self._mixes[name]
        total = float(sum(raw_weights.values()))
        if total <= 0:
            raise ValueError(f"mix {name!r} has non-positive total weight")
        return [
            WeightedTemplate(self.templates[tname], w / total)
            for tname, w in raw_weights.items()
        ]

    def read_write_ratio(self, name: str) -> tuple[float, float]:
        """Return (read_fraction, write_fraction) for a mix (sums to 1)."""
        read_frac = 0.0
        write_frac = 0.0
        for wt in self.mix(name):
            if wt.template.kind == READ:
                read_frac += wt.weight
            else:
                write_frac += wt.weight
        return read_frac, write_frac

    def indexable_columns(self) -> set[tuple[str, str, str]]:
        """Return the set of (table, column, clause) justified by the workload."""
        out: set[tuple[str, str, str]] = set()
        for tmpl in self.templates.values():
            for ref in tmpl.indexable:
                out.add((ref["table"], ref["column"], ref["clause"]))
        return out

    # --------------------------- instance rendering --------------------------
    def generate_instance(
        self,
        template_name: str,
        rng: random.Random,
        *,
        int_bounds: dict[str, int] | None = None,
    ) -> dict[str, object]:
        """Return a concrete parameter dict for a template using ``rng``.

        ``int_bounds`` optionally caps the upper bound of named ``int``-generator
        parameters. The replay controller uses this to keep write parameters
        (e.g. FK ``customer_id`` / ``order_id``) within the rows that actually
        exist in a miniaturized database, so inserts/updates don't violate
        referential integrity. It never *raises* a bound, only lowers it.
        """
        tmpl = self.templates[template_name]
        int_bounds = int_bounds or {}
        values: dict[str, object] = {}
        # Resolve non-derived params first so int_offset can reference them.
        for pname, spec in tmpl.params.items():
            if spec.get("generator") == "int_offset":
                continue
            values[pname] = _gen_param(spec, rng, values, int_bounds.get(pname))
        for pname, spec in tmpl.params.items():
            if spec.get("generator") == "int_offset":
                values[pname] = _gen_param(spec, rng, values, int_bounds.get(pname))
        return values

    def describe(self, name: str) -> list[tuple[str, str, float]]:
        """Return [(template_name, kind, normalized_weight)] sorted by weight desc."""
        rows = [(wt.template.name, wt.template.kind, wt.weight) for wt in self.mix(name)]
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows


# ----------------------------- parameter generators -----------------------------
_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
_SEQUENCE_STATE: dict[int, int] = {}


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _gen_param(
    spec: dict,
    rng: random.Random,
    resolved: dict[str, object],
    int_cap: int | None = None,
) -> object:
    gen = spec["generator"]
    if gen == "choice":
        vals = spec["values"]
        return vals[rng.randrange(len(vals))]
    if gen == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        if int_cap is not None:
            high = min(high, int(int_cap))
        if high < low:
            high = low
        return rng.randint(low, high)
    if gen == "int_offset":
        base = int(resolved[spec["base"]])
        return base + int(spec["offset"])
    if gen == "numeric":
        return round(rng.uniform(float(spec["low"]), float(spec["high"])), 2)
    if gen == "timestamp":
        low = _parse_date(spec["low"])
        high = _parse_date(spec["high"])
        span = int((high - low).total_seconds())
        return (low + timedelta(seconds=rng.randint(0, max(span, 1)))).isoformat()
    if gen == "sequence":
        start = int(spec["start"])
        key = id(spec)
        current = _SEQUENCE_STATE.get(key, start)
        _SEQUENCE_STATE[key] = current + 1
        return current
    raise ValueError(f"unknown generator {gen!r}")


def explain_ok(
    workload: Workload,
    template_name: str,
    cfg: DBConfig | None = None,
    *,
    rng: random.Random | None = None,
) -> bool:
    """Return True if ``EXPLAIN`` of a rendered instance succeeds (DB required).

    Writes are wrapped so EXPLAIN does not mutate data; the surrounding
    transaction is rolled back by :func:`advisor.db.connect` on the raised
    sentinel. We EXPLAIN (no ANALYZE) so nothing is actually executed.
    """
    rng = rng or random.Random(0)
    tmpl = workload.templates[template_name]
    params = workload.generate_instance(template_name, rng)
    with connect(cfg) as conn:
        conn.execute(f"EXPLAIN {tmpl.sql}", params)
    return True
