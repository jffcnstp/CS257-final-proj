"""Calibrate cost-score normalization bounds from real measurements.

Week 1 left ``config/cost_score.yaml`` ``bounds`` as placeholders and explicitly
deferred them to "a calibration run (Week 3 fills these with measured min/max
per component)". This script performs that run: it labels a representative
sample of scenarios, collects every action's RAW cost components, and writes the
observed per-component [min, max] back into ``cost_score.yaml``.

Why it matters: the reference *action* is invariant to the bounds (normalization
is monotonic, and every action for a candidate uses the same bounds), but the
*confidence* decision (score gap vs. measured spread) is in normalized units.
Placeholder bounds (e.g. read 0..60s) compress a real ~1s read advantage into a
~0.002 gap that measurement noise swamps, manufacturing near-ties. Tighter,
measured bounds make genuine wins confident and stable.

A small margin is applied so later runs that drift slightly past the sampled
range still normalize into [0, 1] (the cost model clamps anyway).

Usage:
    python scripts/calibrate_bounds.py --sample 12 --repeats 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import yaml  # noqa: E402

import generate_labels as gl  # noqa: E402
from advisor.candidates import CandidateSet  # noqa: E402
from advisor.cost import COMPONENTS, COST_SCORE_YAML  # noqa: E402
from advisor.db import DBConfig, server_available  # noqa: E402
from advisor.scenarios import ScenarioConfig, generate  # noqa: E402
from advisor.workload import Workload  # noqa: E402

# A little headroom so near-range drift on later runs still normalizes in-range.
MARGIN = 0.10
# Never let a bound collapse to zero width (division-by-zero guard in cost.py).
MIN_SPAN = {
    "read_runtime": 0.05,
    "write_overhead": 0.01,
    "index_storage_bytes": 1024.0,
    "index_build_time": 0.01,
}


def _sample_scenarios(sample: int):
    workload = Workload.load()
    candidate_set = CandidateSet.load(workload=workload)
    scen_cfg = ScenarioConfig.load()
    base_weights = {
        m: {wt.template.name: wt.weight for wt in workload.mix(m)}
        for m in scen_cfg.base_mixes
    }
    scenarios = generate(scen_cfg, base_weights)
    # Spread the sample across the sorted set (so both start-states/mixes appear)
    # rather than taking a contiguous prefix (which is all-absent/all-baseline).
    if sample < len(scenarios):
        step = max(1, len(scenarios) // sample)
        scenarios = scenarios[::step][:sample]
    return workload, candidate_set, scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate cost-score bounds.")
    parser.add_argument("--sample", type=int, default=12, help="scenarios to sample")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--queries-per-pass", type=int, default=30)
    args = parser.parse_args(argv)

    cfg = DBConfig.from_env()
    if not server_available(cfg):
        print("ERROR: no reachable PostgreSQL server (with pg_stat_statements).",
              file=sys.stderr)
        return 2

    workload, candidate_set, scenarios = _sample_scenarios(args.sample)
    print(f"Calibrating from {len(scenarios)} sampled scenarios "
          f"x {len(candidate_set.candidates)} candidates ...")

    mins = {c: float("inf") for c in COMPONENTS}
    maxs = {c: float("-inf") for c in COMPONENTS}
    n_obs = 0
    for i, scenario in enumerate(scenarios, 1):
        triples = gl.label_scenario(
            workload, candidate_set, scenario, cfg,
            repeats=args.repeats, queries_per_pass=args.queries_per_pass,
        )
        for label, _row, _line in triples:
            for outcome in label.outcomes.values():
                raw = outcome.components.as_dict()
                for c in COMPONENTS:
                    mins[c] = min(mins[c], raw[c])
                    maxs[c] = max(maxs[c], raw[c])
                    n_obs += 1
        print(f"  [{i}/{len(scenarios)}] {scenario.scenario_id}")

    bounds = _finalize_bounds(mins, maxs)
    _write_bounds(bounds)

    print(f"\nObserved {n_obs // len(COMPONENTS)} action-outcomes.")
    print("Calibrated bounds written to config/cost_score.yaml:")
    for c in COMPONENTS:
        print(f"  {c:<20} min={bounds[c]['min']:.6g}  max={bounds[c]['max']:.6g} "
              f"(observed {mins[c]:.6g}..{maxs[c]:.6g})")
    return 0


def _finalize_bounds(mins: dict, maxs: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c in COMPONENTS:
        lo = max(0.0, mins[c])
        hi = maxs[c]
        span = hi - lo
        hi = hi + MARGIN * span
        # Enforce a minimum span so cost.py never divides by ~0.
        if hi - lo < MIN_SPAN[c]:
            hi = lo + MIN_SPAN[c]
        out[c] = {"min": round(lo, 6), "max": round(hi, 6)}
    return out


def _write_bounds(bounds: dict[str, dict[str, float]]) -> None:
    """Rewrite only the ``bounds`` mapping in cost_score.yaml, preserving weights."""
    raw = yaml.safe_load(COST_SCORE_YAML.read_text(encoding="utf-8"))
    raw["bounds"] = bounds
    header = (
        "# Evaluation cost score (lower is better).\n"
        "#\n"
        "#   score = w_r * norm(read_runtime)\n"
        "#         + w_w * norm(write_overhead)\n"
        "#         + w_s * norm(index_storage_bytes)\n"
        "#         + w_b * norm(index_build_time)\n"
        "#\n"
        "#   norm(x) = (x - min) / (max - min)   clamped to [0, 1]\n"
        "#\n"
        "# Weights are a TUNABLE decision and sum to 1.0. Every component is also\n"
        "# reported raw (see src/advisor/cost.py) so trade-offs are never hidden.\n"
        "#\n"
        "# NOTE: `bounds` below were produced by scripts/calibrate_bounds.py from a\n"
        "# measured sample of labeling runs (Week-3 calibration), replacing the\n"
        "# Week-1 placeholders. Units: read_runtime / write_overhead / index_build_time\n"
        "# in seconds; index_storage_bytes in bytes.\n"
    )
    body = yaml.safe_dump(raw, sort_keys=False, default_flow_style=False)
    COST_SCORE_YAML.write_text(header + "\n" + body, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
