"""Run a workload scenario end-to-end and dump paired decision states.

Pipeline (Week 2, Task 4):
    1. Replay the chosen mix and measure read_runtime / write_overhead under the
       measurement protocol (src/advisor/replay.py).
    2. Extract per-candidate features from the live DB (src/advisor/features.py).
    3. Serialize each candidate's decision state two ways — tabular + Laya —
       from one canonical dict (src/advisor/state.py).
    4. Write a single reproducible scenario artifact under data/.

Usage:
    python scripts/run_replay.py --mix baseline
    python scripts/run_replay.py --mix shifted --scenario-id demo --out data/demo.json

The script assumes a seeded disposable DB already exists (run scripts/init_db.py
first, e.g. with --scale 0.01 for a fast run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advisor import DATA_DIR  # noqa: E402
from advisor.candidates import CandidateSet  # noqa: E402
from advisor.db import DBConfig, connect, server_available  # noqa: E402
from advisor.features import extract  # noqa: E402
from advisor.replay import ReplayResult, measure  # noqa: E402
from advisor.state import build_state  # noqa: E402
from advisor.workload import Workload  # noqa: E402


def build_artifact(
    workload: Workload,
    candidate_set: CandidateSet,
    mix: str,
    scenario_id: str,
    replay: ReplayResult,
    cfg: DBConfig,
) -> dict[str, object]:
    """Assemble the full scenario artifact (measurement + paired states)."""
    decisions: list[dict[str, object]] = []
    with connect(cfg) as conn:
        for cand in candidate_set.candidates:
            feats = extract(workload, mix, cand, cfg, conn=conn)
            state = build_state(scenario_id, feats)
            # The no-drift guarantee is asserted at build time, not just in tests.
            assert state.views_agree(), f"view drift for {cand.table}.{cand.column}"
            decisions.append(
                {
                    "candidate": f"{cand.table}.{cand.column}",
                    "index_name": cand.index_name,
                    "allowed_actions": list(state.allowed_actions),
                    "tabular": state.tabular_row(),
                    "laya": state.laya_payload(),
                }
            )

    return {
        "scenario_id": scenario_id,
        "mix": mix,
        "measurement": {
            "read_runtime": {
                "median": replay.read_runtime.median,
                "spread": replay.read_runtime.spread,
            },
            "write_overhead": {
                "median": replay.write_overhead.median,
                "spread": replay.write_overhead.spread,
            },
            "client_wall_median": replay.client_wall_median,
            "server_wall_median": replay.server_wall_median,
            "server_le_client": replay.within_spread(),
            "repeats": replay.repeats,
            "measured_passes": replay.measured_passes,
            "per_template": {
                name: {"kind": t.kind, "median": t.median, "spread": t.spread}
                for name, t in replay.per_template.items()
            },
        },
        "decisions": decisions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a scenario and dump paired decision states.")
    parser.add_argument("--mix", default="baseline", help="workload mix name (default baseline)")
    parser.add_argument("--scenario-id", default=None, help="scenario id (default: mix name)")
    parser.add_argument("--out", default=None, help="output artifact path (default data/<id>.json)")
    parser.add_argument("--repeats", type=int, default=3, help="measured-phase repeats")
    parser.add_argument("--queries-per-pass", type=int, default=40)
    args = parser.parse_args(argv)

    cfg = DBConfig.from_env()
    if not server_available(cfg):
        print(
            "ERROR: no reachable PostgreSQL server. Run scripts/init_db.py first "
            "and set PG* env vars (see .env.example).",
            file=sys.stderr,
        )
        return 2

    scenario_id = args.scenario_id or args.mix
    out_path = Path(args.out) if args.out else DATA_DIR / f"{scenario_id}.json"

    workload = Workload.load()
    candidate_set = CandidateSet.load(workload=workload)

    print(f"Replaying mix {args.mix!r} (repeats={args.repeats}) ...")
    replay = measure(
        workload,
        args.mix,
        cfg,
        repeats=args.repeats,
        queries_per_pass=args.queries_per_pass,
    )
    print(
        f"  read_runtime   median={replay.read_runtime.median:.6f}s "
        f"spread={replay.read_runtime.spread:.6f}s"
    )
    print(
        f"  write_overhead median={replay.write_overhead.median:.6f}s "
        f"spread={replay.write_overhead.spread:.6f}s"
    )
    print(
        f"  server_wall={replay.server_wall_median:.6f}s <= "
        f"client_wall={replay.client_wall_median:.6f}s : {replay.within_spread()}"
    )
    print("  per-template medians:")
    for name, t in sorted(replay.per_template.items()):
        print(f"    [{t.kind:>5}] {name:<32} {t.median:.6f}s (spread {t.spread:.6f}s)")

    artifact = build_artifact(workload, candidate_set, args.mix, scenario_id, replay, cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote scenario artifact: {out_path}")
    print(f"  {len(artifact['decisions'])} decision-states (tabular + Laya views each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
