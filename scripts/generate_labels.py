"""Generate and export the labeled decision dataset (Week 3, Tasks 5-7).

Pipeline:
    1. Enumerate scenarios (src/advisor/scenarios.py).
    2. For each scenario, (re)seed a disposable miniaturized DB at the scenario's
       scale/seed, set the candidate's starting index state, then run the
       controlled A/B labeler (src/advisor/labeler.py) per candidate to pick the
       reference action under the declared cost objective.
    3. Attach each candidate's decision state (src/advisor/state.py).
    4. Export:
         data/decisions.csv    tabular features + reference action (+ near-tie
                               probabilities) for scikit-learn / XGBoost,
         data/decisions.jsonl  Laya `laya-train` schema: each line has `state`,
                               `questions`, and `expected` (confident) or `gold`
                               (near-tie teacher probabilities).
    5. Define + record a train/test split by workload MIX so train and test never
       share a mix (disjoint-template holdout).
    6. Print the dataset size, per-action reference-label distribution, the
       confident-vs-near-tie counts, the holdout split, a baseline-vs-reference
       comparison (Task 6), and the weight-sensitivity table (Task 7).

Usage:
    python scripts/generate_labels.py                 # full scenario set
    python scripts/generate_labels.py --limit 4       # quick smoke (first N scenarios)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advisor import (
    DATA_DIR,  # noqa: E402
    schema_loader,  # noqa: E402
)
from advisor.baselines import NoIndexAdvisor, RuleBasedAdvisor  # noqa: E402
from advisor.candidates import CandidateSet  # noqa: E402
from advisor.db import DBConfig, connect, create_test_db, server_available  # noqa: E402
from advisor.features import extract  # noqa: E402
from advisor.labeler import (  # noqa: E402
    CandidateLabel,
    create_index_timed,
    drop_index,
    label_candidate,
)
from advisor.scenarios import Scenario, ScenarioConfig, generate  # noqa: E402
from advisor.sensitivity import analyze  # noqa: E402
from advisor.state import build_state  # noqa: E402
from advisor.workload import Workload  # noqa: E402

# Train/test holdout by mix: the shifted mix is the held-out test set so train
# and test never share a workload template emphasis.
TEST_MIXES = {"shifted"}


def _effective_mix_weights(workload: Workload, scenario: Scenario) -> dict[str, float]:
    """Raw weights for a scenario: jittered overrides if present, else the base."""
    if scenario.weight_overrides:
        return dict(scenario.weight_overrides)
    return {wt.template.name: wt.weight for wt in workload.mix(scenario.mix)}


def _seed_scenario_db(cfg: DBConfig, scenario: Scenario) -> None:
    create_test_db(cfg, drop_existing=True)
    schema_loader.apply_schema(cfg)
    schema_loader.seed_data(cfg, scale=scenario.scale, seed=scenario.seed)


def _set_start_state(cfg: DBConfig, candidate, start_present: bool) -> None:
    with connect(cfg) as conn:
        if start_present:
            create_index_timed(conn, candidate)
        else:
            drop_index(conn, candidate)


def label_scenario(
    workload: Workload,
    candidate_set: CandidateSet,
    scenario: Scenario,
    cfg: DBConfig,
    *,
    repeats: int,
    queries_per_pass: int,
) -> list[tuple[CandidateLabel, dict, dict]]:
    """Label every candidate in one scenario.

    Returns (label, csv_row, jsonl_line) triples. Both export views are built
    from the SAME decision state so they never drift.
    """
    _seed_scenario_db(cfg, scenario)
    start_present = scenario.start_state == "present"

    out: list[tuple[CandidateLabel, dict, dict]] = []
    for candidate in candidate_set.candidates:
        _set_start_state(cfg, candidate, start_present)
        label = label_candidate(
            workload, scenario.mix, candidate, cfg,
            start_index_exists=start_present,
            repeats=repeats, queries_per_pass=queries_per_pass,
            seed=scenario.seed,
        )
        # Decision-state features captured with the index in its START state.
        _set_start_state(cfg, candidate, start_present)
        with connect(cfg) as conn:
            feats = extract(workload, scenario.mix, candidate, cfg, conn=conn)
        state = build_state(scenario.scenario_id, feats)
        assert state.views_agree()
        out.append(
            (label, _csv_row(scenario, state, label), _jsonl_line(scenario, state, label))
        )
    return out


def _csv_row(scenario, state, label: CandidateLabel) -> dict:
    row = dict(state.tabular_row())
    row["mix"] = scenario.mix
    row["mix_variant"] = scenario.mix_variant
    row["start_state"] = scenario.start_state
    row["scale"] = scenario.scale
    row["reference_action"] = label.reference_action
    row["confident"] = label.confident
    # Probabilities over the allowed actions (so near-tie info is not lost).
    for action, p in label.probabilities.items():
        row[f"prob_{action}"] = p
    row["split"] = "test" if scenario.mix in TEST_MIXES else "train"
    return row


def _jsonl_line(scenario, state, label: CandidateLabel) -> dict:
    """One laya-train line: state + questions + (expected | gold)."""
    payload = state.laya_payload()
    line = {
        "scenario_id": scenario.scenario_id,
        "state": payload["state"],
        "questions": payload["questions"],
        "split": "test" if scenario.mix in TEST_MIXES else "train",
    }
    if label.confident:
        line["expected"] = [label.reference_action]
    else:
        # Teacher probabilities over the allowed actions.
        line["gold"] = [label.probabilities]
    return line


def write_csv(path: Path, rows: list[dict]) -> None:
    # Stable, union-of-keys header so every column is present for every row.
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the labeled decision dataset.")
    parser.add_argument("--limit", type=int, default=None, help="label only the first N scenarios")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--queries-per-pass", type=int, default=20)
    parser.add_argument("--csv", default=str(DATA_DIR / "decisions.csv"))
    parser.add_argument("--jsonl", default=str(DATA_DIR / "decisions.jsonl"))
    args = parser.parse_args(argv)

    cfg = DBConfig.from_env()
    if not server_available(cfg):
        print(
            "ERROR: no reachable PostgreSQL server (with pg_stat_statements). "
            "See .env.example / README.",
            file=sys.stderr,
        )
        return 2

    workload = Workload.load()
    candidate_set = CandidateSet.load(workload=workload)
    scen_cfg = ScenarioConfig.load()
    base_weights = {m: {wt.template.name: wt.weight for wt in workload.mix(m)}
                    for m in scen_cfg.base_mixes}
    scenarios = generate(scen_cfg, base_weights)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    print(f"Labeling {len(scenarios)} scenarios x {len(candidate_set.candidates)} "
          f"candidates = {len(scenarios) * len(candidate_set.candidates)} decisions ...")

    all_labels: list[CandidateLabel] = []
    csv_rows: list[dict] = []
    jsonl_lines: list[dict] = []
    for i, scenario in enumerate(scenarios, 1):
        triples = label_scenario(
            workload, candidate_set, scenario, cfg,
            repeats=args.repeats, queries_per_pass=args.queries_per_pass,
        )
        for label, row, line in triples:
            all_labels.append(label)
            csv_rows.append(row)
            jsonl_lines.append(line)
        print(f"  [{i}/{len(scenarios)}] {scenario.scenario_id}")

    write_csv(Path(args.csv), csv_rows)
    write_jsonl(Path(args.jsonl), jsonl_lines)

    _report(all_labels, csv_rows, candidate_set, workload)
    print(f"\nWrote {args.csv} ({len(csv_rows)} rows) and {args.jsonl} "
          f"({len(jsonl_lines)} lines).")
    return 0


def _report(all_labels, csv_rows, candidate_set, workload) -> None:
    # Per-action reference distribution.
    dist: dict[str, int] = {}
    confident = 0
    for lbl in all_labels:
        dist[lbl.reference_action] = dist.get(lbl.reference_action, 0) + 1
        confident += 1 if lbl.confident else 0
    print("\nReference-action distribution:")
    for action in ("CREATE", "KEEP", "DROP", "NO-OP"):
        print(f"  {action:<7} {dist.get(action, 0)}")
    print(f"Confident labels: {confident} / {len(all_labels)} "
          f"(near-tie: {len(all_labels) - confident})")

    # Holdout split.
    train = sum(1 for r in csv_rows if r["split"] == "train")
    test = len(csv_rows) - train
    print(f"\nHoldout split by mix: train={train}  test={test}  (test mixes={sorted(TEST_MIXES)})")

    # Task 6 — baseline vs reference comparison on the first scenario's rows.
    print("\nBaseline vs reference (first scenario):")
    no_index = NoIndexAdvisor()
    rule_based = RuleBasedAdvisor()
    first_id = csv_rows[0]["scenario_id"] if csv_rows else None
    first_rows = [r for r in csv_rows if r["scenario_id"] == first_id]
    for r in first_rows:
        feats = _features_from_row(r)
        ni = no_index.decide(feats).action
        rb = rule_based.decide(feats).action
        ref = r["reference_action"]
        print(f"  {r['table']}.{r['column']:<12} ref={ref:<7} "
              f"no_index={ni:<6} rule_based={rb:<7} "
              f"gap_ni={'=' if ni == ref else 'X'} gap_rb={'=' if rb == ref else 'X'}")

    # Task 7 — weight sensitivity.
    reports = analyze(all_labels)
    print("\nWeight sensitivity (reference-action distribution + label changes):")
    for rep in reports:
        d = ", ".join(f"{a}:{rep.action_distribution.get(a, 0)}"
                      for a in ("CREATE", "KEEP", "DROP", "NO-OP"))
        print(f"  {rep.weighting:<16} [{d}]  changed_vs_default={rep.labels_changed_vs_default}")


def _features_from_row(r: dict):
    from advisor.features import CandidateFeatures

    def f(k):
        return float(r[k]) if r.get(k) not in (None, "") else 0.0

    return CandidateFeatures(
        table=r["table"],
        column=r["column"],
        query_frequency=f("query_frequency"),
        est_selectivity=f("est_selectivity"),
        est_cardinality=f("est_cardinality"),
        explain_startup_cost=f("explain_startup_cost"),
        explain_total_cost=f("explain_total_cost"),
        explain_plan_rows=f("explain_plan_rows"),
        index_exists=str(r["index_exists"]).lower() == "true",
        index_scans_observed=int(f("index_scans_observed")),
        index_size_bytes=int(f("index_size_bytes")),
        workload_write_rate=f("workload_write_rate"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
