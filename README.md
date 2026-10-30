# index-advisor

Evaluating Jev-like typed decision models (**Laya**) for adaptive **PostgreSQL
index lifecycle management** — deciding **CREATE / KEEP / DROP / NO-OP** for
candidate indexes under a changing workload, and comparing Laya against Random
Forest, boosted-tree, and rule-based baselines on both action-prediction
accuracy and real database outcomes.

CS257 final project — Jeff Huang, John Yun.

## Frozen scope

The full experiment specification — research question, success criteria,
schema, workload + frequencies, candidate-index rules, and the cost score — is
in **[docs/experiment-spec.md](docs/experiment-spec.md)**. This is the
authoritative contract for Weeks 2–3.

## Status

| Week | Deliverable                                            | State |
|------|--------------------------------------------------------|-------|
| 1    | Experiment spec + project skeleton                     | done |
| 2    | Workload replay + feature extraction + decision-state  | done |
| 3    | Controlled labeling + baseline advisors                | this milestone |
| 4    | Train Random Forest + boosted trees + holdouts         | planned |
| 5    | Evaluate / fine-tune Laya (LoRA optional)              | planned |
| 6    | End-to-end tests + report/demo                         | planned |

Milestone-1 (~60% complete) is due 11/3/2026; code 12/3/2026; report 12/4/2026.

## Layout

```
config/    schema.sql, workload.yaml, candidates.yaml, cost_score.yaml
           scenarios.yaml, baselines.yaml (Week 3)
src/advisor/  db, schema_loader, workload, candidates, cost (Week 1)
              replay, features, state (Week 2)
              scenarios, labeler, baselines, sensitivity (Week 3)
scripts/   init_db.py (Week 1), run_replay.py (Week 2),
           generate_labels.py (Week 3)
docs/      experiment-spec.md
tests/     pytest suite (DB tests skip when no server is reachable)
```

## Setup

Requires Python 3.10+ and (for database features) a reachable PostgreSQL server.

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e .[dev]       # add .[laya] from Week 5

cp .env.example .env        # fill in PG connection vars; never commit .env
```

All database work runs against an **isolated, disposable** test database named
by `INDEX_ADVISOR_DB` (default `index_advisor_test`). The tooling creates and
drops that database only; it never touches a shared or production instance.

## Build the database

```bash
python scripts/init_db.py                # full-size seed (~50k / 500k / 2M rows)
python scripts/init_db.py --scale 0.01   # fast 1% miniature
```

Prints the resulting row counts and confirms no candidate (non-PK) indexes
exist yet, so CREATE / NO-OP / KEEP / DROP are all reachable.

## Run a workload scenario (Week 2)

```bash
python scripts/run_replay.py --mix baseline     # measure + dump paired states
python scripts/run_replay.py --mix shifted --scenario-id demo --out data/demo.json
```

This replays the mix under the **measurement protocol** (warm-up then measured
phase, server-side `pg_stat_statements` timing, background-activity controls,
repeat-and-median with spread), extracts each candidate's features, and writes a
scenario artifact under `data/` carrying every candidate's decision state in
**both** views — a flat tabular row and a Laya `{state, questions}` payload —
derived from one canonical dict so the two never drift.

The measurement protocol requires the `pg_stat_statements` extension, which must
be loaded at server start via `shared_preload_libraries`. For a local run:

```bash
docker run -d --name pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16 \
  -c shared_preload_libraries=pg_stat_statements -c autovacuum=off
```

Tests that need it skip gracefully when it is not available.

## Generate the labeled decision dataset (Week 3)

```bash
python scripts/generate_labels.py                 # full scenario set
python scripts/generate_labels.py --limit 4       # quick smoke (first N scenarios)
```

This enumerates varied scenarios (across workload mixes, starting index states,
data scales/seeds, and per-template frequency jitter), and for each candidate
runs the **controlled A/B labeler**: it clones the DB state, applies each allowed
action, replays the workload under the Week-2 measurement protocol, scores the
outcome with the Week-1 cost objective, and records the lowest-cost action as
the **reference action** — a reference under a *declared* objective, not
objective ground truth. A clear win becomes a confident `expected` label; a
measured near-tie becomes a `gold` probability distribution.

It writes two synchronized views:

- `data/decisions.csv` — tabular features + reference action (+ near-tie
  probabilities) for scikit-learn / XGBoost,
- `data/decisions.jsonl` — Laya `laya-train` schema (`state`, `questions`, and
  `expected` or `gold` per line),

with a train/test holdout by workload mix (the `shifted` mix is held out). The
run also prints the per-action label distribution, confident-vs-near-tie counts,
a no-index / rule-based **baseline-vs-reference** comparison, and a **weight
sensitivity** table showing how many reference labels change under alternative
cost weightings (so the labels' dependence on the weight choice is reported, not
hidden).

## Tests & linting

```bash
ruff check .
pytest
```

DB-backed tests (`-m db`) skip automatically when no PostgreSQL server is
reachable, so the suite is green locally without a database; CI runs them
against a PostgreSQL service (with `pg_stat_statements` preloaded and
`autovacuum=off`) so the full replay pipeline is exercised on every push.
