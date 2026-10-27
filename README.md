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
| 1    | Experiment spec + project skeleton                     | this milestone |
| 2    | Workload replay + feature extraction + decision-state  | planned |
| 3    | Controlled labeling + baseline advisors                | planned |
| 4    | Train Random Forest + boosted trees + holdouts         | planned |
| 5    | Evaluate / fine-tune Laya (LoRA optional)              | planned |
| 6    | End-to-end tests + report/demo                         | planned |

Milestone-1 (~60% complete) is due 11/3/2026; code 12/3/2026; report 12/4/2026.

## Layout

```
config/    schema.sql, workload.yaml, candidates.yaml, cost_score.yaml
src/advisor/  db, schema_loader, workload, candidates, cost (Week 1)
scripts/   init_db.py  (Week 1)
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

## Tests & linting

```bash
ruff check .
pytest
```

DB-backed tests (`-m db`) skip automatically when no PostgreSQL server is
reachable, so the suite is green locally without a database; CI runs them
against a PostgreSQL service container (added in Week 2).
