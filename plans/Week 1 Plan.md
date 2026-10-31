# Week 1 Plan — Experiment Specification & Project Skeleton

**Project:** Evaluating Jev-Like Decision Models for Adaptive Index Lifecycle Management (Jeff Huang, John Yun)

**Where this sits in the schedule:** This is the first of a six-week schedule and the first of three plans (Weeks 1–3) that target the **Milestone-1 checkpoint (~60% complete) due 11/3/2026**. Later deadlines: code due 12/3/2026, report due 12/4/2026. Treat this as a one-person team.

- **W1 (this plan):** experiment spec + project skeleton
- W2: workload replay + feature extraction + shared decision-state
- W3: controlled labeling + baseline advisors
- W4: train Random Forest + boosted trees + holdouts
- W5: evaluate / fine-tune Laya (LoRA optional)
- W6: end-to-end tests + report/demo

---

## Project context

The project tests whether the **Laya** typed decision model can make effective PostgreSQL index-lifecycle decisions (**CREATE / KEEP / DROP / NO-OP**) compared to Random Forest, boosted-tree, and rule-based baselines. Effectiveness is judged on **both**:

- **Action-prediction accuracy** — macro-F1, per-action precision/recall, probability calibration.
- **Real database outcomes** — query runtime, write overhead, storage/build cost, and net cost versus no-index and heuristic baselines.

**Stack:** PostgreSQL + Python, scikit-learn, XGBoost (gradient boosting), Laya + its fine-tuning tools. Classical models run on CPU; Laya fine-tuning runs on a GPU (Colab / Kaggle free 2×T4).

**Laya API facts (used from W2 onward, documented here for grounding):** `Router().predict(state, questions)` takes a `state` (text or JSON string) and typed `questions` of type `choice` / `score` / `noul`. A `choice` question has `instructions` and `criteria` (a dict of option → description). The model returns typed answers with calibrated probabilities and never generates free text. Fine-tuning uses `laya-train` from a CSV (text column + label column) or JSONL where each line has `state`, `questions`, and `expected` (one answer per question) or `gold` (teacher probabilities); calibration temperatures are fit on a held-out slice.

---

## Shared foundations (authoritative for all three weeks)

Week 1 stands these up in full. Weeks 2–3 only add behavior on top of them — they never re-decide scope.

### Repository layout

```
index-advisor/
├─ README.md
├─ pyproject.toml              # deps: psycopg[binary], pandas, scikit-learn, xgboost, laya, pyyaml, pytest, ruff
├─ .github/workflows/ci.yml    # ruff lint + pytest on push/PR
├─ .env.example                # PG connection vars (no secrets committed)
├─ config/
│  ├─ schema.sql               # DDL for customers/orders/order_items (+ indexes dropped by default)
│  ├─ workload.yaml            # query templates + frequencies + read/write mix
│  ├─ candidates.yaml          # candidate-index rules
│  └─ cost_score.yaml          # cost-score weights + normalization bounds
├─ src/advisor/
│  ├─ db.py                    # connection + isolated-DB helpers (create/clone/drop test DB)
│  ├─ schema_loader.py         # apply schema.sql, load synthetic data
│  ├─ workload.py              # parse workload.yaml, generate query instances
│  ├─ replay.py                # (Week 2) workload replay controller + timing
│  ├─ features.py              # (Week 2) per-candidate feature extraction (EXPLAIN, stats, usage)
│  ├─ state.py                 # (Week 2) shared decision-state serializer (dict → tabular row + Laya JSON)
│  ├─ cost.py                  # (Week 1 formula, Week 3 measured) cost-score computation
│  ├─ labeler.py               # (Week 3) controlled A/B index comparison → reference action label
│  └─ baselines.py             # (Week 3) no-index + rule-based advisors
├─ scripts/
│  ├─ init_db.py               # one-command DB bring-up + seed
│  ├─ run_replay.py            # (Week 2) run a workload scenario, dump features
│  └─ generate_labels.py       # (Week 3) produce labeled decision dataset (CSV + JSONL)
├─ data/                       # generated decision datasets (gitignored except schema samples)
└─ tests/
   ├─ test_schema.py
   ├─ test_workload.py
   ├─ test_features.py
   ├─ test_state.py
   ├─ test_cost.py
   └─ test_labeler.py
```

### Schema (`config/schema.sql`) — order-processing, bounded for local runs

- `customers(customer_id PK, name, email, country, created_at)` — ~50,000 rows
- `orders(order_id PK, customer_id FK, status, total_amount, created_at)` — ~500,000 rows
- `order_items(order_item_id PK, order_id FK, product_id, quantity, unit_price)` — ~2,000,000 rows
- Only primary keys / FKs are created by default; **all candidate (non-key) indexes start absent** so that CREATE / NO-OP / KEEP / DROP are all reachable actions.

### Workload mix (`config/workload.yaml`) — 80% read / 20% write baseline, plus a shifted mix for holdout

- **Reads:** filter orders by `status`; filter orders by `created_at` range; join customers↔orders on `customer_id`; group orders by `customer_id` (sum `total_amount`); order order_items by `unit_price`; filter customers by `country`.
- **Writes:** insert orders; update `orders.status`; insert order_items.
- Each template carries a relative frequency weight; the file defines two named mixes (`baseline`, `shifted`) for the train/test holdout.

### Candidate-index rules (`config/candidates.yaml`)

- Single-column B-tree indexes only, on columns appearing in WHERE / JOIN / GROUP BY / ORDER BY of the workload — e.g., `orders.status`, `orders.created_at`, `orders.customer_id`, `customers.country`, `order_items.order_id`, `order_items.unit_price`.
- State → action rule: **absent ⇒ {CREATE, NO-OP}**; **present ⇒ {KEEP, DROP}**.

### Cost-score formula (`config/cost_score.yaml`) — lower is better

```
score = w_r * norm(read_runtime)
      + w_w * norm(write_overhead)
      + w_s * norm(index_storage_bytes)
      + w_b * norm(index_build_time)
```

- Default weights: `w_r=0.5, w_w=0.3, w_s=0.1, w_b=0.1` (documented as a tunable decision).
- `norm(x) = (x - min) / (max - min)` using per-component bounds recorded from a calibration run; **every component is also reported raw** so a trade-off (faster reads, much slower writes) is never hidden.
- The labeler (Week 3) chooses the action whose resulting state has the lowest measured score.

### Shared decision-state (`src/advisor/state.py`) — one dict per (scenario, candidate), serialized two ways

- **Fields:** `query_frequency`, `est_selectivity`, `est_cardinality`, `explain_startup_cost`, `explain_total_cost`, `explain_plan_rows`, `index_exists` (bool), `index_scans_observed`, `index_size_bytes`, `workload_write_rate`, plus identifiers (`table`, `column`, `scenario_id`).
- **Tabular view:** flat numeric/boolean feature row for scikit-learn / XGBoost.
- **Laya view:** a natural-language/JSON `state` string + a typed `choice` question whose `criteria` are exactly the allowed actions for the current `index_exists` value, matching the `router.predict(state, questions)` API. Labels for fine-tuning follow the `laya-train` JSONL `expected` schema.

### CI/CD (concrete, lightweight, one-person)

- GitHub repo `index-advisor`, trunk-based with short-lived feature branches; direct pushes to `main` avoided in favor of PRs.
- `.github/workflows/ci.yml`: on push/PR, set up Python 3.10, install deps, run `ruff check` and `pytest`. A PostgreSQL service container is added in Week 2 so DB-touching tests run in CI; **Week-1 tests run without a DB**.
- Hugging Face Hub hosts the fine-tuned Laya checkpoint and the decision datasets (pushed starting Week 5); credentials via env vars / repo secrets, never committed. `.env.example` documents required variables.
- All DB work runs against an **isolated, disposable test database** with bounded synthetic data; no shared/production instance is touched.

---

## Problem statement

Freeze exactly what will be tested before building the advisor, and stand up a reproducible project skeleton so Weeks 2–3 only add behavior, never re-decide scope.

## Goals

Locked research question + success criteria; initialized repo + CI; concrete schema with seeded synthetic data; workload mix with frequencies; candidate-index rules; written + coded cost-score formula; a one-page experiment spec committed to the repo.

---

## Tasks

### Task 1 — Initialize repository, tooling, and CI skeleton

- **Objective:** Stand up the repo skeleton so every later task lands in a fixed, lint-and-test-gated structure.
- **Implementation Guidance:** Create the repo layout above, `pyproject.toml` with pinned deps (`psycopg[binary]`, `pandas`, `scikit-learn`, `xgboost`, `laya`, `pyyaml`, `pytest`, `ruff`), a `ruff` config, `.env.example` (PG connection vars, no secrets), and `.github/workflows/ci.yml` running `ruff check` + `pytest` on push/PR with Python 3.10. Use trunk-based development with short-lived feature branches and PRs into `main`.
- **Test Requirements:** A trivial `tests/test_smoke.py` passes locally and in CI; `ruff check` is clean.
- **Demo:** A green CI run on an opening PR.

### Task 2 — Write and version the research question and success criteria

- **Objective:** Commit the exact question and the bar for a "good" result so success is unambiguous later.
- **Implementation Guidance:** Author `docs/experiment-spec.md` stating the primary research question; state that **both** action accuracy (macro-F1, per-action precision/recall, calibration) and **real DB outcomes** (read runtime, write overhead, storage/build cost, net cost vs no-index/heuristic) are measured; and state what counts as a "good" result.
- **Test Requirements:** A `pytest` doc-presence/section check asserts the required headings exist.
- **Demo:** Render the spec; show the required sections present.

### Task 3 — Define and apply the PostgreSQL schema with bounded synthetic data

- **Objective:** Produce a disposable, deterministically-seeded database that matches the shared schema exactly.
- **Implementation Guidance:** Implement `config/schema.sql`, `src/advisor/db.py` (isolated test-DB create/drop), `src/advisor/schema_loader.py`, and `scripts/init_db.py` seeding the row counts above (~50k customers, ~500k orders, ~2M order_items) with deterministic seeds. Create only PKs/FKs; leave all candidate indexes absent.
- **Test Requirements:** `tests/test_schema.py` brings up a disposable DB, applies the schema, and asserts the tables/row-count ranges and that **no candidate indexes exist initially**.
- **Demo:** `python scripts/init_db.py` builds a fresh DB and prints table row counts.

### Task 4 — Specify the workload mix with frequencies and two named mixes

- **Objective:** Turn the workload description into machine-readable templates with frequencies and a holdout split.
- **Implementation Guidance:** Implement `config/workload.yaml` (baseline 80/20 read/write + a `shifted` mix) and `src/advisor/workload.py` to parse templates and emit concrete parameterized query instances covering the reads and writes listed in the shared foundations.
- **Test Requirements:** `tests/test_workload.py` asserts the frequencies sum per mix, the read/write ratio matches, and every template produces runnable SQL against the seeded DB (`EXPLAIN` succeeds).
- **Demo:** Print the generated query set with per-query frequency for both mixes.

### Task 5 — Encode candidate-index rules and the action set

- **Objective:** Enumerate exactly which indexes the system may consider and which actions are legal in each state.
- **Implementation Guidance:** Implement `config/candidates.yaml` plus a `src/advisor/candidates.py` loader that, given schema + workload, enumerates candidate single-column indexes and the allowed actions per index state (absent ⇒ {CREATE, NO-OP}; present ⇒ {KEEP, DROP}).
- **Test Requirements:** `test_candidates` asserts candidates are drawn only from WHERE/JOIN/GROUP BY/ORDER BY columns and that action sets are {CREATE, NO-OP} when absent / {KEEP, DROP} when present.
- **Demo:** Print the full candidate list with current state and allowed actions.

### Task 6 — Define and implement the evaluation cost score

- **Objective:** Make the cost-score formula executable and transparent so it can drive labeling in Week 3.
- **Implementation Guidance:** Implement `config/cost_score.yaml` (weights + normalization bounds) and `src/advisor/cost.py` computing the weighted score from the four components (read runtime, write overhead, index storage, index build time) and **also returning each raw component**. Default weights `w_r=0.5, w_w=0.3, w_s=0.1, w_b=0.1`; `norm(x) = (x-min)/(max-min)`.
- **Test Requirements:** `tests/test_cost.py` checks monotonicity (a worse component ⇒ a not-lower score), weight normalization, and that raw components are always reported.
- **Demo:** Feed two sample component vectors and show scores + per-component breakdown, proving trade-offs are visible.

### Task 7 — Assemble the one-page experiment specification

- **Objective:** Consolidate the frozen scope into a single readable spec linked from the README.
- **Implementation Guidance:** Consolidate Tasks 2–6 into the final `docs/experiment-spec.md` (research question, schema, workload + frequencies, candidate rules, cost formula) and link it from `README.md`.
- **Test Requirements:** A spec-completeness test asserts all five elements are present and cross-referenced.
- **Demo:** Open the finished spec and run `init_db.py` to show the described environment materializing.

---

## Definition of Done (Week 1)

Repo + CI green; disposable DB builds from one command with correct data; workload, candidates, and cost score are coded, tested, and documented; experiment spec committed.

**Demoable increment:** a reviewer clones the repo, runs CI and `init_db.py`, and reads a complete experiment spec.
