# Week 2 Plan — Workload Replay, Feature Extraction & Shared Decision-State

**Project:** Evaluating Jev-Like Decision Models for Adaptive Index Lifecycle Management (Jeff Huang, John Yun)

**Where this sits in the schedule:** Second of six weeks, second of three plans (Weeks 1–3) targeting the **Milestone-1 checkpoint (~60% complete) due 11/3/2026**. Treat this as a one-person team. Week 1 froze the experiment spec and stood up the project skeleton; this week turns that static spec into a running measurement pipeline. Week 3 then uses this pipeline to produce labeled decisions and baselines.

---

## Project context (recap)

The project tests whether the **Laya** typed decision model can make effective PostgreSQL index-lifecycle decisions (**CREATE / KEEP / DROP / NO-OP**) versus Random Forest, boosted-tree, and rule-based baselines, judged on **both** action-prediction accuracy (macro-F1, per-action precision/recall, calibration) and real database outcomes (query runtime, write overhead, storage/build cost, net cost vs no-index/heuristic).

**Stack:** PostgreSQL + Python, scikit-learn, XGBoost, Laya + fine-tuning tools. Classical models run on CPU; Laya fine-tuning runs on GPU (Colab / Kaggle free 2×T4).

**Laya API facts (central to this week's serializer):** `Router().predict(state, questions)` takes a `state` (text or JSON string) and typed `questions` of type `choice` / `score` / `noul`. A `choice` question has `instructions` and `criteria` (a dict of option → description). The model returns typed answers with calibrated probabilities and never generates free text. Fine-tuning (Week 5) uses `laya-train` from CSV or JSONL where each line has `state`, `questions`, and `expected` / `gold`.

---

## Shared foundations (recap — established in full in Week 1)

- **Repo layout:** `index-advisor/` with `config/`, `src/advisor/`, `scripts/`, `data/`, `tests/`, plus `pyproject.toml` and `.github/workflows/ci.yml`. This week adds behavior to `src/advisor/replay.py`, `features.py`, `state.py`, and `scripts/run_replay.py`.
- **Schema (bounded, disposable):** `customers` (~50k), `orders` (~500k), `order_items` (~2M); only PKs/FKs by default so all candidate indexes start absent.
- **Workload (`config/workload.yaml`):** two named mixes — `baseline` (80% read / 20% write) and `shifted` (for holdout); reads filter/join/group/order over the three tables, writes insert/update.
- **Candidate-index rules:** single-column B-tree indexes on WHERE/JOIN/GROUP BY/ORDER BY columns; absent ⇒ {CREATE, NO-OP}, present ⇒ {KEEP, DROP}.
- **Cost score (`config/cost_score.yaml`, lower is better):** `score = w_r*norm(read_runtime) + w_w*norm(write_overhead) + w_s*norm(index_storage_bytes) + w_b*norm(index_build_time)`; defaults `w_r=0.5, w_w=0.3, w_s=0.1, w_b=0.1`; each component also reported raw.
- **Shared decision-state fields:** `query_frequency`, `est_selectivity`, `est_cardinality`, `explain_startup_cost`, `explain_total_cost`, `explain_plan_rows`, `index_exists`, `index_scans_observed`, `index_size_bytes`, `workload_write_rate`, plus identifiers (`table`, `column`, `scenario_id`). Serialized two ways: a flat tabular row (for scikit-learn / XGBoost) and a Laya `state` string + typed `choice` question whose `criteria` equal the allowed actions.
- **CI/CD:** GitHub + trunk-based PRs; `ci.yml` runs `ruff` + `pytest`. **This week extends CI with a PostgreSQL service container** so DB-touching tests run in CI; the container must enable the `pg_stat_statements` extension (`shared_preload_libraries`) required by the measurement protocol. All DB work stays on an isolated, disposable test database.

---

## Problem statement

Turn the static spec into a running measurement pipeline that produces, for every (scenario, candidate), the identical decision state consumed by **both** Laya and the tabular models.

## Goals

A deterministic workload-replay controller with timing; per-candidate feature extraction from EXPLAIN + catalog stats + index usage; a shared state serializer emitting both a tabular row and a Laya typed-choice payload; CI extended with a PostgreSQL service container. All timing is produced under the **measurement protocol** below so that `read_runtime` and `write_overhead` reflect the database's own work and are not confounded by cache state, background activity, build cost, or client overhead.

---

## Measurement protocol (authoritative for all timing in this project)

`read_runtime` and `write_overhead` are the two measured cost components the Week-1 cost objective consumes. They must measure the **database's execution cost for the workload**, isolated from confounders. Every timing figure the pipeline produces — here in Week 2 and in the Week-3 labeler — is produced under this protocol.

### What is measured, and the source of truth

- **Primary source — `pg_stat_statements`.** Enable the extension on the disposable DB. Snapshot its `total_exec_time` / `calls` / `rows` before and after a measured window and **diff** it. Attribute each normalized statement to its workload template, then bucket by template `kind`: the sum of `total_exec_time` over `read` templates is `read_runtime`; over `write` templates is `write_overhead`. This is server-side time, so it excludes client, network, and Python-driver overhead by construction.
- **Cross-check — client wall-clock.** Also record `time.perf_counter()` around each execution. Client time ≥ server time always; a large or drifting gap flags a confounder (network, driver, GC) and invalidates the run. The two are reported side by side.
- **Explicitly NOT a measure of benefit:** `EXPLAIN` *estimated* cost (`explain_total_cost`, etc.) is collected only as a **feature** for the models (Task 2). It is never used as a measured outcome — estimates are what we are testing the models' ability to improve upon, not ground truth.

### Confounders and their controls

1. **Cache warmth (largest confounder).** A cold query reads from disk; a warm one from the buffer cache (orders of magnitude faster). *Control:* every measured configuration runs a **warm-up phase** (N untimed passes to reach steady cache state) before the **measured phase** (M timed passes). Every action for a candidate is measured from the same warmed baseline.
2. **Index build cost leaking into write overhead.** `CREATE INDEX` is an expensive write. *Control:* `index_build_time` is a **separate** cost component; index DDL is applied and timed **outside** the measured read/write window so build cost never contaminates `write_overhead`.
3. **Background Postgres activity.** Autovacuum, autoanalyze, checkpoints, and the WAL writer can fire mid-window and inflate a run. *Control:* on the disposable DB, set `autovacuum = off` during measurement; issue a manual `CHECKPOINT` and a deliberate `VACUUM (ANALYZE)` **before** the measured window (so planner stats are fresh and identical across actions) and never inside it.
4. **Client / network / driver overhead.** *Control:* the primary figure is server-side `pg_stat_statements` time; wall-clock is only a sanity cross-check.
5. **Run-to-run noise / outliers.** *Control:* repeat the measured phase **R times** and report the **median** plus a spread (IQR or stdev), never a single run and never the mean (outlier-sensitive). Near-ties within the spread are flagged, not silently resolved — this feeds the Week-3 label-stability caveat.
6. **Shared, non-isolated resources.** *Control:* a single serial connection against a dedicated disposable DB; no concurrent clients; nothing else hitting the instance.

### The structural defense — differential (A/B) measurement

Absolute timings are never interpreted directly. For each candidate the labeler (Week 3) measures the **same workload** on **cloned states that differ in exactly one thing** — the index present vs. absent — under identical warmed conditions. Any constant overhead (fixed query cost, server baseline, residual client cost) appears in **both** measurements and **cancels in the difference**, so what remains is attributable to the index. Week 2's replay controller must therefore expose timing as a repeatable, state-parameterized measurement that Week 3 can invoke on each cloned state.

### Honest limitation to record

Wall-clock measurement on a general-purpose OS is never perfectly clean. The defensible claim is: timings are measured server-side, under warmed, isolated, background-suppressed, repeated conditions, reported as medians with spread, and compared **differentially** so shared overhead cancels — not that they are noise-free absolutes.

---

## Tasks

### Task 1 — Build the workload-replay controller

- **Objective:** Execute a workload mix reproducibly and measure its `read_runtime` and `write_overhead` under the measurement protocol above.
- **Implementation Guidance:** Implement `src/advisor/replay.py` + `scripts/run_replay.py` to execute a chosen workload mix against the isolated DB, honoring per-query frequencies, with fixed seeds, a **warm-up phase**, and a **measured phase**. Measure server-side time via a `pg_stat_statements` before/after diff, bucketed into `read_runtime` (read templates) and `write_overhead` (write templates); also capture client wall-clock as a cross-check. Repeat the measured phase R times and return the **median per component plus a spread**. Apply the background-activity controls (autovacuum off, pre-window `CHECKPOINT` + `VACUUM (ANALYZE)`). Index DDL/build timing is handled separately (see Week 3) and must stay out of the measured window. Expose timing as a state-parameterized call so Week 3 can run it on cloned states for differential A/B measurement.
- **Test Requirements:** `test_replay` asserts: reproducible medians within tolerance across two runs on identical state; correct read/write bucketing (read templates land in `read_runtime`, writes in `write_overhead`); server time ≤ client wall-clock; and that a spread statistic is returned alongside each median. (DB-backed; skips without a server.)
- **Demo:** Run a scenario and print median aggregate `read_runtime`, `write_overhead`, per-template timings, and the measured-phase spread.

### Task 2 — Implement per-candidate feature extraction

- **Objective:** Collect every decision-state field for each candidate directly from PostgreSQL. These are model **features** (inputs), distinct from the measured cost **outcomes** of the measurement protocol; in particular the `explain_*` fields are *estimates* collected as features, never used as measured outcomes.
- **Implementation Guidance:** Implement `src/advisor/features.py` to collect, per candidate: `query_frequency` (from the workload), `est_selectivity` / `est_cardinality` and EXPLAIN startup/total cost / plan rows (via `EXPLAIN (FORMAT JSON)`), `index_exists`, `index_scans_observed` and `index_size_bytes` (from `pg_stat_user_indexes` / `pg_relation_size`), and `workload_write_rate`.
- **Test Requirements:** `tests/test_features.py` asserts all fields populate with correct types and that creating an index flips `index_exists` and changes the size/usage fields.
- **Demo:** Print the full feature table for all candidates under the `baseline` mix.

### Task 3 — Implement the shared decision-state serializer

- **Objective:** Guarantee Laya and the tabular models see the *same* decision state, serialized two ways without drift.
- **Implementation Guidance:** Implement `src/advisor/state.py` producing one canonical dict per (scenario, candidate) and two views: a flat tabular row, and a Laya `state` string + typed `choice` question whose `criteria` equal the allowed actions for the current `index_exists`.
- **Test Requirements:** `tests/test_state.py` asserts the two views derive from the same dict (no field drift), the Laya question `criteria` match the candidate's allowed action set, and the payload validates against `router.predict`'s expected shape (mocked).
- **Demo:** Show, for one candidate, the tabular row and the Laya JSON side by side.

### Task 4 — Wire replay → features → state into an end-to-end scenario dump

- **Objective:** Produce a single reproducible artifact capturing every candidate's paired decision state for a mix.
- **Implementation Guidance:** Implement `scripts/run_replay.py` to run a mix, extract features for every candidate, serialize decision states, and write a scenario artifact (parquet/JSON) under `data/`.
- **Test Requirements:** An integration test runs a tiny scenario end-to-end on a disposable DB and asserts a well-formed artifact with one decision-state per candidate in **both** views.
- **Demo:** `python scripts/run_replay.py --mix baseline` produces a scenario file; open it to show paired states.

### Task 5 — Extend CI with a PostgreSQL service and integration tests

- **Objective:** Run DB-backed tests automatically so the pipeline stays green on every push.
- **Implementation Guidance:** Add a Postgres service container to `ci.yml`, gate DB tests on its availability, and run the Task-4 integration test in CI with a miniaturized row count.
- **Test Requirements:** A CI run shows DB-backed tests executing and passing.
- **Demo:** A green CI run that includes the end-to-end replay integration test.

---

## Definition of Done (Week 2)

One command runs a workload mix, measures read/write cost, extracts all specified features, and emits paired tabular + Laya decision states to a scenario artifact; DB-backed tests run in CI.

**Demoable increment:** generate a scenario file and show the same decision state feeding both model families.
