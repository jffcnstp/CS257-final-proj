# Week 3 Plan — Controlled Labeling & Baseline Advisors

**Project:** Evaluating Jev-Like Decision Models for Adaptive Index Lifecycle Management (Jeff Huang, John Yun)

**Where this sits in the schedule:** Third of six weeks, final of three plans (Weeks 1–3) and the one that delivers the substrate for the **Milestone-1 checkpoint (~60% complete) due 11/3/2026**. Treat this as a one-person team. Week 1 froze the experiment spec and skeleton; Week 2 built the workload-replay + feature-extraction + shared-decision-state pipeline; this week uses that pipeline to produce **reference-policy labels** (the best action per candidate under the declared cost objective — a reference, not objective ground truth) and the non-learned baselines the models must beat. Weeks 4–5 then train and evaluate the classical models and Laya against this dataset.

---

## Project context (recap)

The project tests whether the **Laya** typed decision model can make effective PostgreSQL index-lifecycle decisions (**CREATE / KEEP / DROP / NO-OP**) versus Random Forest, boosted-tree, and rule-based baselines, judged on **both** action-prediction accuracy (macro-F1, per-action precision/recall, calibration) and real database outcomes (query runtime, write overhead, storage/build cost, net cost vs no-index/heuristic).

**Stack:** PostgreSQL + Python, scikit-learn, XGBoost, Laya + fine-tuning tools. Classical models run on CPU; Laya fine-tuning runs on GPU (Colab / Kaggle free 2×T4).

**Laya API facts (shape the exported JSONL this week):** `Router().predict(state, questions)` takes a `state` (text or JSON string) and typed `questions` of type `choice` / `score` / `noul`; a `choice` question has `instructions` and `criteria` (option → description). Fine-tuning uses `laya-train` from a CSV (text column + label column) or JSONL where each line has `state`, `questions`, and `expected` (one answer per question) or `gold` (teacher probabilities); calibration temperatures are fit on a held-out slice; the Kaggle notebook runs on free 2×T4 GPUs. The dataset produced this week must therefore be exportable in exactly this `laya-train` JSONL schema.

---

## Shared foundations (recap — established in full in Week 1)

- **Repo layout:** `index-advisor/` with `config/`, `src/advisor/`, `scripts/`, `data/`, `tests/`. This week adds behavior to `src/advisor/scenarios.py` (scenario generator), `labeler.py`, `baselines.py`, and `scripts/generate_labels.py`, plus `tests/test_labeler.py`.
- **Schema (bounded, disposable):** `customers` (~50k), `orders` (~500k), `order_items` (~2M); only PKs/FKs by default so all candidate indexes start absent.
- **Workload (`config/workload.yaml`):** two named mixes — `baseline` (80% read / 20% write) and `shifted` — used to build a train/test holdout by workload template/mix.
- **Candidate-index rules:** single-column B-tree indexes on WHERE/JOIN/GROUP BY/ORDER BY columns; absent ⇒ {CREATE, NO-OP}, present ⇒ {KEEP, DROP}.
- **Cost score (`config/cost_score.yaml`, lower is better):** `score = w_r*norm(read_runtime) + w_w*norm(write_overhead) + w_s*norm(index_storage_bytes) + w_b*norm(index_build_time)`; defaults `w_r=0.5, w_w=0.3, w_s=0.1, w_b=0.1`; each component also reported raw so trade-offs stay visible. This is a **declared cost objective**, not objective truth; the weights encode a value judgment that is stated, inspectable, and tested for robustness (Task 7). **This week's labeler selects the min-cost action under this objective as the reference action.**
- **Shared decision-state:** one canonical dict per (scenario, candidate) with fields `query_frequency`, `est_selectivity`, `est_cardinality`, `explain_startup_cost`, `explain_total_cost`, `explain_plan_rows`, `index_exists`, `index_scans_observed`, `index_size_bytes`, `workload_write_rate`, plus identifiers; serialized as a flat tabular row **and** a Laya `state` + typed `choice` payload. Produced by Week 2's `src/advisor/state.py`.
- **CI/CD:** GitHub + trunk-based PRs; `ci.yml` runs `ruff` + `pytest` with a PostgreSQL service container (added in Week 2). All DB work stays on an isolated, disposable test database; no shared/production instance is touched.

---

## Problem statement

Produce the **reference-policy action labels** by actually measuring each action's outcome on cloned database states and selecting the best action under the *declared cost objective* (not an objective ground truth), and implement the two non-learned advisors the models must beat. A dataset of only `candidates × 2 mixes` (≈12 rows) is far too small to fine-tune Laya, which needs on the order of hundreds of labeled decisions to rise above chance; so this week must also **generate enough varied scenarios** to make the dataset large enough to train on.

## Goals

A **scenario generator** that produces enough distinct, varied decision situations for the dataset to be large enough to fine-tune Laya; a labeler that, per candidate, clones state, applies each allowed action, replays the workload, scores outcomes with the Week-1 cost objective, and records the lowest-cost action as the **reference action** (emitting a confident label on a clear win and a probability distribution on a near-tie); a no-index baseline and a rule-based advisor; a weight **sensitivity analysis** that shows whether the method ranking is stable across cost objectives; a labeled decision dataset exported as CSV (tabular) and JSONL (Laya `laya-train` schema; its per-line answer field is named `expected` for a hard label, or `gold` for teacher probabilities) with a documented train/test holdout.

---

## Tasks

### Task 1 — Build the scenario generator (dataset scale)

- **Objective:** Produce enough distinct, varied decision situations that the labeled dataset is large enough to fine-tune Laya (target: on the order of several hundred labeled decisions, not ~12), while keeping each scenario cheap enough to label within the week's compute budget.
- **Implementation Guidance:** Implement `src/advisor/scenarios.py` to enumerate scenarios by varying the independent factors that change the right index decision, so each (scenario, candidate) pair is a distinct labeled row. Vary at least: **workload mix** (baseline, shifted, plus additional generated read/write balances), **starting index state** (candidate absent vs. already present — the only way to generate KEEP/DROP labels, not just CREATE/NO-OP), **data scale / distribution** (e.g. different seeds and column skew so selectivity and cardinality differ), and **per-template frequency perturbations**. Record every varied factor in the scenario descriptor so a row is fully reproducible and the factors are available as context. Keep per-scenario cost bounded by using the Week-1 `scale` factor (miniaturized DBs) for the bulk of scenarios and reserving full-size runs for a validation subset. Deterministic seeds so the scenario set regenerates identically.
- **Test Requirements:** `tests/test_scenarios.py` asserts the generator yields the configured number of scenarios, that both index-state starts occur (so KEEP/DROP are reachable, not only CREATE/NO-OP), that varied factors are recorded per scenario, and that regeneration with the same seed is identical.
- **Demo:** Print the scenario count and a breakdown by varied factor (mixes × index-states × data variants), and the resulting projected dataset size (`scenarios × candidates`).

### Task 2 — Implement the controlled index-comparison labeler

- **Objective:** Assign each candidate a **reference action** by measuring the real outcome of every allowed action under the declared cost objective (a reference under a stated objective, not objective ground truth).
- **Implementation Guidance:** Implement `src/advisor/labeler.py` to, per (scenario, candidate): clone the DB state; for each allowed action (CREATE/NO-OP or KEEP/DROP) apply it, run the Week-2 replay **under the Week-2 measurement protocol** (warm-up/measured phases, server-side `pg_stat_statements` timing, background-activity controls, repeat-and-median with spread), and compute the Week-1 cost score (+ raw components); select the **min-cost** action as the reference action; record per-action scores (and their raw components) for transparency so the choice can be audited and re-scored under different weights. Apply index DDL and record `index_build_time` **outside** the measured read/write window so build cost never contaminates `write_overhead`. Because the same workload is measured on cloned states that differ only in the index, this is a **differential (A/B) measurement**: shared overhead cancels in the comparison. **Emit label confidence from the measured spread:** when the best action's score is clearly separated from the alternative (gap larger than the measured spread), record a confident single answer; when the actions are a **near-tie** (scores within the spread), record a probability distribution over the actions (derived from the score gap) rather than forcing a hard winner. Both forms are carried through to the export (Task 5): a confident label becomes `expected`, a near-tie becomes `gold` teacher probabilities — the two label shapes Laya's `laya-train` reads.
- **Test Requirements:** `tests/test_labeler.py` on a seeded scenario asserts deterministic reference labels across repeats; that an obviously-beneficial index yields a confident CREATE and an obviously-useless one a confident NO-OP; and that a constructed near-tie (two actions with scores inside the spread) produces a probability distribution rather than an arbitrary hard label.
- **Demo:** Print, for one clear-win candidate and one near-tie candidate, each action's measured score + spread and the resulting label (confident `expected` vs. `gold` distribution).

### Task 3 — Implement the no-index baseline advisor

- **Objective:** Provide the lower-bound advisor that never maintains candidate indexes.
- **Implementation Guidance:** Implement in `src/advisor/baselines.py` an advisor that always chooses NO-OP/DROP (never maintains candidate indexes); produce its decisions and measured cost over a scenario.
- **Test Requirements:** Asserts it emits a valid action for every candidate and that its decisions are reproducible.
- **Demo:** Run it over a scenario and print its total measured cost.

### Task 4 — Implement the rule-based (heuristic) advisor

- **Objective:** Provide a transparent heuristic baseline the learned models must outperform.
- **Implementation Guidance:** Implement a heuristic advisor in `src/advisor/baselines.py` — e.g., CREATE when `query_frequency` and `est_selectivity` exceed thresholds and `write_rate` is below a bound; otherwise NO-OP; KEEP if observed `index_scans` is high, else DROP. Thresholds live in `config/`.
- **Test Requirements:** Asserts decisions follow the documented rules on constructed feature rows and that all actions are reachable.
- **Demo:** Run it over a scenario and print decisions with the triggering rule per candidate.

### Task 5 — Generate and export the labeled decision dataset with holdout

- **Objective:** Produce the dual-format dataset (features + reference action) over all generated scenarios, with a disjoint-template train/test split, large enough to fine-tune Laya.
- **Implementation Guidance:** Implement `scripts/generate_labels.py` to run the labeler (Task 2) across **every scenario from the generator (Task 1)**, attach each candidate's decision state, and export: `data/decisions.csv` (tabular + reference action) for scikit-learn/XGBoost, and `data/decisions.jsonl` in Laya `laya-train` schema. For the JSONL, choose the label field per decision from Task 2's confidence: a confident decision is written as `expected` (a single answer), a near-tie is written as `gold` (teacher probabilities over the actions) — matching the two label shapes `laya-train` accepts. The CSV carries the reference action plus, for near-ties, the probability so no information is lost across formats. Define and record a train/test split by workload template/mix so train and test never share templates.
- **Test Requirements:** Asserts row counts equal `generated-scenarios × candidates`; that the dataset reaches the configured minimum size target for fine-tuning; that the reference-action distribution is reported across all four actions and that all four (CREATE/KEEP/DROP/NO-OP) actually occur (confirming Task 1 produced both index-state starts); that CSV and JSONL encode the same decisions/labels including near-tie probabilities; that each JSONL line uses `expected` or `gold` consistent with its confidence; and that train/test template sets are disjoint.
- **Demo:** `python scripts/generate_labels.py` produces both files; print the total row count, the per-action reference-label distribution, the count of confident-vs-near-tie labels, and the holdout split.

### Task 6 — Integrate baselines and labels into a single comparison report

- **Objective:** Quantify the gap between each baseline and the **reference action** — the gap the learned models will try to close.
- **Implementation Guidance:** Add a small reporting step that, for one scenario, tabulates each baseline's chosen action and measured cost against the labeler's reference action/cost.
- **Test Requirements:** Asserts the report contains one row per candidate with baseline action, baseline cost, reference action, reference cost, and cost gap.
- **Demo:** Generate the report and show where no-index and rule-based diverge from the reference action.

### Task 7 — Weight sensitivity analysis of the reference policy

- **Objective:** Show whether the reference labels — and therefore any downstream method comparison — are an artifact of one arbitrary weight choice, converting the cost objective's subjectivity from a hidden assumption into a tested, reported property.
- **Implementation Guidance:** Because the labeler records per-action raw components (Task 1), reference labels can be **re-derived under alternative weightings without re-running the expensive replays**. Implement a step that re-scores the stored per-action components under several declared weightings (e.g. read-heavy `0.5/0.3/0.1/0.1`, balanced `0.25/0.25/0.25/0.25`, write-sensitive `0.3/0.6/0.05/0.05`) and reports, per weighting, the reference-action distribution and how many candidates change label versus the default. (Model-ranking stability under each weighting is completed in Weeks 4–5 once models exist; this task establishes the label-level sensitivity and the re-scoring machinery.)
- **Test Requirements:** Asserts re-scoring is driven purely by stored raw components (no replay), that at least the three named weightings produce reports, and that label-change counts are computed relative to the default weighting.
- **Demo:** Print a small table: weighting → reference-action distribution → number of candidates whose label changed from the default.

---

## Definition of Done (Week 3)

A **scenario generator** produces a varied set of decision situations (across mixes, starting index states, and data variants) large enough to fine-tune Laya; a reproducible decision dataset labeled with the **reference action** exists in both tabular (CSV) and Laya (JSONL) formats — using confident `expected` labels for clear wins and `gold` probability labels for measured near-ties — with all four actions (CREATE/KEEP/DROP/NO-OP) represented and a disjoint-template holdout; no-index and rule-based baselines are implemented, tested, and benchmarked against the reference action; a comparison report quantifies the baseline-vs-reference gap; and a weight **sensitivity analysis** reports how the reference labels shift across alternative cost objectives (establishing that the labels are a declared reference, not objective truth).

**Demoable increment:** run label generation end-to-end and present the dataset size, the per-action reference-label distribution, the confident-vs-near-tie label counts, the baseline-vs-reference comparison, and the label sensitivity table — the ~60% substrate for Milestone 1 on 11/3.
