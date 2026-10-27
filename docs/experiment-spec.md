# Experiment Specification

> Frozen scope for "Evaluating Jev-Like Decision Models for Adaptive Index
> Lifecycle Management" (Jeff Huang, John Yun). Weeks 2–3 add behavior on top of
> this; they do not re-decide scope.

## Research Question

Can a Jev-like typed decision model (**Laya**), given a structured workload
state and a typed choice question, make index-lifecycle decisions —
**CREATE / KEEP / DROP / NO-OP** — for a changing PostgreSQL workload as
effectively as Random Forest, boosted-tree, and rule-based baselines, while
keeping query performance and index costs competitive?

Each decision concerns one candidate single-column index in one workload
scenario. The allowed actions depend on the current index state:

- index **absent** → `CREATE` or `NO-OP`
- index **present** → `KEEP` or `DROP`

## Success Criteria

Effectiveness is judged on **both** dimensions below; a method is not
considered good if it wins one while badly losing the other.

### Action-Prediction Accuracy

- **macro-F1** across the four actions (primary accuracy metric).
- **Per-action precision/recall** (CREATE, KEEP, DROP, NO-OP) to expose class
  imbalance and which actions a model gets wrong.
- **Probability calibration** of the model's reported confidence (Laya emits
  calibrated probabilities; baselines' `predict_proba` is assessed the same way).

### Real Database Outcomes

- **Read runtime** — aggregate measured read runtime for the workload.
- **Write overhead** — aggregate measured write overhead for the workload.
- **Storage/build cost** — index size in bytes and index build time.
- **Net cost vs. baselines** — the combined cost score (below) of following a
  method's recommendations relative to a no-index baseline and a rule-based
  heuristic.

### What Counts as a "Good" Result

A learned model (classical or Laya) is a practical index-advisor component if it:

1. matches or beats the **rule-based** baseline on macro-F1, **and**
2. produces recommendations whose **net cost score** is no worse than the
   better of {no-index, rule-based} on a held-out workload mix, **and**
3. reports **calibrated** probabilities (confidence tracks correctness).

The study's conclusion is comparative: whether Laya is competitive with
classical tabular models, not whether any single absolute threshold is hit.

## Schema

Order-processing schema, bounded so a full run is practical on a laptop. Only
primary keys and foreign keys are indexed by default; every candidate (non-key)
index starts **absent** so all four actions are reachable. DDL:
[`config/schema.sql`](../config/schema.sql).

| Table         | Key columns                          | Other columns                         | Approx. rows |
|---------------|--------------------------------------|---------------------------------------|-------------:|
| `customers`   | `customer_id` PK                     | `name, email, country, created_at`    |       50,000 |
| `orders`      | `order_id` PK, `customer_id` FK      | `status, total_amount, created_at`    |      500,000 |
| `order_items` | `order_item_id` PK, `order_id` FK    | `product_id, quantity, unit_price`    |    2,000,000 |

> PostgreSQL does not auto-index foreign keys, so `orders.customer_id` and
> `order_items.order_id` remain candidate (absent) indexes.

## Workload Mix and Frequencies

Defined in [`config/workload.yaml`](../config/workload.yaml). Two named mixes
give a train/test holdout by mix (train and test never share the same mix).

### Reads

- filter `orders` by `status` (WHERE)
- filter `orders` by `created_at` range (WHERE)
- join `customers` ↔ `orders` on `customer_id` (JOIN)
- group `orders` by `customer_id`, sum `total_amount` (GROUP BY)
- order `order_items` by `unit_price` (ORDER BY)
- join `orders` ↔ `order_items` on `order_id` (JOIN)
- filter `customers` by `country` (WHERE)

### Writes

- insert `orders`
- update `orders.status`
- insert `order_items`

### Mixes (relative weights, normalized to fractions by the loader)

| Template                        | kind  | baseline | shifted |
|---------------------------------|-------|---------:|--------:|
| read_orders_by_status           | read  |       22 |       7 |
| read_orders_by_created_range    | read  |       13 |       9 |
| read_join_customers_orders      | read  |       13 |      10 |
| read_group_orders_by_customer   | read  |       10 |      10 |
| read_order_items_by_unit_price  | read  |        8 |       8 |
| read_join_orders_items          | read  |        7 |       8 |
| read_customers_by_country       | read  |        7 |       8 |
| write_insert_order              | write |        8 |      16 |
| write_update_order_status       | write |        7 |      14 |
| write_insert_order_item         | write |        5 |      10 |
| **read / write split**          |       | **80 / 20** | **60 / 40** |

## Candidate-Index Rules

Defined in [`config/candidates.yaml`](../config/candidates.yaml). Single-column
B-tree indexes only, each justified by a WHERE/JOIN/GROUP BY/ORDER BY clause in
the workload. Action set: absent ⇒ {CREATE, NO-OP}; present ⇒ {KEEP, DROP}.

| Candidate index          | Column           | Justifying clause        |
|--------------------------|------------------|--------------------------|
| ix_orders_status         | orders.status    | WHERE                    |
| ix_orders_created_at     | orders.created_at| WHERE (range)            |
| ix_orders_customer_id    | orders.customer_id | JOIN / GROUP BY        |
| ix_customers_country     | customers.country| WHERE                    |
| ix_order_items_order_id  | order_items.order_id | JOIN (FK)            |
| ix_order_items_unit_price| order_items.unit_price | ORDER BY           |

## Evaluation Cost Score

Defined in [`config/cost_score.yaml`](../config/cost_score.yaml). Lower is
better.

```
score = w_r * norm(read_runtime)
      + w_w * norm(write_overhead)
      + w_s * norm(index_storage_bytes)
      + w_b * norm(index_build_time)

norm(x) = clamp((x - min) / (max - min), 0, 1)
```

- Default weights (tunable): `w_r=0.5, w_w=0.3, w_s=0.1, w_b=0.1` (sum = 1.0).
- Per-component min/max bounds come from a calibration run (Week 3 records the
  measured bounds; placeholders are in the config until then).
- **Every component is also reported raw** so a trade-off — e.g. faster reads
  bought with much slower writes — is never hidden behind the scalar.
- The Week 3 labeler assigns the gold action as the one whose resulting state
  yields the lowest measured score.

## How This Maps to Laya

The shared decision state is serialized two ways (implemented in Week 2):

- **Tabular** — a flat numeric/boolean feature row for scikit-learn / XGBoost.
- **Laya** — a `state` string plus a typed `choice` question whose `criteria`
  are exactly the allowed actions for the current `index_exists`. Fine-tuning
  labels follow the `laya-train` JSONL `expected` schema.
