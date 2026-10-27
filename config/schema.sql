-- Order-processing schema for the index-advisor experiments.
--
-- Design rules (see docs/experiment-spec.md):
--   * Bounded sizes so a full run is practical on a laptop.
--   * ONLY primary keys and foreign keys are indexed by default. Every
--     candidate (non-key) index starts ABSENT so that the actions
--     CREATE / NO-OP (absent) and KEEP / DROP (present) are all reachable.
--   * All objects live in the target (disposable) database; this file is
--     idempotent so it can be re-applied to a fresh database.

DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ~50,000 rows
CREATE TABLE customers (
    customer_id  BIGINT       PRIMARY KEY,
    name         TEXT         NOT NULL,
    email        TEXT         NOT NULL,
    country      TEXT         NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL
);

-- ~500,000 rows
CREATE TABLE orders (
    order_id      BIGINT       PRIMARY KEY,
    customer_id   BIGINT       NOT NULL REFERENCES customers (customer_id),
    status        TEXT         NOT NULL,
    total_amount  NUMERIC(12, 2) NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL
);

-- ~2,000,000 rows
CREATE TABLE order_items (
    order_item_id  BIGINT        PRIMARY KEY,
    order_id       BIGINT        NOT NULL REFERENCES orders (order_id),
    product_id     BIGINT        NOT NULL,
    quantity       INTEGER       NOT NULL,
    unit_price     NUMERIC(12, 2) NOT NULL
);

-- NOTE: No index is created on the foreign-key columns orders.customer_id or
-- order_items.order_id. PostgreSQL does NOT auto-create indexes for foreign
-- keys, so these columns remain candidate indexes (absent by default), exactly
-- as the experiment requires.
