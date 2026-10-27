"""Apply the schema and seed bounded, deterministic synthetic data.

Row targets (see docs/experiment-spec.md):
    customers   ~   50,000
    orders      ~  500,000
    order_items ~ 2,000,000

Seeding is deterministic (fixed RNG seed) so every run produces the same data
and labels downstream are reproducible. Data is streamed to the server with
``COPY`` for speed. A ``scale`` factor lets tests seed a miniature database
(e.g. scale=0.001) while preserving the table proportions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import CONFIG_DIR
from .db import DBConfig, connect

SCHEMA_SQL = CONFIG_DIR / "schema.sql"

# Full-size row targets.
N_CUSTOMERS = 50_000
N_ORDERS = 500_000
N_ORDER_ITEMS = 2_000_000

COUNTRIES = ["US", "CA", "GB", "DE", "FR", "IN", "BR", "JP", "AU", "MX"]
ORDER_STATUSES = ["NEW", "PAID", "SHIPPED", "CANCELLED", "REFUNDED"]

_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
DEFAULT_SEED = 257


@dataclass(frozen=True)
class SeedCounts:
    customers: int
    orders: int
    order_items: int

    @classmethod
    def for_scale(cls, scale: float) -> SeedCounts:
        """Scale the full targets, keeping at least 1 row per table."""
        return cls(
            customers=max(1, int(N_CUSTOMERS * scale)),
            orders=max(1, int(N_ORDERS * scale)),
            order_items=max(1, int(N_ORDER_ITEMS * scale)),
        )


def apply_schema(cfg: DBConfig | None = None, *, schema_path: Path = SCHEMA_SQL) -> None:
    """Execute schema.sql against the target database (creates tables, PK/FK only)."""
    sql = schema_path.read_text(encoding="utf-8")
    with connect(cfg) as conn:
        conn.execute(sql)


def seed_data(
    cfg: DBConfig | None = None,
    *,
    scale: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> SeedCounts:
    """Populate the three tables with deterministic synthetic rows via COPY."""
    counts = SeedCounts.for_scale(scale)
    rng = random.Random(seed)

    with connect(cfg) as conn:
        _copy_customers(conn, counts.customers, rng)
        _copy_orders(conn, counts.orders, counts.customers, rng)
        _copy_order_items(conn, counts.order_items, counts.orders, rng)
    return counts


def _ts(rng: random.Random, span_days: int = 365) -> str:
    offset = timedelta(
        days=rng.randint(0, span_days),
        seconds=rng.randint(0, 86_399),
    )
    return (_EPOCH + offset).isoformat()


def _copy_customers(conn, n: int, rng: random.Random) -> None:
    with conn.cursor().copy(
        "COPY customers (customer_id, name, email, country, created_at) FROM STDIN"
    ) as copy:
        for cid in range(1, n + 1):
            country = COUNTRIES[rng.randrange(len(COUNTRIES))]
            copy.write_row((cid, f"Customer {cid}", f"user{cid}@example.com", country, _ts(rng)))


def _copy_orders(conn, n: int, n_customers: int, rng: random.Random) -> None:
    with conn.cursor().copy(
        "COPY orders (order_id, customer_id, status, total_amount, created_at) FROM STDIN"
    ) as copy:
        for oid in range(1, n + 1):
            customer_id = rng.randint(1, n_customers)
            status = ORDER_STATUSES[rng.randrange(len(ORDER_STATUSES))]
            total = round(rng.uniform(1.0, 2000.0), 2)
            copy.write_row((oid, customer_id, status, total, _ts(rng)))


def _copy_order_items(conn, n: int, n_orders: int, rng: random.Random) -> None:
    with conn.cursor().copy(
        "COPY order_items (order_item_id, order_id, product_id, quantity, unit_price) FROM STDIN"
    ) as copy:
        for iid in range(1, n + 1):
            order_id = rng.randint(1, n_orders)
            product_id = rng.randint(1, 10_000)
            quantity = rng.randint(1, 10)
            unit_price = round(rng.uniform(1.0, 500.0), 2)
            copy.write_row((iid, order_id, product_id, quantity, unit_price))


def table_counts(cfg: DBConfig | None = None) -> dict[str, int]:
    """Return row counts for the three tables."""
    out: dict[str, int] = {}
    with connect(cfg) as conn:
        for table in ("customers", "orders", "order_items"):
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            out[table] = int(row[0])
    return out


def candidate_indexes_present(cfg: DBConfig | None = None) -> list[str]:
    """Return names of any non-primary-key indexes (should be empty at seed time).

    Primary-key indexes (``*_pkey``) are expected; anything else would be a
    candidate index that must NOT exist in a freshly seeded database.
    """
    with connect(cfg) as conn:
        rows = conn.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename IN ('customers', 'orders', 'order_items')
              AND indexname NOT LIKE '%_pkey'
            ORDER BY indexname
            """
        ).fetchall()
    return [r[0] for r in rows]
