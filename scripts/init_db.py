"""One-command disposable-database bring-up and seed.

Usage:
    python scripts/init_db.py                 # full-size seed
    python scripts/init_db.py --scale 0.01    # 1% miniature (fast)
    python scripts/init_db.py --no-drop       # keep an existing DB, (re)apply schema

Builds a fresh isolated test database, applies config/schema.sql (PK/FK only),
seeds deterministic synthetic data, and prints the resulting row counts plus a
confirmation that no candidate indexes exist yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advisor import schema_loader  # noqa: E402
from advisor.db import DBConfig, create_test_db, server_available  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create and seed the disposable test DB.")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="row-count scale factor (default 1.0)"
    )
    parser.add_argument("--seed", type=int, default=schema_loader.DEFAULT_SEED, help="RNG seed")
    parser.add_argument("--no-drop", action="store_true", help="do not drop an existing DB first")
    args = parser.parse_args(argv)

    cfg = DBConfig.from_env()
    if not server_available(cfg):
        print(
            "ERROR: no reachable PostgreSQL server. Set PGHOST/PGPORT/PGUSER/PGPASSWORD "
            "(see .env.example) and ensure the server is running.",
            file=sys.stderr,
        )
        return 2

    print(f"Creating disposable database {cfg.dbname!r} on {cfg.host}:{cfg.port} ...")
    create_test_db(cfg, drop_existing=not args.no_drop)

    print("Applying schema (config/schema.sql) ...")
    schema_loader.apply_schema(cfg)

    print(f"Seeding synthetic data (scale={args.scale}, seed={args.seed}) ...")
    counts = schema_loader.seed_data(cfg, scale=args.scale, seed=args.seed)

    actual = schema_loader.table_counts(cfg)
    print("\nTable row counts:")
    for table in ("customers", "orders", "order_items"):
        print(f"  {table:<12} {actual[table]:>12,}")

    leftover = schema_loader.candidate_indexes_present(cfg)
    if leftover:
        print(f"\nWARNING: unexpected non-PK indexes present: {leftover}", file=sys.stderr)
        return 1
    print("\nNo candidate (non-PK) indexes present - CREATE/NO-OP/KEEP/DROP all reachable.")
    print(f"Seed targets for scale={args.scale}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
