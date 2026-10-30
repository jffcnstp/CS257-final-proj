"""index-advisor: typed decision models for PostgreSQL index lifecycle management.

Week 1 modules:
    * db             - connection + isolated test-DB helpers
    * schema_loader  - apply schema.sql + seed bounded synthetic data
    * workload       - parse workload.yaml, generate query instances
    * candidates     - enumerate candidate indexes + allowed actions
    * cost           - the evaluation cost score

Week 2 modules:
    * replay         - workload-replay controller + timing (measurement protocol)
    * features       - per-candidate feature extraction (EXPLAIN, stats, usage)
    * state          - shared decision-state serializer (dict -> tabular + Laya)

Week 3 modules (this milestone):
    * scenarios      - scenario generator (varied, reproducible decision situations)
    * labeler        - controlled A/B index-comparison -> reference action label
    * baselines      - no-index + rule-based advisors
    * sensitivity    - re-score labels under alternative cost weightings (no replay)
"""

from pathlib import Path

__version__ = "0.1.0"

# Repository paths. `__file__` is <repo>/src/advisor/__init__.py, so the repo
# root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs"

__all__ = ["__version__", "REPO_ROOT", "CONFIG_DIR", "DATA_DIR", "DOCS_DIR"]
