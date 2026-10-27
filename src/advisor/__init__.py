"""index-advisor: typed decision models for PostgreSQL index lifecycle management.

Week 1 modules (this milestone):
    * db             - connection + isolated test-DB helpers
    * schema_loader  - apply schema.sql + seed bounded synthetic data
    * workload       - parse workload.yaml, generate query instances
    * candidates     - enumerate candidate indexes + allowed actions
    * cost           - the evaluation cost score

Week 2-3 modules (replay, features, state, labeler, baselines) are added later.
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
