"""Task 1 — trivial smoke test proving the package imports and CI runs."""

from __future__ import annotations

import advisor


def test_package_imports_and_has_version():
    assert isinstance(advisor.__version__, str)
    assert advisor.__version__


def test_repo_paths_exist():
    assert advisor.CONFIG_DIR.is_dir()
    assert (advisor.CONFIG_DIR / "schema.sql").is_file()
