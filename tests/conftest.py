"""
Shared pytest fixtures for btkit tests.

Fixtures are organised by scope:
  - session-scoped: heavyweight resources built once (InputDatabase, test DB path)
  - function-scoped: output DBs that need a clean slate per test
"""

from __future__ import annotations

from pathlib import Path

import pytest

from btkit.db.input_db import InputDatabase
from btkit.db.output_db import OutputDatabase

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
STRATEGIES_DIR = FIXTURES_DIR / "strategies"
INPUT_DB_PATH = Path(__file__).parent / "output" / "input.db"


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def input_db() -> InputDatabase:
    """Shared read-only InputDatabase for the pre-built test fixture DB."""
    if not INPUT_DB_PATH.exists():
        pytest.skip(
            f"Test input DB not found at {INPUT_DB_PATH}. "
            "Run: btkit build --data-path tests/fixtures/data/ "
            "--db-path tests/output/input.db "
            "--indicators tests/fixtures/indicators.py"
        )
    return InputDatabase(str(INPUT_DB_PATH))


@pytest.fixture
def output_db(tmp_path: Path) -> OutputDatabase:
    """Fresh OutputDatabase for each test — written to a pytest tmp_path."""
    db = OutputDatabase(str(tmp_path / "output.db"))
    db.create_schema()
    return db
