"""
IndicatorRunner — loads and executes user-supplied indicator scripts.

Each script must expose a compute(df: pl.DataFrame) -> pl.DataFrame function.
The function receives underlying_bars columns and must return the same DataFrame
with one or more indicator columns appended. Each appended column becomes its own
independent series in indicator_bars, stored under its column name.

Multiple output columns from a single script are supported — each becomes its own
row in indicator_definition and its own series in indicator_bars.

External imports within scripts are permitted; the user is responsible for ensuring
those modules are installed in the active environment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import polars as pl

from btkit.db.input_db import InputDatabase


class IndicatorRunner:
    def __init__(self, db: InputDatabase, script_path: str | Path) -> None:
        """
        Read and store the script source at init time. The source is later written
        to indicator_definition for reproducibility. The script is executed in the
        current Python environment.
        """
        self.db = db
        self.script_path = Path(script_path)
        self.script_source: str = self.script_path.read_text()
        self._module: ModuleType = self._load_module()

    def run(self, underlying_id: int) -> None:
        """
        Execute the indicator script against the underlying's bars.

        Steps:
            1. Load underlying_bars for underlying_id into a Polars DataFrame.
            2. Call compute(df) → wide DataFrame with indicator columns appended.
            3. For each appended indicator column:
                a. Insert or update a row in indicator_definition
                   (name, underlying_id, underlying_symbol, script_source).
                b. Melt the column to tall format (ts_event, indicator_id, value).
                c. Write tall rows to indicator_bars.
        """
        raise NotImplementedError

    def _load_module(self) -> ModuleType:
        """Dynamically load the script as a Python module."""
        spec = importlib.util.spec_from_file_location(
            self.script_path.stem, self.script_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load indicator script: {self.script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if not hasattr(module, "compute"):
            raise AttributeError(
                f"Indicator script {self.script_path} must define a compute(df) function"
            )
        return module
