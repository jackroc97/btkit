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

import duckdb
import polars as pl


class IndicatorRunner:
    def __init__(self, con: duckdb.DuckDBPyConnection, script_path: str | Path) -> None:
        """
        Read and store the script source at init time. The source is later written
        to indicator_definition for reproducibility. The script is executed in the
        current Python environment.
        """
        self.con = con
        self.script_path = Path(script_path)
        self.script_source: str = self.script_path.read_text()
        self._module: ModuleType = self._load_module()

    def run(self, underlying_id: int) -> None:
        """
        Execute the indicator script against the underlying's bars.

        Steps:
            1. Load all underlying_bars for underlying_id.
            2. Call compute(df) → wide DataFrame with indicator columns appended.
            3. For each appended indicator column:
                a. Upsert a row in indicator_definition
                   (name, underlying_id, underlying_symbol, params, script_source).
                b. Melt the column to tall format (ts_event, indicator_id, value).
                c. Write tall rows to indicator_bars.
        """
        # Step 1: Load underlying bars
        underlying_df = self.con.execute(
            "SELECT ts_event, instrument_id, symbol, open, high, low, close, volume "
            "FROM underlying_bars WHERE instrument_id = ? ORDER BY ts_event",
            [underlying_id],
        ).pl()

        if underlying_df.is_empty():
            return

        underlying_symbol = underlying_df["symbol"][0]

        # Step 2: Call compute()
        result_df = self._module.compute(underlying_df)

        # Identify appended columns
        input_cols = set(underlying_df.columns)
        new_cols = [c for c in result_df.columns if c not in input_cols]
        if not new_cols:
            return

        # Step 3: For each indicator column
        for col_name in new_cols:
            indicator_id = self._upsert_indicator_definition(
                name=col_name,
                underlying_id=underlying_id,
                underlying_symbol=underlying_symbol,
            )
            self._write_indicator_bars(result_df, col_name, indicator_id)

    def _upsert_indicator_definition(
        self,
        name: str,
        underlying_id: int,
        underlying_symbol: str,
    ) -> int:
        """
        Insert indicator_definition if (name, underlying_id) not present.
        Returns the id.
        """
        row = self.con.execute(
            "SELECT id FROM indicator_definition WHERE name = ? AND underlying_id = ?",
            [name, underlying_id],
        ).fetchone()

        if row:
            return row[0]

        next_id = self.con.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM indicator_definition"
        ).fetchone()[0]

        self.con.execute(
            """
            INSERT INTO indicator_definition
                (id, name, underlying_id, underlying_symbol, params, script_source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [next_id, name, underlying_id, underlying_symbol, None, self.script_source],
        )
        return next_id

    def _write_indicator_bars(
        self,
        df: pl.DataFrame,
        col_name: str,
        indicator_id: int,
    ) -> None:
        """Melt one indicator column to tall format and insert into indicator_bars."""
        tall = df.select(
            [
                pl.col("ts_event"),
                pl.lit(indicator_id).cast(pl.Int64).alias("indicator_id"),
                pl.col(col_name).cast(pl.Float64).alias("value"),
            ]
        ).filter(pl.col("value").is_not_null())

        if tall.is_empty():
            return

        self.con.register("_indicator_batch", tall)
        self.con.execute("INSERT INTO indicator_bars SELECT * FROM _indicator_batch")
        self.con.unregister("_indicator_batch")

    def _load_module(self) -> ModuleType:
        """Dynamically load the script as a Python module."""
        spec = importlib.util.spec_from_file_location(self.script_path.stem, self.script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load indicator script: {self.script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if not hasattr(module, "compute"):
            raise AttributeError(
                f"Indicator script {self.script_path} must define a compute(df) function"
            )
        return module
