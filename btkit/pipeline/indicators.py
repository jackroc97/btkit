"""
IndicatorRunner — loads and executes user-supplied indicator scripts.

Each script must expose a compute(df: pl.DataFrame) -> pl.DataFrame function.
The function receives underlying_bars columns and must return the same DataFrame
with one or more indicator columns appended. Each appended column becomes its own
independent series in indicator_bars, stored under its column name.

Optionally, compute() may accept a second argument — an IndicatorContext object —
to access option_bars and option_greeks data in addition to underlying bars:

    def compute(df: pl.DataFrame, ctx: IndicatorContext) -> pl.DataFrame:
        greeks = ctx.option_greeks(dte_max=30)
        ...

The runner detects the arity of compute() and passes the context automatically.
Scripts that only declare compute(df) are unaffected.

Multiple output columns from a single script are supported — each becomes its own
row in indicator_definition and its own series in indicator_bars.

External imports within scripts are permitted; the user is responsible for ensuring
those modules are installed in the active environment.
"""

from __future__ import annotations

import importlib.util
import inspect
from datetime import datetime
from pathlib import Path
from types import ModuleType

import duckdb
import polars as pl


class IndicatorContext:
    """
    Data-access helper passed to indicator scripts that need option data.

    Provides lazy, filtered access to option_bars and option_greeks for the
    same underlying and time window as the underlying_bars DataFrame.  Data
    is only fetched when the script calls a method — scripts that never call
    option_greeks() or option_bars() incur no overhead.

    Attributes
    ----------
    underlying_id : int
        The instrument_id of the underlying being processed.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        underlying_id: int,
        start: datetime,
        end: datetime,
    ) -> None:
        self._con = con
        self.underlying_id = underlying_id
        self._start = start
        self._end = end

    def option_greeks(
        self,
        dte_min: int | None = None,
        dte_max: int | None = None,
        delta_min: float | None = None,
        delta_max: float | None = None,
    ) -> pl.DataFrame:
        """
        Return option greeks for all options on this underlying over the same
        time window as the underlying_bars DataFrame passed to compute().

        Columns: ts_event, instrument_id, dte, T, iv, delta, gamma, theta, vega

        All filter parameters are optional and combined with AND logic.

        Parameters
        ----------
        dte_min:
            Keep only options with dte >= this value.
        dte_max:
            Keep only options with dte <= this value.  Use this to limit
            result size when you only care about near-term options (e.g.
            dte_max=30 for a 0–30 DTE vol surface).
        delta_min:
            Keep only options with delta >= this value (e.g. delta_min=-0.50
            to exclude deep ITM puts).
        delta_max:
            Keep only options with delta <= this value.
        """
        clauses = ["underlying_id = ?", "ts_event >= ?", "ts_event <= ?"]
        params: list = [self.underlying_id, self._start, self._end]

        if dte_min is not None:
            clauses.append("dte >= ?")
            params.append(dte_min)
        if dte_max is not None:
            clauses.append("dte <= ?")
            params.append(dte_max)
        if delta_min is not None:
            clauses.append("delta >= ?")
            params.append(float(delta_min))
        if delta_max is not None:
            clauses.append("delta <= ?")
            params.append(float(delta_max))

        where = " AND ".join(clauses)
        return self._con.execute(
            f"""
            SELECT ts_event, instrument_id, dte, T, iv, delta, gamma, theta, vega
            FROM option_greeks
            WHERE {where}
            ORDER BY ts_event
            """,
            params,
        ).pl()

    def option_bars(
        self,
        dte_min: int | None = None,
        dte_max: int | None = None,
        right: str | None = None,
    ) -> pl.DataFrame:
        """
        Return option OHLCV bars for all options on this underlying over the
        same time window as the underlying_bars DataFrame passed to compute().

        Columns: ts_event, instrument_id, symbol, expiration, strike_price,
                 right, multiplier, open, high, low, close, volume

        All filter parameters are optional and combined with AND logic.

        Parameters
        ----------
        dte_min:
            Keep only options whose expiration is at least dte_min calendar
            days after the bar's date.
        dte_max:
            Keep only options whose expiration is at most dte_max calendar
            days after the bar's date.  Use this to avoid loading the entire
            option chain history (e.g. dte_max=45 for a 0–45 DTE study).
        right:
            ``'call'`` or ``'put'`` — filter by option type.
        """
        clauses = ["underlying_id = ?", "ts_event >= ?", "ts_event <= ?"]
        params: list = [self.underlying_id, self._start, self._end]

        if dte_min is not None:
            clauses.append("DATEDIFF('day', ts_event::DATE, expiration) >= ?")
            params.append(dte_min)
        if dte_max is not None:
            clauses.append("DATEDIFF('day', ts_event::DATE, expiration) <= ?")
            params.append(dte_max)
        if right is not None:
            clauses.append('"right" = ?')
            params.append(right)

        where = " AND ".join(clauses)
        return self._con.execute(
            f"""
            SELECT ts_event, instrument_id, symbol, expiration, strike_price,
                   "right", multiplier, open, high, low, close, volume
            FROM option_bars
            WHERE {where}
            ORDER BY ts_event
            """,
            params,
        ).pl()


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
        self._wants_context: bool = self._detect_context_param()

    def run(self, underlying_id: int) -> None:
        """
        Execute the indicator script against the underlying's bars.

        Steps:
            1. Load all underlying_bars for underlying_id.
            2. Build an IndicatorContext if compute() accepts a second argument.
            3. Call compute(df) or compute(df, ctx) → wide DataFrame with
               indicator columns appended.
            4. For each appended indicator column:
                a. Upsert a row in indicator_definition.
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

        # Step 2 & 3: Call compute(), optionally with context
        if self._wants_context:
            start: datetime = underlying_df["ts_event"].min()
            end: datetime = underlying_df["ts_event"].max()
            ctx = IndicatorContext(self.con, underlying_id, start, end)
            result_df = self._module.compute(underlying_df, ctx)
        else:
            result_df = self._module.compute(underlying_df)

        # Identify appended columns
        input_cols = set(underlying_df.columns)
        new_cols = [c for c in result_df.columns if c not in input_cols]
        if not new_cols:
            return

        # Step 4: For each indicator column
        for col_name in new_cols:
            indicator_id = self._upsert_indicator_definition(
                name=col_name,
                underlying_id=underlying_id,
                underlying_symbol=underlying_symbol,
            )
            self._write_indicator_bars(result_df, col_name, indicator_id)

    def _detect_context_param(self) -> bool:
        """Return True if compute() declares a second positional parameter."""
        sig = inspect.signature(self._module.compute)
        positional_kinds = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        positional_params = [p for p in sig.parameters.values() if p.kind in positional_kinds]
        return len(positional_params) >= 2

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
        self.con.execute(
            """
            INSERT INTO indicator_bars
            SELECT t.* FROM _indicator_batch t
            WHERE NOT EXISTS (
                SELECT 1 FROM indicator_bars ib
                WHERE ib.ts_event = t.ts_event AND ib.indicator_id = t.indicator_id
            )
            """
        )
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
