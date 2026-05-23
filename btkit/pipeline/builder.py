"""
DatabaseBuilder — orchestrates the full input database build from raw Databento files.

Build sequence:
    1. _ingest_definitions()  — read .dbn definition files, build instrument map
    2. _ingest_ohlcv()        — read .dbn OHLCV files, write underlying_bars and
                                option_bars (with definition metadata pre-joined)
    3. _compute_greeks()      — run GreeksCalculator over option_bars
    4. _run_indicators()      — run each user indicator script via IndicatorRunner

raw_data_path must be a directory containing one or more .zip files downloaded from
Databento. The builder auto-detects schema type by inspecting filenames inside each zip:
    - Files matching *.ohlcv-*.dbn.zst  → OHLCV bars
    - Files matching *.definition.dbn.zst → instrument definitions

Called from cli.build() and cli.pipeline().
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator

import duckdb
import polars as pl

from btkit.db.input_db import INPUT_SCHEMA_SQL
from btkit.pipeline.greeks import GreeksCalculator
from btkit.pipeline.indicators import IndicatorRunner

try:
    import databento as db_sdk
except ImportError as exc:  # pragma: no cover
    raise ImportError("databento package required: pip install databento") from exc


# ---------------------------------------------------------------------------
# Internal instrument map record
# ---------------------------------------------------------------------------

class _InstrumentInfo:
    __slots__ = ("instrument_class", "raw_symbol", "expiration", "strike_price", "right", "multiplier", "underlying_id")

    def __init__(
        self,
        instrument_class: str,
        raw_symbol: str,
        expiration,        # datetime.date or None
        strike_price: float | None,
        right: str | None,  # 'C', 'P', or None for futures
        multiplier: int,
        underlying_id: int,
    ) -> None:
        self.instrument_class = instrument_class
        self.raw_symbol = raw_symbol
        self.expiration = expiration
        self.strike_price = strike_price
        self.right = right
        self.multiplier = multiplier
        self.underlying_id = underlying_id


# ---------------------------------------------------------------------------
# DatabaseBuilder
# ---------------------------------------------------------------------------

class DatabaseBuilder:
    def __init__(
        self,
        raw_data_path: str | Path,
        db_path: str | Path,
        indicator_scripts: list[str | Path] | None = None,
    ) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.db_path = Path(db_path)
        self.indicator_scripts = [Path(p) for p in (indicator_scripts or [])]
        self._instrument_map: dict[int, _InstrumentInfo] = {}

    def build(self) -> None:
        """
        Full build: ingest → greeks → indicators.
        Creates or overwrites the database at db_path.
        """
        # Open a single writable connection for the entire build.
        self._con = duckdb.connect(str(self.db_path))
        try:
            self._con.execute(INPUT_SCHEMA_SQL)
            self._ingest_definitions()
            self._ingest_ohlcv()
            self._compute_greeks()
        finally:
            self._con.close()
            self._con = None

        self._run_indicators()

    # ------------------------------------------------------------------
    # Step 1: Definitions
    # ------------------------------------------------------------------

    def _ingest_definitions(self) -> None:
        """
        Read all *.definition.dbn.zst files from zip archives in raw_data_path.
        Build self._instrument_map: instrument_id → _InstrumentInfo.

        Each day's definition file covers the same instruments with the same static
        attributes. We process files in chronological order and skip instrument_ids
        already present — first-seen wins.

        Only instrument_class F (future), C (call), and P (put) are kept.
        """
        processed = 0
        for data in self._iter_dbn_files("definition"):
            store = db_sdk.DBNStore.from_bytes(data)
            df = store.to_df()

            # Keep only futures and options; drop spreads, T, M, etc.
            df = df[df["instrument_class"].isin(["F", "C", "P"])]

            for row in df.itertuples(index=False):
                iid = row.instrument_id
                if iid in self._instrument_map:
                    continue  # first-seen wins

                icls = row.instrument_class
                # Extract date from expiration datetime (timezone-aware)
                exp = row.expiration
                exp_date = exp.date() if hasattr(exp, "date") and exp.date().year > 1970 else None

                right = icls if icls in ("C", "P") else None
                strike = float(row.strike_price) if icls in ("C", "P") else None
                # unit_of_measure_qty is the actual multiplier (e.g. 50 for ES options).
                # contract_multiplier is always INT32_MAX (sentinel = not set).
                multiplier = int(row.unit_of_measure_qty) if row.unit_of_measure_qty > 0 else 50
                underlying_id = int(row.underlying_id)

                self._instrument_map[iid] = _InstrumentInfo(
                    instrument_class=icls,
                    raw_symbol=row.raw_symbol,
                    expiration=exp_date,
                    strike_price=strike,
                    right=right,
                    multiplier=multiplier,
                    underlying_id=underlying_id,
                )
                processed += 1

        print(f"[ingest] Loaded {processed} instruments from definitions ({len(self._instrument_map)} unique)")

    # ------------------------------------------------------------------
    # Step 2: OHLCV
    # ------------------------------------------------------------------

    def _ingest_ohlcv(self) -> None:
        """
        Read all *.ohlcv-*.dbn.zst files, split rows into underlying_bars and
        option_bars using the instrument_map built in _ingest_definitions(), and
        write both tables to the input database.

        OHLCV rows whose instrument_id is not in the instrument_map are skipped
        (they are spreads or other unsupported instrument classes).
        """
        if not self._instrument_map:
            raise RuntimeError("_ingest_ohlcv() called before _ingest_definitions()")

        # Collect all OHLCV DataFrames (usually just one file)
        ohlcv_frames: list[pl.DataFrame] = []
        for data in self._iter_dbn_files("ohlcv"):
            store = db_sdk.DBNStore.from_bytes(data)
            df = _dbn_to_polars(store)
            ohlcv_frames.append(df)

        if not ohlcv_frames:
            raise RuntimeError(f"No OHLCV files found in {self.raw_data_path}")

        ohlcv = pl.concat(ohlcv_frames)

        # Build lookup DataFrames from instrument_map for fast joining.
        underlying_ids = {
            iid for iid, info in self._instrument_map.items()
            if info.instrument_class == "F"
        }
        option_ids = {
            iid for iid, info in self._instrument_map.items()
            if info.instrument_class in ("C", "P")
        }

        # ── Underlying bars ──────────────────────────────────────────────────
        ub = ohlcv.filter(pl.col("instrument_id").is_in(underlying_ids)).select([
            "ts_event", "instrument_id", "symbol",
            "open", "high", "low", "close", "volume",
        ])

        # ── Option bars ──────────────────────────────────────────────────────
        # Join definition metadata (expiration, strike_price, right, multiplier,
        # underlying_id) from the instrument map into each option bar row.
        opt_raw = ohlcv.filter(pl.col("instrument_id").is_in(option_ids))

        if not opt_raw.is_empty():
            # Build a small metadata DataFrame from the instrument_map
            meta_rows = [
                {
                    "instrument_id": iid,
                    "underlying_id": info.underlying_id,
                    "expiration": info.expiration,
                    "strike_price": info.strike_price,
                    "right": info.right,
                    "multiplier": info.multiplier,
                }
                for iid, info in self._instrument_map.items()
                if info.instrument_class in ("C", "P")
            ]
            meta_df = pl.DataFrame(meta_rows, schema={
                "instrument_id": pl.Int64,
                "underlying_id": pl.Int64,
                "expiration": pl.Date,
                "strike_price": pl.Float64,
                "right": pl.Utf8,
                "multiplier": pl.Int64,
            })
            ob = opt_raw.join(meta_df, on="instrument_id", how="left").select([
                "ts_event", "instrument_id", "underlying_id", "symbol",
                "expiration", "strike_price", "right", "multiplier",
                "open", "high", "low", "close", "volume",
            ])
        else:
            ob = pl.DataFrame(schema={
                "ts_event": pl.Datetime("us", "UTC"),
                "instrument_id": pl.Int64,
                "underlying_id": pl.Int64,
                "symbol": pl.Utf8,
                "expiration": pl.Date,
                "strike_price": pl.Float64,
                "right": pl.Utf8,
                "multiplier": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            })

        _write_df(self._con, "underlying_bars", ub)
        _write_df(self._con, "option_bars", ob)

        print(
            f"[ingest] Wrote {len(ub):,} underlying bars, "
            f"{len(ob):,} option bars"
        )

    # ------------------------------------------------------------------
    # Step 3: Greeks
    # ------------------------------------------------------------------

    def _compute_greeks(self) -> None:
        """
        Instantiate GreeksCalculator and process all option_bars in batches,
        writing results to option_greeks.
        """
        calc = GreeksCalculator(self._con)
        calc.run()

    # ------------------------------------------------------------------
    # Step 4: Indicators
    # ------------------------------------------------------------------

    def _run_indicators(self) -> None:
        """
        Instantiate one IndicatorRunner per script in indicator_scripts and run
        each against the underlying_bars for the relevant underlying.
        Skipped if indicator_scripts is empty.
        """
        if not self.indicator_scripts:
            return

        # Re-open a writable connection for indicator writes.
        con = duckdb.connect(str(self.db_path))
        try:
            underlyings = con.execute(
                "SELECT DISTINCT instrument_id FROM underlying_bars"
            ).fetchall()
            for script_path in self.indicator_scripts:
                runner = IndicatorRunner(con, script_path)
                for (underlying_id,) in underlyings:
                    runner.run(underlying_id)
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _iter_dbn_files(self, schema: str) -> Iterator[bytes]:
        """
        Yield raw bytes for each matching .dbn.zst file found in zip archives
        under raw_data_path.

        schema='definition' matches filenames containing '.definition.'
        schema='ohlcv'      matches filenames containing '.ohlcv-'
        """
        pattern = ".definition." if schema == "definition" else ".ohlcv-"

        zip_files = sorted(self.raw_data_path.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError(f"No zip files found in {self.raw_data_path}")

        for zip_path in zip_files:
            with zipfile.ZipFile(zip_path) as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith(".dbn.zst") and pattern in name:
                        yield zf.read(name)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _dbn_to_polars(store) -> pl.DataFrame:
    """Convert a DBNStore to a Polars DataFrame with a ts_event column."""
    pdf = store.to_df()
    # ts_event is the pandas index — reset it to a regular column
    pdf = pdf.reset_index()
    return pl.from_pandas(pdf).select([
        "ts_event", "instrument_id", "symbol",
        "open", "high", "low", "close", "volume",
    ])


def _write_df(con: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame) -> None:
    """Register a Polars DataFrame and INSERT INTO the target table."""
    if df.is_empty():
        return
    con.register("_write_tmp", df)
    con.execute(f"INSERT INTO {table} SELECT * FROM _write_tmp")
    con.unregister("_write_tmp")
