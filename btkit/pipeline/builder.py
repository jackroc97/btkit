"""
DatabaseBuilder — orchestrates the full input database build from raw Databento files.

Build sequence:
    0. _collect_ohlcv_instrument_ids() — lightweight pre-scan of all OHLCV files;
                                         returns the exact set of instrument_ids
                                         that appear in price data
    1. _ingest_definitions(required_ids) — read all definition files in parallel
                                           (one worker per zip archive), merge
                                           results into a single instrument map
    2. _ingest_ohlcv()        — read .dbn OHLCV files, write underlying_bars and
                                option_bars (with definition metadata pre-joined)
    3. _compute_greeks()      — run GreeksCalculator over option_bars
    4. _run_indicators()      — run each user indicator script via IndicatorRunner

Parallelism strategy:
    - Definition ingest: one ThreadPoolExecutor worker per zip archive. Each worker
      opens its zip once, reads every *.definition.dbn.zst file sequentially, and
      returns a partial {instrument_id → _InstrumentInfo} dict. Results are merged
      in the main thread (first-seen wins; metadata is static so any snapshot gives
      the correct values). This scales to multi-year datasets where the early-stop
      optimisation saves very few files in practice.

    - OHLCV ingest: one worker per zip archive, same pattern.

    - Greeks: days are processed in a pipelined ThreadPoolExecutor — the main thread
      reads each day's batch from DuckDB and submits the numba computation to the
      pool; completed results are written back to DuckDB in the main thread while
      the next batch is already computing. numba JIT functions release the GIL, so
      threads provide near-linear speedup up to the physical core count.

raw_data_path must be a directory containing one or more .zip files downloaded from
Databento. The builder auto-detects schema type by inspecting filenames inside each zip:
    - Files matching *.ohlcv-*.dbn.zst  → OHLCV bars
    - Files matching *.definition.dbn.zst → instrument definitions

Called from cli.build() and cli.pipeline().
"""

from __future__ import annotations

import os
import time
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from btkit.db.input_db import INPUT_SCHEMA_SQL
from btkit.pipeline.greeks import GreeksCalculator
from btkit.pipeline.indicators import IndicatorRunner

try:
    import databento as db_sdk
except ImportError as exc:  # pragma: no cover
    raise ImportError("databento package required: pip install databento") from exc

# Number of threads used to read and convert zip files in parallel.
# I/O and C-extension work (dbn decompression, ndarray conversion) releases
# the GIL so threads scale well here without process-spawn overhead.
_INGEST_WORKERS = min(os.cpu_count() or 4, 8)


# ---------------------------------------------------------------------------
# Internal instrument map record
# ---------------------------------------------------------------------------


class _InstrumentInfo:
    __slots__ = (
        "instrument_class",
        "raw_symbol",
        "expiration",
        "strike_price",
        "right",
        "multiplier",
        "underlying_id",
    )

    def __init__(
        self,
        instrument_class: str,
        raw_symbol: str,
        expiration,  # datetime.date or None
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
        append: bool = False,
    ) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.db_path = Path(db_path)
        self.indicator_scripts = [Path(p) for p in (indicator_scripts or [])]
        self.append = append
        self._instrument_map: dict[int, _InstrumentInfo] = {}

    def build(self) -> None:
        """
        Full build: ingest → greeks → indicators.
        Creates or overwrites the database at db_path.
        """
        t_total = time.perf_counter()
        self._con = duckdb.connect(str(self.db_path))
        try:
            self._con.execute(INPUT_SCHEMA_SQL)

            t = time.perf_counter()
            required_ids = self._collect_ohlcv_instrument_ids()
            print(f"[ingest] Pre-scan: {len(required_ids):,} instrument IDs in OHLCV data")
            self._ingest_definitions(required_ids)
            print(f"[timing] definitions:  {_fmt_elapsed(time.perf_counter() - t)}")

            t = time.perf_counter()
            self._ingest_ohlcv()
            print(f"[timing] ohlcv ingest: {_fmt_elapsed(time.perf_counter() - t)}")

            t = time.perf_counter()
            self._compute_greeks()
            print(f"[timing] greeks:       {_fmt_elapsed(time.perf_counter() - t)}")
        finally:
            self._con.close()
            self._con = None

        t = time.perf_counter()
        self._run_indicators()
        print(f"[timing] indicators:   {_fmt_elapsed(time.perf_counter() - t)}")

        print(f"[timing] total:        {_fmt_elapsed(time.perf_counter() - t_total)}")

    # ------------------------------------------------------------------
    # Step 1: Definitions
    # ------------------------------------------------------------------

    def _collect_ohlcv_instrument_ids(self) -> set[int]:
        """
        Lightweight pre-scan: read all OHLCV files and return the complete set of
        instrument_ids that appear in price data.

        This set drives _ingest_definitions — we read definition files only until
        every id here is resolved, rather than reading all daily snapshots. See the
        module docstring for the full rationale.

        Uses to_ndarray() (not to_df()) and processes zip files in parallel via
        ThreadPoolExecutor for a significant speedup over the legacy pandas path.
        """
        zip_files = sorted(self.raw_data_path.glob("*.zip"))
        ids: set[int] = set()
        with ThreadPoolExecutor(max_workers=_INGEST_WORKERS) as pool:
            for partial in pool.map(_scan_zip_for_ids, zip_files):
                ids.update(partial)
        return ids

    def _ingest_definitions(self, required_ids: set[int]) -> None:
        """
        Read all *.definition.dbn.zst files in parallel (one worker per zip archive)
        and build self._instrument_map: instrument_id → _InstrumentInfo.

        Each worker opens its zip once, reads every definition file inside it, and
        returns a partial {instrument_id: _InstrumentInfo} dict. The main thread
        merges all partial dicts (first-seen wins). Because option/future metadata
        is static — strike, expiry, and multiplier never change — any snapshot is
        equivalent and the merge order does not affect correctness.

        Only instrument_class F (future), C (call), and P (put) are kept.
        Spreads, calendar spreads, and strategy instruments are filtered out.
        """
        zip_files = sorted(self.raw_data_path.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError(f"No zip files found in {self.raw_data_path}")

        total_files = 0
        with ThreadPoolExecutor(max_workers=_INGEST_WORKERS) as pool:
            for partial_map, files_read in pool.map(_scan_zip_for_definitions, zip_files):
                total_files += files_read
                for iid, info in partial_map.items():
                    if iid not in self._instrument_map:
                        self._instrument_map[iid] = info

        missing = required_ids - set(self._instrument_map)
        if missing:
            print(
                f"[ingest] Warning: {len(missing)} instrument IDs found in OHLCV "
                f"data have no matching definition record and will be dropped."
            )

        print(
            f"[ingest] Loaded {len(self._instrument_map):,} instruments "
            f"from {total_files} definition file(s) across {len(zip_files)} zip(s)"
        )

    # ------------------------------------------------------------------
    # Step 2: OHLCV
    # ------------------------------------------------------------------

    def _ingest_ohlcv(self) -> None:
        """
        Read all *.ohlcv-*.dbn.zst files one at a time, split rows into
        underlying_bars and option_bars using the instrument_map built in
        _ingest_definitions(), and write each file's rows immediately.

        Processing per file avoids loading all files into memory at once,
        which would be prohibitive for multi-month datasets.
        """
        if not self._instrument_map:
            raise RuntimeError("_ingest_ohlcv() called before _ingest_definitions()")

        # Build metadata lookup DataFrames once — reused for every zip file.
        futures_meta = pl.DataFrame(
            [
                {"instrument_id": iid, "expiration": info.expiration}
                for iid, info in self._instrument_map.items()
                if info.instrument_class == "F"
            ],
            schema={"instrument_id": pl.Int64, "expiration": pl.Date},
        )
        opt_meta = pl.DataFrame(
            [
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
            ],
            schema={
                "instrument_id": pl.Int64,
                "underlying_id": pl.Int64,
                "expiration": pl.Date,
                "strike_price": pl.Float64,
                "right": pl.Utf8,
                "multiplier": pl.Int64,
            },
        )
        underlying_ids = set(futures_meta["instrument_id"].to_list())
        option_ids = set(opt_meta["instrument_id"].to_list())

        zip_files = sorted(self.raw_data_path.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError(f"No zip files found in {self.raw_data_path}")

        # Pre-compute the max timestamp already in each table (once, before the loop).
        # Rows strictly after this cutoff are unambiguously new and skip NOT EXISTS.
        if self.append:
            from datetime import timezone as _tz
            ub_cutoff = self._con.execute(
                "SELECT MAX(ts_event) FROM underlying_bars"
            ).fetchone()[0]
            ob_cutoff = self._con.execute(
                "SELECT MAX(ts_event) FROM option_bars"
            ).fetchone()[0]
            # Normalise to UTC so Polars can compare against Datetime("us", "UTC") columns.
            if ub_cutoff is not None:
                ub_cutoff = ub_cutoff.astimezone(_tz.utc)
            if ob_cutoff is not None:
                ob_cutoff = ob_cutoff.astimezone(_tz.utc)
            if ub_cutoff:
                print(f"[ingest] existing coverage up to {ub_cutoff.date()} — "
                      f"skipping NOT EXISTS for newer rows")
        else:
            ub_cutoff = ob_cutoff = None

        # Symbol lookup — _dbn_to_polars no longer reads symbol from the store
        # (to_ndarray() doesn't expose it). Join here from the instrument map instead.
        symbol_df = pl.DataFrame(
            {
                "instrument_id": pl.Series(
                    list(self._instrument_map.keys()), dtype=pl.Int64
                ),
                "symbol": pl.Series(
                    [info.raw_symbol for info in self._instrument_map.values()],
                    dtype=pl.Utf8,
                ),
            }
        )

        total_ub = total_ob = 0
        completed = 0
        with ThreadPoolExecutor(max_workers=_INGEST_WORKERS) as pool:
            futures_zip = {
                pool.submit(_process_zip_to_polars, zp): zp for zp in zip_files
            }
            for future in as_completed(futures_zip):
                ohlcv = future.result()
                completed += 1
                if ohlcv is None:
                    continue

                # Re-attach symbol from the instrument map.
                ohlcv = ohlcv.join(symbol_df, on="instrument_id", how="left")

                # ── Underlying bars ────────────────────────────────────────────
                ub_raw = ohlcv.filter(
                    pl.col("instrument_id").is_in(underlying_ids)
                ).select(
                    ["ts_event", "instrument_id", "symbol",
                     "open", "high", "low", "close", "volume"]
                )
                ub = ub_raw.join(futures_meta, on="instrument_id", how="left").select(
                    ["ts_event", "instrument_id", "symbol", "expiration",
                     "open", "high", "low", "close", "volume"]
                )

                # ── Option bars ────────────────────────────────────────────────
                opt_raw = ohlcv.filter(pl.col("instrument_id").is_in(option_ids))
                if not opt_raw.is_empty():
                    ob = opt_raw.join(opt_meta, on="instrument_id", how="left").select(
                        ["ts_event", "instrument_id", "underlying_id", "symbol",
                         "expiration", "strike_price", "right", "multiplier",
                         "open", "high", "low", "close", "volume"]
                    )
                else:
                    ob = pl.DataFrame(schema={
                        "ts_event": pl.Datetime("us", "UTC"), "instrument_id": pl.Int64,
                        "underlying_id": pl.Int64, "symbol": pl.Utf8,
                        "expiration": pl.Date, "strike_price": pl.Float64,
                        "right": pl.Utf8, "multiplier": pl.Int64,
                        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
                        "close": pl.Float64, "volume": pl.Int64,
                    })

                _write_df(self._con, "underlying_bars", ub,
                          skip_existing=self.append, cutoff_ts=ub_cutoff)
                _write_df(self._con, "option_bars", ob,
                          skip_existing=self.append, cutoff_ts=ob_cutoff)
                total_ub += len(ub)
                total_ob += len(ob)

                print(
                    f"\r[ingest] {completed}/{len(zip_files)}  "
                    f"ub={total_ub:,}  ob={total_ob:,}",
                    end="",
                    flush=True,
                )

        print(f"\n[ingest] Wrote {total_ub:,} underlying bars, {total_ob:,} option bars")

    # ------------------------------------------------------------------
    # Step 3: Greeks
    # ------------------------------------------------------------------

    def _compute_greeks(self) -> None:
        """
        Instantiate GreeksCalculator and process all option_bars in batches,
        writing results to option_greeks.
        """
        calc = GreeksCalculator(self._con)
        calc.run(skip_existing=self.append)

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
        under raw_data_path, in sorted (chronological) order.

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
    """
    Convert a DBNStore to a Polars DataFrame via to_ndarray().

    Avoids store.to_df() which converts through pandas and is ~170× slower on
    large files (benchmarked: 64 s vs 0.37 s for 1.26 M rows).

    Returns columns: ts_event, instrument_id, open, high, low, close, volume.
    'symbol' is intentionally omitted — to_ndarray() does not expose it. The
    caller (DatabaseBuilder._ingest_ohlcv) joins symbol from the instrument map.

    Price fields in the DBN format are fixed-point int64 scaled by 1e9; dividing
    by 1e9 recovers the floating-point price. ts_event is nanoseconds since the
    Unix epoch stored as uint64; dividing by 1000 gives microseconds for Polars
    Datetime("us", "UTC").
    """
    arr = store.to_ndarray()
    return pl.DataFrame(
        {
            "ts_event": (
                pl.Series(arr["ts_event"].astype(np.int64) // 1_000)
                .cast(pl.Datetime("us", "UTC"))
            ),
            "instrument_id": pl.Series(arr["instrument_id"].astype(np.int64)),
            "open":   pl.Series(arr["open"].astype(np.float64))  / 1e9,
            "high":   pl.Series(arr["high"].astype(np.float64))  / 1e9,
            "low":    pl.Series(arr["low"].astype(np.float64))   / 1e9,
            "close":  pl.Series(arr["close"].astype(np.float64)) / 1e9,
            "volume": pl.Series(arr["volume"].astype(np.int64)),
        }
    )



def _scan_zip_for_definitions(
    zip_path: Path,
) -> tuple[dict[int, _InstrumentInfo], int]:
    """
    Read every *.definition.dbn.zst file inside one zip archive.

    Returns (partial_map, files_read) where partial_map maps instrument_id →
    _InstrumentInfo for all F/C/P instruments found. First occurrence per iid
    is kept (files are sorted, so first occurrence is chronologically earliest).

    Iterates rows with an early-exit dict check (if iid in result: continue).
    After the first file all ~19k unique instruments are known, so subsequent
    files become nearly 100% fast hash-lookup skips with no object allocation.
    This is faster than a vectorised concat+unique approach, which accumulates
    ~100 MB of intermediate arrays across a 95-file zip and incurs GC pressure.

    from_bytes() (ZST decompression) is C-extension work that releases the GIL,
    so workers on different zip files run truly in parallel.
    """
    result: dict[int, _InstrumentInfo] = {}
    files_read = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith(".dbn.zst") and ".definition." in name:
                files_read += 1
                arr = db_sdk.DBNStore.from_bytes(zf.read(name)).to_ndarray()
                fcp_mask = np.isin(arr["instrument_class"], [b"F", b"C", b"P"])
                for row in arr[fcp_mask]:
                    iid = int(row["instrument_id"])
                    if iid in result:
                        continue
                    icls = row["instrument_class"].decode("ascii")
                    exp_ns = int(row["expiration"])
                    exp_date = (
                        datetime.fromtimestamp(exp_ns / 1_000_000_000, tz=timezone.utc).date()
                        if 0 < exp_ns < 9_000_000_000_000_000_000
                        else None
                    )
                    right = icls if icls in ("C", "P") else None
                    strike = float(row["strike_price"]) / 1e9 if icls in ("C", "P") else None
                    uom = int(row["unit_of_measure_qty"])
                    result[iid] = _InstrumentInfo(
                        instrument_class=icls,
                        raw_symbol=row["raw_symbol"].decode("ascii").rstrip("\x00"),
                        expiration=exp_date,
                        strike_price=strike,
                        right=right,
                        multiplier=round(uom / 1e9) if uom > 0 else 50,
                        underlying_id=int(row["underlying_id"]),
                    )
    return result, files_read


def _scan_zip_for_ids(zip_path: Path) -> set[int]:
    """Return all instrument_ids found in OHLCV files within one zip archive."""
    ids: set[int] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith(".dbn.zst") and ".ohlcv-" in name:
                store = db_sdk.DBNStore.from_bytes(zf.read(name))
                ids.update(store.to_ndarray()["instrument_id"].tolist())
    return ids


def _process_zip_to_polars(zip_path: Path) -> pl.DataFrame | None:
    """
    Read one zip archive, convert all OHLCV .dbn.zst files to Polars, and
    return a single deduplicated frame. Returns None if the zip has no OHLCV files.

    Designed to run inside a ThreadPoolExecutor — all work is C-extension or I/O
    that releases the GIL. The returned frame has no 'symbol' column; the caller
    joins it from the instrument map.
    """
    frames: list[pl.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith(".dbn.zst") and ".ohlcv-" in name:
                store = db_sdk.DBNStore.from_bytes(zf.read(name))
                frames.append(_dbn_to_polars(store))
    if not frames:
        return None
    return pl.concat(frames).unique(subset=["ts_event", "instrument_id"], keep="first")


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as mm:ss or ss.s depending on magnitude."""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


def _write_df(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pl.DataFrame,
    skip_existing: bool = False,
    cutoff_ts=None,
) -> None:
    """Register a Polars DataFrame and INSERT INTO the target table.

    When skip_existing=True, rows whose (ts_event, instrument_id) already exist
    in the table are silently skipped — safe for incremental / append builds.

    When cutoff_ts is provided, rows strictly after that timestamp are
    unambiguously new and bypass the NOT EXISTS check entirely. Only rows at or
    before the cutoff (the potential overlap window) go through NOT EXISTS.
    This avoids redundant index lookups for the bulk of new data in a typical
    append where new dates extend well beyond existing coverage.
    """
    if df.is_empty():
        return

    if not skip_existing:
        con.register("_write_tmp", df)
        con.execute(f"INSERT INTO {table} SELECT * FROM _write_tmp")
        con.unregister("_write_tmp")
        return

    if cutoff_ts is not None:
        clearly_new = df.filter(pl.col("ts_event") > cutoff_ts)
        overlap = df.filter(pl.col("ts_event") <= cutoff_ts)

        if not clearly_new.is_empty():
            con.register("_write_tmp", clearly_new)
            con.execute(f"INSERT INTO {table} SELECT * FROM _write_tmp")
            con.unregister("_write_tmp")

        if not overlap.is_empty():
            con.register("_write_tmp", overlap)
            con.execute(
                f"""
                INSERT INTO {table}
                SELECT t.* FROM _write_tmp t
                WHERE NOT EXISTS (
                    SELECT 1 FROM {table} e
                    WHERE e.ts_event = t.ts_event
                      AND e.instrument_id = t.instrument_id
                )
                """
            )
            con.unregister("_write_tmp")
    else:
        # No cutoff info — check every row (safe fallback).
        con.register("_write_tmp", df)
        con.execute(
            f"""
            INSERT INTO {table}
            SELECT t.* FROM _write_tmp t
            WHERE NOT EXISTS (
                SELECT 1 FROM {table} e
                WHERE e.ts_event = t.ts_event
                  AND e.instrument_id = t.instrument_id
            )
            """
        )
        con.unregister("_write_tmp")
