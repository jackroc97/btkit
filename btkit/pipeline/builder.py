"""
DatabaseBuilder — orchestrates the full input database build from raw Databento files.

Build sequence:
    0. _collect_ohlcv_instrument_ids() — lightweight pre-scan of all OHLCV files;
                                         returns the exact set of instrument_ids
                                         that appear in price data
    1. _ingest_definitions(required_ids) — read all definition files in parallel
                                           (one worker per .dbn.zst file), merge
                                           results into a single instrument map
    2. _ingest_ohlcv()        — read .dbn OHLCV files in parallel batches, write
                                underlying_bars and option_bars (with definition
                                metadata pre-joined)
    3. _compute_greeks()      — run GreeksCalculator over option_bars
    4. _run_indicators()      — run each user indicator script via IndicatorRunner

Parallelism strategy:
    - Definition ingest: one ThreadPoolExecutor worker per .dbn.zst file (not per
      zip). Each worker opens its own ZipFile handle, reads one named file,
      decompresses and parses it. Because the data vendor (Databento GLBX MDP3)
      recycles numeric instrument_ids — an expired instrument's ID is later
      reassigned to a new one — each ID is kept as a *list of validity segments*
      keyed by the definition's [activation, expiration] window, NOT collapsed to a
      single winner. OHLCV bars are then routed per-segment: a bar is classified by
      the definition whose validity window contains the bar's ts_event. This
      correctly separates, e.g., a June-2020 future and a 2023 call that share one
      recycled ID, filing each contract's bars in the right table for the right span.

    - OHLCV ingest: work units are (zip_path, filename) tuples. Files are processed
      in batches of _OHLCV_BATCH_SIZE with _INGEST_WORKERS threads per batch.
      After each batch the concatenated DataFrame is written to DuckDB and freed,
      keeping peak memory to ~_OHLCV_BATCH_SIZE × 75 MB (≈2.4 GB for batch=32).

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
import re
import time
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from btkit.db.contract_key import contract_id as _contract_id
from btkit.db.input_db import INPUT_DEFERRED_INDEXES_SQL, INPUT_TABLES_SQL
from btkit.pipeline.greeks import GreeksCalculator
from btkit.pipeline.indicators import IndicatorRunner

try:
    import databento as db_sdk
except ImportError as exc:  # pragma: no cover
    raise ImportError("databento package required: pip install databento") from exc

# Number of threads used for parallel file parsing.
# I/O and C-extension work (dbn decompression, ndarray conversion) releases
# the GIL so threads scale well here without process-spawn overhead.
_INGEST_WORKERS = min(os.cpu_count() or 4, 8)

# Number of .dbn.zst files parsed per write cycle in _ingest_ohlcv.
# Peak memory per batch: _OHLCV_BATCH_SIZE × ~75 MB ≈ 2.4 GB at default.
# Reduce to _INGEST_WORKERS * 2 if memory is tight.
_OHLCV_BATCH_SIZE: int = _INGEST_WORKERS * 4

# Where DuckDB spills intermediates that exceed memory_limit.
_BUILD_TEMP_DIR = "/tmp/duckdb_build_tmp"

# Memory ceiling for build connections. Below physical RAM so DuckDB spills to
# temp_directory rather than pushing the OS into swap, but high enough to hold the
# deferred index builds — lowering it to 6GB made ingest fail EARLIER, because ART
# indexes are unspillable and must fit under the limit whatever it is.
_DEFAULT_MEMORY_LIMIT = "10GB"
_DEFAULT_MAX_TEMP_SIZE = "150GB"


# ---------------------------------------------------------------------------
# Internal instrument map record
# ---------------------------------------------------------------------------


class _InstrumentInfo:
    __slots__ = (
        "instrument_class",
        "raw_symbol",
        "activation",
        "expiration",
        "strike_price",
        "right",
        "multiplier",
        "underlying_id",
        "pub_from",
    )

    def __init__(
        self,
        instrument_class: str,
        raw_symbol: str,
        activation,  # datetime.date or None — first valid date of this definition
        expiration,  # datetime.date or None — last valid date of this definition
        strike_price: float | None,
        right: str | None,  # 'C', 'P', or None for futures
        multiplier: int,
        underlying_id: int,
        pub_from=None,  # datetime (UTC) — earliest ts_event this definition was published
    ) -> None:
        self.instrument_class = instrument_class
        self.raw_symbol = raw_symbol
        self.activation = activation
        self.expiration = expiration
        self.strike_price = strike_price
        self.right = right
        self.multiplier = multiplier
        self.underlying_id = underlying_id
        self.pub_from = pub_from


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
        underlying_staleness_minutes: int = 15,
        memory_limit: str = _DEFAULT_MEMORY_LIMIT,
        max_temp_size: str = _DEFAULT_MAX_TEMP_SIZE,
        fill_range: tuple[datetime, datetime] | None = None,
        instruments: list[int] | None = None,
        recompute_tail_days: int = 60,
    ) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.db_path = Path(db_path)
        self.indicator_scripts = [Path(p) for p in (indicator_scripts or [])]
        # Interior "fill-the-gaps" mode: splice corrected source data into an
        # existing timeline and refresh greeks + indicators scoped to that window.
        # It is an append operation by construction, so append is forced on.
        self.fill_range = fill_range
        self.instruments = list(instruments) if instruments else None
        self.recompute_tail_days = int(recompute_tail_days)
        self.append = append or (fill_range is not None)
        # Max age (minutes) of the nearest-prior underlying bar used as the spot
        # price F when computing greeks. 0 = require an exact same-minute bar.
        self.underlying_staleness_minutes = int(underlying_staleness_minutes)
        # instrument_id → list of validity segments (sorted by activation). Each
        # segment is one definition of that (possibly recycled) ID for a distinct
        # [activation, expiration] window.
        self.memory_limit = memory_limit
        self.max_temp_size = max_temp_size
        self._instrument_map: dict[int, list[_InstrumentInfo]] = {}

    def _create_deferred_indexes(self) -> None:
        """Create the large-table indexes once all data is loaded.

        Deferred from the schema so the bulk load does not pay incremental ART index
        maintenance, which is memory-resident and unspillable. Idempotent
        (IF NOT EXISTS), so an --append build where they already exist is a no-op.
        """
        con = duckdb.connect(str(self.db_path))
        self._apply_session_limits(con)
        try:
            con.execute(INPUT_DEFERRED_INDEXES_SQL)
        finally:
            con.close()

    def _apply_session_limits(self, con: duckdb.DuckDBPyConnection) -> None:
        """Apply the build's memory and spill settings to one connection.

        Cap DuckDB well below physical RAM so it spills to temp_directory instead of
        competing with the OS page cache. Left at the default (~80% of RAM) on a
        17 GB machine, the full ES build never spilled at all — temp_directory
        stayed at 0 bytes with 181 GiB free — and instead failed three different
        ways while the OS swapped: OutOfMemory, "failed to commit: failed to pin",
        and an IO error reading a short file during the greeks ASOF join. A
        full-history build needs a larger working set than this machine has, so
        spilling is the only way it completes; the limit trades speed for finishing.

        Settings are per-connection, so every connection the build opens needs this.
        """
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory='{_BUILD_TEMP_DIR}'")
        con.execute(f"SET memory_limit='{self.memory_limit}'")
        con.execute(f"SET max_temp_directory_size='{self.max_temp_size}'")

    def build(self) -> None:
        """
        Full build: ingest → greeks → indicators.
        Creates or overwrites the database at db_path.
        """
        t_total = time.perf_counter()
        self._con = duckdb.connect(str(self.db_path))
        self._apply_session_limits(self._con)
        try:
            # Tables only. The indexes on the three largest tables are created
            # after every phase has loaded (see _create_deferred_indexes), because
            # DuckDB's ART indexes cannot spill and maintaining them across a
            # ~70M-row load exhausts memory on a 17 GB machine.
            self._con.execute(INPUT_TABLES_SQL)
            # Self-heal databases built before underlying_lag_s existed so an
            # --append build against an older DB does not fail on INSERT.
            self._con.execute(
                "ALTER TABLE option_greeks ADD COLUMN IF NOT EXISTS underlying_lag_s INTEGER"
            )

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

        t = time.perf_counter()
        self._create_deferred_indexes()
        print(f"[timing] indexes:      {_fmt_elapsed(time.perf_counter() - t)}")

        print(f"[timing] total:        {_fmt_elapsed(time.perf_counter() - t_total)}")

    # ------------------------------------------------------------------
    # Step 0: Pre-scan OHLCV IDs
    # ------------------------------------------------------------------

    def _collect_ohlcv_instrument_ids(self) -> set[int]:
        """
        Lightweight pre-scan: read all OHLCV files and return the complete set of
        instrument_ids that appear in price data.

        Parallelises at the individual file level — each worker opens its own
        ZipFile handle and reads exactly one .dbn.zst file.
        """
        zip_files = sorted(self.raw_data_path.glob("*.zip"))
        work_units: list[tuple[Path, str]] = []
        for zp in zip_files:
            work_units.extend(_list_zip_ohlcv_files(zp))

        ids: set[int] = set()
        with ThreadPoolExecutor(max_workers=_INGEST_WORKERS) as pool:
            for partial in pool.map(_scan_file_for_ids, work_units):
                ids.update(partial)
        return ids

    # ------------------------------------------------------------------
    # Step 1: Definitions
    # ------------------------------------------------------------------

    def _ingest_definitions(self, required_ids: set[int]) -> None:
        """
        Read all *.definition.dbn.zst files in parallel (one worker per file) and
        build self._instrument_map: instrument_id → list of validity segments.

        Each worker opens its own ZipFile handle and reads exactly one file and
        returns, per instrument_id, the distinct definitions it saw keyed by their
        [activation, expiration] window. The main thread unions these across all
        files, so a recycled ID that was (say) a future in 2020 and an option in
        2023 is kept as TWO segments rather than collapsed to one.  _ingest_ohlcv
        then routes each bar to the segment whose window contains it.

        Only instrument_class F (future), C (call), and P (put) are kept.
        """
        zip_files = sorted(self.raw_data_path.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError(f"No zip files found in {self.raw_data_path}")

        work_units: list[tuple[Path, str]] = []
        for zp in zip_files:
            work_units.extend(_list_zip_definition_files(zp))

        if not work_units:
            raise FileNotFoundError(
                f"No definition files found in zip archives under {self.raw_data_path}"
            )

        # instrument_id → {(activation, expiration, raw_symbol): _InstrumentInfo}.
        # Keying by contract identity de-duplicates the same definition reported
        # across many daily snapshot files while preserving genuinely distinct
        # (recycled) ones. A definition republished in several files keeps its
        # EARLIEST ts_event, which is when the id started referring to that contract.
        seg_maps: dict[int, dict[tuple, _InstrumentInfo]] = {}
        with ThreadPoolExecutor(max_workers=_INGEST_WORKERS) as pool:
            for partial_map in pool.map(_scan_definition_file, work_units):
                for iid, segs in partial_map.items():
                    target = seg_maps.setdefault(iid, {})
                    for key, info in segs.items():
                        existing = target.get(key)
                        if existing is None:
                            target[key] = info
                        elif info.pub_from is not None and (
                            existing.pub_from is None or info.pub_from < existing.pub_from
                        ):
                            existing.pub_from = info.pub_from

        self._instrument_map = {
            iid: _order_segments(segs.values()) for iid, segs in seg_maps.items()
        }

        n_segments = sum(len(v) for v in self._instrument_map.values())
        n_recycled = sum(1 for v in self._instrument_map.values() if len(v) > 1)

        missing = required_ids - set(self._instrument_map)
        if missing:
            print(
                f"[ingest] Warning: {len(missing)} instrument IDs found in OHLCV "
                f"data have no matching definition record and will be dropped."
            )
        if n_recycled:
            print(
                f"[ingest] {n_recycled:,} instrument IDs were recycled across "
                f"multiple validity windows — bars routed per-window."
            )

        print(
            f"[ingest] Loaded {len(self._instrument_map):,} instruments "
            f"({n_segments:,} validity segments) from {len(work_units)} "
            f"definition file(s) across {len(zip_files)} zip(s)"
        )

    # ------------------------------------------------------------------
    # Step 2: OHLCV
    # ------------------------------------------------------------------

    def _write_contracts(self, segments: pl.DataFrame) -> None:
        """Upsert the contract dimension from the routing segments.

        One row per real contract, keyed by the stable synthetic contract_id. Written
        before the bars so option_bars.contract_id always resolves. Existing rows are
        left alone: contract_id is a pure function of (symbol, expiration), so a row
        already present is already correct, and this keeps --append idempotent.

        Covers futures as well as options — futures symbols recycle on the same
        10-year CME cycle (ESH4 expires both 2014-03-21 and 2024-03-15) — so the
        dimension is complete even though underlying_bars does not yet carry the key.
        """
        if segments.is_empty():
            return
        contracts = segments.select(
            [
                "contract_id",
                "symbol",
                pl.col("seg_expiration").alias("expiration"),
                "underlying_id",
                "strike_price",
                "right",
                "multiplier",
                "instrument_class",
            ]
        ).unique(subset=["contract_id"], keep="first")

        self._con.register("_contract_tmp", contracts)
        self._con.execute(
            """
            INSERT INTO contract
            SELECT t.* FROM _contract_tmp t
            WHERE NOT EXISTS (
                SELECT 1 FROM contract c WHERE c.contract_id = t.contract_id
            )
            """
        )
        self._con.unregister("_contract_tmp")
        print(f"[ingest] contract dimension: {len(contracts):,} distinct contracts")

    def _ingest_ohlcv(self) -> None:
        """
        Read all *.ohlcv-*.dbn.zst files in parallel batches of _OHLCV_BATCH_SIZE,
        route each bar to the definition whose validity window contains it, split
        into underlying_bars and option_bars, and write each batch immediately
        before freeing it.

        Batched writes keep peak memory bounded: each batch holds at most
        _OHLCV_BATCH_SIZE × ~75 MB of decompressed data at once.
        """
        if not self._instrument_map:
            raise RuntimeError("_ingest_ohlcv() called before _ingest_definitions()")

        # Build the segment lookup table once — one row per (instrument_id,
        # validity window). Each bar is routed to the segment whose window contains
        # it (see _route_bars_to_segments). Routing bounds (_act / _exp_route) are
        # sentinel-filled so a missing activation/expiration never excludes an
        # otherwise-valid bar; the real (possibly null) expiration is carried
        # separately (seg_expiration) for the output row.
        segments = _build_segments_frame(self._instrument_map)
        self._write_contracts(segments)

        zip_files = sorted(self.raw_data_path.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError(f"No zip files found in {self.raw_data_path}")

        # Pre-scan → flat (zip_path, filename) work units.
        work_units: list[tuple[Path, str]] = []
        for zp in zip_files:
            work_units.extend(_list_zip_ohlcv_files(zp))

        if not work_units:
            print("[ingest] No OHLCV files found — nothing to write")
            return

        # Process chronologically by the date range encoded in each filename, so the
        # running cutoff below advances monotonically and the NOT EXISTS fast path
        # applies to all but genuinely overlapping source files. Zip-name order does
        # not imply date order: a full-history re-download sorts after the
        # incremental zips whose dates it already covers.
        work_units.sort(key=_work_unit_sort_key)

        # The cutoff is the maximum ts_event currently in each table; rows newer than
        # it cannot collide with an existing row, so they skip the NOT EXISTS check.
        # It is advanced after every batch (see _advance_cutoff).
        #
        # A fresh build starts at datetime.min rather than None so it uses the same
        # mechanism: overlapping source files duplicate rows on a full build exactly
        # as they do on an append, because a plain INSERT has no check at all. That is
        # how 4.84M duplicate option bars entered the ES database.
        ub_cutoff = self._con.execute("SELECT MAX(ts_event) FROM underlying_bars").fetchone()[0]
        ob_cutoff = self._con.execute("SELECT MAX(ts_event) FROM option_bars").fetchone()[0]
        ub_cutoff = (
            ub_cutoff.astimezone(UTC) if ub_cutoff is not None else datetime.min.replace(tzinfo=UTC)
        )
        ob_cutoff = (
            ob_cutoff.astimezone(UTC) if ob_cutoff is not None else datetime.min.replace(tzinfo=UTC)
        )
        if self.append and ub_cutoff > datetime.min.replace(tzinfo=UTC):
            print(
                f"[ingest] existing coverage up to {ub_cutoff.date()} — "
                f"skipping NOT EXISTS for newer rows"
            )

        total_ub = total_ob = 0
        total_gap = total_overlap = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=_INGEST_WORKERS) as pool:
            for batch_start in range(0, len(work_units), _OHLCV_BATCH_SIZE):
                batch = work_units[batch_start : batch_start + _OHLCV_BATCH_SIZE]
                frames = [f for f in pool.map(_parse_ohlcv_file, batch) if f is not None]
                completed += len(batch)

                if not frames:
                    print(
                        f"\r[ingest] {completed}/{len(work_units)}  "
                        f"ub={total_ub:,}  ob={total_ob:,}",
                        end="",
                        flush=True,
                    )
                    continue

                ohlcv = pl.concat(frames).unique(subset=["ts_event", "instrument_id"], keep="first")
                del frames

                # Time-aware routing: attach each bar's covering segment.
                resolved, n_overlap = _route_bars_to_segments(ohlcv, segments)
                del ohlcv
                total_gap += resolved.filter(~pl.col("_contained")).height
                total_overlap += n_overlap

                # ── Underlying bars (futures) ──────────────────────────────
                ub = resolved.filter(pl.col("instrument_class") == "F").select(
                    [
                        "ts_event",
                        "instrument_id",
                        "symbol",
                        pl.col("seg_expiration").alias("expiration"),
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                )

                # ── Option bars (calls / puts) ─────────────────────────────
                ob = resolved.filter(pl.col("instrument_class").is_in(["C", "P"])).select(
                    [
                        "ts_event",
                        "instrument_id",
                        "contract_id",
                        "underlying_id",
                        "symbol",
                        pl.col("seg_expiration").alias("expiration"),
                        "strike_price",
                        "right",
                        "multiplier",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                )
                del resolved

                _write_df(self._con, "underlying_bars", ub, skip_existing=True, cutoff_ts=ub_cutoff)
                _write_df(self._con, "option_bars", ob, skip_existing=True, cutoff_ts=ob_cutoff)
                # Advance each cutoff past what this batch just inserted. The fast
                # path in _write_df skips the NOT EXISTS check for rows newer than
                # the cutoff, which is only sound while the cutoff tracks the table's
                # current maximum. Left at its pre-loop value, a later batch that
                # re-supplies the same bars from an overlapping source file would
                # bypass the check and insert a second copy. Tracked in Python rather
                # than re-querying MAX(ts_event): the index is on (instrument_id,
                # ts_event), so the aggregate would be a full scan per batch.
                ub_cutoff = _advance_cutoff(ub_cutoff, ub)
                ob_cutoff = _advance_cutoff(ob_cutoff, ob)
                total_ub += len(ub)
                total_ob += len(ob)
                del ub, ob

                print(
                    f"\r[ingest] {completed}/{len(work_units)}  ub={total_ub:,}  ob={total_ob:,}",
                    end="",
                    flush=True,
                )

        print(f"\n[ingest] Wrote {total_ub:,} underlying bars, {total_ob:,} option bars")
        if total_gap or total_overlap:
            print(
                f"[ingest] Segment routing: {total_gap:,} bars fell in a validity-window "
                f"gap (reclassified to nearest segment), {total_overlap:,} bars matched "
                f"overlapping windows (tightest window kept)."
            )

    # ------------------------------------------------------------------
    # Step 3: Greeks
    # ------------------------------------------------------------------

    def _compute_greeks(self) -> None:
        """
        Instantiate GreeksCalculator and process all option_bars in batches,
        writing results to option_greeks.
        """
        calc = GreeksCalculator(
            self._con, underlying_max_staleness_minutes=self.underlying_staleness_minutes
        )
        calc.run(
            skip_existing=self.append,
            fill_range=self.fill_range,
            instruments=self.instruments,
        )

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

        con = duckdb.connect(str(self.db_path))
        # Session settings are per-connection, and the indicator phase runs on its
        # own connection after build() closed the ingest one — so the spill
        # configuration has to be applied again here or indicators run with
        # DuckDB's default limit and can exhaust memory the same way.
        self._apply_session_limits(con)
        try:
            if self.fill_range is not None:
                self._run_indicators_scoped(con)
            else:
                underlyings = con.execute(
                    "SELECT DISTINCT instrument_id FROM underlying_bars"
                ).fetchall()
                usable = self._preflight_indicators(con, underlyings)
                for script_path in usable:
                    runner = IndicatorRunner(con, script_path)
                    for (underlying_id,) in underlyings:
                        runner.run(underlying_id)
        finally:
            con.close()

    def _preflight_indicators(
        self,
        con: duckdb.DuckDBPyConnection,
        underlyings: list[tuple],
    ) -> list[str | Path]:
        """Return the indicator scripts that actually run, warning about the rest.

        Each script is exercised against one underlying before the real pass. A
        script can fail for reasons that only appear when it is called — an
        undeclared import (ves.py needs scipy, which is not a btkit dependency) or
        a context method that does not exist (weekly_ema21.py calls
        ctx.continuous_underlying_bars(), which its own docstring notes "is not a
        real method in btkit today").

        Previously either of those aborted the whole build *after* the definitions,
        ingest and greeks phases had completed, throwing away the expensive work for
        a broken auxiliary script. Now the failures are reported together, up front,
        and the remaining scripts still run.
        """
        scripts = list(self.indicator_scripts or [])
        if not scripts or not underlyings:
            return scripts

        probe_id = underlyings[0][0]
        usable: list[str | Path] = []
        failed: list[tuple[str, str]] = []
        for script_path in scripts:
            try:
                IndicatorRunner(con, script_path).run(probe_id)
            except Exception as exc:  # noqa: BLE001 - any failure disqualifies it
                failed.append((str(script_path), f"{type(exc).__name__}: {exc}"))
            else:
                usable.append(script_path)

        if failed:
            print(f"[indicators] {len(failed)} script(s) failed and will be skipped:")
            for name, err in failed:
                print(f"[indicators]   {Path(name).name}: {err}")
        if not usable:
            raise RuntimeError(
                "every indicator script failed its pre-flight; "
                "see the messages above. Pass --indicators with working scripts, "
                "or omit it to build without indicators."
            )
        return usable

    def _run_indicators_scoped(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Interior fill mode: recompute each indicator only for the underlyings whose
        bars were back-filled, over the fill window plus a trailing-dependency tail,
        overwriting stale values (see IndicatorRunner.run_scoped).

        Affected underlyings are the caller-supplied --instruments, or (when none
        are given) every underlying with bars in the fill window.
        """
        start, end = self.fill_range
        if self.instruments:
            underlying_ids = list(self.instruments)
        else:
            underlying_ids = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT instrument_id FROM underlying_bars "
                    "WHERE ts_event >= ? AND ts_event <= ?",
                    [start, end],
                ).fetchall()
            ]

        if not underlying_ids:
            print("[indicators] fill mode: no affected underlyings in the window — skipping")
            return

        print(
            f"[indicators] fill mode: recomputing {len(self.indicator_scripts)} script(s) "
            f"for {len(underlying_ids)} underlying(s) over "
            f"{start.date()}→{end.date()} ±{self.recompute_tail_days}d"
        )
        for script_path in self.indicator_scripts:
            runner = IndicatorRunner(con, script_path)
            for underlying_id in underlying_ids:
                runner.run_scoped(underlying_id, start, end, self.recompute_tail_days)

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
# Module-level helpers — pre-scan
# ---------------------------------------------------------------------------


def _list_zip_ohlcv_files(zip_path: Path) -> list[tuple[Path, str]]:
    """Return sorted (zip_path, filename) tuples for OHLCV files within one zip."""
    with zipfile.ZipFile(zip_path) as zf:
        return [
            (zip_path, name)
            for name in sorted(zf.namelist())
            if name.endswith(".dbn.zst") and ".ohlcv-" in name
        ]


def _list_zip_definition_files(zip_path: Path) -> list[tuple[Path, str]]:
    """Return sorted (zip_path, filename) tuples for definition files within one zip."""
    with zipfile.ZipFile(zip_path) as zf:
        return [
            (zip_path, name)
            for name in sorted(zf.namelist())
            if name.endswith(".dbn.zst") and ".definition." in name
        ]


# ---------------------------------------------------------------------------
# Module-level helpers — per-file workers (called via ThreadPoolExecutor.map)
# ---------------------------------------------------------------------------


def _scan_file_for_ids(work: tuple[Path, str]) -> set[int]:
    """Return all instrument_ids found in one OHLCV .dbn.zst file."""
    zip_path, filename = work
    with zipfile.ZipFile(zip_path) as zf:
        store = db_sdk.DBNStore.from_bytes(zf.read(filename))
    return set(store.to_ndarray()["instrument_id"].tolist())


def _ns_to_date(ns: int) -> date | None:
    """Convert a Databento ns UNIX timestamp to a UTC date, or None if unset.

    Databento marks undefined timestamps with UINT64_MAX (and sometimes 0); the
    finite-range guard rejects both.
    """
    return (
        datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC).date()
        if 0 < ns < 9_000_000_000_000_000_000
        else None
    )


def _ns_to_datetime(ns: int) -> datetime | None:
    """Convert a Databento ns UNIX timestamp to a UTC datetime, or None if unset.

    Same sentinel handling as _ns_to_date, but keeps the time of day — definition
    publication times matter to the minute when an instrument_id is reassigned.
    """
    return (
        datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC)
        if 0 < ns < 9_000_000_000_000_000_000
        else None
    )


_SEGMENTS_SCHEMA = {
    "instrument_id": pl.Int64,
    "contract_id": pl.Int64,
    "instrument_class": pl.Utf8,
    "symbol": pl.Utf8,
    "seg_expiration": pl.Date,
    "underlying_id": pl.Int64,
    "strike_price": pl.Float64,
    "right": pl.Utf8,
    "multiplier": pl.Int64,
    "_act": pl.Date,
    "_exp_route": pl.Date,
    "_pub_from": pl.Datetime("us", "UTC"),
    "_pub_to": pl.Datetime("us", "UTC"),
    "_seg_rank": pl.Int64,
}

_PUB_MIN = datetime.min.replace(tzinfo=UTC)
_PUB_MAX = datetime.max.replace(tzinfo=UTC, microsecond=0)


def _build_segments_frame(instrument_map: dict[int, list[_InstrumentInfo]]) -> pl.DataFrame:
    """Flatten the instrument map into the routing lookup table.

    Each definition gets a publication validity window [_pub_from, _pub_to):

      * `_pub_from` is when the exchange first published this id → contract mapping.
        The FIRST segment for an id is widened to the beginning of time so bars that
        precede our definition coverage are still classified rather than dropped.
      * `_pub_to` is the next segment's `_pub_from`, so each contract owns the id
        until the id is reassigned. The last segment runs to the end of time.

    `_seg_rank` is the segment's position in publication order, used as a final
    deterministic tie-break so routing can never depend on row order.

    `_act` / `_exp_route` are retained as a fallback for ids whose definitions carry
    no publication timestamp at all.
    """
    rows: list[dict] = []
    for iid, segs in instrument_map.items():
        for rank, seg in enumerate(segs):
            # The next segment's publication time closes this one's window.
            nxt = segs[rank + 1] if rank + 1 < len(segs) else None
            pub_to = nxt.pub_from if nxt is not None and nxt.pub_from is not None else _PUB_MAX
            pub_from = seg.pub_from if seg.pub_from is not None else _PUB_MIN
            if rank == 0:
                pub_from = _PUB_MIN
            rows.append(
                {
                    "instrument_id": iid,
                    # Stable surrogate key, computed once per contract here rather
                    # than per bar: ~665k contracts vs 74.5M option bars.
                    "contract_id": _contract_id(seg.raw_symbol, seg.expiration),
                    "instrument_class": seg.instrument_class,
                    "symbol": seg.raw_symbol,
                    "seg_expiration": seg.expiration,
                    "underlying_id": seg.underlying_id,
                    "strike_price": seg.strike_price,
                    "right": seg.right,
                    "multiplier": seg.multiplier,
                    "_act": seg.activation or date(1970, 1, 1),
                    "_exp_route": seg.expiration or date(9999, 12, 31),
                    "_pub_from": pub_from,
                    "_pub_to": max(pub_to, pub_from),
                    "_seg_rank": rank,
                }
            )
    return pl.DataFrame(rows, schema=_SEGMENTS_SCHEMA)


def _order_segments(segs) -> list[_InstrumentInfo]:
    """Order one instrument_id's definitions by publication time.

    Sorting by pub_from puts a recycled id's contracts in the order the exchange
    actually assigned them, which is what _route_bars_to_segments needs to derive
    each one's validity window. Definitions with no publication timestamp sort
    first by (activation, expiration) so they are superseded by any dated
    definition rather than shadowing one.
    """
    return sorted(
        segs,
        key=lambda s: (
            s.pub_from is not None,
            s.pub_from or datetime.min.replace(tzinfo=UTC),
            _segment_sort_key(s),
            s.raw_symbol,
        ),
    )


def _segment_sort_key(seg: _InstrumentInfo) -> tuple[date, date]:
    """Sort validity segments chronologically (None sorts first)."""
    return (seg.activation or date.min, seg.expiration or date.min)


def _scan_definition_file(
    work: tuple[Path, str],
) -> dict[int, dict[tuple, _InstrumentInfo]]:
    """
    Parse one *.definition.dbn.zst file and return, per instrument_id, the
    distinct definitions it contains keyed by contract identity.

    Each definition also carries `pub_from`: the earliest ts_event at which this
    file published it. That publication timestamp — not activation/expiration — is
    what resolves a recycled instrument_id, because activation describes when the
    *contract* was listed, which can precede the period the id referred to it by
    weeks. See _route_bars_to_segments and docs/data_duplication_fix.md §5.

    Returning per-contract segments (not a single winner) is what lets the main
    thread reconstruct a recycled ID's full timeline: an ID that was a future in
    one window and an option in another yields two entries here, unioned across
    files by _ingest_definitions.

    Each call opens its own ZipFile handle (thread-safe independent descriptor).
    from_bytes() (ZST decompression) is C-extension work that releases the GIL,
    so workers on different files run truly in parallel.
    """
    zip_path, filename = work
    with zipfile.ZipFile(zip_path) as zf:
        arr = db_sdk.DBNStore.from_bytes(zf.read(filename)).to_ndarray()

    result: dict[int, dict[tuple, _InstrumentInfo]] = {}
    fcp_mask = np.isin(arr["instrument_class"], [b"F", b"C", b"P"])
    for row in arr[fcp_mask]:
        iid = int(row["instrument_id"])
        icls = row["instrument_class"].decode("ascii")
        act_date = _ns_to_date(int(row["activation"]))
        exp_date = _ns_to_date(int(row["expiration"]))
        raw_symbol = row["raw_symbol"].decode("ascii").rstrip("\x00")
        # Key on full contract identity, not just the validity window: the same id
        # can carry two contracts whose (activation, expiration) windows coincide,
        # and collapsing them would silently discard one.
        key = (act_date, exp_date, raw_symbol)
        pub_ts = _ns_to_datetime(int(row["ts_event"]))
        segs = result.setdefault(iid, {})
        existing = segs.get(key)
        if existing is not None:
            # Same definition republished — keep the earliest publication time.
            if pub_ts is not None and (existing.pub_from is None or pub_ts < existing.pub_from):
                existing.pub_from = pub_ts
            continue
        right = icls if icls in ("C", "P") else None
        strike = float(row["strike_price"]) / 1e9 if icls in ("C", "P") else None
        uom = int(row["unit_of_measure_qty"])
        segs[key] = _InstrumentInfo(
            instrument_class=icls,
            raw_symbol=raw_symbol,
            activation=act_date,
            expiration=exp_date,
            strike_price=strike,
            right=right,
            multiplier=round(uom / 1e9) if uom > 0 else 50,
            underlying_id=int(row["underlying_id"]),
            pub_from=pub_ts,
        )
    return result


def _parse_ohlcv_file(work: tuple[Path, str]) -> pl.DataFrame | None:
    """
    Parse one *.ohlcv-*.dbn.zst file and return a Polars DataFrame.
    Returns None if the file contains no rows.

    Each call opens its own ZipFile handle (thread-safe). No 'symbol' column is
    included; the caller joins it from the instrument map after concatenation.
    """
    zip_path, filename = work
    with zipfile.ZipFile(zip_path) as zf:
        store = db_sdk.DBNStore.from_bytes(zf.read(filename))
    df = _dbn_to_polars(store)
    return df if not df.is_empty() else None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _route_bars_to_segments(
    ohlcv: pl.DataFrame,
    segments: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """
    Resolve each OHLCV bar to the instrument definition (segment) that the exchange
    had published for its instrument_id at the time of the bar.

    Because the data vendor recycles instrument_ids, a single ID may map to several
    definitions over time (e.g. a future, then an option). Bars are inner-joined to
    segments on instrument_id (bars whose ID has no definition are dropped, matching
    the original routing) and, per bar, the winning segment is the one whose
    publication window [_pub_from, _pub_to) contains the bar's ts_event.

    Why publication time and not [activation, expiration]
    -----------------------------------------------------
    Routing used to pick the segment whose [activation, expiration] window contained
    the bar date, breaking ties on the tightest window. `activation` is when the
    *contract* was listed, which can precede the period the id referred to it by
    weeks, so a recycled id's windows overlap. Worse, CME lists options with fixed
    tenors, so a recycled id's two contracts have IDENTICAL window lengths and the
    tie-break was a coin flip resolved by `pl.DataFrame.sort()`, which is not stable.
    A random sample of 24 affected id-days in the ES database found a tie in 24 of
    24 cases. That is how 3,875 instrument-days ended up carrying two contracts.

    A definition record's own ts_event is when the exchange announced the mapping, so
    the mapping in force for a bar is simply the latest one published at or before
    it. Those publication windows are disjoint per id by construction, so there is no
    tie to break. Worked examples are in docs/data_duplication_fix.md §5.

    Fallbacks, in order:
      * the publication window containing the bar (the normal path);
      * failing that — an id whose definitions carry no publication timestamp at all —
        the [activation, expiration] containment rule, nearest window by date
        distance, so the bar is reclassified rather than dropped (row conservation);
      * `_seg_rank` last, so the outcome never depends on row order.

    Returns (resolved, n_overlap):
      resolved  — one row per input bar with the winning segment's columns and a
                  boolean `_contained` (False ⇒ a date-distance fallback was used).
      n_overlap — number of bars matching more than one publication window. Expected
                  to be 0; a non-zero count means the source definitions disagree.
    """
    cand = ohlcv.with_columns(pl.col("ts_event").dt.date().alias("_bar_date")).join(
        segments, on="instrument_id", how="inner"
    )

    cand = cand.with_columns(
        # Published window containing the bar — the authoritative match.
        (
            (pl.col("ts_event") >= pl.col("_pub_from")) & (pl.col("ts_event") < pl.col("_pub_to"))
        ).alias("_in_pub"),
        (
            (pl.col("_bar_date") >= pl.col("_act")) & (pl.col("_bar_date") <= pl.col("_exp_route"))
        ).alias("_contained"),
        (pl.col("_exp_route") - pl.col("_act")).dt.total_days().alias("_win"),
        pl.max_horizontal(
            (pl.col("_act") - pl.col("_bar_date")).dt.total_days(),
            (pl.col("_bar_date") - pl.col("_exp_route")).dt.total_days(),
            pl.lit(0, dtype=pl.Int64),
        ).alias("_dist"),
    )

    # Bars matching more than one publication window. Publication windows are built
    # back-to-back per instrument_id so this should be zero; it is reported rather
    # than assumed so a vendor-side contradiction is visible in the ingest summary.
    n_overlap = (
        cand.filter(pl.col("_in_pub"))
        .group_by(["ts_event", "instrument_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    # Winner per (ts_event, instrument_id). Sort keys, in order:
    #   _in_pub desc  — a published match beats any fallback
    #   _dist asc     — fallback only: a containing window beats a distant one
    #   _win asc      — fallback only: tightest window
    #   _seg_rank asc — final deterministic tie-break, so an unstable sort cannot
    #                   make the result depend on row order
    resolved = cand.sort(
        ["ts_event", "instrument_id", "_in_pub", "_dist", "_win", "_seg_rank"],
        descending=[False, False, True, False, False, False],
    ).unique(subset=["ts_event", "instrument_id"], keep="first")

    # A bar matched by publication window is correctly classified even if it sits
    # outside the contract's [activation, expiration] dates, so do not report it as
    # a gap fallback.
    resolved = resolved.with_columns((pl.col("_contained") | pl.col("_in_pub")).alias("_contained"))
    return resolved, n_overlap


def _dbn_to_polars(store) -> pl.DataFrame:
    """
    Convert a DBNStore to a Polars DataFrame via to_ndarray().

    Avoids store.to_df() which converts through pandas and is ~170× slower on
    large files (benchmarked: 64 s vs 0.37 s for 1.26 M rows).

    Returns columns: ts_event, instrument_id, open, high, low, close, volume.
    'symbol' is intentionally omitted — to_ndarray() does not expose it. The
    caller joins symbol from the instrument map.

    Price fields in the DBN format are fixed-point int64 scaled by 1e9; dividing
    by 1e9 recovers the floating-point price. ts_event is nanoseconds since the
    Unix epoch stored as uint64; dividing by 1000 gives microseconds for Polars
    Datetime("us", "UTC").
    """
    arr = store.to_ndarray()
    return pl.DataFrame(
        {
            "ts_event": (
                pl.Series(arr["ts_event"].astype(np.int64) // 1_000).cast(pl.Datetime("us", "UTC"))
            ),
            "instrument_id": pl.Series(arr["instrument_id"].astype(np.int64)),
            "open": pl.Series(arr["open"].astype(np.float64)) / 1e9,
            "high": pl.Series(arr["high"].astype(np.float64)) / 1e9,
            "low": pl.Series(arr["low"].astype(np.float64)) / 1e9,
            "close": pl.Series(arr["close"].astype(np.float64)) / 1e9,
            "volume": pl.Series(arr["volume"].astype(np.int64)),
        }
    )


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as mm:ss or ss.s depending on magnitude."""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


_WORK_UNIT_DATE_RE = re.compile(r"(\d{8})(?:-(\d{8}))?")


def _work_unit_sort_key(work: tuple[Path, str]) -> tuple[str, str, str]:
    """Chronological sort key for an OHLCV work unit, from the date range in its name.

    Databento names payloads `glbx-mdp3-YYYYMMDD.ohlcv-1m.dbn.zst` or
    `glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1m.dbn.zst`, so the first 8-digit run is the
    start date and the optional second is the end date. Sorting on (start, end) makes
    the ingest loop advance through time, which keeps the _write_df cutoff fast path
    effective; without it the loop follows zip-name order, where a full-history
    re-download trails the incremental zips it overlaps.

    Names without a parseable date sort last but keep a stable relative order, so an
    unexpected naming scheme degrades to "always check" rather than failing.
    """
    _, filename = work
    match = _WORK_UNIT_DATE_RE.search(Path(filename).name)
    if match is None:
        return ("99999999", "99999999", filename)
    start = match.group(1)
    return (start, match.group(2) or start, filename)


def _advance_cutoff(cutoff, df: pl.DataFrame):
    """Return the later of `cutoff` and the maximum ts_event in `df`.

    Keeps the _write_df fast-path cutoff in step with what has actually been
    inserted, so rows re-supplied by a later batch are still checked for existence.
    A None cutoff means "no fast path" and stays None: appends that never had a
    cutoff must keep checking every row.
    """
    if cutoff is None or df.is_empty():
        return cutoff
    batch_max = df["ts_event"].max()
    if batch_max is None:
        return cutoff
    return max(cutoff, batch_max)


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

    cutoff_ts MUST be the maximum ts_event present in the table *right now*, not
    merely the maximum from before the ingest loop started. The fast path's premise
    is "no existing row has a timestamp this late", which only holds against the
    table's current contents. Callers therefore advance the cutoff after every write
    (see _ingest_ohlcv). A stale cutoff silently duplicates every row that a later
    batch re-supplies from an overlapping source file — the defect that put 4.84M
    duplicate option bars into the ES database. See docs/data_duplication_fix.md §2.

    Note also that NOT EXISTS cannot deduplicate *within* df: the subquery is
    evaluated against the pre-insert table, so two copies of a key inside df both
    pass. Callers must hand over a frame already unique on (ts_event, instrument_id).
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
