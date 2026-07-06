"""
AuditRunner — orchestrates all audit phases and writes the option_audit table.

Usage:
    runner = AuditRunner("/path/to/input.db")
    result = runner.run()           # writes option_audit, prints progress
    summary = runner.quintile_summary()  # post-run quintile breakdown

    runner = AuditRunner("/path/to/input.db", dry_run=True)
    result = runner.run(verbose=False)  # no output, no table write

Progress output (verbose=True):
    Phase 1  IV flags  ...
    Phase 1  IV flags: 4,583,097 rows  (0.8s)
    Phase 2  Delta consistency: materialising 3-way join ...
      247 / 1008 dates  (24%)
     1008 / 1008 dates  (100%)
    Phase 2  Delta consistency: 183,429 rows  (47.3s)
    Phase 3  Bar coverage  ...
    Phase 3  Bar coverage: 84,526 rows  (2.1s)
    Phase 4  Integrity  ...
    Phase 4  Integrity: 337 rows  (0.6s)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from btkit.audit.rules import phase1_iv, phase2_delta, phase3_coverage, phase4_integrity
from btkit.audit.schema import OPTION_AUDIT_DDL, empty_audit_df

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FlagCount:
    flag_code: str
    severity: str
    row_count: int
    instrument_count: int


@dataclass
class AuditResult:
    db_path: str
    dry_run: bool
    phase2_skipped: bool
    elapsed_seconds: float
    flag_counts: list[FlagCount]
    total_flagged_instruments: int


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AuditRunner:
    def __init__(
        self,
        db_path: str,
        *,
        dry_run: bool = False,
        skip_phase2: bool = False,
    ) -> None:
        self.db_path = db_path
        self.dry_run = dry_run
        self.skip_phase2 = skip_phase2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> AuditResult:
        """
        Run all audit phases, write the option_audit table (unless dry_run),
        and return an AuditResult with per-flag statistics.

        verbose=True  — print a per-phase progress line to stdout
                        (phase 2 shows an inline date counter that updates in place)
        verbose=False — silent; useful for JSON/CSV output and testing
        """
        t0 = time.perf_counter()

        con = duckdb.connect(self.db_path)
        try:
            all_flags = self._run_phases(con, verbose=verbose)

            if not self.dry_run:
                self._write_audit_table(con, all_flags)
        finally:
            con.close()

        elapsed = time.perf_counter() - t0
        flag_counts = self._compute_flag_counts(all_flags)
        total_instruments = all_flags["instrument_id"].n_unique() if not all_flags.is_empty() else 0

        return AuditResult(
            db_path=self.db_path,
            dry_run=self.dry_run,
            phase2_skipped=self.skip_phase2,
            elapsed_seconds=elapsed,
            flag_counts=flag_counts,
            total_flagged_instruments=total_instruments,
        )

    def quintile_summary(self) -> pl.DataFrame:
        """
        Compute the quintile breakdown of flagged instruments by first-bar delta
        and option right (P/C).

        Returns a DataFrame with columns:
            right, quintile, delta_min, delta_max, delta_median,
            total_instruments, flagged_instruments, pct_flagged

        Requires the option_audit table to exist (run() without dry_run first).

        Puts: Q1 = deep ITM (delta ≈ -1), Q5 = deep OTM (delta ≈ 0).
        Calls: Q1 = deep OTM (delta ≈ 0), Q5 = deep ITM (delta ≈ 1).
        """
        if not Path(self.db_path).exists():
            return pl.DataFrame()

        con = duckdb.connect(self.db_path)
        try:
            audit_exists = (
                con.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name = 'option_audit'"
                ).fetchone()[0]
                > 0
            )

            if not audit_exists:
                return pl.DataFrame()

            return con.execute(
                """
                WITH first_delta AS (
                    SELECT
                        CAST(g.instrument_id AS BIGINT) AS instrument_id,
                        FIRST(g.delta    ORDER BY g.ts_event) AS first_delta,
                        FIRST(ob."right" ORDER BY g.ts_event) AS right
                    FROM option_greeks g
                    JOIN option_bars ob
                      ON ob.instrument_id = g.instrument_id
                     AND ob.ts_event      = g.ts_event
                    WHERE g.delta IS NOT NULL AND NOT isnan(g.delta)
                    GROUP BY g.instrument_id
                ),
                quintile_assignment AS (
                    SELECT
                        instrument_id,
                        first_delta,
                        right,
                        NTILE(5) OVER (
                            PARTITION BY right ORDER BY first_delta
                        ) AS quintile
                    FROM first_delta
                ),
                flagged_set AS (
                    SELECT DISTINCT instrument_id FROM option_audit
                ),
                summary AS (
                    SELECT
                        q.right,
                        q.quintile,
                        MIN(q.first_delta)   AS delta_min,
                        MAX(q.first_delta)   AS delta_max,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (
                            ORDER BY q.first_delta
                        )                    AS delta_median,
                        COUNT(*)             AS total_instruments,
                        COUNT(f.instrument_id) AS flagged_instruments
                    FROM quintile_assignment q
                    LEFT JOIN flagged_set f USING (instrument_id)
                    GROUP BY q.right, q.quintile
                    ORDER BY q.right DESC, q.quintile
                )
                SELECT
                    right,
                    quintile,
                    delta_min,
                    delta_max,
                    delta_median,
                    total_instruments,
                    flagged_instruments,
                    ROUND(
                        flagged_instruments::DOUBLE / NULLIF(total_instruments, 0) * 100.0,
                        1
                    ) AS pct_flagged
                FROM summary
                """
            ).pl()
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_phases(
        self,
        con: duckdb.DuckDBPyConnection,
        verbose: bool,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []

        def _print(msg: str, *, end: str = "\n") -> None:
            if verbose:
                sys.stdout.write(msg + end)
                sys.stdout.flush()

        # ---- Phase 1: IV flags -----------------------------------------------
        _print("  Phase 1  IV flags  ...", end="")
        t = time.perf_counter()
        df1 = phase1_iv.run(con)
        elapsed = time.perf_counter() - t
        _print(f"\r  Phase 1  IV flags: {len(df1):>10,} rows  ({elapsed:.1f}s)")
        if not df1.is_empty():
            frames.append(df1)

        # ---- Phase 2: Black-76 delta consistency (optional) ------------------
        if not self.skip_phase2:
            _print("  Phase 2  Delta consistency: materialising 3-way join ...")
            t = time.perf_counter()

            def _p2_tick(done: int, total: int) -> None:
                pct = int(done / total * 100) if total else 0
                _print(f"\r    {done:>5} / {total} dates  ({pct}%)", end="")

            df2 = phase2_delta.run(con, progress_cb=_p2_tick if verbose else None)
            elapsed = time.perf_counter() - t
            # Overwrite the last \r progress line with the completion summary.
            _print(f"\r  Phase 2  Delta consistency: {len(df2):>10,} rows  ({elapsed:.1f}s)")
            if not df2.is_empty():
                frames.append(df2)

        # ---- Phase 3: bar coverage -------------------------------------------
        _print("  Phase 3  Bar coverage  ...", end="")
        t = time.perf_counter()
        df3 = phase3_coverage.run(con)
        elapsed = time.perf_counter() - t
        _print(f"\r  Phase 3  Bar coverage: {len(df3):>10,} rows  ({elapsed:.1f}s)")
        if not df3.is_empty():
            frames.append(df3)

        # ---- Phase 4: integrity checks ---------------------------------------
        _print("  Phase 4  Integrity  ...", end="")
        t = time.perf_counter()
        df4 = phase4_integrity.run(con)
        elapsed = time.perf_counter() - t
        _print(f"\r  Phase 4  Integrity: {len(df4):>10,} rows  ({elapsed:.1f}s)")
        if not df4.is_empty():
            frames.append(df4)

        if not frames:
            return empty_audit_df()

        combined = pl.concat(frames)

        # De-duplicate: same (instrument_id, ts_event, flag_code) can appear in
        # multiple phases (e.g. a zombie bar may also have a negative close).
        return combined.unique(
            subset=["instrument_id", "ts_event", "flag_code"],
            keep="first",
            maintain_order=True,
        )

    def _write_audit_table(
        self,
        con: duckdb.DuckDBPyConnection,
        flags: pl.DataFrame,
    ) -> None:
        for stmt in OPTION_AUDIT_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                con.execute(stmt)

        con.execute("DELETE FROM option_audit")

        if flags.is_empty():
            return

        con.register("_audit_batch", flags)
        con.execute(
            """
            INSERT INTO option_audit
            SELECT instrument_id, ts_event, flag_code, flag_severity, flag_value, threshold
            FROM _audit_batch
            """
        )
        con.unregister("_audit_batch")

    @staticmethod
    def _compute_flag_counts(flags: pl.DataFrame) -> list[FlagCount]:
        if flags.is_empty():
            return []

        summary = (
            flags.group_by(["flag_code", "flag_severity"])
            .agg(
                [
                    pl.len().alias("row_count"),
                    pl.col("instrument_id").n_unique().alias("instrument_count"),
                ]
            )
            .sort("flag_code")
        )

        return [
            FlagCount(
                flag_code=row["flag_code"],
                severity=row["flag_severity"],
                row_count=row["row_count"],
                instrument_count=row["instrument_count"],
            )
            for row in summary.iter_rows(named=True)
        ]
