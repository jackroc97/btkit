"""
Verify a rebuilt input database and compare it against the one it replaces.

Checks the invariants that the duplication and routing fixes were meant to
establish, then diffs coverage against the old database so an unexpected loss of
data is caught before the files are swapped.

    python scripts/verify_rebuild.py --new path/to/es_rebuild.db [--old path/to/es.db]

Exits non-zero if any invariant fails, so it can gate a swap.
See docs/data_duplication_fix.md.
"""

from __future__ import annotations

import argparse
import sys

import duckdb
import polars as pl

from btkit.audit.rules import phase5_duplication


def _scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def _check(label: str, actual, expected, failures: list[str]) -> None:
    ok = actual == expected
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {label}: {actual:,}"
        if isinstance(actual, int)
        else f"  [{status}] {label}: {actual}"
    )
    if not ok:
        failures.append(f"{label}: got {actual}, expected {expected}")


def _report_lost_instruments(new_path: str, old_path: str) -> None:
    """Report instrument_ids present in the old database but absent from the new one.

    Routing inner-joins bars to definition records, so an id whose definition is no
    longer in raw_data/ loses its bars. The original ES database was built
    incrementally over months, and it holds bars for 7,638 ids more than today's
    definition files can route — those bars cannot be reproduced without
    re-downloading the definition ranges that covered them.

    The per-month breakdown is the actionable part: it says which ranges to re-request
    from Databento.
    """
    probe = duckdb.connect(":memory:")
    probe.execute("SET threads TO 4")
    probe.execute(f"ATTACH '{old_path}' AS o (READ_ONLY)")
    probe.execute(f"ATTACH '{new_path}' AS n (READ_ONLY)")

    lost = probe.execute(
        """
        WITH old_ids AS (SELECT DISTINCT instrument_id FROM o.option_bars),
             new_ids AS (SELECT DISTINCT instrument_id FROM n.option_bars)
        SELECT instrument_id FROM old_ids
        WHERE instrument_id NOT IN (SELECT instrument_id FROM new_ids)
        """
    ).pl()

    print("\n  Instruments present in the old database but not the new one:")
    if lost.is_empty():
        print("    none")
        probe.close()
        return

    ids = lost["instrument_id"].to_list()
    probe.register("_lost", lost)
    detail = probe.execute(
        """
        SELECT strftime(ts_event, '%Y-%m') AS month,
               count(*) AS bars,
               count(DISTINCT instrument_id) AS instruments
        FROM o.option_bars
        WHERE instrument_id IN (SELECT instrument_id FROM _lost)
        GROUP BY 1 ORDER BY 1
        """
    ).pl()
    total_bars = int(detail["bars"].sum())
    print(f"    {len(ids):,} instrument_ids, {total_bars:,} bars")
    print("    These lack a definition record in the current raw_data/, so routing")
    print("    drops them. Re-download the definition files for these months to recover:")
    print(f"      {'month':>9} {'bars':>10} {'instruments':>12}")
    for row in detail.iter_rows(named=True):
        print(f"      {row['month']:>9} {row['bars']:>10,} {row['instruments']:>12,}")
    probe.close()


def verify(new_path: str, old_path: str | None) -> int:
    failures: list[str] = []
    con = duckdb.connect(new_path, read_only=True)
    con.execute("SET threads TO 8")

    print(f"\n=== Row counts: {new_path} ===")
    tables = ["contract", "underlying_bars", "option_bars", "option_greeks", "indicator_bars"]
    new_counts = {}
    for t in tables:
        try:
            new_counts[t] = _scalar(con, f"SELECT count(*) FROM {t}")
        except duckdb.CatalogException:
            new_counts[t] = None
            failures.append(f"table {t} is missing")
        print(
            f"  {t:18} {new_counts[t]:>14,}" if new_counts[t] is not None else f"  {t:18} MISSING"
        )

    # ── Invariant 1: no duplicates, no same-day id collisions ────────────────
    print("\n=== Phase 5 audit (all must be zero) ===")
    flags = phase5_duplication.run(con)
    counts = (
        dict(flags.group_by("flag_code").agg(pl.len().alias("n")).iter_rows())
        if not flags.is_empty()
        else {}
    )
    for code in ("DUPLICATE_BAR", "DUPLICATE_GREEKS", "ID_DAY_COLLISION"):
        _check(code, counts.get(code, 0), 0, failures)
    # Soft: an exchange moved an expiration and the vendor republished the contract,
    # so bars either side of the announcement carry different expirations. Reported
    # because the superseded date gives a wrong DTE, but it is not a misroute.
    print(f"  [INFO] EXPIRY_RESCHEDULED: {counts.get('EXPIRY_RESCHEDULED', 0):,}")

    # ── Invariant 2: contract_id is populated and resolves ───────────────────
    print("\n=== contract_id integrity ===")
    _check(
        "option_bars with null contract_id",
        _scalar(con, "SELECT count(*) FROM option_bars WHERE contract_id IS NULL"),
        0,
        failures,
    )
    _check(
        "option_greeks with null contract_id",
        _scalar(con, "SELECT count(*) FROM option_greeks WHERE contract_id IS NULL"),
        0,
        failures,
    )
    _check(
        "option_bars contract_id missing from dimension",
        _scalar(
            con,
            "SELECT count(*) FROM option_bars b WHERE NOT EXISTS "
            "(SELECT 1 FROM contract c WHERE c.contract_id = b.contract_id)",
        ),
        0,
        failures,
    )
    _check(
        "contracts whose (symbol, expiration) disagrees with their economics",
        _scalar(
            con,
            "SELECT count(*) FROM (SELECT symbol, expiration FROM option_bars "
            "GROUP BY 1, 2 HAVING count(DISTINCT concat(strike_price,'|',\"right\")) > 1)",
        ),
        0,
        failures,
    )

    # ── Invariant 3: one contract per instrument-day ─────────────────────────
    print("\n=== Routing ===")
    # Only genuinely different instruments count as a routing failure; the same
    # contract under a rescheduled expiration is counted separately above.
    _check(
        "(instrument_id, UTC day) resolving to >1 DIFFERENT contract",
        _scalar(
            con,
            "SELECT count(*) FROM (SELECT instrument_id, "
            "CAST(ts_event AT TIME ZONE 'UTC' AS DATE) FROM option_bars "
            "GROUP BY 1, 2 HAVING count(DISTINCT symbol) > 1 "
            'OR count(DISTINCT strike_price) > 1 OR count(DISTINCT "right") > 1)',
        ),
        0,
        failures,
    )

    # ── Comparison against the database being replaced ───────────────────────
    if old_path:
        print(f"\n=== Coverage vs {old_path} ===")
        old = duckdb.connect(old_path, read_only=True)
        old.execute("SET threads TO 8")
        print(f"  {'table':18} {'old':>14} {'new':>14} {'delta':>14}")
        for t in tables:
            try:
                o = _scalar(old, f"SELECT count(*) FROM {t}")
            except duckdb.CatalogException:
                continue
            n = new_counts.get(t) or 0
            print(f"  {t:18} {o:>14,} {n:>14,} {n - o:>+14,}")

        # Per-year row counts. Deduplication should remove rows roughly in
        # proportion to where the duplicate window was (2026-01..05 for ES); a
        # shortfall concentrated in other years points at source files that were
        # ingested historically but are no longer in raw_data/.
        print("\n  option_bars by year:")
        print(f"    {'year':>6} {'old':>14} {'new':>14} {'delta':>12} {'unique_old':>14}")
        old_year = dict(
            old.execute("SELECT year(ts_event), count(*) FROM option_bars GROUP BY 1").fetchall()
        )
        old_year_uniq = dict(
            old.execute(
                "SELECT year(ts_event), count(*) FROM "
                "(SELECT DISTINCT instrument_id, ts_event, year(ts_event) AS year "
                " FROM option_bars) GROUP BY 1"
            ).fetchall()
        )
        new_year = dict(
            con.execute("SELECT year(ts_event), count(*) FROM option_bars GROUP BY 1").fetchall()
        )
        for y in sorted(set(old_year) | set(new_year)):
            o, n, u = old_year.get(y, 0), new_year.get(y, 0), old_year_uniq.get(y, 0)
            # Against the OLD unique-key count: a negative delta here is a real
            # coverage loss, not deduplication.
            mark = "  <-- below old unique keys" if n < u else ""
            print(f"    {y:>6} {o:>14,} {n:>14,} {n - o:>+12,} {u:>14,}{mark}")
            if n < u:
                failures.append(f"{y}: {u - n:,} option bars below the old unique-key count")

        # Distinct contracts and date span must not shrink: deduplication removes
        # rows, never coverage.
        for label, sql in (
            (
                "distinct (symbol, expiration)",
                "SELECT count(DISTINCT concat(symbol,'|',expiration)) FROM option_bars",
            ),
            ("min ts_event", "SELECT min(ts_event) FROM option_bars"),
            ("max ts_event", "SELECT max(ts_event) FROM option_bars"),
        ):
            o, n = _scalar(old, sql), _scalar(con, sql)
            flag = "" if o == n else ("  <-- CHANGED" if "ts_event" in label else "")
            print(f"  {label:28} old={o}  new={n}{flag}")
            if "distinct" in label and n < o:
                failures.append(f"{label} shrank: {o} -> {n}")
        old.close()
        _report_lost_instruments(new_path, old_path)

    con.close()

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL INVARIANTS PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="Rebuilt database to verify.")
    ap.add_argument("--old", default=None, help="Database being replaced, for comparison.")
    args = ap.parse_args()
    sys.exit(verify(args.new, args.old))
