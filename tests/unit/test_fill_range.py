"""
Unit tests for interior "fill-the-gaps" build scoping.

Covers the two pieces added for splicing corrected data into the middle of an
existing timeline:

  * GreeksCalculator.run(fill_range=..., instruments=...) — recompute greeks only
    for option bars in the target window / underlyings.
  * IndicatorRunner.run_scoped(...) — recompute an indicator over the back-filled
    window PLUS a trailing-dependency tail, overwriting stale values (the normal
    NOT EXISTS write never overwrites), scoped to one underlying.

Everything runs against an in-memory DuckDB — no DBN files needed.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import duckdb

from btkit.db.input_db import INPUT_SCHEMA_SQL
from btkit.pipeline.greeks import GreeksCalculator
from btkit.pipeline.indicators import IndicatorRunner


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(INPUT_SCHEMA_SQL)
    return con


def _day(d: int) -> datetime:
    return datetime(2020, 1, d, 14, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Scoped greeks recompute
# ---------------------------------------------------------------------------


class TestScopedGreeks:
    def _seed(self, con):
        # One underlying with a bar every day 1..20; one option per day (missing greeks).
        for d in range(1, 21):
            con.execute(
                "INSERT INTO underlying_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [_day(d), 100, "ESM0", date(2020, 3, 20), 4000.0, 4000.0, 4000.0, 4000.0, 1000],
            )
            con.execute(
                "INSERT INTO option_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    _day(d), 200, 100, "ESM0 C4000", date(2020, 3, 20), 4000.0, "C", 50,
                    20.0, 20.0, 20.0, 20.0, 5,
                ],
            )

    def _greek_days(self, con) -> set[int]:
        rows = con.execute("SELECT ts_event FROM option_greeks").fetchall()
        return {r[0].astimezone(UTC).day for r in rows}

    def test_fill_range_limits_greeks_to_window(self):
        con = _con()
        self._seed(con)
        GreeksCalculator(con).run(
            skip_existing=True,
            fill_range=(_day(8), datetime(2020, 1, 12, 23, 59, 59, tzinfo=UTC)),
        )
        assert self._greek_days(con) == {8, 9, 10, 11, 12}

    def test_instruments_filter(self):
        con = _con()
        self._seed(con)
        # underlying 999 has no bars → scoping to it yields nothing
        GreeksCalculator(con).run(
            skip_existing=True, fill_range=(_day(1), _day(20)), instruments=[999]
        )
        assert self._greek_days(con) == set()

    def test_unscoped_still_full(self):
        con = _con()
        self._seed(con)
        GreeksCalculator(con).run(skip_existing=True)
        assert self._greek_days(con) == set(range(1, 21))


# ---------------------------------------------------------------------------
# Scoped indicator recompute (trailing-dependency refresh)
# ---------------------------------------------------------------------------

_SCRIPT = """
import polars as pl

def compute(df):
    return df.with_columns(pl.col("close").rolling_mean(window_size=3).alias("roll3"))
"""


class TestScopedIndicatorRecompute:
    def _add_underlying(self, con, days: list[int]):
        for d in days:
            con.execute(
                "INSERT INTO underlying_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [_day(d), 1, "ESM0", None, 0.0, 0.0, 0.0, float(d), 100],  # close == day number
            )

    def _roll3_by_day(self, con) -> dict[int, float]:
        rows = con.execute(
            "SELECT ib.ts_event, ib.value FROM indicator_bars ib "
            "JOIN indicator_definition d ON ib.indicator_id = d.id WHERE d.name = 'roll3'"
        ).fetchall()
        return {r[0].astimezone(UTC).day: r[1] for r in rows}

    def test_backfill_refreshes_window_and_trailing_tail(self, tmp_path):
        script = tmp_path / "roll.py"
        script.write_text(_SCRIPT)
        con = _con()

        # Timeline day 1..20 but with a GAP at days 10,11,12 (as if the future's
        # bars were misfiled/missing there).
        present = [d for d in range(1, 21) if d not in (10, 11, 12)]
        self._add_underlying(con, present)

        # First indicator pass over the gapped data. roll3 at day 13 is stale:
        # rows are consecutive so it averages closes 8,9,13 instead of 11,12,13.
        IndicatorRunner(con, script).run(1)
        stale = self._roll3_by_day(con)
        assert stale[13] == (8 + 9 + 13) / 3  # == 10.0, wrong

        # Splice in the missing gap bars, then scoped-recompute over the window.
        self._add_underlying(con, [10, 11, 12])
        IndicatorRunner(con, script).run_scoped(1, _day(10), _day(12), tail_days=3)

        got = self._roll3_by_day(con)
        # Gap days now present and correct: mean of the 3 trailing closes.
        assert got[10] == (8 + 9 + 10) / 3
        assert got[11] == (9 + 10 + 11) / 3
        assert got[12] == (10 + 11 + 12) / 3
        # Trailing tail (day 13) refreshed from stale 10.0 → correct 12.0.
        assert got[13] == (11 + 12 + 13) / 3
        # Pre-window value untouched (still correct, backward window can't reach the gap).
        assert got[9] == (7 + 8 + 9) / 3
        # Far-forward value outside window+tail untouched.
        assert got[20] == (18 + 19 + 20) / 3

    def test_idempotent(self, tmp_path):
        script = tmp_path / "roll.py"
        script.write_text(_SCRIPT)
        con = _con()
        self._add_underlying(con, list(range(1, 21)))
        IndicatorRunner(con, script).run(1)
        before = self._roll3_by_day(con)
        r = IndicatorRunner(con, script)
        r.run_scoped(1, _day(10), _day(12), tail_days=3)
        r.run_scoped(1, _day(10), _day(12), tail_days=3)
        after = self._roll3_by_day(con)
        assert before == after
        # No duplicate rows in the overwritten window.
        n = con.execute(
            "SELECT COUNT(*) FROM indicator_bars ib JOIN indicator_definition d "
            "ON ib.indicator_id = d.id WHERE d.name='roll3' "
            "AND ib.ts_event >= ? AND ib.ts_event <= ?",
            [_day(10), _day(12) + timedelta(days=3)],
        ).fetchone()[0]
        assert n == len(range(10, 16))  # days 10..15, one row each

    def test_scoped_to_underlying(self, tmp_path):
        """run_scoped for underlying 1 must not touch underlying 2's rows."""
        script = tmp_path / "roll.py"
        script.write_text(_SCRIPT)
        con = _con()
        self._add_underlying(con, list(range(1, 21)))
        for d in range(1, 21):
            con.execute(
                "INSERT INTO underlying_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [_day(d), 2, "NQM0", None, 0.0, 0.0, 0.0, float(d * 10), 100],
            )
        r = IndicatorRunner(con, script)
        r.run(1)
        r.run(2)
        n2_before = con.execute(
            "SELECT COUNT(*) FROM indicator_bars ib JOIN indicator_definition d "
            "ON ib.indicator_id = d.id WHERE d.underlying_id = 2"
        ).fetchone()[0]
        r.run_scoped(1, _day(10), _day(12), tail_days=3)
        n2_after = con.execute(
            "SELECT COUNT(*) FROM indicator_bars ib JOIN indicator_definition d "
            "ON ib.indicator_id = d.id WHERE d.underlying_id = 2"
        ).fetchone()[0]
        assert n2_before == n2_after
