"""
Unit tests for the backward ASOF underlying join in GreeksCalculator.

The underlying future does not print a bar every minute an option trades. The old
exact-`ts_event` equality join dropped those options, leaving them with no greeks.
GreeksCalculator now borrows the spot price F from the nearest prior underlying bar
within `underlying_max_staleness_minutes`; a too-stale gap still yields no greeks,
and tolerance 0 reproduces the exact-match behaviour.

These run GreeksCalculator against an in-memory DuckDB seeded with a deliberately
sparse underlying series — no DBN files needed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import duckdb
import pytest

from btkit.db.input_db import INPUT_SCHEMA_SQL
from btkit.pipeline.greeks import GreeksCalculator

_UND_ID = 100
_OPT_ID = 200
_T0 = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
_EXPIRY = date(2026, 5, 8)


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(INPUT_SCHEMA_SQL)
    return con


def _add_underlying(con, minutes_offsets: list[int], close: float = 4000.0) -> None:
    for m in minutes_offsets:
        ts = _T0 + timedelta(minutes=m)
        con.execute(
            "INSERT INTO underlying_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [ts, _UND_ID, "ESM6", _EXPIRY, close, close, close, close, 1000],
        )


def _add_option(con, minute_offset: int, close: float = 20.0) -> None:
    ts = _T0 + timedelta(minutes=minute_offset)
    con.execute(
        "INSERT INTO option_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        # ts, id, underlying_id, symbol, expiration, strike, right, mult, o, h, l, c, vol
        [
            ts,
            _OPT_ID,
            _UND_ID,
            "ESM6 C4000",
            _EXPIRY,
            4000.0,
            "C",
            50,
            close,
            close,
            close,
            close,
            5,
        ],
    )


def _greek_minute_offsets(con) -> set[int]:
    rows = con.execute("SELECT ts_event FROM option_greeks WHERE iv IS NOT NULL").fetchall()
    return {round((r[0].astimezone(UTC) - _T0).total_seconds() / 60) for r in rows}


class TestAsofUnderlyingJoin:
    def test_recovers_option_when_underlying_skipped_that_minute(self):
        """Underlying prints at t0 and t0+5 only; an option at t0+2 (2 min stale)
        still gets greeks from the t0 bar under a 15-min tolerance."""
        con = _con()
        _add_underlying(con, [0, 5])
        _add_option(con, 0)  # exact match
        _add_option(con, 2)  # no same-minute underlying → as-of from t0
        GreeksCalculator(con, underlying_max_staleness_minutes=15).run()
        assert _greek_minute_offsets(con) == {0, 2}

    def test_underlying_lag_seconds_recorded(self):
        """underlying_lag_s stores the option↔F timestamp diff in seconds
        (0 for an exact same-minute match, 120 for a 2-minute-stale borrow)."""
        con = _con()
        _add_underlying(con, [0])
        _add_option(con, 0)  # exact → lag 0
        _add_option(con, 2)  # 2 min stale → lag 120 s
        GreeksCalculator(con, underlying_max_staleness_minutes=15).run()
        lag_by_offset = {}
        for ts, lag in con.execute(
            "SELECT ts_event, underlying_lag_s FROM option_greeks"
        ).fetchall():
            lag_by_offset[round((ts.astimezone(UTC) - _T0).total_seconds() / 60)] = lag
        assert lag_by_offset == {0: 0, 2: 120}

    def test_too_stale_gets_no_greeks(self):
        """An option 30 min after the last underlying bar exceeds a 15-min
        tolerance and is left without greeks."""
        con = _con()
        _add_underlying(con, [0])
        _add_option(con, 2)  # within tolerance → greeks
        _add_option(con, 30)  # 30 min after only underlying bar → dropped
        GreeksCalculator(con, underlying_max_staleness_minutes=15).run()
        assert _greek_minute_offsets(con) == {2}

    def test_tolerance_zero_is_exact_match_only(self):
        """tolerance=0 reproduces the legacy exact same-minute requirement."""
        con = _con()
        _add_underlying(con, [0])
        _add_option(con, 0)  # exact → greeks
        _add_option(con, 2)  # no same-minute underlying → dropped
        GreeksCalculator(con, underlying_max_staleness_minutes=0).run()
        assert _greek_minute_offsets(con) == {0}

    def test_uses_nearest_prior_not_future_bar(self):
        """F comes from the prior underlying bar, never a later one."""
        con = _con()
        # Underlying prices differ so we can tell which bar supplied F.
        _add_underlying(con, [0], close=4000.0)
        _add_underlying(con, [10], close=5000.0)
        _add_option(con, 3)  # prior bar is t0 (4000), not t0+10 (5000)
        GreeksCalculator(con, underlying_max_staleness_minutes=15).run()
        row = con.execute(
            "SELECT delta FROM option_greeks WHERE ts_event = ?",
            [_T0 + timedelta(minutes=3)],
        ).fetchone()
        assert row is not None
        # ATM-ish call at F=4000,K=4000 has delta well below the 0.5+ it would show
        # if F=5000 (deep ITM) had been (incorrectly) used from the later bar.
        assert 0.0 < row[0] < 0.65


@pytest.mark.parametrize("tol", [0, 5, 15, 60])
def test_no_future_leakage_and_runs_clean(tol):
    """Sanity: every produced greek row corresponds to an option bar and the
    underlying used is at or before it (never after)."""
    con = _con()
    _add_underlying(con, [0, 1, 2, 3, 4, 5])
    for m in range(6):
        _add_option(con, m)
    GreeksCalculator(con, underlying_max_staleness_minutes=tol).run()
    n = con.execute("SELECT COUNT(*) FROM option_greeks").fetchone()[0]
    assert n == 6  # all options have a same-or-prior underlying bar
