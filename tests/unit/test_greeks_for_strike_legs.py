"""
Unit tests for InputDatabase.greeks_for_strike_legs against a REAL DuckDB
connection (not a mock) — covering both the time_tolerance=0 exact-equality
path and the time_tolerance>0 tolerant path.

The tolerant path is a Python-level loop over (underlying_id, calendar date)
partitions, one bounded SQL query per partition — this is the known-good
implementation, validated against a real multi-year, multi-combination study
run (108/162 combinations completed with stable memory before being killed
for an unrelated reason). A single-query alternative (expressing the same
partitioning as join equality keys and trusting DuckDB to hash-join and
bound memory internally) was tried and reverted: it benchmarked ~5x faster
in synthetic testing (up to 250 partitions) with byte-identical results, but
caused a faster and more severe memory blowup than this loop version at real
multi-year scale (confirmed live). Do not re-attempt that collapse without
verifying the query plan (EXPLAIN ANALYZE) and testing at real scale first —
synthetic partition-count benchmarking did not surface the regression.
The connection's session TimeZone is still pinned to UTC in
InputDatabase.__init__ (a separate, legitimate correctness fix, unrelated to
the blowup) — see TestSessionTimezoneRegression, though this loop version's
own SQL doesn't use CAST(... AS DATE) and so isn't exposed to that specific
failure mode regardless.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import duckdb
import polars as pl

from btkit.db.input_db import INPUT_SCHEMA_SQL, InputDatabase

_EXPIRY = date(2026, 5, 8)


def _seed_db(db_path: str, option_bars: list[tuple], option_greeks: list[tuple]) -> None:
    con = duckdb.connect(db_path)
    con.execute(INPUT_SCHEMA_SQL)
    for row in option_bars:
        con.execute("INSERT INTO option_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
    for row in option_greeks:
        con.execute("INSERT INTO option_greeks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
    con.close()


def _option_bar(
    ts: datetime,
    instrument_id: int,
    underlying_id: int,
    strike: float,
    right: str = "P",
    close: float = 2.00,
) -> tuple:
    # ts, id, underlying_id, symbol, expiration, strike, right, mult, o, h, l, c, vol
    return (
        ts,
        instrument_id,
        underlying_id,
        f"ES {strike}{right}",
        _EXPIRY,
        strike,
        right,
        50,
        close,
        close,
        close,
        close,
        10,
    )


def _option_greek(ts: datetime, instrument_id: int, underlying_id: int, dte: int = 21) -> tuple:
    # ts_event, instrument_id, underlying_id, dte, T, iv, delta, gamma, theta, vega, lag_s
    return (ts, instrument_id, underlying_id, dte, 0.05, 0.20, -0.05, 0.001, -0.05, 0.15, 0)


def _strike_targets(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_event": pl.Series([r["ts_event"] for r in rows], dtype=pl.Datetime("us", "UTC")),
            "anchor_ts_event": pl.Series(
                [r["anchor_ts_event"] for r in rows], dtype=pl.Datetime("us", "UTC")
            ),
            "underlying_id": pl.Series([r["underlying_id"] for r in rows], dtype=pl.Int64),
            "leg_name": pl.Series([r.get("leg_name", "long_put") for r in rows], dtype=pl.Utf8),
            "right": pl.Series([r.get("right", "P") for r in rows], dtype=pl.Utf8),
            "target_strike": pl.Series([r["target_strike"] for r in rows], dtype=pl.Float64),
            "reference_expiration": pl.Series(
                [r.get("reference_expiration", _EXPIRY) for r in rows], dtype=pl.Date
            ),
        }
    )


class TestExactMatch:
    def test_exact_match_found(self, tmp_path):
        ts = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
        db_path = str(tmp_path / "input.db")
        _seed_db(
            db_path,
            option_bars=[_option_bar(ts, 200, 100, 4950.0)],
            option_greeks=[_option_greek(ts, 200, 100)],
        )
        db = InputDatabase(db_path)
        targets = _strike_targets(
            [{"ts_event": ts, "anchor_ts_event": ts, "underlying_id": 100, "target_strike": 4950.0}]
        )
        result = db.greeks_for_strike_legs(targets, time_tolerance=0)
        assert not result.is_empty()
        assert result["instrument_id"][0] == 200
        assert result["ts_event"][0] == ts

    def test_no_match_when_not_exact(self, tmp_path):
        ts = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
        db_path = str(tmp_path / "input.db")
        _seed_db(
            db_path,
            option_bars=[_option_bar(ts + timedelta(seconds=20), 200, 100, 4950.0)],
            option_greeks=[_option_greek(ts + timedelta(seconds=20), 200, 100)],
        )
        db = InputDatabase(db_path)
        targets = _strike_targets(
            [{"ts_event": ts, "anchor_ts_event": ts, "underlying_id": 100, "target_strike": 4950.0}]
        )
        result = db.greeks_for_strike_legs(targets, time_tolerance=0)
        assert result.is_empty()


class TestTolerantMatch:
    def test_within_tolerance_found(self, tmp_path):
        ts = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
        candidate_ts = ts + timedelta(seconds=45)
        db_path = str(tmp_path / "input.db")
        _seed_db(
            db_path,
            option_bars=[_option_bar(candidate_ts, 200, 100, 4950.0)],
            option_greeks=[_option_greek(candidate_ts, 200, 100)],
        )
        db = InputDatabase(db_path)
        targets = _strike_targets(
            [{"ts_event": ts, "anchor_ts_event": ts, "underlying_id": 100, "target_strike": 4950.0}]
        )
        result = db.greeks_for_strike_legs(targets, time_tolerance=60)
        assert not result.is_empty()
        assert result["instrument_id"][0] == 200
        assert result["ts_event"][0] == ts, "row identity must stay the anchor row's ts_event"
        assert result["_match_ts_event"][0] == candidate_ts

    def test_beyond_tolerance_not_found(self, tmp_path):
        ts = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
        candidate_ts = ts + timedelta(seconds=90)
        db_path = str(tmp_path / "input.db")
        _seed_db(
            db_path,
            option_bars=[_option_bar(candidate_ts, 200, 100, 4950.0)],
            option_greeks=[_option_greek(candidate_ts, 200, 100)],
        )
        db = InputDatabase(db_path)
        targets = _strike_targets(
            [{"ts_event": ts, "anchor_ts_event": ts, "underlying_id": 100, "target_strike": 4950.0}]
        )
        result = db.greeks_for_strike_legs(targets, time_tolerance=60)
        assert result.is_empty()

    def test_nearest_in_time_wins_over_nearest_strike(self, tmp_path):
        """Two candidates near one anchor, both within strike_tolerance (a hard
        filter, not a soft ranking input — a candidate outside it is never a
        match regardless of time): one slightly off-strike but nearest in
        time, one exact-strike but farther in time. Nearest-in-time must win
        (matches the documented tier-1 priority)."""
        ts = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
        near_time_off_strike_ts = ts + timedelta(seconds=5)
        far_time_exact_strike_ts = ts + timedelta(seconds=50)
        db_path = str(tmp_path / "input.db")
        _seed_db(
            db_path,
            option_bars=[
                _option_bar(near_time_off_strike_ts, 201, 100, 4950.8),  # within $1 tolerance
                _option_bar(far_time_exact_strike_ts, 202, 100, 4950.0),  # exact strike
            ],
            option_greeks=[
                _option_greek(near_time_off_strike_ts, 201, 100),
                _option_greek(far_time_exact_strike_ts, 202, 100),
            ],
        )
        db = InputDatabase(db_path)
        targets = _strike_targets(
            [{"ts_event": ts, "anchor_ts_event": ts, "underlying_id": 100, "target_strike": 4950.0}]
        )
        result = db.greeks_for_strike_legs(targets, time_tolerance=60)
        assert not result.is_empty()
        assert result["instrument_id"][0] == 201, "nearest-in-time candidate must win"

    def test_multiple_underlying_ids_partitioned_correctly(self, tmp_path):
        """Two anchors on different underlying_ids (contract roll) must each
        find their own match — the per-underlying_id partitioning loop must
        not lose or cross-contaminate anchors."""
        ts1 = datetime(2026, 2, 1, 14, 30, tzinfo=UTC)
        ts2 = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
        db_path = str(tmp_path / "input.db")
        _seed_db(
            db_path,
            option_bars=[
                _option_bar(ts1 + timedelta(seconds=10), 201, 100, 4800.0),
                _option_bar(ts2 + timedelta(seconds=10), 202, 200, 4950.0),
            ],
            option_greeks=[
                _option_greek(ts1 + timedelta(seconds=10), 201, 100),
                _option_greek(ts2 + timedelta(seconds=10), 202, 200),
            ],
        )
        db = InputDatabase(db_path)
        targets = _strike_targets(
            [
                {
                    "ts_event": ts1,
                    "anchor_ts_event": ts1,
                    "underlying_id": 100,
                    "target_strike": 4800.0,
                },
                {
                    "ts_event": ts2,
                    "anchor_ts_event": ts2,
                    "underlying_id": 200,
                    "target_strike": 4950.0,
                },
            ]
        )
        result = db.greeks_for_strike_legs(targets, time_tolerance=60)
        assert len(result) == 2
        by_ts = {row["ts_event"]: row["instrument_id"] for row in result.iter_rows(named=True)}
        assert by_ts[ts1] == 201
        assert by_ts[ts2] == 202

    def test_same_underlying_different_days_both_resolved(self, tmp_path):
        """Two anchors on the SAME underlying_id but different calendar days
        (the new day-level sub-partitioning) must each independently find
        their own match, not just underlying_id-level partitioning."""
        day1 = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
        day2 = datetime(2026, 5, 4, 14, 30, tzinfo=UTC)
        db_path = str(tmp_path / "input.db")
        _seed_db(
            db_path,
            option_bars=[
                _option_bar(day1 + timedelta(seconds=10), 201, 100, 4800.0),
                _option_bar(day2 + timedelta(seconds=10), 202, 100, 4820.0),
            ],
            option_greeks=[
                _option_greek(day1 + timedelta(seconds=10), 201, 100),
                _option_greek(day2 + timedelta(seconds=10), 202, 100),
            ],
        )
        db = InputDatabase(db_path)
        targets = _strike_targets(
            [
                {
                    "ts_event": day1,
                    "anchor_ts_event": day1,
                    "underlying_id": 100,
                    "target_strike": 4800.0,
                },
                {
                    "ts_event": day2,
                    "anchor_ts_event": day2,
                    "underlying_id": 100,
                    "target_strike": 4820.0,
                },
            ]
        )
        result = db.greeks_for_strike_legs(targets, time_tolerance=60)
        assert len(result) == 2
        by_ts = {row["ts_event"]: row["instrument_id"] for row in result.iter_rows(named=True)}
        assert by_ts[day1] == 201
        assert by_ts[day2] == 202

    def test_anchor_ts_event_differs_from_ts_event(self, tmp_path):
        """Row identity (ts_event, the bar's own timestamp) is independent of
        the anchor (anchor_ts_event, e.g. another leg's resolved timestamp) —
        matching happens against the anchor, but the returned row keys off
        the original row identity."""
        bar_ts = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
        anchor_ts = bar_ts + timedelta(seconds=8)  # main leg resolved 8s off the bar
        candidate_ts = anchor_ts + timedelta(seconds=7)  # 7s from the anchor, not the bar
        db_path = str(tmp_path / "input.db")
        _seed_db(
            db_path,
            option_bars=[_option_bar(candidate_ts, 200, 100, 4950.0)],
            option_greeks=[_option_greek(candidate_ts, 200, 100)],
        )
        db = InputDatabase(db_path)
        targets = _strike_targets(
            [
                {
                    "ts_event": bar_ts,
                    "anchor_ts_event": anchor_ts,
                    "underlying_id": 100,
                    "target_strike": 4950.0,
                }
            ]
        )
        result = db.greeks_for_strike_legs(targets, time_tolerance=10)
        assert not result.is_empty()
        assert result["ts_event"][0] == bar_ts
        assert result["_match_ts_event"][0] == candidate_ts


class TestDenseMultiDayScale:
    def test_many_anchors_across_many_days_all_resolved_correctly(self, tmp_path):
        """Moderate-density stress test: one underlying_id, 10 trading days,
        20 anchors/day (200 total), each with its own correct nearest-time
        candidate — exercises the day-level sub-partitioning loop across many
        iterations and confirms no anchor is lost, duplicated, or mismatched
        to the wrong day's candidate."""
        base_day = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
        option_bars = []
        option_greeks = []
        targets_rows = []
        expected: dict[datetime, int] = {}
        instrument_id = 1000
        for day_offset in range(10):
            day_ts = base_day + timedelta(days=day_offset)
            for minute_offset in range(20):
                anchor_ts = day_ts + timedelta(minutes=minute_offset)
                candidate_ts = anchor_ts + timedelta(seconds=12)
                strike = 4800.0 + day_offset * 10 + minute_offset
                instrument_id += 1
                option_bars.append(_option_bar(candidate_ts, instrument_id, 100, strike))
                option_greeks.append(_option_greek(candidate_ts, instrument_id, 100))
                targets_rows.append(
                    {
                        "ts_event": anchor_ts,
                        "anchor_ts_event": anchor_ts,
                        "underlying_id": 100,
                        "target_strike": strike,
                    }
                )
                expected[anchor_ts] = instrument_id

        db_path = str(tmp_path / "input.db")
        _seed_db(db_path, option_bars=option_bars, option_greeks=option_greeks)
        db = InputDatabase(db_path)
        targets = _strike_targets(targets_rows)

        result = db.greeks_for_strike_legs(targets, time_tolerance=30)
        assert len(result) == len(expected), "every anchor must resolve to exactly one match"
        by_ts = {row["ts_event"]: row["instrument_id"] for row in result.iter_rows(named=True)}
        assert by_ts == expected


class TestSessionTimezoneRegression:
    def test_anchor_near_utc_midnight_still_matches(self, tmp_path):
        """
        InputDatabase pins the connection's session TimeZone to UTC so that
        SQL-side CAST(ts_event AS DATE) (used to hash-join on calendar date)
        agrees with Python-side .dt.date() bucketing. Without that pin,
        CAST uses the connection's local timezone default (observed to be
        America/New_York, not UTC, in this environment) — which disagrees
        with the UTC date for any timestamp between UTC midnight and the
        local offset's midnight, silently dropping otherwise-valid matches.

        02:30 UTC on May 2 is 22:30 ET on May 1 — a date this test picks
        specifically because the UTC and America/New_York calendar dates
        disagree, so a regression here would show up as this anchor's
        candidate (matched to the same instant) going missing.
        """
        anchor_ts = datetime(2026, 5, 2, 2, 30, tzinfo=UTC)
        candidate_ts = anchor_ts + timedelta(seconds=5)
        db_path = str(tmp_path / "input.db")
        _seed_db(
            db_path,
            option_bars=[_option_bar(candidate_ts, 200, 100, 4950.0)],
            option_greeks=[_option_greek(candidate_ts, 200, 100)],
        )
        db = InputDatabase(db_path)
        targets = _strike_targets(
            [
                {
                    "ts_event": anchor_ts,
                    "anchor_ts_event": anchor_ts,
                    "underlying_id": 100,
                    "target_strike": 4950.0,
                }
            ]
        )
        result = db.greeks_for_strike_legs(targets, time_tolerance=30)
        assert not result.is_empty(), (
            "anchor near UTC midnight must still match its candidate — a wrong "
            "session timezone would bucket the anchor and candidate onto "
            "different calendar dates and silently drop this match"
        )
        assert result["instrument_id"][0] == 200
