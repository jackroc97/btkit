"""
Unit tests for EntryScanner's session-scoped backward as-of indicator join
(feature: conditional leg parameters, item 1).

Indicators coarser than the 1-minute entry grid (a daily regime signal, a
5-minute value) must gate *all* intraday entries in a session, not only those
that land exactly on an indicator timestamp. The merge is a backward as-of join
scoped to the session-local calendar date, never filling across the session
boundary.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from unittest.mock import MagicMock

import polars as pl

from btkit.backtest.entry import EntryScanner
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    SessionConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)


def _make_scanner(conditions: list[str] | None = None) -> EntryScanner:
    strategy = StrategyDefinition(
        name="asof_test",
        universe=UniverseConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            session=SessionConfig(
                timezone="America/New_York",
                start_time=time(9, 30),
                end_time=time(16, 0),
                weekdays_only=False,
            ),
        ),
        trades=[
            TradeDefinition(
                name="t",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(
                    window=EntryWindowConfig(start=time(9, 0), end=time(16, 0)),
                    conditions=conditions or [],
                ),
                legs=[
                    LegConfig(
                        name="sp",
                        right="put",
                        action="sell_to_open",
                        dte=21,
                        delta={"target": -0.20},
                    ),
                ],
                exit=ExitConfig(stop_loss=2.0),
            )
        ],
    )
    return EntryScanner(MagicMock(), strategy, strategy.trades[0])


# 2024-01-02: NY is UTC-5, so 14:30Z = 09:30 EST (session open).
def _utc(h: int, m: int, day: int = 2) -> datetime:
    return datetime(2024, 1, day, h, m, tzinfo=UTC)


class TestAsofJoinHelper:
    def test_daily_indicator_gates_all_intraday_rows(self):
        """One daily value at session open fills forward across the whole session."""
        scanner = _make_scanner()
        indicators = pl.DataFrame(
            {
                "ts_event": pl.Series([_utc(14, 30)], dtype=pl.Datetime("us", "UTC")),
                "regime": pl.Series([25.0], dtype=pl.Float64),
            }
        )
        frame = pl.DataFrame(
            {
                "ts_event": pl.Series(
                    [_utc(14, 30), _utc(15, 0), _utc(16, 0), _utc(19, 0)],
                    dtype=pl.Datetime("us", "UTC"),
                ),
            }
        )
        out = scanner._asof_join_indicators(frame, indicators, "ts_event").sort("ts_event")
        assert out["regime"].to_list() == [25.0, 25.0, 25.0, 25.0]

    def test_row_before_first_value_is_null(self):
        """A candidate earlier than the session's first indicator value gets null."""
        scanner = _make_scanner()
        indicators = pl.DataFrame(
            {
                "ts_event": pl.Series([_utc(14, 30)], dtype=pl.Datetime("us", "UTC")),
                "regime": pl.Series([25.0], dtype=pl.Float64),
            }
        )
        frame = pl.DataFrame(
            {
                "ts_event": pl.Series([_utc(14, 0), _utc(15, 0)], dtype=pl.Datetime("us", "UTC")),
            }
        )
        out = scanner._asof_join_indicators(frame, indicators, "ts_event").sort("ts_event")
        assert out["regime"].to_list() == [None, 25.0]

    def test_no_fill_across_session_boundary(self):
        """Session 2 must not inherit session 1's last indicator value."""
        scanner = _make_scanner()
        indicators = pl.DataFrame(
            {
                "ts_event": pl.Series([_utc(14, 30, day=2)], dtype=pl.Datetime("us", "UTC")),
                "regime": pl.Series([25.0], dtype=pl.Float64),
            }
        )
        frame = pl.DataFrame(
            {
                "ts_event": pl.Series(
                    [_utc(15, 0, day=2), _utc(15, 0, day=3)],  # next calendar day has no indicator
                    dtype=pl.Datetime("us", "UTC"),
                ),
            }
        )
        out = scanner._asof_join_indicators(frame, indicators, "ts_event").sort("ts_event")
        assert out["regime"].to_list() == [25.0, None]

    def test_mixed_cadence_carries_latest_per_column(self):
        """A daily column and a 5-min column in one frame each fill within session."""
        scanner = _make_scanner()
        # daily value present only at 14:30; 5-min column present at 14:30, 14:35, 14:40
        indicators = pl.DataFrame(
            {
                "ts_event": pl.Series(
                    [_utc(14, 30), _utc(14, 35), _utc(14, 40)],
                    dtype=pl.Datetime("us", "UTC"),
                ),
                "daily": pl.Series([25.0, None, None], dtype=pl.Float64),
                "vol5m": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            }
        )
        frame = pl.DataFrame(
            {
                "ts_event": pl.Series([_utc(14, 37), _utc(14, 41)], dtype=pl.Datetime("us", "UTC")),
            }
        )
        out = scanner._asof_join_indicators(frame, indicators, "ts_event").sort("ts_event")
        assert out["daily"].to_list() == [25.0, 25.0]  # daily carried forward
        assert out["vol5m"].to_list() == [2.0, 3.0]  # nearest-at-or-before per row

    def test_empty_indicators_returns_frame_unchanged(self):
        scanner = _make_scanner()
        frame = pl.DataFrame({"ts_event": pl.Series([_utc(15, 0)], dtype=pl.Datetime("us", "UTC"))})
        out = scanner._asof_join_indicators(frame, pl.DataFrame(), "ts_event")
        assert out.columns == ["ts_event"]


class TestConditionsUseAsof:
    def test_daily_condition_gates_all_intraday_entries(self):
        """A daily indicator used in an entry condition keeps every session entry."""
        scanner = _make_scanner(conditions=["regime > 20"])
        indicators = pl.DataFrame(
            {
                "ts_event": pl.Series([_utc(14, 30)], dtype=pl.Datetime("us", "UTC")),
                "regime": pl.Series([25.0], dtype=pl.Float64),
            }
        )
        entries = pl.DataFrame(
            {
                "entry_time": pl.Series(
                    [_utc(14, 30), _utc(15, 0), _utc(16, 0)],
                    dtype=pl.Datetime("us", "UTC"),
                ),
                "open_mark": pl.Series([1.0, 1.0, 1.0], dtype=pl.Float64),
            }
        )
        out = scanner._evaluate_conditions(entries, indicators)
        assert len(out) == 3  # all three intraday entries survive, not just 14:30

    def test_daily_condition_false_drops_all(self):
        scanner = _make_scanner(conditions=["regime > 30"])
        indicators = pl.DataFrame(
            {
                "ts_event": pl.Series([_utc(14, 30)], dtype=pl.Datetime("us", "UTC")),
                "regime": pl.Series([25.0], dtype=pl.Float64),
            }
        )
        entries = pl.DataFrame(
            {
                "entry_time": pl.Series([_utc(15, 0), _utc(16, 0)], dtype=pl.Datetime("us", "UTC")),
                "open_mark": pl.Series([1.0, 1.0], dtype=pl.Float64),
            }
        )
        out = scanner._evaluate_conditions(entries, indicators)
        assert len(out) == 0

    def test_five_minute_indicator_retention_vs_exact(self):
        """
        Regression: a condition on a 5-minute indicator should retain ~5× the
        entries an exact-equality join would. Here 10 one-minute candidates span
        two 5-min indicator stamps; the as-of join keeps all 10 (post first value),
        whereas exact equality would keep only the 2 aligned minutes.
        """
        scanner = _make_scanner(conditions=["vol5m > 0"])
        indicators = pl.DataFrame(
            {
                "ts_event": pl.Series([_utc(14, 30), _utc(14, 35)], dtype=pl.Datetime("us", "UTC")),
                "vol5m": pl.Series([1.0, 1.0], dtype=pl.Float64),
            }
        )
        entries = pl.DataFrame(
            {
                "entry_time": pl.Series(
                    [_utc(14, 30 + i) for i in range(10)], dtype=pl.Datetime("us", "UTC")
                ),
                "open_mark": pl.Series([1.0] * 10, dtype=pl.Float64),
            }
        )
        out = scanner._evaluate_conditions(entries, indicators)
        # exact-equality join would retain only 14:30 and 14:35 (2 rows)
        exact = entries.join(
            indicators.rename({"ts_event": "entry_time"}), on="entry_time", how="inner"
        )
        assert len(exact) == 2
        assert len(out) == 10
