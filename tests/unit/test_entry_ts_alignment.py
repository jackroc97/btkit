"""
Unit tests for entry.time_tolerance and the ASOF-join option matching in
EntryScanner._select_legs.

time_tolerance (seconds) controls how far an option_greeks timestamp may
differ from the candidate bar timestamp:
  - 0  (default) → exact match required; legacy behaviour
  - >0            → nearest greeks snapshot within that many seconds is used

Tests inject a minimal mock InputDatabase whose greeks_for_all_legs returns
option data at a controlled timestamp offset from the candidate bars.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import polars as pl

from btkit.backtest.entry import EntryScanner
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    delta: float = -0.05,
    delta_tolerance: float = 0.03,
    dte: int = 45,
    dte_tolerance: int = 10,
    time_tolerance: int = 0,
) -> tuple[StrategyDefinition, TradeDefinition]:
    strat = StrategyDefinition(
        name="ts_align_test",
        universe=UniverseConfig(
            start_date=date(2022, 12, 14),
            end_date=date(2022, 12, 14),
        ),
        trades=[
            TradeDefinition(
                name="long_put",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(
                    window=EntryWindowConfig(start=time(9, 30), end=time(11, 0)),
                    conditions=[],
                    time_tolerance=time_tolerance,
                ),
                legs=[
                    LegConfig(
                        name="put",
                        right="put",
                        action="buy_to_open",
                        dte=dte,
                        delta={"target": delta, "tolerance": delta_tolerance},
                        dte_tolerance=dte_tolerance,
                    )
                ],
                exit=ExitConfig(take_profit=1.0),
            )
        ],
    )
    return strat, strat.trades[0]


def _greeks_df(ts_event: datetime, underlying_id: int, delta: float = -0.05) -> pl.DataFrame:
    """Minimal one-row greeks DataFrame as returned by greeks_for_all_legs."""
    return pl.DataFrame(
        {
            "leg_name": pl.Series(["put"], dtype=pl.Utf8),
            "ts_event": pl.Series([ts_event], dtype=pl.Datetime("us", "UTC")),
            "instrument_id": pl.Series([12345], dtype=pl.Int64),
            "underlying_id": pl.Series([underlying_id], dtype=pl.Int64),
            "dte": pl.Series([45], dtype=pl.Int32),
            "iv": pl.Series([0.25], dtype=pl.Float64),
            "delta": pl.Series([delta], dtype=pl.Float64),
            "gamma": pl.Series([0.001], dtype=pl.Float64),
            "theta": pl.Series([-0.10], dtype=pl.Float64),
            "vega": pl.Series([0.20], dtype=pl.Float64),
            "strike_price": pl.Series([5000.0], dtype=pl.Float64),
            "expiration": pl.Series([date(2023, 3, 17)], dtype=pl.Date),
            "right": pl.Series(["P"], dtype=pl.Utf8),
            "multiplier": pl.Series([50.0], dtype=pl.Float64),
            "symbol": pl.Series(["ES230317P05000"], dtype=pl.Utf8),
            "close": pl.Series([10.00], dtype=pl.Float64),
        }
    )


def _candidates_df(ts_event: datetime, underlying_id: int) -> pl.DataFrame:
    """Minimal candidate bar row that _select_legs expects."""
    return pl.DataFrame(
        {
            "ts_event": pl.Series([ts_event], dtype=pl.Datetime("us", "UTC")),
            "underlying_id": pl.Series([underlying_id], dtype=pl.Int64),
            "open": pl.Series([5100.0], dtype=pl.Float64),
            "high": pl.Series([5110.0], dtype=pl.Float64),
            "low": pl.Series([5090.0], dtype=pl.Float64),
            "close": pl.Series([5105.0], dtype=pl.Float64),
            "volume": pl.Series([1000], dtype=pl.Int64),
            "expiry": pl.Series([date(2022, 12, 16)], dtype=pl.Date),
            "next_underlying_id": pl.Series([None], dtype=pl.Int64),
        }
    )


class _MockDB:
    """Minimal stub that returns pre-built greeks DataFrames."""

    def __init__(self, greeks: pl.DataFrame) -> None:
        self._greeks = greeks

    def greeks_for_all_legs(self, ts_event_underlying, leg_specs, **kwargs) -> pl.DataFrame:
        return self._greeks

    def greeks_for_strike_legs(self, strike_targets, **kwargs) -> pl.DataFrame:
        return pl.DataFrame()

    def front_future_schedule(self, *args, **kwargs) -> pl.DataFrame:
        return pl.DataFrame()

    def underlying_bars_for_root(self, *args, **kwargs) -> pl.DataFrame:
        return pl.DataFrame()

    def indicators(self, *args, **kwargs) -> pl.DataFrame:
        return pl.DataFrame()


# ---------------------------------------------------------------------------
# Exact match (time_tolerance=0, the default)
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_aligned_timestamps_produce_entry(self):
        """Exact match: same ts_event in bars and greeks → entry found."""
        ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        strat, trade = _make_trade(time_tolerance=0)
        db = _MockDB(greeks=_greeks_df(ts_event=ts, underlying_id=999))
        result = EntryScanner(db, strat, trade)._select_legs(_candidates_df(ts, 999))
        assert not result.is_empty()

    def test_offset_timestamps_produce_no_entry(self):
        """Exact match: 5-min greeks offset → no entry (timestamps don't align)."""
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        greeks_ts = datetime(2022, 12, 14, 14, 35, tzinfo=UTC)
        strat, trade = _make_trade(time_tolerance=0)
        db = _MockDB(greeks=_greeks_df(ts_event=greeks_ts, underlying_id=999))
        result = EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        assert result.is_empty(), "time_tolerance=0 must require exact timestamp match"


# ---------------------------------------------------------------------------
# Positive tolerance
# ---------------------------------------------------------------------------


class TestPositiveTolerance:
    def test_offset_within_tolerance_produces_entry(self):
        """5-min offset is within a 10-min tolerance → entry found."""
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        greeks_ts = datetime(2022, 12, 14, 14, 35, tzinfo=UTC)
        strat, trade = _make_trade(time_tolerance=600)  # 10 min
        db = _MockDB(greeks=_greeks_df(ts_event=greeks_ts, underlying_id=999))
        result = EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        assert not result.is_empty()

    def test_offset_exceeding_tolerance_produces_no_entry(self):
        """15-min offset exceeds a 10-min tolerance → no entry."""
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        greeks_ts = datetime(2022, 12, 14, 14, 45, tzinfo=UTC)
        strat, trade = _make_trade(time_tolerance=600)  # 10 min
        db = _MockDB(greeks=_greeks_df(ts_event=greeks_ts, underlying_id=999))
        result = EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        assert result.is_empty(), "15-min offset must exceed 10-min tolerance"

    def test_entry_time_is_bar_timestamp_not_greeks_timestamp(self):
        """Recorded entry time is the BAR's ts_event, not the greeks snapshot time."""
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        greeks_ts = datetime(2022, 12, 14, 14, 35, tzinfo=UTC)
        strat, trade = _make_trade(time_tolerance=600)
        db = _MockDB(greeks=_greeks_df(ts_event=greeks_ts, underlying_id=999))
        result = EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        assert not result.is_empty()
        assert result["ts_event"][0] == bar_ts

    def test_nearest_snapshot_selected_over_farther(self):
        """Two greeks snapshots within tolerance: the nearer one is selected."""
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        near_ts = datetime(2022, 12, 14, 14, 31, tzinfo=UTC)  # 1 min away
        far_ts = datetime(2022, 12, 14, 14, 36, tzinfo=UTC)  # 6 min away

        near_greeks = _greeks_df(ts_event=near_ts, underlying_id=999, delta=-0.075)
        far_greeks = _greeks_df(ts_event=far_ts, underlying_id=999, delta=-0.11)
        both = pl.concat([near_greeks, far_greeks])

        strat, trade = _make_trade(time_tolerance=600, delta=-0.10, delta_tolerance=0.03)
        db = _MockDB(greeks=both)
        result = EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        assert not result.is_empty()
        # Nearest snapshot (9:31) wins; its delta is -0.075
        assert abs(result["leg_put_delta"][0] - (-0.075)) < 1e-6

    def test_different_day_greeks_always_rejected(self):
        """Greeks from a different date are rejected regardless of tolerance."""
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        greeks_ts = datetime(2022, 12, 15, 14, 30, tzinfo=UTC)
        strat, trade = _make_trade(time_tolerance=86400)  # 24 h — can't cross dates
        db = _MockDB(greeks=_greeks_df(ts_event=greeks_ts, underlying_id=999))
        result = EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        assert result.is_empty(), "by-date grouping must prevent cross-date matches"

    def test_multiple_bars_all_matched_within_tolerance(self):
        """Multiple candidate bars on the same day all match the nearest greeks."""
        bar_ts_1 = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        bar_ts_2 = datetime(2022, 12, 14, 14, 31, tzinfo=UTC)
        greeks_ts = datetime(2022, 12, 14, 14, 35, tzinfo=UTC)
        candidates = pl.concat(
            [
                _candidates_df(bar_ts_1, 999),
                _candidates_df(bar_ts_2, 999),
            ]
        )
        strat, trade = _make_trade(time_tolerance=600)
        db = _MockDB(greeks=_greeks_df(ts_event=greeks_ts, underlying_id=999))
        result = EntryScanner(db, strat, trade)._select_legs(candidates)
        assert len(result) == 2
        assert set(result["ts_event"].to_list()) == {bar_ts_1, bar_ts_2}
