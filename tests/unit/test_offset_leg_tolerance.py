"""
Unit tests for reference_leg.entry_time_tolerance on strike_offset legs —
the Pass 2 path in EntryScanner._select_legs, backed by
InputDatabase.greeks_for_strike_legs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

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


def _make_offset_trade(
    *,
    bar_time_tolerance: int = 0,
    entry_time_tolerance: int | None,
) -> tuple[StrategyDefinition, TradeDefinition]:
    strat = StrategyDefinition(
        name="offset_tol_test",
        universe=UniverseConfig(start_date=date(2022, 12, 14), end_date=date(2022, 12, 14)),
        trades=[
            TradeDefinition(
                name="spread",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(
                    window=EntryWindowConfig(start=time(9, 30), end=time(11, 0)),
                    bar_time_tolerance=bar_time_tolerance,
                ),
                legs=[
                    LegConfig(
                        name="short_put",
                        right="put",
                        action="sell_to_open",
                        dte=21,
                        delta={"target": -0.10, "tolerance": 0.05},
                    ),
                    LegConfig(
                        name="long_put",
                        right="put",
                        action="buy_to_open",
                        reference_leg={
                            "name": "short_put",
                            "strike_offset": -50.0,
                            "entry_time_tolerance": entry_time_tolerance,
                        },
                    ),
                ],
                exit=ExitConfig(take_profit=1.0),
            )
        ],
    )
    return strat, strat.trades[0]


def _candidates_df(ts_event: datetime, underlying_id: int) -> pl.DataFrame:
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


def _short_put_greeks(ts_event: datetime, underlying_id: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "leg_name": pl.Series(["short_put"], dtype=pl.Utf8),
            "ts_event": pl.Series([ts_event], dtype=pl.Datetime("us", "UTC")),
            "instrument_id": pl.Series([111], dtype=pl.Int64),
            "underlying_id": pl.Series([underlying_id], dtype=pl.Int64),
            "dte": pl.Series([21], dtype=pl.Int32),
            "iv": pl.Series([0.25], dtype=pl.Float64),
            "delta": pl.Series([-0.10], dtype=pl.Float64),
            "gamma": pl.Series([0.001], dtype=pl.Float64),
            "theta": pl.Series([-0.10], dtype=pl.Float64),
            "vega": pl.Series([0.20], dtype=pl.Float64),
            "strike_price": pl.Series([5000.0], dtype=pl.Float64),
            "expiration": pl.Series([date(2023, 1, 4)], dtype=pl.Date),
            "right": pl.Series(["P"], dtype=pl.Utf8),
            "multiplier": pl.Series([50.0], dtype=pl.Float64),
            "symbol": pl.Series(["ES230104P05000"], dtype=pl.Utf8),
            "close": pl.Series([5.00], dtype=pl.Float64),
        }
    )


def _offset_candidate_row(
    row_ts_event: datetime,
    match_ts_event: datetime,
    strike_price: float,
) -> dict:
    return {
        "leg_name": "long_put",
        "ts_event": row_ts_event,
        "_match_ts_event": match_ts_event,
        "instrument_id": 222,
        "underlying_id": 999,
        "dte": 21,
        "iv": 0.20,
        "delta": -0.05,
        "gamma": 0.001,
        "theta": -0.05,
        "vega": 0.15,
        "strike_price": strike_price,
        "expiration": date(2023, 1, 4),
        "right": "P",
        "multiplier": 50.0,
        "symbol": "ES230104P04950",
        "close": 2.00,
    }


def _offset_candidates_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "leg_name": pl.Series([r["leg_name"] for r in rows], dtype=pl.Utf8),
            "ts_event": pl.Series([r["ts_event"] for r in rows], dtype=pl.Datetime("us", "UTC")),
            "_match_ts_event": pl.Series(
                [r["_match_ts_event"] for r in rows], dtype=pl.Datetime("us", "UTC")
            ),
            "instrument_id": pl.Series([r["instrument_id"] for r in rows], dtype=pl.Int64),
            "underlying_id": pl.Series([r["underlying_id"] for r in rows], dtype=pl.Int64),
            "dte": pl.Series([r["dte"] for r in rows], dtype=pl.Int32),
            "iv": pl.Series([r["iv"] for r in rows], dtype=pl.Float64),
            "delta": pl.Series([r["delta"] for r in rows], dtype=pl.Float64),
            "gamma": pl.Series([r["gamma"] for r in rows], dtype=pl.Float64),
            "theta": pl.Series([r["theta"] for r in rows], dtype=pl.Float64),
            "vega": pl.Series([r["vega"] for r in rows], dtype=pl.Float64),
            "strike_price": pl.Series([r["strike_price"] for r in rows], dtype=pl.Float64),
            "expiration": pl.Series([r["expiration"] for r in rows], dtype=pl.Date),
            "right": pl.Series([r["right"] for r in rows], dtype=pl.Utf8),
            "multiplier": pl.Series([r["multiplier"] for r in rows], dtype=pl.Float64),
            "symbol": pl.Series([r["symbol"] for r in rows], dtype=pl.Utf8),
            "close": pl.Series([r["close"] for r in rows], dtype=pl.Float64),
        }
    )


class _CapturingDB:
    """Records greeks_for_strike_legs calls; returns pre-built offset candidates."""

    def __init__(self, short_put_greeks: pl.DataFrame, offset_candidates: pl.DataFrame) -> None:
        self._short_put_greeks = short_put_greeks
        self._offset_candidates = offset_candidates
        self.calls: list[dict] = []

    def greeks_for_all_legs(self, ts_event_underlying, leg_specs, **kwargs) -> pl.DataFrame:
        return self._short_put_greeks

    def greeks_for_strike_legs(self, strike_targets, **kwargs) -> pl.DataFrame:
        self.calls.append({"strike_targets": strike_targets, **kwargs})
        return self._offset_candidates

    def front_future_schedule(self, *a, **k) -> pl.DataFrame:
        return pl.DataFrame()

    def underlying_bars_for_root(self, *a, **k) -> pl.DataFrame:
        return pl.DataFrame()

    def indicators(self, *a, **k) -> pl.DataFrame:
        return pl.DataFrame()


class TestOffsetLegTimeTolerance:
    def test_entry_time_tolerance_unset_passes_bar_time_tolerance(self):
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        strat, trade = _make_offset_trade(bar_time_tolerance=45, entry_time_tolerance=None)
        offset_rows = [_offset_candidate_row(bar_ts, bar_ts, 4950.0)]
        db = _CapturingDB(_short_put_greeks(bar_ts, 999), _offset_candidates_df(offset_rows))
        EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        assert db.calls, "expected greeks_for_strike_legs to be called"
        assert db.calls[0]["time_tolerance"] == 45

    def test_entry_time_tolerance_set_overrides_bar_time_tolerance(self):
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        strat, trade = _make_offset_trade(bar_time_tolerance=5, entry_time_tolerance=120)
        offset_rows = [_offset_candidate_row(bar_ts, bar_ts, 4950.0)]
        db = _CapturingDB(_short_put_greeks(bar_ts, 999), _offset_candidates_df(offset_rows))
        EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        assert db.calls[0]["time_tolerance"] == 120

    def test_strike_targets_carries_row_identity_and_anchor(self):
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        root_ts = bar_ts + timedelta(seconds=6)
        strat, trade = _make_offset_trade(bar_time_tolerance=10, entry_time_tolerance=60)
        offset_rows = [_offset_candidate_row(bar_ts, root_ts, 4950.0)]
        db = _CapturingDB(_short_put_greeks(root_ts, 999), _offset_candidates_df(offset_rows))
        EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        st = db.calls[0]["strike_targets"]
        assert st["ts_event"][0] == bar_ts, "row identity must stay the bar's own ts_event"
        assert st["anchor_ts_event"][0] == root_ts, (
            "anchor must be short_put's resolved ts (root_ts), not the bar"
        )

    def test_nearest_in_time_wins_over_nearest_strike(self):
        """Two candidate bars for the same anchor row, both within
        strike_tolerance (a hard filter enforced by the real DB — a candidate
        outside it would never reach this reduction at all, regardless of
        time): one slightly off-strike but nearest in time, one exact-strike
        but farther in time. The nearest-in-time one must win (matches
        tier-1's documented priority)."""
        bar_ts = datetime(2022, 12, 14, 14, 30, tzinfo=UTC)
        strat, trade = _make_offset_trade(bar_time_tolerance=0, entry_time_tolerance=60)
        # target_strike = short_put strike (5000) - 50 = 4950
        near_time_off_strike = _offset_candidate_row(
            bar_ts, bar_ts + timedelta(seconds=5), 4950.8
        )
        far_time_exact_strike = _offset_candidate_row(
            bar_ts, bar_ts + timedelta(seconds=50), 4950.0
        )
        offset_rows = [near_time_off_strike, far_time_exact_strike]
        db = _CapturingDB(_short_put_greeks(bar_ts, 999), _offset_candidates_df(offset_rows))
        result = EntryScanner(db, strat, trade)._select_legs(_candidates_df(bar_ts, 999))
        assert not result.is_empty()
        assert result["leg_long_put_strike_price"][0] == 4950.8
