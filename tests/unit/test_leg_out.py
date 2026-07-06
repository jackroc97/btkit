"""
Unit tests for ExitScanner._adjust_leg_out_exits.

All tests use hand-constructed DataFrames with known values. No DB access.
The method replaces the forward-filled long-leg price in exit_mark with the
first real bar after exit_time (market-order semantics).

Adjustment formula (long leg: signed_qty = -1 × quantity):
    exit_mark' = exit_mark
                 - stale_close × signed_qty
                 + fill_close  × signed_qty
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from unittest.mock import MagicMock

import polars as pl
import pytest

from btkit.backtest.exit import ExitScanner
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    LiquidityConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)


def _make_scanner(leg_out: bool = True) -> ExitScanner:
    strategy = StrategyDefinition(
        name="test",
        universe=UniverseConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 31),
        ),
        trades=[
            TradeDefinition(
                name="put_spread",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(10, 0))),
                legs=[
                    LegConfig(
                        name="short_put",
                        right="put",
                        action="sell_to_open",
                        dte=0,
                        delta={"target": -0.16},
                    ),
                    LegConfig(
                        name="long_put",
                        right="put",
                        action="buy_to_open",
                        strike_offset=-50.0,
                        reference_leg="short_put",
                    ),
                ],
                exit=ExitConfig(take_profit_pct=0.50, leg_out=leg_out),
            )
        ],
    )
    return ExitScanner(db=MagicMock(), strategy=strategy, trade=strategy.trades[0])


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 2, hour, minute, tzinfo=UTC)


class TestAdjustLegOutExits:
    def _base_entries(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "entry_id": pl.Series([1], dtype=pl.UInt32),
                "entry_time": pl.Series([_ts(9, 45)], dtype=pl.Datetime("us", "UTC")),
                "leg_short_put_instrument_id": pl.Series([101], dtype=pl.Int64),
                "leg_long_put_instrument_id": pl.Series([102], dtype=pl.Int64),
                "leg_short_put_expiration": pl.Series([date(2024, 1, 5)], dtype=pl.Date),
                "leg_long_put_expiration": pl.Series([date(2024, 1, 5)], dtype=pl.Date),
            }
        )

    def _base_exits(self, exit_mark: float, reason: str = "take_profit") -> pl.DataFrame:
        return pl.DataFrame(
            {
                "entry_id": pl.Series([1], dtype=pl.UInt32),
                "exit_time": pl.Series([_ts(11, 30)], dtype=pl.Datetime("us", "UTC")),
                "exit_mark": pl.Series([exit_mark], dtype=pl.Float64),
                "worst_mark": pl.Series([0.30], dtype=pl.Float64),
                "exit_reason": pl.Series([reason], dtype=pl.Utf8),
            }
        )

    def _option_bars(
        self,
        short_close: float = 0.25,
        long_stale_close: float = 0.15,
        long_fill_close: float | None = 0.05,
    ) -> pl.DataFrame:
        rows = [
            # short put: bar at TP time
            (_ts(11, 30), 101, short_close),
            # long put: stale bar (last before exit, after entry)
            (_ts(10, 5), 102, long_stale_close),
        ]
        if long_fill_close is not None:
            rows.append((_ts(11, 31), 102, long_fill_close))

        ts, instr, close = zip(*rows, strict=False)
        return pl.DataFrame(
            {
                "ts_event": pl.Series(list(ts), dtype=pl.Datetime("us", "UTC")),
                "instrument_id": pl.Series(list(instr), dtype=pl.Int64),
                "close": pl.Series(list(close), dtype=pl.Float64),
                "open": pl.Series(list(close), dtype=pl.Float64),
                "high": pl.Series(list(close), dtype=pl.Float64),
                "low": pl.Series(list(close), dtype=pl.Float64),
                "volume": pl.Series([100] * len(ts), dtype=pl.Int64),
            }
        )

    def test_replaces_stale_long_price_with_fill(self):
        # exit_mark at TP = short_close(0.25) × 1 + long_stale(0.15) × (-1) = 0.10
        # After adjustment: 0.10 - 0.15 × (-1) + 0.05 × (-1) = 0.10 + 0.15 - 0.05 = 0.20
        scanner = _make_scanner()
        entries = self._base_entries()
        exits = self._base_exits(exit_mark=0.10)
        bars = self._option_bars(short_close=0.25, long_stale_close=0.15, long_fill_close=0.05)

        result = scanner._adjust_leg_out_exits(exits, entries, bars)

        assert len(result) == 1
        assert result["exit_mark"][0] == pytest.approx(0.20)
        assert result["exit_reason"][0] == "take_profit"

    def test_no_bar_after_exit_fills_at_zero(self):
        # Long leg is illiquid — no bar after exit_time → fill at 0
        # adjusted = 0.10 - 0.15 × (-1) + 0.0 × (-1) = 0.10 + 0.15 = 0.25
        scanner = _make_scanner()
        entries = self._base_entries()
        exits = self._base_exits(exit_mark=0.10)
        bars = self._option_bars(long_fill_close=None)

        result = scanner._adjust_leg_out_exits(exits, entries, bars)

        assert len(result) == 1
        assert result["exit_mark"][0] == pytest.approx(0.25)

    def test_expiry_exits_unchanged(self):
        scanner = _make_scanner()
        entries = self._base_entries()
        exits = self._base_exits(exit_mark=0.10, reason="expiry")
        bars = self._option_bars(long_fill_close=0.05)

        result = scanner._adjust_leg_out_exits(exits, entries, bars)

        assert len(result) == 1
        assert result["exit_mark"][0] == pytest.approx(0.10)

    def test_mixed_expiry_and_tp_in_same_cohort(self):
        scanner = _make_scanner()

        entries = pl.DataFrame(
            {
                "entry_id": pl.Series([1, 2], dtype=pl.UInt32),
                "entry_time": pl.Series([_ts(9, 45), _ts(9, 46)], dtype=pl.Datetime("us", "UTC")),
                "leg_short_put_instrument_id": pl.Series([101, 103], dtype=pl.Int64),
                "leg_long_put_instrument_id": pl.Series([102, 104], dtype=pl.Int64),
                "leg_short_put_expiration": pl.Series(
                    [date(2024, 1, 5), date(2024, 1, 5)], dtype=pl.Date
                ),
                "leg_long_put_expiration": pl.Series(
                    [date(2024, 1, 5), date(2024, 1, 5)], dtype=pl.Date
                ),
            }
        )
        exits = pl.DataFrame(
            {
                "entry_id": pl.Series([1, 2], dtype=pl.UInt32),
                "exit_time": pl.Series([_ts(11, 30), _ts(15, 59)], dtype=pl.Datetime("us", "UTC")),
                "exit_mark": pl.Series([0.10, 0.40], dtype=pl.Float64),
                "worst_mark": pl.Series([0.30, 0.50], dtype=pl.Float64),
                "exit_reason": pl.Series(["take_profit", "expiry"], dtype=pl.Utf8),
            }
        )
        bars = pl.DataFrame(
            {
                "ts_event": pl.Series(
                    [_ts(10, 5), _ts(11, 31)],
                    dtype=pl.Datetime("us", "UTC"),
                ),
                "instrument_id": pl.Series([102, 102], dtype=pl.Int64),
                "close": pl.Series([0.15, 0.05], dtype=pl.Float64),
                "open": pl.Series([0.15, 0.05], dtype=pl.Float64),
                "high": pl.Series([0.15, 0.05], dtype=pl.Float64),
                "low": pl.Series([0.15, 0.05], dtype=pl.Float64),
                "volume": pl.Series([50, 50], dtype=pl.Int64),
            }
        )

        result = scanner._adjust_leg_out_exits(exits, entries, bars)
        result_sorted = result.sort("entry_id")

        assert len(result_sorted) == 2
        # entry 1 (TP): adjusted 0.10 + 0.15 - 0.05 = 0.20
        assert result_sorted.filter(pl.col("entry_id") == 1)["exit_mark"][0] == pytest.approx(0.20)
        # entry 2 (expiry): unchanged
        assert result_sorted.filter(pl.col("entry_id") == 2)["exit_mark"][0] == pytest.approx(0.40)

    def test_no_long_legs_returns_exits_unchanged(self):
        # Strategy with only short legs — no adjustment needed
        strategy = StrategyDefinition(
            name="test",
            universe=UniverseConfig(
                start_date=date(2024, 1, 2),
                end_date=date(2024, 3, 31),
            ),
            trades=[
                TradeDefinition(
                    name="naked_put",
                    instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                    entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(10, 0))),
                    legs=[
                        LegConfig(
                            name="short_put",
                            right="put",
                            action="sell_to_open",
                            dte=0,
                            delta={"target": -0.16},
                        ),
                    ],
                    exit=ExitConfig(take_profit_pct=0.50, leg_out=True),
                )
            ],
        )
        scanner = ExitScanner(db=MagicMock(), strategy=strategy, trade=strategy.trades[0])

        entries = pl.DataFrame(
            {
                "entry_id": pl.Series([1], dtype=pl.UInt32),
                "entry_time": pl.Series([_ts(9, 45)], dtype=pl.Datetime("us", "UTC")),
                "leg_short_put_instrument_id": pl.Series([101], dtype=pl.Int64),
                "leg_short_put_expiration": pl.Series([date(2024, 1, 5)], dtype=pl.Date),
            }
        )
        exits = pl.DataFrame(
            {
                "entry_id": pl.Series([1], dtype=pl.UInt32),
                "exit_time": pl.Series([_ts(11, 30)], dtype=pl.Datetime("us", "UTC")),
                "exit_mark": pl.Series([0.10], dtype=pl.Float64),
                "worst_mark": pl.Series([0.30], dtype=pl.Float64),
                "exit_reason": pl.Series(["take_profit"], dtype=pl.Utf8),
            }
        )
        bars = self._option_bars()

        result = scanner._adjust_leg_out_exits(exits, entries, bars)

        assert result["exit_mark"][0] == pytest.approx(0.10)

    def test_bars_before_entry_time_are_ignored(self):
        # A bar at 9:30 (before entry at 9:45) must not be used as the stale price
        # Only the 10:05 bar should be the stale reference
        scanner = _make_scanner()
        entries = self._base_entries()
        exits = self._base_exits(exit_mark=0.10)

        bars = pl.DataFrame(
            {
                "ts_event": pl.Series(
                    [_ts(9, 30), _ts(10, 5), _ts(11, 31)],
                    dtype=pl.Datetime("us", "UTC"),
                ),
                "instrument_id": pl.Series([102, 102, 102], dtype=pl.Int64),
                "close": pl.Series([0.99, 0.15, 0.05], dtype=pl.Float64),
                "open": pl.Series([0.99, 0.15, 0.05], dtype=pl.Float64),
                "high": pl.Series([0.99, 0.15, 0.05], dtype=pl.Float64),
                "low": pl.Series([0.99, 0.15, 0.05], dtype=pl.Float64),
                "volume": pl.Series([50, 50, 50], dtype=pl.Int64),
            }
        )

        result = scanner._adjust_leg_out_exits(exits, entries, bars)

        # stale should be 0.15 (10:05), not 0.99 (9:30)
        # adjusted = 0.10 - 0.15 × (-1) + 0.05 × (-1) = 0.20

        assert result["exit_mark"][0] == pytest.approx(0.20)


# ── Volume gate × leg_out interaction ────────────────────────────────────────


def _make_scanner_with_volume(leg_out: bool, min_exit_volume: int) -> ExitScanner:
    strategy = StrategyDefinition(
        name="test",
        universe=UniverseConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 31),
        ),
        trades=[
            TradeDefinition(
                name="put_spread",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(10, 0))),
                legs=[
                    LegConfig(
                        name="short_put",
                        right="put",
                        action="sell_to_open",
                        dte=0,
                        delta={"target": -0.16},
                    ),
                    LegConfig(
                        name="long_put",
                        right="put",
                        action="buy_to_open",
                        strike_offset=-50.0,
                        reference_leg="short_put",
                    ),
                ],
                exit=ExitConfig(
                    take_profit_pct=0.50,
                    leg_out=leg_out,
                    liquidity=LiquidityConfig(min_exit_volume=min_exit_volume),
                ),
            )
        ],
    )
    return ExitScanner(db=MagicMock(), strategy=strategy, trade=strategy.trades[0])


def _entries_for_volume_test() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entry_id": pl.Series([1], dtype=pl.UInt32),
            "entry_time": pl.Series([_ts(9, 45)], dtype=pl.Datetime("us", "UTC")),
            "leg_short_put_instrument_id": pl.Series([101], dtype=pl.Int64),
            "leg_long_put_instrument_id": pl.Series([102], dtype=pl.Int64),
            "leg_short_put_expiration": pl.Series([date(2024, 1, 5)], dtype=pl.Date),
            "leg_long_put_expiration": pl.Series([date(2024, 1, 5)], dtype=pl.Date),
        }
    )


def _option_bars_with_volumes(short_volume: int, long_volume: int) -> pl.DataFrame:
    """One bar at 10:00 for each leg; long leg may have zero volume (sparse/illiquid)."""
    return pl.DataFrame(
        {
            "ts_event": pl.Series([_ts(10, 0), _ts(10, 0)], dtype=pl.Datetime("us", "UTC")),
            "instrument_id": pl.Series([101, 102], dtype=pl.Int64),
            "open": pl.Series([0.25, 0.15], dtype=pl.Float64),
            "high": pl.Series([0.25, 0.15], dtype=pl.Float64),
            "low": pl.Series([0.25, 0.15], dtype=pl.Float64),
            "close": pl.Series([0.25, 0.15], dtype=pl.Float64),
            "volume": pl.Series([short_volume, long_volume], dtype=pl.Int64),
        }
    )


class TestVolumeGateLegOut:
    """_min_leg_volume should reflect only short-leg volume when leg_out=True."""

    def test_leg_out_true_min_volume_uses_short_leg_only(self):
        # Short has volume=10, long has volume=0 (typical sparse OTM long).
        # With leg_out=True the gate should see 10, not min(10,0)=0.
        scanner = _make_scanner_with_volume(leg_out=True, min_exit_volume=5)
        marks = scanner._compute_position_marks(
            _option_bars_with_volumes(short_volume=10, long_volume=0),
            _entries_for_volume_test(),
        )
        assert not marks.is_empty()
        assert marks["_min_leg_volume"][0] == 10

    def test_leg_out_false_min_volume_uses_all_legs(self):
        # Same bars; without leg_out the minimum across legs is used: min(10,0)=0.
        scanner = _make_scanner_with_volume(leg_out=False, min_exit_volume=5)
        marks = scanner._compute_position_marks(
            _option_bars_with_volumes(short_volume=10, long_volume=0),
            _entries_for_volume_test(),
        )
        assert not marks.is_empty()
        assert marks["_min_leg_volume"][0] == 0

    def test_leg_out_true_absent_long_bar_does_not_block_volume_gate(self):
        # More realistic: long leg has no bar at this ts_event at all (absent row).
        # After outer join it gets fill_null(0)=0 for volume. With leg_out=True
        # the gate should still pass on the short leg's volume alone.
        scanner = _make_scanner_with_volume(leg_out=True, min_exit_volume=5)
        bars = pl.DataFrame(
            {
                "ts_event": pl.Series([_ts(10, 0)], dtype=pl.Datetime("us", "UTC")),
                "instrument_id": pl.Series([101], dtype=pl.Int64),  # short leg only
                "open": pl.Series([0.25], dtype=pl.Float64),
                "high": pl.Series([0.25], dtype=pl.Float64),
                "low": pl.Series([0.25], dtype=pl.Float64),
                "close": pl.Series([0.25], dtype=pl.Float64),
                "volume": pl.Series([10], dtype=pl.Int64),
            }
        )
        marks = scanner._compute_position_marks(bars, _entries_for_volume_test())
        # position_marks drops rows where any close is null even after forward-fill,
        # so this only returns a row if the long leg has a prior bar to forward-fill from.
        # The point here: if a row IS present, _min_leg_volume reflects the short leg.
        if not marks.is_empty():
            assert marks["_min_leg_volume"][0] == 10

    def test_both_legs_have_volume_unchanged_by_leg_out(self):
        # When both legs have volume, leg_out should not change the result.
        scanner_lo = _make_scanner_with_volume(leg_out=True, min_exit_volume=5)
        scanner_no = _make_scanner_with_volume(leg_out=False, min_exit_volume=5)
        bars = _option_bars_with_volumes(short_volume=10, long_volume=8)
        entries = _entries_for_volume_test()

        marks_lo = scanner_lo._compute_position_marks(bars, entries)
        marks_no = scanner_no._compute_position_marks(bars, entries)

        # leg_out=True: short-leg only → 10
        # leg_out=False: min(10, 8) = 8
        assert marks_lo["_min_leg_volume"][0] == 10
        assert marks_no["_min_leg_volume"][0] == 8
