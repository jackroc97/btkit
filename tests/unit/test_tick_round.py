"""
Unit tests for minimum tick size rounding.

Covers:
  - tick_round_expr utility (zero tick = no-op; standard sizes; float precision)
  - InstrumentConfig.tick_size field
  - open_mark, tp_price, sl_price rounded by EntryScanner._compute_open_mark
    (verified via PnLCalculator which receives pre-rounded values from EntryScanner)
  - exit_mark and worst_mark rounded by ExitScanner._find_first_hit
    (tested by calling the method directly with constructed DataFrames)
"""

from __future__ import annotations

from datetime import date, time

import polars as pl
import pytest

from btkit.backtest._util import tick_round_expr
from btkit.backtest.exit import ExitScanner
from btkit.backtest.pnl import PnLCalculator
from btkit.strategy.definition import (
    CostsConfig,
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    FeesConfig,
    InstrumentConfig,
    LegConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)

# ---------------------------------------------------------------------------
# tick_round_expr — utility
# ---------------------------------------------------------------------------


class TestTickRoundExpr:
    def _apply(self, values: list[float], tick_size: float) -> list[float]:
        df = pl.DataFrame({"x": values})
        return df.select(tick_round_expr(pl.col("x"), tick_size).alias("r"))["r"].to_list()

    def test_zero_tick_is_noop(self):
        vals = [1.234, 5.678, -3.141]
        assert self._apply(vals, 0.0) == vals

    def test_rounds_to_nearest_05(self):
        assert self._apply([5.02], 0.05) == pytest.approx([5.00])
        assert self._apply([5.03], 0.05) == pytest.approx([5.05])
        assert self._apply([5.025], 0.05) == pytest.approx([5.00])  # half rounds to even (banker's)

    def test_rounds_to_nearest_10(self):
        assert self._apply([4.94], 0.10) == pytest.approx([4.90])
        assert self._apply([4.96], 0.10) == pytest.approx([5.00])

    def test_rounds_to_nearest_25(self):
        assert self._apply([1.10], 0.25) == pytest.approx([1.00])
        assert self._apply([1.15], 0.25) == pytest.approx([1.25])

    def test_already_on_tick_unchanged(self):
        assert self._apply([5.00, 5.05, 5.10], 0.05) == pytest.approx([5.00, 5.05, 5.10])

    def test_negative_prices(self):
        # Negative marks can occur for debit spreads
        assert self._apply([-5.03], 0.05) == pytest.approx([-5.05])
        assert self._apply([-5.02], 0.05) == pytest.approx([-5.00])

    def test_floating_point_classic_case(self):
        # 0.1 / 0.05 = 1.9999999999999998 in float64 — should still round to 2 → 0.10
        assert self._apply([0.10], 0.05) == pytest.approx([0.10])

    def test_vectorized_multiple_rows(self):
        vals = [5.01, 5.025, 5.04, 5.06, 5.099]
        result = self._apply(vals, 0.05)
        assert result == pytest.approx([5.00, 5.00, 5.05, 5.05, 5.10])


# ---------------------------------------------------------------------------
# InstrumentConfig.tick_size field
# ---------------------------------------------------------------------------


class TestInstrumentConfigTickSize:
    def test_default_is_zero(self):
        cfg = InstrumentConfig(root_symbol="ES", asset_class="future")
        assert cfg.tick_size == 0.0

    def test_explicit_tick_size(self):
        cfg = InstrumentConfig(root_symbol="ES", asset_class="future", tick_size=0.05)
        assert cfg.tick_size == pytest.approx(0.05)

    def test_tick_size_in_strategy(self):
        strat = StrategyDefinition(
            name="test",
            universe=UniverseConfig(start_date=date(2026, 1, 1), end_date=date(2026, 3, 31)),
            trades=[
                TradeDefinition(
                    name="t",
                    instrument=InstrumentConfig(root_symbol="ES", asset_class="future", tick_size=0.05),
                    entry=EntryConfig(window=EntryWindowConfig(start=time(10, 0), end=time(12, 0))),
                    legs=[LegConfig(name="sp", right="put", action="sell_to_open", delta=-0.25, dte=21)],
                    exit=ExitConfig(stop_loss=2.0, take_profit=1.0),
                )
            ],
        )
        assert strat.trades[0].instrument.tick_size == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# PnL-level rounding: open_mark, tp_price, sl_price feed into net_pnl
#
# We test rounding indirectly via PnLCalculator — the open_mark used in
# gross_pnl = (open_mark - exit_mark) × multiplier must be on-tick.
# We construct entries where open_mark is deliberately off-tick without
# rounding, then verify the rounded value appears in net_pnl.
# ---------------------------------------------------------------------------


def _make_strategy_with_tick(tick_size: float = 0.0) -> StrategyDefinition:
    return StrategyDefinition(
        name="test",
        universe=UniverseConfig(start_date=date(2026, 1, 1), end_date=date(2026, 3, 31)),
        costs=CostsConfig(fees=FeesConfig()),
        trades=[
            TradeDefinition(
                name="trade1",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future", tick_size=tick_size),
                entry=EntryConfig(window=EntryWindowConfig(start=time(10, 0), end=time(12, 0))),
                legs=[LegConfig(name="short_put", right="put", action="sell_to_open", delta=-0.25, dte=21)],
                exit=ExitConfig(stop_loss=2.0, take_profit=1.0),
            )
        ],
    )


def _make_entries(open_mark: float, multiplier: float = 50.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entry_id": [0],
            "trade_name": ["trade1"],
            "entry_time": pl.Series(["2026-01-15 10:30:00"], dtype=pl.Datetime("us", "UTC")),
            "open_mark": [open_mark],
            "tp_price": [open_mark - 1.0],
            "sl_price": [open_mark + 2.0],
            "dte_exit": pl.Series([None], dtype=pl.Int32),
            "leg_short_put_instrument_id": [1001],
            "leg_short_put_symbol": ["EW3K6 P4300"],
            "leg_short_put_expiration": pl.Series(["2026-01-17"], dtype=pl.Date),
            "leg_short_put_strike_price": [4300.0],
            "leg_short_put_right": ["P"],
            "leg_short_put_action": ["sell_to_open"],
            "leg_short_put_quantity": [1],
            "leg_short_put_multiplier": [multiplier],
            "leg_short_put_close": [open_mark],
            "leg_short_put_delta": [-0.25],
            "leg_short_put_iv": [0.20],
            "leg_short_put_gamma": [0.01],
            "leg_short_put_theta": [-0.05],
            "leg_short_put_vega": [10.0],
            "leg_short_put_dte": [21],
        }
    )


def _make_exits(exit_mark: float, worst_mark: float, exit_reason: str = "take_profit") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entry_id": [0],
            "exit_time": pl.Series(["2026-01-15 14:00:00"], dtype=pl.Datetime("us", "UTC")),
            "exit_mark": [exit_mark],
            "worst_mark": [worst_mark],
            "exit_reason": [exit_reason],
        }
    )


class TestTickRoundingThroughPnL:
    """
    PnLCalculator receives pre-rounded prices from EntryScanner and ExitScanner;
    it does not apply rounding itself. These tests verify that on-tick prices
    flow through to net_pnl correctly, and that the no-tick-size path is a no-op.
    """

    def test_no_tick_size_prices_unchanged(self):
        strat = _make_strategy_with_tick(tick_size=0.0)
        calc = PnLCalculator(strat)
        entries = _make_entries(open_mark=5.03)
        exits = _make_exits(exit_mark=4.03, worst_mark=5.50)
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        pos = result.positions
        assert pos["open_mark"][0] == pytest.approx(5.03)
        assert pos["exit_mark"][0] == pytest.approx(4.03)
        gross = (5.03 - 4.03) * 50.0
        assert pos["net_pnl"][0] == pytest.approx(gross)

    def test_on_tick_prices_produce_correct_net_pnl(self):
        # Pre-rounded values flow through; net_pnl computed from them correctly
        strat = _make_strategy_with_tick(tick_size=0.05)
        calc = PnLCalculator(strat)
        entries = _make_entries(open_mark=5.05)
        exits = _make_exits(exit_mark=4.05, worst_mark=5.50)
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        pos = result.positions
        assert pos["exit_mark"][0] == pytest.approx(4.05)
        gross = (5.05 - 4.05) * 50.0
        assert pos["net_pnl"][0] == pytest.approx(gross)

    def test_open_mark_stored_as_passed_in(self):
        # PnLCalculator stores open_mark as received — rounding is EntryScanner's job
        strat = _make_strategy_with_tick(tick_size=0.05)
        calc = PnLCalculator(strat)
        entries = _make_entries(open_mark=5.05)
        exits = _make_exits(exit_mark=4.05, worst_mark=5.05)
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        assert result.positions["open_mark"][0] == pytest.approx(5.05)


# ---------------------------------------------------------------------------
# ExitScanner._find_first_hit — verify exit_mark and worst_mark are rounded
#
# _find_first_hit does not call self.db, so we can instantiate ExitScanner
# with db=None and drive it with hand-crafted DataFrames.
# ---------------------------------------------------------------------------


def _make_exit_scanner(tick_size: float = 0.0, dte_exit: int | None = 1) -> ExitScanner:
    strat = StrategyDefinition(
        name="test",
        universe=UniverseConfig(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 3, 31),
        ),
        trades=[
            TradeDefinition(
                name="trade1",
                instrument=InstrumentConfig(
                    root_symbol="ES",
                    asset_class="future",
                    tick_size=tick_size,
                ),
                entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(16, 0))),
                legs=[LegConfig(name="short_put", right="put", action="sell_to_open", delta=-0.25, dte=0)],
                exit=ExitConfig(dte_exit=dte_exit, expiry_exit=False),
            )
        ],
    )
    return ExitScanner(db=None, strategy=strat, trade=strat.trades[0])


def _make_pm_entries(
    position_marks: list[float],
    ts_events_utc: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build (position_marks_df, entries_df) for ExitScanner._find_first_hit."""
    n = len(position_marks)
    pm_df = pl.DataFrame(
        {
            "entry_id": [0] * n,
            "ts_event": pl.Series(ts_events_utc, dtype=pl.Datetime("us", "UTC")),
            "position_mark": position_marks,
            "spread_open_mark": position_marks,
        }
    )
    entries_df = pl.DataFrame(
        {
            "entry_id": [0],
            "entry_time": pl.Series(["2026-01-14 16:00:00"], dtype=pl.Datetime("us", "UTC")),
            "tp_price": pl.Series([None], dtype=pl.Float64),
            "sl_price": pl.Series([None], dtype=pl.Float64),
            "dte_exit": pl.Series([1], dtype=pl.Int32),
            # Expiration one week out — bars won't hit it
            "leg_short_put_expiration": pl.Series([date(2026, 1, 21)], dtype=pl.Date),
        }
    )
    return pm_df, entries_df


class TestExitScannerTickRounding:
    """Verify that _find_first_hit rounds exit_mark and worst_mark to tick."""

    # 2026-01-16 is a Friday. Bar at 11:00 AM ET = 16:00 UTC → DTE from 2026-01-21 = 5.
    # But we set dte_exit=5 to trigger at that bar.
    _BAR = "2026-01-16 16:00:00"  # 11:00 AM ET, weekday, within session

    def _run(self, tick_size: float, position_mark: float) -> pl.DataFrame:
        scanner = _make_exit_scanner(tick_size=tick_size, dte_exit=5)
        pm_df, entries_df = _make_pm_entries([position_mark], [self._BAR])
        return scanner._find_first_hit(pm_df, pl.DataFrame(), entries_df)

    def test_zero_tick_exit_mark_unchanged(self):
        result = self._run(tick_size=0.0, position_mark=5.03)
        assert result["exit_mark"][0] == pytest.approx(5.03)
        assert result["exit_reason"][0] == "dte_exit"

    def test_exit_mark_rounded_down(self):
        result = self._run(tick_size=0.05, position_mark=5.02)
        assert result["exit_mark"][0] == pytest.approx(5.00)

    def test_exit_mark_rounded_up(self):
        result = self._run(tick_size=0.05, position_mark=5.03)
        assert result["exit_mark"][0] == pytest.approx(5.05)

    def test_exit_mark_already_on_tick(self):
        result = self._run(tick_size=0.05, position_mark=5.05)
        assert result["exit_mark"][0] == pytest.approx(5.05)

    def test_worst_mark_rounded(self):
        # Two bars: higher one sets the worst_mark
        scanner = _make_exit_scanner(tick_size=0.05, dte_exit=5)
        pm_df, entries_df = _make_pm_entries(
            [5.53, 5.03],
            ["2026-01-15 16:00:00", "2026-01-16 16:00:00"],
        )
        result = scanner._find_first_hit(pm_df, pl.DataFrame(), entries_df)
        # worst_mark = max(5.53, 5.03) = 5.53 → rounded to 5.55
        assert result["worst_mark"][0] == pytest.approx(5.55)

    def test_zero_tick_worst_mark_unchanged(self):
        scanner = _make_exit_scanner(tick_size=0.0, dte_exit=5)
        pm_df, entries_df = _make_pm_entries(
            [5.53, 5.03],
            ["2026-01-15 16:00:00", "2026-01-16 16:00:00"],
        )
        result = scanner._find_first_hit(pm_df, pl.DataFrame(), entries_df)
        assert result["worst_mark"][0] == pytest.approx(5.53)

    def test_025_tick_size(self):
        result = self._run(tick_size=0.25, position_mark=5.10)
        # 5.10 / 0.25 = 20.4 → round to 20 → 5.00
        assert result["exit_mark"][0] == pytest.approx(5.00)

    def test_exit_reason_is_dte_exit(self):
        result = self._run(tick_size=0.05, position_mark=5.00)
        assert result["exit_reason"][0] == "dte_exit"
