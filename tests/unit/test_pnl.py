"""
Unit tests for PnLCalculator.

PnLCalculator is pure DataFrame arithmetic — no DB access — so all tests use
hand-constructed DataFrames with known values. Each case verifies the exact
dollar amounts produced by the cost model:

    gross_pnl    = (open_mark - exit_mark) × multiplier
    slippage     = |exit_mark| × multiplier × slippage_pct
    fee          = fee_per_contract  (flat, per round-trip)
    net_pnl      = gross_pnl - slippage - fee
"""

from __future__ import annotations

from datetime import date, time

import polars as pl
import pytest

from btkit.backtest.pnl import PnLCalculator
from btkit.strategy.definition import (
    CostsConfig,
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


def _make_strategy(
    slippage_pct: float = 0.0,
    fee_per_contract: float = 0.0,
    n_legs: int = 1,
) -> StrategyDefinition:
    legs = [
        LegConfig(
            name="short_put",
            right="put",
            action="sell_to_open",
            dte=21,
            delta=-0.25,
        )
    ]
    if n_legs == 2:
        legs.append(
            LegConfig(
                name="long_put",
                right="put",
                action="buy_to_open",
                dte=21,
                strike_offset=-25.0,
                reference_leg="short_put",
            )
        )
    return StrategyDefinition(
        name="test",
        universe=UniverseConfig(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 3, 31),
        ),
        costs=CostsConfig(slippage_pct=slippage_pct, fee_per_contract=fee_per_contract),
        trades=[
            TradeDefinition(
                name="trade1",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(window=EntryWindowConfig(start=time(10, 0), end=time(12, 0))),
                legs=legs,
                exit=ExitConfig(stop_loss=2.0, take_profit=1.0),
            )
        ],
    )


def _make_entries(
    entry_id: int = 0,
    open_mark: float = 5.0,
    multiplier: float = 50.0,
    trade_name: str = "trade1",
    leg_close: float = 5.0,
    leg_name: str = "short_put",
) -> pl.DataFrame:
    """Minimal entries DataFrame with the columns PnLCalculator needs."""
    return pl.DataFrame(
        {
            "entry_id": [entry_id],
            "trade_name": [trade_name],
            "entry_time": pl.Series(["2026-01-15 10:30:00"], dtype=pl.Datetime("us", "UTC")),
            "open_mark": [open_mark],
            "tp_price": [open_mark - 1.0],
            "sl_price": [open_mark + 2.0],
            "dte_exit": pl.Series([None], dtype=pl.Int32),
            f"leg_{leg_name}_instrument_id": [1001],
            f"leg_{leg_name}_symbol": ["EW3K6 P4300"],
            f"leg_{leg_name}_expiration": pl.Series(["2026-01-17"], dtype=pl.Date),
            f"leg_{leg_name}_strike_price": [4300.0],
            f"leg_{leg_name}_right": ["P"],
            f"leg_{leg_name}_action": ["sell_to_open"],
            f"leg_{leg_name}_quantity": [1],
            f"leg_{leg_name}_multiplier": [multiplier],
            f"leg_{leg_name}_close": [leg_close],
            f"leg_{leg_name}_delta": [-0.25],
            f"leg_{leg_name}_iv": [0.20],
            f"leg_{leg_name}_gamma": [0.01],
            f"leg_{leg_name}_theta": [-0.05],
            f"leg_{leg_name}_vega": [10.0],
            f"leg_{leg_name}_dte": [21],
        }
    )


def _make_exits(
    entry_id: int = 0,
    exit_mark: float = 4.0,
    worst_mark: float = 5.5,
    exit_reason: str = "take_profit",
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entry_id": [entry_id],
            "exit_time": pl.Series(["2026-01-15 14:00:00"], dtype=pl.Datetime("us", "UTC")),
            "exit_mark": [exit_mark],
            "worst_mark": [worst_mark],
            "exit_reason": [exit_reason],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPnLCalculatorCostModel:
    def test_gross_pnl_no_costs(self):
        strat = _make_strategy(slippage_pct=0.0, fee_per_contract=0.0)
        calc = PnLCalculator(strat)
        entries = _make_entries(open_mark=5.0, multiplier=50.0)
        exits = _make_exits(exit_mark=4.0)
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        pos = result.positions
        # gross = (open - exit) * multiplier = (5-4)*50 = 50, no costs
        assert pos["slippage_cost"][0] == pytest.approx(0.0)
        assert pos["fee_cost"][0] == pytest.approx(0.0)
        assert pos["net_pnl"][0] == pytest.approx(50.0)

    def test_slippage_applied_to_exit_mark(self):
        strat = _make_strategy(slippage_pct=0.01, fee_per_contract=0.0)
        calc = PnLCalculator(strat)
        entries = _make_entries(open_mark=5.0, multiplier=50.0)
        exits = _make_exits(exit_mark=4.0)
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        pos = result.positions
        expected_slip = abs(4.0) * 50.0 * 0.01
        assert pos["slippage_cost"][0] == pytest.approx(expected_slip)
        assert pos["net_pnl"][0] == pytest.approx(50.0 - expected_slip)

    def test_flat_fee(self):
        strat = _make_strategy(slippage_pct=0.0, fee_per_contract=0.65)
        calc = PnLCalculator(strat)
        entries = _make_entries(open_mark=5.0, multiplier=50.0)
        exits = _make_exits(exit_mark=4.0)
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        pos = result.positions
        assert pos["fee_cost"][0] == pytest.approx(0.65)
        assert pos["net_pnl"][0] == pytest.approx(50.0 - 0.65)

    def test_both_costs(self):
        strat = _make_strategy(slippage_pct=0.01, fee_per_contract=0.65)
        calc = PnLCalculator(strat)
        entries = _make_entries(open_mark=5.0, multiplier=50.0)
        exits = _make_exits(exit_mark=4.0)
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        pos = result.positions
        gross = (5.0 - 4.0) * 50.0
        slip = 4.0 * 50.0 * 0.01
        fee = 0.65
        assert pos["net_pnl"][0] == pytest.approx(gross - slip - fee)

    def test_loss_position(self):
        strat = _make_strategy()
        calc = PnLCalculator(strat)
        entries = _make_entries(open_mark=3.0, multiplier=50.0)
        exits = _make_exits(exit_mark=6.0, exit_reason="stop_loss")
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        pos = result.positions
        # gross = (3-6)*50 = -150, no costs
        assert pos["net_pnl"][0] == pytest.approx(-150.0)

    def test_slippage_uses_abs_exit_mark(self):
        """Slippage should use |exit_mark|, not exit_mark (credit spread can have negative mark)."""
        strat = _make_strategy(slippage_pct=0.01)
        calc = PnLCalculator(strat)
        entries = _make_entries(open_mark=-2.0, multiplier=50.0)
        exits = _make_exits(exit_mark=-3.0, exit_reason="stop_loss")
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        pos = result.positions
        assert pos["slippage_cost"][0] == pytest.approx(abs(-3.0) * 50.0 * 0.01)


class TestPnLCalculatorOutput:
    def test_positions_output_columns(self):
        strat = _make_strategy()
        calc = PnLCalculator(strat)
        result = calc.compute(
            {"trade1": _make_entries()},
            {"trade1": _make_exits()},
        )
        expected = {
            "entry_id",
            "trade_name",
            "open_time",
            "exit_time",
            "exit_reason",
            "open_mark",
            "exit_mark",
            "worst_mark",
            "slippage_cost",
            "fee_cost",
            "net_pnl",
        }
        assert expected.issubset(set(result.positions.columns))

    def test_legs_output_columns(self):
        strat = _make_strategy()
        calc = PnLCalculator(strat)
        result = calc.compute(
            {"trade1": _make_entries()},
            {"trade1": _make_exits()},
        )
        expected = {
            "entry_id",
            "instrument_id",
            "symbol",
            "expiration",
            "strike_price",
            "right",
            "action",
            "quantity",
            "multiplier",
            "open_price",
            "exit_price",
            "entry_delta",
            "entry_iv",
            "entry_gamma",
            "entry_theta",
            "entry_vega",
            "entry_dte",
        }
        assert expected.issubset(set(result.legs.columns))

    def test_exit_reason_preserved(self):
        strat = _make_strategy()
        calc = PnLCalculator(strat)
        result = calc.compute(
            {"trade1": _make_entries()},
            {"trade1": _make_exits(exit_reason="stop_loss")},
        )
        assert result.positions["exit_reason"][0] == "stop_loss"

    def test_worst_mark_preserved(self):
        strat = _make_strategy()
        calc = PnLCalculator(strat)
        result = calc.compute(
            {"trade1": _make_entries()},
            {"trade1": _make_exits(worst_mark=7.5)},
        )
        assert result.positions["worst_mark"][0] == pytest.approx(7.5)

    def test_empty_entries_returns_empty(self):
        strat = _make_strategy()
        calc = PnLCalculator(strat)
        result = calc.compute(
            {"trade1": pl.DataFrame()},
            {"trade1": _make_exits()},
        )
        assert result.positions.is_empty()
        assert result.legs.is_empty()

    def test_multiple_positions(self):
        strat = _make_strategy(slippage_pct=0.0)
        calc = PnLCalculator(strat)
        entries = pl.concat(
            [
                _make_entries(entry_id=0, open_mark=5.0),
                _make_entries(entry_id=1, open_mark=8.0),
            ]
        )
        exits = pl.concat(
            [
                _make_exits(entry_id=0, exit_mark=4.0),
                _make_exits(entry_id=1, exit_mark=6.0),
            ]
        )
        result = calc.compute({"trade1": entries}, {"trade1": exits})
        assert len(result.positions) == 2
        pnls = sorted(result.positions["net_pnl"].to_list())
        assert pnls == pytest.approx([50.0, 100.0])

    def test_action_code_sto(self):
        strat = _make_strategy()
        calc = PnLCalculator(strat)
        result = calc.compute(
            {"trade1": _make_entries()},
            {"trade1": _make_exits()},
        )
        assert result.legs["action"][0] == "STO"

    def test_entry_greeks_written_to_legs(self):
        strat = _make_strategy()
        calc = PnLCalculator(strat)
        result = calc.compute(
            {"trade1": _make_entries(leg_close=5.0)},
            {"trade1": _make_exits()},
        )
        leg = result.legs
        assert leg["entry_delta"][0] == pytest.approx(-0.25)
        assert leg["entry_iv"][0] == pytest.approx(0.20)
        assert leg["entry_dte"][0] == 21
