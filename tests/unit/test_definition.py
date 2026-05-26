"""
Unit tests for StrategyDefinition and sub-model Pydantic validators.

Tests cover the model_validator logic that cannot be caught by field-level
type validation alone — mutual exclusions, cross-field consistency, and
uniqueness constraints.
"""

from __future__ import annotations

from datetime import date, time

import pytest
from pydantic import ValidationError

from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    StopLossConfig,
    StrategyDefinition,
    SweepRange,
    TakeProfitConfig,
    TradeDefinition,
    UniverseConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instrument() -> InstrumentConfig:
    return InstrumentConfig(root_symbol="ES", asset_class="future")


def _make_universe() -> UniverseConfig:
    return UniverseConfig(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )


def _make_entry() -> EntryConfig:
    return EntryConfig(window=EntryWindowConfig(start=time(10, 0), end=time(12, 0)))


def _make_exit() -> ExitConfig:
    return ExitConfig(stop_loss=2.0, take_profit=1.0)


def _make_leg(name: str = "short_put", delta: float = -0.25, dte: int = 21) -> LegConfig:
    return LegConfig(
        name=name,
        right="put",
        action="sell_to_open",
        delta=delta,
        dte=dte,
    )


def _make_trade(name: str = "trade1", legs: list | None = None) -> TradeDefinition:
    if legs is None:
        legs = [_make_leg()]
    return TradeDefinition(
        name=name,
        instrument=_make_instrument(),
        entry=_make_entry(),
        legs=legs,
        exit=_make_exit(),
    )


def _make_strategy(**kwargs) -> StrategyDefinition:
    defaults = dict(
        name="test_strategy",
        universe=_make_universe(),
        trades=[_make_trade()],
    )
    defaults.update(kwargs)
    return StrategyDefinition(**defaults)


# ---------------------------------------------------------------------------
# EntryWindowConfig
# ---------------------------------------------------------------------------


class TestEntryWindowConfig:
    def test_valid_window(self):
        w = EntryWindowConfig(start=time(9, 30), end=time(16, 0))
        assert w.start < w.end

    def test_start_must_be_before_end(self):
        with pytest.raises(ValidationError, match="start must be before end"):
            EntryWindowConfig(start=time(12, 0), end=time(10, 0))

    def test_equal_times_rejected(self):
        with pytest.raises(ValidationError):
            EntryWindowConfig(start=time(10, 0), end=time(10, 0))


# ---------------------------------------------------------------------------
# ExitConfig
# ---------------------------------------------------------------------------


class TestExitConfig:
    def test_all_optional(self):
        cfg = ExitConfig()
        assert cfg.stop_loss is None
        assert cfg.take_profit is None
        assert cfg.take_profit_pct is None

    def test_stop_loss_only(self):
        cfg = ExitConfig(stop_loss=2.0)
        assert cfg.stop_loss == 2.0
        assert cfg.take_profit is None

    def test_take_profit_and_pct_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ExitConfig(stop_loss=2.0, take_profit=1.0, take_profit_pct=0.5)

    def test_take_profit_pct_only(self):
        cfg = ExitConfig(stop_loss=2.0, take_profit_pct=0.5)
        assert cfg.take_profit_pct == 0.5
        assert cfg.take_profit is None

    def test_take_profit_only(self):
        cfg = ExitConfig(stop_loss=2.0, take_profit=1.0)
        assert cfg.take_profit == 1.0
        assert cfg.take_profit_pct is None

    # --- Object form ---

    def test_stop_loss_object_form(self):
        cfg = ExitConfig(
            stop_loss=StopLossConfig(price=2.0, condition="close < vwap"),
            take_profit=1.0,
        )
        assert isinstance(cfg.stop_loss, StopLossConfig)
        assert cfg.stop_loss.price == 2.0
        assert cfg.stop_loss.condition == "close < vwap"

    def test_stop_loss_object_no_condition(self):
        cfg = ExitConfig(stop_loss=StopLossConfig(price=2.0), take_profit=1.0)
        assert cfg.stop_loss.condition is None

    def test_take_profit_object_price(self):
        cfg = ExitConfig(
            stop_loss=2.0,
            take_profit=TakeProfitConfig(price=1.0, condition="close > vwap"),
        )
        assert isinstance(cfg.take_profit, TakeProfitConfig)
        assert cfg.take_profit.price == 1.0
        assert cfg.take_profit.condition == "close > vwap"

    def test_take_profit_object_pct(self):
        cfg = ExitConfig(
            stop_loss=2.0,
            take_profit=TakeProfitConfig(pct=0.70, condition="close > vwap"),
        )
        assert cfg.take_profit.pct == 0.70
        assert cfg.take_profit.price is None

    def test_take_profit_object_requires_price_or_pct(self):
        with pytest.raises(ValidationError, match="one of price or pct"):
            ExitConfig(stop_loss=2.0, take_profit=TakeProfitConfig())

    def test_take_profit_object_price_and_pct_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ExitConfig(stop_loss=2.0, take_profit=TakeProfitConfig(price=1.0, pct=0.5))

    def test_both_object_forms(self):
        cfg = ExitConfig(
            stop_loss=StopLossConfig(price=5.0, condition="close < vwap_1d"),
            take_profit=TakeProfitConfig(pct=0.70, condition="close > vwap_1u"),
        )
        assert isinstance(cfg.stop_loss, StopLossConfig)
        assert isinstance(cfg.take_profit, TakeProfitConfig)


# ---------------------------------------------------------------------------
# LegConfig
# ---------------------------------------------------------------------------


class TestLegConfig:
    def test_delta_mode(self):
        leg = _make_leg()
        assert leg.delta == -0.25
        assert leg.strike_offset is None
        assert leg.reference_leg is None

    def test_strike_offset_mode(self):
        leg = LegConfig(
            name="wing",
            right="put",
            action="buy_to_open",
            dte=21,
            strike_offset=-25.0,
            reference_leg="short_put",
        )
        assert leg.strike_offset == -25.0
        assert leg.reference_leg == "short_put"

    def test_delta_and_offset_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            LegConfig(
                name="leg",
                right="put",
                action="sell_to_open",
                dte=21,
                delta=-0.25,
                strike_offset=-10.0,
                reference_leg="other",
            )

    def test_neither_delta_nor_offset_rejected(self):
        with pytest.raises(ValidationError, match="one of delta or strike_offset"):
            LegConfig(name="leg", right="put", action="sell_to_open", dte=21)

    def test_offset_without_reference_rejected(self):
        with pytest.raises(ValidationError, match="reference_leg is required"):
            LegConfig(
                name="leg",
                right="put",
                action="buy_to_open",
                dte=21,
                strike_offset=-10.0,
            )

    def test_default_tolerances(self):
        leg = _make_leg()
        assert leg.delta_tolerance == 0.10
        assert leg.dte_tolerance == 5

    def test_custom_tolerances(self):
        leg = LegConfig(
            name="leg",
            right="put",
            action="sell_to_open",
            dte=0,
            delta=-0.25,
            delta_tolerance=0.20,
            dte_tolerance=0,
        )
        assert leg.delta_tolerance == 0.20
        assert leg.dte_tolerance == 0


# ---------------------------------------------------------------------------
# TradeDefinition
# ---------------------------------------------------------------------------


class TestTradeDefinition:
    def test_leg_names_must_be_unique(self):
        legs = [_make_leg("leg1"), _make_leg("leg1")]
        with pytest.raises(ValidationError, match="unique"):
            TradeDefinition(
                name="t",
                instrument=_make_instrument(),
                entry=_make_entry(),
                legs=legs,
                exit=_make_exit(),
            )

    def test_reference_leg_must_exist(self):
        legs = [
            _make_leg("short_put"),
            LegConfig(
                name="long_put",
                right="put",
                action="buy_to_open",
                dte=21,
                strike_offset=-25.0,
                reference_leg="nonexistent_leg",
            ),
        ]
        with pytest.raises(ValidationError, match="nonexistent_leg"):
            TradeDefinition(
                name="t",
                instrument=_make_instrument(),
                entry=_make_entry(),
                legs=legs,
                exit=_make_exit(),
            )

    def test_reference_leg_must_be_delta_selected(self):
        """An offset leg cannot reference another offset leg."""
        legs = [
            _make_leg("anchor"),
            LegConfig(
                name="offset1",
                right="put",
                action="buy_to_open",
                dte=21,
                strike_offset=-25.0,
                reference_leg="anchor",
            ),
            LegConfig(
                name="offset2",
                right="put",
                action="buy_to_open",
                dte=21,
                strike_offset=-50.0,
                reference_leg="offset1",  # offset1 is itself offset-selected
            ),
        ]
        with pytest.raises(ValidationError):
            TradeDefinition(
                name="t",
                instrument=_make_instrument(),
                entry=_make_entry(),
                legs=legs,
                exit=_make_exit(),
            )

    def test_valid_two_leg_spread(self):
        legs = [
            _make_leg("short_put"),
            LegConfig(
                name="long_put",
                right="put",
                action="buy_to_open",
                dte=21,
                strike_offset=-25.0,
                reference_leg="short_put",
            ),
        ]
        trade = TradeDefinition(
            name="spread",
            instrument=_make_instrument(),
            entry=_make_entry(),
            legs=legs,
            exit=_make_exit(),
        )
        assert len(trade.legs) == 2


# ---------------------------------------------------------------------------
# StrategyDefinition
# ---------------------------------------------------------------------------


class TestStrategyDefinition:
    def test_minimal_valid_strategy(self):
        s = _make_strategy()
        assert s.name == "test_strategy"
        assert len(s.trades) == 1

    def test_trade_names_must_be_unique(self):
        with pytest.raises(ValidationError, match="trade names must be unique"):
            _make_strategy(trades=[_make_trade("t1"), _make_trade("t1")])

    def test_all_trades_must_share_underlying(self):
        trade2 = TradeDefinition(
            name="different_trade",
            instrument=InstrumentConfig(root_symbol="NQ", asset_class="future"),
            entry=_make_entry(),
            legs=[_make_leg()],
            exit=_make_exit(),
        )
        with pytest.raises(ValidationError, match="same underlying"):
            _make_strategy(trades=[_make_trade(), trade2])

    def test_combinations_and_sweeps_mutually_exclusive(self):
        leg_with_sweep = LegConfig(
            name="short_put",
            right="put",
            action="sell_to_open",
            dte=21,
            delta=[-0.20, -0.25, -0.30],
        )
        trade = _make_trade(legs=[leg_with_sweep])
        with pytest.raises(ValidationError, match="mutually exclusive"):
            _make_strategy(
                trades=[trade],
                combinations=[{"short_put": {"delta": -0.25}}],
            )

    def test_is_parameterized_scalar(self):
        s = _make_strategy()
        assert s.is_parameterized() is False

    def test_is_parameterized_list(self):
        leg = LegConfig(
            name="short_put",
            right="put",
            action="sell_to_open",
            dte=21,
            delta=[-0.20, -0.25],
        )
        trade = _make_trade(legs=[leg])
        s = _make_strategy(trades=[trade])
        assert s.is_parameterized() is True

    def test_is_parameterized_sweep_range(self):
        leg = LegConfig(
            name="short_put",
            right="put",
            action="sell_to_open",
            dte=21,
            delta=SweepRange(start=-0.20, stop=-0.30, step=-0.05),
        )
        trade = _make_trade(legs=[leg])
        s = _make_strategy(trades=[trade])
        assert s.is_parameterized() is True

    def test_default_version(self):
        s = _make_strategy()
        assert s.version == "1.0"

    def test_multi_trade_same_underlying(self):
        t1 = _make_trade("trade1")
        t2 = _make_trade("trade2")
        s = _make_strategy(trades=[t1, t2])
        assert len(s.trades) == 2


# ---------------------------------------------------------------------------
# SweepRange
# ---------------------------------------------------------------------------


class TestSweepRange:
    def test_ascending_range(self):
        sr = SweepRange(start=0.10, stop=0.30, step=0.10)
        assert sr.values() == pytest.approx([0.10, 0.20, 0.30])

    def test_descending_range(self):
        sr = SweepRange(start=0.30, stop=0.10, step=-0.10)
        assert sr.values() == pytest.approx([0.30, 0.20, 0.10])

    def test_single_value_range(self):
        sr = SweepRange(start=0.25, stop=0.25, step=0.05)
        assert sr.values() == pytest.approx([0.25])

    def test_integer_range(self):
        sr = SweepRange(start=10, stop=30, step=10)
        assert sr.values() == pytest.approx([10.0, 20.0, 30.0])

    def test_float_accumulation(self):
        """Floating-point step accumulation should not produce extra values."""
        sr = SweepRange(start=0.0, stop=1.0, step=0.1)
        vals = sr.values()
        assert len(vals) == 11
        assert vals[-1] == pytest.approx(1.0)

    def test_small_step(self):
        sr = SweepRange(start=0.0, stop=0.05, step=0.01)
        vals = sr.values()
        assert len(vals) == 6
