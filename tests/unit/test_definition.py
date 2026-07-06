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
    CostsConfig,
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    FeesConfig,
    InstrumentConfig,
    LegConfig,
    LiquidityConfig,
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
        delta={"target": delta},
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


class TestEntryConfig:
    def test_defaults(self):
        cfg = EntryConfig(window=EntryWindowConfig(start=time(9, 45), end=time(14, 30)))
        assert cfg.max_entries_per_day is None
        assert cfg.conditions == []
        assert cfg.min_credit is None

    def test_max_entries_per_day_set(self):
        cfg = EntryConfig(
            window=EntryWindowConfig(start=time(9, 45), end=time(14, 30)),
            max_entries_per_day=1,
        )
        assert cfg.max_entries_per_day == 1

    def test_max_entries_per_day_unlimited(self):
        cfg = EntryConfig(
            window=EntryWindowConfig(start=time(9, 45), end=time(14, 30)),
            max_entries_per_day=None,
        )
        assert cfg.max_entries_per_day is None


class TestTakeProfitConfig:
    def test_confirmation_bars_default(self):
        cfg = TakeProfitConfig(pct=0.70)
        assert cfg.confirmation_bars == 1

    def test_confirmation_bars_set(self):
        cfg = TakeProfitConfig(pct=0.70, confirmation_bars=2)
        assert cfg.confirmation_bars == 2

    def test_confirmation_bars_with_price(self):
        cfg = TakeProfitConfig(price=1.0, confirmation_bars=3)
        assert cfg.confirmation_bars == 3

    def test_confirmation_bars_with_condition(self):
        cfg = TakeProfitConfig(pct=0.70, condition="close > vwap", confirmation_bars=2)
        assert cfg.confirmation_bars == 2
        assert cfg.condition == "close > vwap"


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

    def test_leg_out_defaults_false(self):
        cfg = ExitConfig()
        assert cfg.leg_out is False

    def test_leg_out_can_be_enabled(self):
        cfg = ExitConfig(stop_loss=5.0, take_profit_pct=0.50, leg_out=True)
        assert cfg.leg_out is True

    def test_leg_out_round_trips_via_model_dump(self):
        cfg = ExitConfig(stop_loss=5.0, take_profit_pct=0.50, leg_out=True)
        dumped = cfg.model_dump()
        restored = ExitConfig(**dumped)
        assert restored.leg_out is True


# ---------------------------------------------------------------------------
# LegConfig
# ---------------------------------------------------------------------------


class TestLegConfig:
    def test_delta_mode(self):
        from btkit.strategy.definition import SimpleDeltaConfig

        leg = _make_leg()
        assert isinstance(leg.delta, SimpleDeltaConfig)
        assert leg.delta.target == -0.25
        assert leg.strike_offset is None
        assert leg.reference_leg is None

    def test_strike_offset_mode_no_dte(self):
        # dte is optional for offset legs — omitting it means "inherit from reference"
        leg = LegConfig(
            name="wing",
            right="put",
            action="buy_to_open",
            strike_offset=-25.0,
            reference_leg="short_put",
        )
        assert leg.strike_offset == -25.0
        assert leg.reference_leg == "short_put"
        assert leg.dte is None

    def test_strike_offset_mode_with_dte(self):
        # dte may still be specified on an offset leg (reserved for calendar spreads)
        leg = LegConfig(
            name="wing",
            right="put",
            action="buy_to_open",
            dte=45,
            strike_offset=-25.0,
            reference_leg="short_put",
        )
        assert leg.dte == 45

    def test_delta_leg_without_dte_rejected(self):
        with pytest.raises(ValidationError, match="dte is required for delta-targeted legs"):
            LegConfig(name="leg", right="put", action="sell_to_open", delta={"target": -0.25})

    def test_delta_and_offset_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            LegConfig(
                name="leg",
                right="put",
                action="sell_to_open",
                dte=21,
                delta={"target": -0.25},
                strike_offset=-10.0,
                reference_leg="other",
            )

    def test_neither_delta_nor_offset_rejected(self):
        with pytest.raises(
            ValidationError, match="one of delta, strike_offset, stepped, or targets is required"
        ):
            LegConfig(name="leg", right="put", action="sell_to_open", dte=21)

    def test_offset_without_reference_rejected(self):
        with pytest.raises(ValidationError, match="reference_leg is required"):
            LegConfig(
                name="leg",
                right="put",
                action="buy_to_open",
                strike_offset=-10.0,
            )

    def test_default_tolerances(self):
        from btkit.strategy.definition import SimpleDeltaConfig

        leg = _make_leg()
        assert isinstance(leg.delta, SimpleDeltaConfig)
        assert leg.delta.tolerance == 0.10
        assert leg.dte_tolerance == 5

    def test_custom_tolerances(self):
        from btkit.strategy.definition import SimpleDeltaConfig

        leg = LegConfig(
            name="leg",
            right="put",
            action="sell_to_open",
            dte=0,
            delta={"target": -0.25, "tolerance": 0.20},
            dte_tolerance=0,
        )
        assert isinstance(leg.delta, SimpleDeltaConfig)
        assert leg.delta.tolerance == 0.20
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
            delta={"target": [-0.20, -0.25, -0.30]},
        )
        trade = _make_trade(legs=[leg_with_sweep])
        with pytest.raises(ValidationError, match="mutually exclusive"):
            _make_strategy(
                trades=[trade],
                combinations=[{"short_put": {"delta": {"target": -0.25}}}],
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
            delta={"target": [-0.20, -0.25]},
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
            delta={"target": SweepRange(start=-0.20, stop=-0.30, step=-0.05)},
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


# ---------------------------------------------------------------------------
# LiquidityConfig
# ---------------------------------------------------------------------------


class TestLiquidityConfig:
    def test_defaults(self):
        liq = LiquidityConfig()
        assert liq.min_exit_volume is None
        assert liq.lookback_minutes == 3
        assert liq.pre_expiry_lock_minutes is None
        assert liq.slippage_model == "flat"

    def test_is_default_true_when_all_defaults(self):
        assert LiquidityConfig().is_default is True

    def test_is_default_false_when_volume_set(self):
        liq = LiquidityConfig(min_exit_volume=100)
        assert liq.is_default is False

    def test_is_default_false_when_pre_expiry_lock_set(self):
        liq = LiquidityConfig(pre_expiry_lock_minutes=15)
        assert liq.is_default is False

    def test_is_default_false_when_spread_slippage(self):
        liq = LiquidityConfig(slippage_model="spread")
        assert liq.is_default is False

    def test_needs_volume_false_by_default(self):
        assert LiquidityConfig().needs_volume is False

    def test_needs_volume_true_when_set(self):
        assert LiquidityConfig(min_exit_volume=50).needs_volume is True

    def test_needs_spread_false_by_default(self):
        assert LiquidityConfig().needs_spread is False

    def test_needs_spread_true_when_spread(self):
        assert LiquidityConfig(slippage_model="spread").needs_spread is True

    def test_invalid_slippage_model_rejected(self):
        with pytest.raises(ValidationError):
            LiquidityConfig(slippage_model="half_spread")

    def test_custom_lookback_minutes(self):
        liq = LiquidityConfig(min_exit_volume=200, lookback_minutes=5)
        assert liq.lookback_minutes == 5
        assert liq.needs_volume is True

    def test_all_features_enabled(self):
        liq = LiquidityConfig(
            min_exit_volume=100,
            lookback_minutes=5,
            pre_expiry_lock_minutes=30,
            slippage_model="spread",
        )
        assert liq.needs_volume is True
        assert liq.needs_spread is True
        assert liq.is_default is False

    def test_exit_config_default_liquidity(self):
        cfg = ExitConfig(stop_loss=2.0, take_profit=1.0)
        assert cfg.liquidity.is_default is True

    def test_exit_config_liquidity_override(self):
        cfg = ExitConfig(
            stop_loss=2.0,
            take_profit=1.0,
            liquidity=LiquidityConfig(min_exit_volume=100, slippage_model="spread"),
        )
        assert cfg.liquidity.needs_volume is True
        assert cfg.liquidity.needs_spread is True


# ---------------------------------------------------------------------------
# FeesConfig / CostsConfig.effective_fees
# ---------------------------------------------------------------------------


class TestFeesConfig:
    def test_defaults_all_zero(self):
        f = FeesConfig()
        assert f.entry_fee_per_contract == 0.0
        assert f.exit_fee_per_contract == 0.0
        assert f.expiration_fee_per_contract == 0.0

    def test_all_fields_set(self):
        f = FeesConfig(
            entry_fee_per_contract=0.65,
            exit_fee_per_contract=0.65,
            expiration_fee_per_contract=0.0,
        )
        assert f.entry_fee_per_contract == 0.65
        assert f.exit_fee_per_contract == 0.65
        assert f.expiration_fee_per_contract == 0.0


class TestCostsConfigFees:
    def test_default_costs_effective_fees_all_zero(self):
        fees = CostsConfig().effective_fees
        assert fees.entry_fee_per_contract == 0.0
        assert fees.exit_fee_per_contract == 0.0
        assert fees.expiration_fee_per_contract == 0.0

    def test_legacy_fee_per_contract_splits_evenly(self):
        fees = CostsConfig(fee_per_contract=0.65).effective_fees
        assert fees.entry_fee_per_contract == pytest.approx(0.325)
        assert fees.exit_fee_per_contract == pytest.approx(0.325)
        assert fees.expiration_fee_per_contract == 0.0

    def test_structured_fees_takes_precedence(self):
        fees = CostsConfig(
            fees=FeesConfig(
                entry_fee_per_contract=0.65,
                exit_fee_per_contract=0.65,
                expiration_fee_per_contract=0.0,
            )
        ).effective_fees
        assert fees.entry_fee_per_contract == 0.65
        assert fees.exit_fee_per_contract == 0.65
        assert fees.expiration_fee_per_contract == 0.0

    def test_structured_fees_with_nonzero_expiration(self):
        fees = CostsConfig(
            fees=FeesConfig(
                entry_fee_per_contract=0.65,
                exit_fee_per_contract=0.65,
                expiration_fee_per_contract=0.10,
            )
        ).effective_fees
        assert fees.expiration_fee_per_contract == pytest.approx(0.10)

    def test_fee_per_contract_and_fees_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            CostsConfig(
                fee_per_contract=0.65,
                fees=FeesConfig(entry_fee_per_contract=0.65),
            )

    def test_fees_none_with_zero_legacy_returns_zeros(self):
        # Default: no fee_per_contract set, no fees block → all zeros
        costs = CostsConfig(slippage_pct=0.01)
        fees = costs.effective_fees
        assert fees.entry_fee_per_contract == 0.0
        assert fees.exit_fee_per_contract == 0.0
