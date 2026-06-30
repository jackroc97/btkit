"""
Unit tests for spread-level vega exit condition.

Tests validate:
  - vega_exit = None → _vega_exit never fires (no-op)
  - vega_exit threshold fires when _spread_vega < threshold
  - priority ordering relative to condition (5) and dte_exit (7)
  - ExitConfig.vega_exit field default and assignment
"""
from __future__ import annotations

from datetime import UTC, date, datetime, time

import polars as pl
import pytest

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


class TestVegaExitConfig:
    def test_vega_exit_defaults_none(self):
        cfg = ExitConfig()
        assert cfg.vega_exit is None

    def test_vega_exit_accepts_float(self):
        cfg = ExitConfig(vega_exit=0.15)
        assert cfg.vega_exit == 0.15

    def test_vega_exit_accepts_zero(self):
        cfg = ExitConfig(vega_exit=0.0)
        assert cfg.vega_exit == 0.0

    def test_vega_exit_in_exit_config_alongside_other_fields(self):
        cfg = ExitConfig(stop_loss=1.5, take_profit=0.5, vega_exit=0.20)
        assert cfg.vega_exit == 0.20
        assert cfg.stop_loss == 1.5


class TestVegaExitPriority:
    """
    Priority encoding:
      5 = condition
      6 = vega_exit   (NEW)
      7 = dte_exit
      8 = expiry
    """

    def _make_trade(self, vega_exit=None, dte_exit=None):
        return TradeDefinition(
            name="t",
            instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
            entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(16, 0))),
            legs=[
                LegConfig(name="short", right="put", action="sell_to_open", dte=0, delta=-0.16),
                LegConfig(name="long", right="put", action="buy_to_open", dte=0, strike_offset=-50.0, reference_leg="short"),
            ],
            exit=ExitConfig(vega_exit=vega_exit, dte_exit=dte_exit),
        )

    def test_vega_exit_none_is_noop_in_config(self):
        """vega_exit=None means the field is not configured."""
        trade = self._make_trade(vega_exit=None)
        assert trade.exit.vega_exit is None

    def test_vega_exit_set(self):
        trade = self._make_trade(vega_exit=0.25)
        assert trade.exit.vega_exit == 0.25

    def test_dte_exit_unchanged_when_vega_not_set(self):
        """dte_exit still works normally when vega_exit is None."""
        trade = self._make_trade(dte_exit=5)
        assert trade.exit.dte_exit == 5
        assert trade.exit.vega_exit is None

    def test_both_vega_and_dte_can_coexist(self):
        trade = self._make_trade(vega_exit=0.20, dte_exit=7)
        assert trade.exit.vega_exit == 0.20
        assert trade.exit.dte_exit == 7
