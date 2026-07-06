"""
Unit tests for IV-stepped delta configuration.

Covers:
  - DeltaStep / SimpleDeltaConfig / SteppedDeltaConfig validation
  - LegConfig acceptance of both config shapes
  - EntryScanner._build_step_exprs() step resolution logic
"""

from __future__ import annotations

from datetime import date, time

import polars as pl
import pytest
from pydantic import ValidationError

from btkit.strategy.definition import (
    DeltaStep,
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    SimpleDeltaConfig,
    SteppedDeltaConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)

# ---------------------------------------------------------------------------
# SimpleDeltaConfig
# ---------------------------------------------------------------------------


class TestSimpleDeltaConfig:
    def test_scalar_target(self):
        cfg = SimpleDeltaConfig(target=-0.25)
        assert cfg.target == -0.25
        assert cfg.tolerance == 0.10

    def test_list_target(self):
        cfg = SimpleDeltaConfig(target=[-0.20, -0.25])
        assert cfg.target == [-0.20, -0.25]

    def test_custom_tolerance(self):
        cfg = SimpleDeltaConfig(target=-0.25, tolerance=0.05)
        assert cfg.tolerance == 0.05


# ---------------------------------------------------------------------------
# DeltaStep
# ---------------------------------------------------------------------------


class TestDeltaStep:
    def test_step_with_below(self):
        step = DeltaStep(below=15.0, target=-0.10, tolerance=0.03)
        assert step.below == 15.0
        assert step.target == -0.10
        assert step.tolerance == 0.03

    def test_catch_all_step(self):
        step = DeltaStep(target=-0.12)
        assert step.below is None

    def test_step_tolerance_defaults_none(self):
        step = DeltaStep(below=20.0, target=-0.10)
        assert step.tolerance is None


# ---------------------------------------------------------------------------
# SteppedDeltaConfig validation
# ---------------------------------------------------------------------------


class TestSteppedDeltaConfig:
    def test_valid_stepped_config(self):
        cfg = SteppedDeltaConfig(
            step_source="ves1d_close",
            steps=[
                DeltaStep(below=10.0, target=-0.10, tolerance=0.03),
                DeltaStep(below=15.0, target=-0.12, tolerance=0.05),
                DeltaStep(target=-0.15),  # catch-all
            ],
        )
        assert cfg.step_source == "ves1d_close"
        assert len(cfg.steps) == 3

    def test_empty_steps_raises(self):
        with pytest.raises(ValidationError, match="steps must not be empty"):
            SteppedDeltaConfig(step_source="iv", steps=[])

    def test_multiple_catch_alls_raises(self):
        with pytest.raises(ValidationError, match="at most one catch-all"):
            SteppedDeltaConfig(
                step_source="iv",
                steps=[
                    DeltaStep(target=-0.10),
                    DeltaStep(target=-0.12),
                ],
            )

    def test_catch_all_not_last_raises(self):
        with pytest.raises(ValidationError, match="catch-all.*must be last"):
            SteppedDeltaConfig(
                step_source="iv",
                steps=[
                    DeltaStep(target=-0.10),  # catch-all in position 0
                    DeltaStep(below=15.0, target=-0.12),
                ],
            )

    def test_fallback_tolerance_used(self):
        cfg = SteppedDeltaConfig(
            step_source="iv",
            tolerance=0.07,
            steps=[DeltaStep(below=15.0, target=-0.10)],
        )
        assert cfg.tolerance == 0.07


# ---------------------------------------------------------------------------
# LegConfig with both config shapes
# ---------------------------------------------------------------------------


class TestLegConfigDelta:
    def test_simple_delta_dict(self):
        leg = LegConfig(
            name="sp",
            right="put",
            action="sell_to_open",
            dte=21,
            delta={"target": -0.25},
        )
        assert isinstance(leg.delta, SimpleDeltaConfig)
        assert leg.delta.target == -0.25

    def test_simple_delta_with_tolerance(self):
        leg = LegConfig(
            name="sp",
            right="put",
            action="sell_to_open",
            dte=21,
            delta={"target": -0.25, "tolerance": 0.05},
        )
        assert leg.delta.tolerance == 0.05

    def test_stepped_delta_dict(self):
        leg = LegConfig(
            name="sp",
            right="put",
            action="sell_to_open",
            dte=21,
            delta={
                "step_source": "ves1d_close",
                "steps": [
                    {"below": 10.0, "target": -0.10, "tolerance": 0.03},
                    {"below": 15.0, "target": -0.12, "tolerance": 0.05},
                ],
            },
        )
        assert isinstance(leg.delta, SteppedDeltaConfig)
        assert leg.delta.step_source == "ves1d_close"
        assert len(leg.delta.steps) == 2

    def test_stepped_delta_object(self):
        stepped = SteppedDeltaConfig(
            step_source="iv",
            steps=[DeltaStep(below=15.0, target=-0.10)],
        )
        leg = LegConfig(
            name="sp",
            right="put",
            action="sell_to_open",
            dte=21,
            delta=stepped,
        )
        assert isinstance(leg.delta, SteppedDeltaConfig)


# ---------------------------------------------------------------------------
# _build_step_exprs step resolution
# ---------------------------------------------------------------------------


def _make_entry_scanner(trade: TradeDefinition):
    """Return an EntryScanner without a real DB (for testing step exprs)."""
    from unittest.mock import MagicMock

    from btkit.backtest.entry import EntryScanner

    strategy = StrategyDefinition(
        name="test",
        universe=UniverseConfig(start_date=date(2024, 1, 1), end_date=date(2024, 3, 31)),
        trades=[trade],
    )
    scanner = EntryScanner.__new__(EntryScanner)
    scanner.db = MagicMock()
    scanner.strategy = strategy
    scanner.trade = trade
    return scanner


def _make_trade_with_stepped_delta(
    source: str,
    steps: list[DeltaStep],
    fallback_tolerance: float = 0.10,
) -> TradeDefinition:
    return TradeDefinition(
        name="t",
        instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
        entry=EntryConfig(window=EntryWindowConfig(start=time(10, 0), end=time(12, 0))),
        legs=[
            LegConfig(
                name="sp",
                right="put",
                action="sell_to_open",
                dte=21,
                delta=SteppedDeltaConfig(
                    step_source=source,
                    tolerance=fallback_tolerance,
                    steps=steps,
                ),
            )
        ],
        exit=ExitConfig(stop_loss=2.0),
    )


class TestBuildStepExprs:
    def test_two_steps_first_bucket(self):
        """IV < 10 → target -0.10, tolerance 0.03"""
        trade = _make_trade_with_stepped_delta(
            source="iv",
            steps=[
                DeltaStep(below=10.0, target=-0.10, tolerance=0.03),
                DeltaStep(below=15.0, target=-0.12, tolerance=0.05),
            ],
        )
        scanner = _make_entry_scanner(trade)
        assert isinstance(trade.legs[0].delta, SteppedDeltaConfig)
        d_expr, t_expr = scanner._build_step_exprs(trade.legs[0].delta)

        df = pl.DataFrame({"iv": [8.0, 12.0, 20.0]})
        d_vals = df.with_columns(d_expr.alias("d"))["d"].to_list()
        t_vals = df.with_columns(t_expr.alias("t"))["t"].to_list()

        assert d_vals == [-0.10, -0.12, None]
        assert t_vals == [0.03, 0.05, None]

    def test_catch_all_used_for_no_match(self):
        """IV >= all thresholds → catch-all fires"""
        trade = _make_trade_with_stepped_delta(
            source="iv",
            steps=[
                DeltaStep(below=10.0, target=-0.10, tolerance=0.03),
                DeltaStep(target=-0.15),  # catch-all, no explicit tolerance
            ],
            fallback_tolerance=0.08,
        )
        scanner = _make_entry_scanner(trade)
        assert isinstance(trade.legs[0].delta, SteppedDeltaConfig)
        d_expr, t_expr = scanner._build_step_exprs(trade.legs[0].delta)

        df = pl.DataFrame({"iv": [5.0, 25.0]})
        d_vals = df.with_columns(d_expr.alias("d"))["d"].to_list()
        t_vals = df.with_columns(t_expr.alias("t"))["t"].to_list()

        assert d_vals == [-0.10, -0.15]
        assert t_vals[0] == 0.03
        assert (
            t_vals[1] == 0.08
        )  # catch-all tolerance is None → falls back to SteppedDeltaConfig.tolerance

    def test_null_when_no_match_and_no_catch_all(self):
        """No step matches and no catch-all → null"""
        trade = _make_trade_with_stepped_delta(
            source="iv",
            steps=[DeltaStep(below=10.0, target=-0.10, tolerance=0.03)],
        )
        scanner = _make_entry_scanner(trade)
        assert isinstance(trade.legs[0].delta, SteppedDeltaConfig)
        d_expr, t_expr = scanner._build_step_exprs(trade.legs[0].delta)

        df = pl.DataFrame({"iv": [20.0]})
        d_val = df.with_columns(d_expr.alias("d"))["d"].to_list()[0]
        assert d_val is None

    def test_single_catch_all_only(self):
        """A single catch-all step always fires"""
        trade = _make_trade_with_stepped_delta(
            source="iv",
            steps=[DeltaStep(target=-0.12, tolerance=0.04)],
        )
        scanner = _make_entry_scanner(trade)
        assert isinstance(trade.legs[0].delta, SteppedDeltaConfig)
        d_expr, t_expr = scanner._build_step_exprs(trade.legs[0].delta)

        df = pl.DataFrame({"iv": [5.0, 15.0, 99.0]})
        d_vals = df.with_columns(d_expr.alias("d"))["d"].to_list()
        assert d_vals == [-0.12, -0.12, -0.12]

    def test_null_source_propagates(self):
        """When the indicator column is null, result is null (< comparisons return null)"""
        trade = _make_trade_with_stepped_delta(
            source="iv",
            steps=[DeltaStep(below=10.0, target=-0.10, tolerance=0.03)],
        )
        scanner = _make_entry_scanner(trade)
        assert isinstance(trade.legs[0].delta, SteppedDeltaConfig)
        d_expr, _ = scanner._build_step_exprs(trade.legs[0].delta)

        df = pl.DataFrame({"iv": pl.Series([None], dtype=pl.Float64)})
        d_val = df.with_columns(d_expr.alias("d"))["d"].to_list()[0]
        assert d_val is None
