"""
Unit tests for named conditional leg targets (`targets:`) — feature: conditional
leg parameters, item 3.

A `targets` map holds named strike/expiry selections, each guarded by a
`condition` and an explicit `priority`. At each entry the true condition with the
highest priority wins; a reserved `default` fires when none match. The winning
target's name is tagged onto the position for per-target P&L attribution.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from unittest.mock import MagicMock

import polars as pl
import pytest
from pydantic import ValidationError

from btkit.backtest.entry import EntryScanner
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    LegTarget,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _leg(**kw):
    base = dict(name="sp", right="put", action="sell_to_open")
    base.update(kw)
    return LegConfig(**base)


class TestTargetsSchema:
    def test_target_requires_dte_and_delta(self):
        with pytest.raises(ValidationError):
            LegTarget(priority=10, condition="iv > 0.5", delta=-0.1)  # missing dte
        with pytest.raises(ValidationError):
            LegTarget(priority=10, condition="iv > 0.5", dte=1)  # missing delta

    def test_non_default_requires_condition_and_priority(self):
        with pytest.raises(ValidationError, match="requires a condition"):
            _leg(targets={"a": LegTarget(priority=10, dte=1, delta=-0.1)})
        with pytest.raises(ValidationError, match="requires a priority"):
            _leg(targets={"a": LegTarget(condition="iv > 0.5", dte=1, delta=-0.1)})

    def test_default_must_not_set_condition_or_priority(self):
        with pytest.raises(ValidationError, match="reserved 'default'"):
            _leg(
                targets={
                    "a": LegTarget(priority=10, condition="iv > 0.5", dte=1, delta=-0.1),
                    "default": LegTarget(priority=1, dte=1, delta=-0.1),
                }
            )

    def test_duplicate_priority_raises(self):
        with pytest.raises(ValidationError, match="priorities must be unique"):
            _leg(
                targets={
                    "a": LegTarget(priority=10, condition="iv > 0.5", dte=1, delta=-0.1),
                    "b": LegTarget(priority=10, condition="iv > 0.2", dte=2, delta=-0.2),
                }
            )

    def test_requires_at_least_one_conditional(self):
        with pytest.raises(ValidationError, match="at least one conditional"):
            _leg(targets={"default": LegTarget(dte=1, delta=-0.1)})

    def test_valid_targets_leg(self):
        leg = _leg(
            targets={
                "hi": LegTarget(priority=90, condition="iv >= 0.67", dte=0, delta=-0.05),
                "lo": LegTarget(priority=10, condition="iv < 0.33", dte=2, delta=-0.10),
                "default": LegTarget(dte=1, delta=-0.08),
            }
        )
        assert leg.targets is not None
        assert leg.dte is None and leg.delta is None


class TestTargetsMutualExclusion:
    def test_targets_with_delta_raises(self):
        with pytest.raises(ValidationError, match="targets is mutually exclusive with delta"):
            _leg(
                delta={"target": -0.2},
                targets={"a": LegTarget(priority=1, condition="iv > 0.5", dte=1, delta=-0.1)},
            )

    def test_targets_with_scalar_dte_raises(self):
        with pytest.raises(
            ValidationError, match="targets is mutually exclusive with leg-level dte"
        ):
            _leg(
                dte=5,
                targets={"a": LegTarget(priority=1, condition="iv > 0.5", dte=1, delta=-0.1)},
            )

    def test_targets_with_stepped_raises(self):
        from btkit.strategy.definition import SteppedLegConfig, SteppedStep

        with pytest.raises(ValidationError, match="mutually exclusive"):
            _leg(
                stepped=SteppedLegConfig(source="iv", steps=[SteppedStep(dte=1, delta=-0.1)]),
                targets={"a": LegTarget(priority=1, condition="iv > 0.5", dte=1, delta=-0.1)},
            )

    def test_at_most_one_targets_leg_per_trade(self):
        with pytest.raises(ValidationError, match="at most one leg per trade may use targets"):
            TradeDefinition(
                name="t",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(window=EntryWindowConfig(start=time(10, 0), end=time(12, 0))),
                legs=[
                    _leg(
                        name="a",
                        targets={"x": LegTarget(priority=1, condition="iv>0.5", dte=1, delta=-0.1)},
                    ),
                    _leg(
                        name="b",
                        targets={"y": LegTarget(priority=1, condition="iv>0.5", dte=1, delta=-0.1)},
                    ),
                ],
                exit=ExitConfig(stop_loss=2.0),
            )


# ---------------------------------------------------------------------------
# _build_targets_exprs priority resolution
# ---------------------------------------------------------------------------


def _scanner_with_targets(targets: dict) -> EntryScanner:
    trade = TradeDefinition(
        name="t",
        instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
        entry=EntryConfig(window=EntryWindowConfig(start=time(10, 0), end=time(12, 0))),
        legs=[
            LegConfig(
                name="sp", right="put", action="sell_to_open", dte_tolerance=5, targets=targets
            )
        ],
        exit=ExitConfig(stop_loss=2.0),
    )
    strategy = StrategyDefinition(
        name="test",
        universe=UniverseConfig(start_date=date(2024, 1, 1), end_date=date(2024, 3, 31)),
        trades=[trade],
    )
    return EntryScanner(MagicMock(), strategy, trade)


class TestBuildTargetsExprs:
    def test_priority_resolves_overlap(self):
        """When two conditions overlap, the higher priority wins regardless of map order."""
        scanner = _scanner_with_targets(
            {
                "lo": LegTarget(priority=10, condition="iv > 0.2", dte=2, delta=-0.10),
                "hi": LegTarget(priority=90, condition="iv > 0.5", dte=0, delta=-0.05),
                "default": LegTarget(dte=1, delta=-0.08),
            }
        )
        d, _, dte, _, name, _ = scanner._build_targets_exprs(scanner.trade.legs[0])
        df = pl.DataFrame({"iv": [0.6, 0.3, 0.1]})  # both / only-lo / none
        out = df.with_columns(d.alias("d"), dte.alias("dte"), name.alias("name"))
        assert out["name"].to_list() == ["hi", "lo", "default"]
        assert out["dte"].to_list() == [0, 2, 1]
        assert out["d"].to_list() == [-0.05, -0.10, -0.08]

    def test_no_match_no_default_is_null(self):
        scanner = _scanner_with_targets(
            {"hi": LegTarget(priority=90, condition="iv > 0.5", dte=0, delta=-0.05)}
        )
        d, _, dte, _, name, _ = scanner._build_targets_exprs(scanner.trade.legs[0])
        df = pl.DataFrame({"iv": [0.1]})
        out = df.with_columns(d.alias("d"), dte.alias("dte"), name.alias("name"))
        assert out["name"].to_list() == [None]
        assert out["dte"].to_list() == [None]
        assert out["d"].to_list() == [None]

    def test_size_multiplier_carried(self):
        scanner = _scanner_with_targets(
            {
                "hi": LegTarget(
                    priority=90, condition="iv > 0.5", dte=0, delta=-0.05, size_multiplier=0.35
                ),
                "default": LegTarget(dte=1, delta=-0.08),
            }
        )
        *_, size = scanner._build_targets_exprs(scanner.trade.legs[0])
        df = pl.DataFrame({"iv": [0.6, 0.1]})
        out = df.with_columns(size.alias("size"))
        assert out["size"].to_list() == [0.35, 1.0]


# ---------------------------------------------------------------------------
# _select_legs dispatch + tagging
# ---------------------------------------------------------------------------


class _CapturingDB:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def greeks_for_all_legs(self, ts_event_underlying, leg_specs, **kwargs) -> pl.DataFrame:
        self.calls.append(leg_specs)
        return pl.DataFrame()

    def greeks_for_strike_legs(self, *a, **k) -> pl.DataFrame:
        return pl.DataFrame()


class TestTargetsDispatch:
    def test_buckets_dispatch_distinct_targets(self):
        db = _CapturingDB()
        scanner = _scanner_with_targets(
            {
                "hi": LegTarget(
                    priority=90, condition="iv >= 0.67", dte=0, delta=-0.05, delta_tolerance=0.01
                ),
                "lo": LegTarget(
                    priority=10, condition="iv < 0.33", dte=2, delta=-0.10, delta_tolerance=0.015
                ),
                "default": LegTarget(dte=1, delta=-0.08, delta_tolerance=0.012),
            }
        )
        scanner.db = db
        ts1 = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
        ts2 = datetime(2024, 1, 2, 15, 1, tzinfo=UTC)
        ts3 = datetime(2024, 1, 2, 15, 2, tzinfo=UTC)
        candidates = pl.DataFrame(
            {
                "ts_event": pl.Series([ts1, ts2, ts3], dtype=pl.Datetime("us", "UTC")),
                "underlying_id": pl.Series([100, 100, 100], dtype=pl.Int64),
                "open": pl.Series([5000.0] * 3, dtype=pl.Float64),
                "high": pl.Series([5010.0] * 3, dtype=pl.Float64),
                "low": pl.Series([4990.0] * 3, dtype=pl.Float64),
                "close": pl.Series([5005.0] * 3, dtype=pl.Float64),
                "volume": pl.Series([1, 1, 1], dtype=pl.Int64),
                "iv": pl.Series([0.8, 0.2, 0.5], dtype=pl.Float64),  # hi / lo / default
            }
        )
        scanner._select_legs(candidates)
        got = {
            (s["target_dte"], s["target_delta"], s["delta_tolerance"])
            for specs in db.calls
            for s in specs
        }
        assert got == {(0, -0.05, 0.01), (2, -0.10, 0.015), (1, -0.08, 0.012)}
