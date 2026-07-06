"""
Unit tests for the unified `stepped:` leg block (feature: conditional leg
parameters, item 2).

A `stepped:` block emits the full (dte, delta, delta_tolerance, dte_tolerance)
tuple per bucket from a single indicator source, so a leg's DTE *and* delta can
both vary by market state. Covers schema validation, per-bucket expression
lowering, and per-bucket dispatch into the leg-selection query.
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
    SteppedLegConfig,
    SteppedStep,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSteppedSchema:
    def test_step_requires_dte_and_delta(self):
        with pytest.raises(ValidationError):
            SteppedStep(below=0.33, delta=-0.10)  # missing dte
        with pytest.raises(ValidationError):
            SteppedStep(below=0.33, dte=2)  # missing delta

    def test_empty_steps_raises(self):
        with pytest.raises(ValidationError, match="steps must not be empty"):
            SteppedLegConfig(source="iv", steps=[])

    def test_multiple_catch_alls_raises(self):
        with pytest.raises(ValidationError, match="at most one catch-all"):
            SteppedLegConfig(
                source="iv",
                steps=[SteppedStep(dte=1, delta=-0.1), SteppedStep(dte=2, delta=-0.2)],
            )

    def test_catch_all_not_last_raises(self):
        with pytest.raises(ValidationError, match="catch-all.*must be last"):
            SteppedLegConfig(
                source="iv",
                steps=[
                    SteppedStep(dte=1, delta=-0.1),  # catch-all first
                    SteppedStep(below=0.5, dte=2, delta=-0.2),
                ],
            )

    def test_valid_stepped_block(self):
        cfg = SteppedLegConfig(
            source="iv_percentile",
            steps=[
                SteppedStep(below=0.33, dte=2, delta=-0.10, delta_tolerance=0.015),
                SteppedStep(dte=1, delta=-0.07),
            ],
        )
        assert cfg.source == "iv_percentile"
        assert len(cfg.steps) == 2


class TestSteppedLegMutualExclusion:
    def _leg(self, **kw):
        base = dict(name="sp", right="put", action="sell_to_open")
        base.update(kw)
        return LegConfig(**base)

    def test_stepped_only_is_valid(self):
        leg = self._leg(
            stepped=SteppedLegConfig(source="iv", steps=[SteppedStep(dte=1, delta=-0.1)])
        )
        assert leg.stepped is not None
        assert leg.delta is None
        assert leg.dte is None

    def test_stepped_with_scalar_dte_raises(self):
        with pytest.raises(
            ValidationError, match="stepped is mutually exclusive with leg-level dte"
        ):
            self._leg(
                dte=5,
                stepped=SteppedLegConfig(source="iv", steps=[SteppedStep(dte=1, delta=-0.1)]),
            )

    def test_stepped_with_delta_raises(self):
        with pytest.raises(ValidationError, match="stepped is mutually exclusive with delta"):
            self._leg(
                delta={"target": -0.2},
                stepped=SteppedLegConfig(source="iv", steps=[SteppedStep(dte=1, delta=-0.1)]),
            )

    def test_stepped_with_strike_offset_raises(self):
        with pytest.raises(ValidationError, match="stepped is mutually exclusive"):
            self._leg(
                strike_offset=-50.0,
                reference_leg="x",
                stepped=SteppedLegConfig(source="iv", steps=[SteppedStep(dte=1, delta=-0.1)]),
            )


# ---------------------------------------------------------------------------
# _build_stepped_leg_exprs lowering
# ---------------------------------------------------------------------------


def _scanner_with_stepped(steps: list[SteppedStep], dte_tolerance: int = 5) -> EntryScanner:
    trade = TradeDefinition(
        name="t",
        instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
        entry=EntryConfig(window=EntryWindowConfig(start=time(10, 0), end=time(12, 0))),
        legs=[
            LegConfig(
                name="sp",
                right="put",
                action="sell_to_open",
                dte_tolerance=dte_tolerance,
                stepped=SteppedLegConfig(source="iv", steps=steps),
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


class TestBuildSteppedLegExprs:
    def test_buckets_resolve_full_tuple(self):
        scanner = _scanner_with_stepped(
            steps=[
                SteppedStep(below=0.33, dte=2, delta=-0.10, delta_tolerance=0.015),
                SteppedStep(
                    below=0.67, dte=5, delta=-0.085, delta_tolerance=0.012, dte_tolerance=3
                ),
                SteppedStep(dte=1, delta=-0.07),  # catch-all
            ]
        )
        d, dtol, dte, dtetol = scanner._build_stepped_leg_exprs(scanner.trade.legs[0])
        df = pl.DataFrame({"iv": [0.2, 0.5, 0.9]})
        out = df.with_columns(
            d.alias("d"), dtol.alias("dtol"), dte.alias("dte"), dtetol.alias("dtetol")
        )
        assert out["d"].to_list() == [-0.10, -0.085, -0.07]
        assert out["dtol"].to_list() == [0.015, 0.012, 0.10]  # catch-all → 0.10 default
        assert out["dte"].to_list() == [2, 5, 1]
        assert out["dtetol"].to_list() == [5, 3, 5]  # step1 & catch-all → leg default 5

    def test_no_match_no_catch_all_is_null(self):
        scanner = _scanner_with_stepped(steps=[SteppedStep(below=0.33, dte=2, delta=-0.10)])
        d, _, dte, _ = scanner._build_stepped_leg_exprs(scanner.trade.legs[0])
        df = pl.DataFrame({"iv": [0.9]})
        out = df.with_columns(d.alias("d"), dte.alias("dte"))
        assert out["d"].to_list() == [None]
        assert out["dte"].to_list() == [None]

    def test_null_source_propagates_null(self):
        scanner = _scanner_with_stepped(steps=[SteppedStep(below=0.33, dte=2, delta=-0.10)])
        d, _, dte, _ = scanner._build_stepped_leg_exprs(scanner.trade.legs[0])
        df = pl.DataFrame({"iv": pl.Series([None], dtype=pl.Float64)})
        out = df.with_columns(d.alias("d"), dte.alias("dte"))
        assert out["d"].to_list() == [None]
        assert out["dte"].to_list() == [None]


# ---------------------------------------------------------------------------
# _select_legs per-bucket dispatch
# ---------------------------------------------------------------------------


class _CapturingDB:
    """Records the leg_specs passed to greeks_for_all_legs; returns empty."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def greeks_for_all_legs(self, ts_event_underlying, leg_specs, **kwargs) -> pl.DataFrame:
        self.calls.append(leg_specs)
        return pl.DataFrame()

    def greeks_for_strike_legs(self, *a, **k) -> pl.DataFrame:
        return pl.DataFrame()


class TestSteppedDispatch:
    def test_two_buckets_dispatch_distinct_targets(self):
        """Candidates in two source buckets produce two queries with each bucket's targets."""
        db = _CapturingDB()
        scanner = _scanner_with_stepped(
            steps=[
                SteppedStep(below=0.33, dte=2, delta=-0.10, delta_tolerance=0.015),
                SteppedStep(below=0.67, dte=5, delta=-0.085, delta_tolerance=0.012),
                SteppedStep(dte=1, delta=-0.07),
            ]
        )
        scanner.db = db

        ts1 = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
        ts2 = datetime(2024, 1, 2, 15, 1, tzinfo=UTC)
        candidates = pl.DataFrame(
            {
                "ts_event": pl.Series([ts1, ts2], dtype=pl.Datetime("us", "UTC")),
                "underlying_id": pl.Series([100, 100], dtype=pl.Int64),
                "open": pl.Series([5000.0, 5000.0], dtype=pl.Float64),
                "high": pl.Series([5010.0, 5010.0], dtype=pl.Float64),
                "low": pl.Series([4990.0, 4990.0], dtype=pl.Float64),
                "close": pl.Series([5005.0, 5005.0], dtype=pl.Float64),
                "volume": pl.Series([1, 1], dtype=pl.Int64),
                "iv": pl.Series([0.2, 0.5], dtype=pl.Float64),  # bucket 1 and bucket 2
            }
        )
        scanner._select_legs(candidates)

        # One spec dict per query; collect (target_dte, target_delta, tol, dte_tol)
        got = {
            (s["target_dte"], s["target_delta"], s["delta_tolerance"], s["dte_tolerance"])
            for specs in db.calls
            for s in specs
        }
        assert got == {
            (2, -0.10, 0.015, 5),
            (5, -0.085, 0.012, 5),
        }
