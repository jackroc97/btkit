"""
Integration tests for conditional leg selection (feature: conditional leg
parameters) against the pre-built fixture database.

Covers the acceptance criteria that need real option data:
  - Item 2: a `stepped:` leg selects options at different (dte, delta) per bucket.
  - Item 3: a `targets:` leg reproduces the positions of the equivalent
    N-gated single-leg strategies (selection parity), tagged per target.

Requires tests/output/input.db (the conftest input_db fixture skips otherwise).
The fixture provides per-minute sma_5 / sma_20 indicators over ES.
"""

from __future__ import annotations

from datetime import date, time

import polars as pl

from btkit.backtest.entry import EntryScanner
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    LegTarget,
    SteppedLegConfig,
    SteppedStep,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)

START = date(2026, 4, 22)
END = date(2026, 5, 21)


def _trade(legs, conditions=None):
    return TradeDefinition(
        name="t",
        instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
        entry=EntryConfig(
            window=EntryWindowConfig(start=time(10, 0), end=time(12, 0)),
            conditions=conditions or [],
        ),
        legs=legs,
        exit=ExitConfig(stop_loss=2.0),
    )


def _scan(input_db, trade):
    strat = StrategyDefinition(
        name="s", universe=UniverseConfig(start_date=START, end_date=END), trades=[trade]
    )
    return EntryScanner(input_db, strat, trade).scan()


class TestSteppedDteEndToEnd:
    def test_dte_bifurcates_by_bucket(self, input_db):
        """A stepped leg selects lower-DTE options in the low-sma bucket."""
        stepped = SteppedLegConfig(
            source="sma_5",
            steps=[
                SteppedStep(below=7300.0, dte=7, delta=-0.20, delta_tolerance=0.10),
                SteppedStep(dte=21, delta=-0.20, delta_tolerance=0.10),  # catch-all
            ],
        )
        trade = _trade(
            [
                LegConfig(
                    name="sp", right="put", action="sell_to_open", dte_tolerance=5, stepped=stepped
                )
            ]
        )
        res = _scan(input_db, trade)
        assert len(res) > 0
        res = res.with_columns((pl.col("sma_5") < 7300.0).alias("low"))
        by = {
            r["low"]: r
            for r in res.group_by("low")
            .agg(pl.col("leg_sp_dte").mean().alias("dte"))
            .iter_rows(named=True)
        }
        # low-sma bucket targets dte 7; high bucket targets dte 21
        assert by[True]["dte"] < by[False]["dte"]
        assert by[True]["dte"] < 12  # near the 7 target (± tolerance)
        assert by[False]["dte"] > 15  # near the 21 target


class TestTargetsParity:
    def test_targets_reproduces_gated_selection(self, input_db):
        """A single targets leg selects exactly what the equivalent gated
        single-leg strategies select, per bucket, and tags each position."""
        conds = {
            "lo": "sma_5 < 7250",
            "mid": "sma_5 >= 7250 and sma_5 < 7450",
            "hi": "sma_5 >= 7450",
        }
        params = {"lo": (5, -0.15), "mid": (10, -0.20), "hi": (20, -0.25)}
        prio = {"lo": 30, "mid": 20, "hi": 10}

        targets = {
            n: LegTarget(
                priority=prio[n],
                condition=conds[n],
                dte=params[n][0],
                delta=params[n][1],
                delta_tolerance=0.10,
            )
            for n in conds
        }
        trade_a = _trade(
            [
                LegConfig(
                    name="sp", right="put", action="sell_to_open", dte_tolerance=5, targets=targets
                )
            ]
        )
        a = _scan(input_db, trade_a)
        assert set(a["_target_name"].unique().to_list()) == set(conds)

        for name in conds:
            dte, delta = params[name]
            leg = LegConfig(
                name="sp",
                right="put",
                action="sell_to_open",
                dte=dte,
                delta={"target": delta, "tolerance": 0.10},
                dte_tolerance=5,
            )
            b = _scan(input_db, _trade([leg], conditions=[conds[name]]))
            a_bucket = (
                a.filter(pl.col("_target_name") == name)
                .select(["entry_time", "leg_sp_instrument_id", "leg_sp_dte"])
                .sort("entry_time")
            )
            b_sel = b.select(["entry_time", "leg_sp_instrument_id", "leg_sp_dte"]).sort(
                "entry_time"
            )
            assert a_bucket.equals(b_sel), f"bucket {name} selection differs from gated equivalent"
