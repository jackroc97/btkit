"""
Unit tests for the reference_leg schema: legacy sibling-field coercion,
the folded strike_offset/entry_time_tolerance object form, validation of
reference targets, anchor-chain rejection, and the time_tolerance ->
bar_time_tolerance compatibility shim.
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
    ReferenceLegConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)


def _strategy(legs: list[LegConfig], **entry_kwargs) -> StrategyDefinition:
    return StrategyDefinition(
        name="test",
        universe=UniverseConfig(start_date=date(2024, 1, 1), end_date=date(2024, 3, 31)),
        trades=[
            TradeDefinition(
                name="t1",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(
                    window=EntryWindowConfig(start=time(10, 0), end=time(12, 0)), **entry_kwargs
                ),
                legs=legs,
                exit=ExitConfig(stop_loss=2.0),
            )
        ],
    )


def _short_put() -> LegConfig:
    return LegConfig(
        name="short_put", right="put", action="sell_to_open", dte=21, delta={"target": -0.10}
    )


class TestLegacyCoercion:
    def test_legacy_sibling_fields_fold_into_reference_leg(self):
        leg = LegConfig(
            name="long_put",
            right="put",
            action="buy_to_open",
            reference_leg="short_put",
            strike_offset=-50.0,
        )
        assert isinstance(leg.reference_leg, ReferenceLegConfig)
        assert leg.reference_leg.name == "short_put"
        assert leg.reference_leg.strike_offset == -50.0
        assert leg.reference_leg.entry_time_tolerance is None

    def test_object_form_strike_offset(self):
        leg = LegConfig(
            name="long_put",
            right="put",
            action="buy_to_open",
            reference_leg={"name": "short_put", "strike_offset": -50.0},
        )
        assert leg.reference_leg.strike_offset == -50.0
        assert leg.reference_leg.entry_time_tolerance is None

    def test_object_form_with_entry_time_tolerance(self):
        leg = LegConfig(
            name="long_put",
            right="put",
            action="buy_to_open",
            reference_leg={
                "name": "short_put",
                "strike_offset": -50.0,
                "entry_time_tolerance": 120,
            },
        )
        assert leg.reference_leg.entry_time_tolerance == 120

    def test_strike_offset_without_reference_leg_rejected(self):
        with pytest.raises(ValidationError, match="strike_offset requires reference_leg"):
            LegConfig(name="leg", right="put", action="buy_to_open", strike_offset=-10.0)

    def test_bare_reference_leg_no_strike_offset_is_harmless(self):
        # A leg may reference another leg purely as a label / future time-anchor
        # without deriving its strike from it (no strike_offset at all).
        leg = LegConfig(
            name="long_put",
            right="put",
            action="buy_to_open",
            dte=21,
            delta={"target": -0.05},
            reference_leg="short_put",
        )
        assert leg.reference_leg == "short_put"


class TestReferenceLegValidation:
    def test_valid_offset_leg(self):
        strat = _strategy(
            [
                _short_put(),
                LegConfig(
                    name="long_put",
                    right="put",
                    action="buy_to_open",
                    reference_leg={"name": "short_put", "strike_offset": -50.0},
                ),
            ]
        )
        assert strat.trades[0].legs[1].reference_leg.strike_offset == -50.0

    def test_reference_to_unknown_leg_rejected(self):
        with pytest.raises(ValidationError, match="references 'nonexistent'"):
            _strategy(
                [
                    _short_put(),
                    LegConfig(
                        name="long_put",
                        right="put",
                        action="buy_to_open",
                        reference_leg={"name": "nonexistent", "strike_offset": -50.0},
                    ),
                ]
            )

    def test_reference_to_offset_leg_rejected(self):
        with pytest.raises(ValidationError, match="strike_offset legs cannot be referenced"):
            _strategy(
                [
                    _short_put(),
                    LegConfig(
                        name="long_put",
                        right="put",
                        action="buy_to_open",
                        reference_leg={"name": "short_put", "strike_offset": -50.0},
                    ),
                    LegConfig(
                        name="farther_put",
                        right="put",
                        action="buy_to_open",
                        reference_leg={"name": "long_put", "strike_offset": -25.0},
                    ),
                ]
            )

    def test_anchor_chain_rejected(self):
        with pytest.raises(ValidationError, match="anchor chains are not supported"):
            _strategy(
                [
                    _short_put(),
                    LegConfig(
                        name="mid_put",
                        right="put",
                        action="buy_to_open",
                        dte=21,
                        delta={"target": -0.07},
                        reference_leg={"name": "short_put", "entry_time_tolerance": 30},
                    ),
                    LegConfig(
                        name="far_put",
                        right="put",
                        action="buy_to_open",
                        dte=21,
                        delta={"target": -0.05},
                        reference_leg={"name": "mid_put", "entry_time_tolerance": 30},
                    ),
                ]
            )

    def test_chain_via_bare_reference_is_allowed(self):
        # mid_put has only a bare (no entry_time_tolerance) reference to
        # short_put, so it's still bar-anchored — far_put can validly anchor
        # to it via entry_time_tolerance.
        strat = _strategy(
            [
                _short_put(),
                LegConfig(
                    name="mid_put",
                    right="put",
                    action="buy_to_open",
                    dte=21,
                    delta={"target": -0.07},
                    reference_leg="short_put",
                ),
                LegConfig(
                    name="far_put",
                    right="put",
                    action="buy_to_open",
                    dte=21,
                    delta={"target": -0.05},
                    reference_leg={"name": "mid_put", "entry_time_tolerance": 30},
                ),
            ]
        )
        assert strat.trades[0].legs[2].reference_leg.entry_time_tolerance == 30


class TestBarTimeToleranceCompat:
    def test_legacy_time_tolerance_maps_to_bar_time_tolerance(self):
        ec = EntryConfig(
            window=EntryWindowConfig(start=time(9, 30), end=time(16, 0)), time_tolerance=300
        )
        assert ec.bar_time_tolerance == 300

    def test_both_keys_set_raises(self):
        with pytest.raises(ValidationError, match="not both"):
            EntryConfig(
                window=EntryWindowConfig(start=time(9, 30), end=time(16, 0)),
                time_tolerance=300,
                bar_time_tolerance=100,
            )

    def test_default_is_zero(self):
        ec = EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(16, 0)))
        assert ec.bar_time_tolerance == 0
