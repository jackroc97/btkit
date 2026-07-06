"""
Unit tests for ExitScanner._compute_settlement_marks.

All tests use hand-constructed DataFrames and a mocked InputDatabase so that
no real DB access is needed. The method returns [entry_id, settlement_mark]
where settlement_mark = Σ intrinsic_per_leg × signed_qty.

Intrinsic value:
    call:  max(0, S − strike)
    put:   max(0, strike − S)

Signed quantity:
    sell_to_open  → +qty
    buy_to_open   → −qty
"""

from __future__ import annotations

from datetime import date, time
from unittest.mock import MagicMock

import polars as pl
import pytest

from btkit.backtest.exit import ExitScanner
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    SessionConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TZ = "America/New_York"
EXP_DATE = date(2026, 4, 9)

# Stable fake option instrument IDs used across tests
SHORT_OPT_ID = 1001
LONG_OPT_ID  = 1002
UNDERLYING_ID = 999


def _make_scanner(legs: list[LegConfig], expiry_close_time: time | None = time(16, 0)) -> ExitScanner:
    strategy = StrategyDefinition(
        name="test",
        universe=UniverseConfig(
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 30),
            session=SessionConfig(timezone=TZ),
        ),
        trades=[
            TradeDefinition(
                name="spread",
                instrument=InstrumentConfig(
                    root_symbol="ES",
                    asset_class="future",
                    expiry_close_time=expiry_close_time,
                    tick_size=0.05,
                ),
                entry=EntryConfig(
                    window=EntryWindowConfig(start=time(9, 30), end=time(10, 0))
                ),
                legs=legs,
                exit=ExitConfig(expiry_exit=True),
            )
        ],
    )
    return ExitScanner(db=MagicMock(), strategy=strategy, trade=strategy.trades[0])


def _entries(
    entry_id: int,
    exp_date: date,
    short_strike: float,
    long_strike: float,
    right: str = "P",
    short_opt_id: int = SHORT_OPT_ID,
    long_opt_id: int = LONG_OPT_ID,
) -> pl.DataFrame:
    """Minimal entries DataFrame for a 2-leg spread."""
    right_upper = right.upper()
    return pl.DataFrame(
        {
            "entry_id": pl.Series([entry_id], dtype=pl.UInt32),
            "leg_short_instrument_id": pl.Series([short_opt_id], dtype=pl.Int64),
            "leg_short_strike_price": pl.Series([short_strike], dtype=pl.Float64),
            "leg_short_right": pl.Series([right_upper], dtype=pl.Utf8),
            "leg_short_expiration": pl.Series([exp_date], dtype=pl.Date),
            "leg_long_instrument_id": pl.Series([long_opt_id], dtype=pl.Int64),
            "leg_long_strike_price": pl.Series([long_strike], dtype=pl.Float64),
            "leg_long_right": pl.Series([right_upper], dtype=pl.Utf8),
            "leg_long_expiration": pl.Series([exp_date], dtype=pl.Date),
        }
    )



def _scanner_with_settlement(legs: list[LegConfig], settlement: float) -> ExitScanner:
    scanner = _make_scanner(legs)
    scanner._opt_to_underlying = {SHORT_OPT_ID: UNDERLYING_ID}
    scanner._settlement_closes_by_key = {(UNDERLYING_ID, EXP_DATE): settlement}
    return scanner


# ---------------------------------------------------------------------------
# Put credit spread
# ---------------------------------------------------------------------------

PUT_LEGS = [
    LegConfig(name="short", right="put", action="sell_to_open", dte=0, delta={"target": -0.25}),
    LegConfig(name="long", right="put", action="buy_to_open", strike_offset=-50.0, reference_leg="short"),
]


class TestPutCreditSpread:
    def test_both_otm_mark_is_zero(self):
        """Underlying above both strikes → both OTM → mark = 0."""
        scanner = _scanner_with_settlement(PUT_LEGS, settlement=5500.0)
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5350.0, right="P")
        result = scanner._compute_settlement_marks(entries, TZ)
        assert result.filter(pl.col("entry_id") == 1)["settlement_mark"][0] == pytest.approx(0.0)

    def test_both_itm_mark_equals_spread_width(self):
        """Underlying below both strikes → both ITM → mark = spread width."""
        scanner = _scanner_with_settlement(PUT_LEGS, settlement=5300.0)
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5350.0, right="P")
        result = scanner._compute_settlement_marks(entries, TZ)
        # STO 5400P: max(0, 5400-5300)=100, BTO 5350P: max(0,5350-5300)=50
        # mark = +100 -50 = 50 = spread width
        assert result.filter(pl.col("entry_id") == 1)["settlement_mark"][0] == pytest.approx(50.0)

    def test_partial_itm_short_only(self):
        """Underlying between strikes → only STO ITM → mark = STO intrinsic."""
        scanner = _scanner_with_settlement(PUT_LEGS, settlement=5375.0)
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5350.0, right="P")
        result = scanner._compute_settlement_marks(entries, TZ)
        # STO 5400P: max(0, 5400-5375)=25, BTO 5350P: max(0,5350-5375)=0
        assert result.filter(pl.col("entry_id") == 1)["settlement_mark"][0] == pytest.approx(25.0)

    def test_mark_never_negative(self):
        """settlement_mark is always >= 0 for a credit spread at any underlying price."""
        scanner = _make_scanner(PUT_LEGS)
        for underlying in [5200, 5350, 5375, 5400, 5450, 5600]:
            scanner._opt_to_underlying = {SHORT_OPT_ID: UNDERLYING_ID}
            scanner._settlement_closes_by_key = {(UNDERLYING_ID, EXP_DATE): float(underlying)}
            entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5350.0, right="P")
            result = scanner._compute_settlement_marks(entries, TZ)
            mark = result["settlement_mark"][0]
            assert mark >= 0.0, f"Negative mark {mark} at underlying {underlying}"


# ---------------------------------------------------------------------------
# Call credit spread
# ---------------------------------------------------------------------------

CALL_LEGS = [
    LegConfig(name="short", right="call", action="sell_to_open", dte=0, delta={"target": 0.25}),
    LegConfig(name="long", right="call", action="buy_to_open", strike_offset=50.0, reference_leg="short"),
]


class TestCallCreditSpread:
    def test_both_otm_mark_is_zero(self):
        """Underlying below both strikes → both OTM → mark = 0."""
        scanner = _scanner_with_settlement(CALL_LEGS, settlement=5300.0)
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5450.0, right="C")
        result = scanner._compute_settlement_marks(entries, TZ)
        assert result.filter(pl.col("entry_id") == 1)["settlement_mark"][0] == pytest.approx(0.0)

    def test_both_itm_mark_equals_spread_width(self):
        """Underlying above both strikes → both ITM → mark = spread width."""
        scanner = _scanner_with_settlement(CALL_LEGS, settlement=5500.0)
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5450.0, right="C")
        result = scanner._compute_settlement_marks(entries, TZ)
        # STO 5400C: max(0,5500-5400)=100, BTO 5450C: max(0,5500-5450)=50
        # mark = +100 -50 = 50 = spread width
        assert result.filter(pl.col("entry_id") == 1)["settlement_mark"][0] == pytest.approx(50.0)

    def test_partial_itm_short_only(self):
        """Underlying between strikes → only STO ITM → mark = STO intrinsic."""
        scanner = _scanner_with_settlement(CALL_LEGS, settlement=5425.0)
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5450.0, right="C")
        result = scanner._compute_settlement_marks(entries, TZ)
        # STO 5400C: 25, BTO 5450C: 0 → mark = 25
        assert result.filter(pl.col("entry_id") == 1)["settlement_mark"][0] == pytest.approx(25.0)

    def test_bug_trade_exact_values(self):
        """Reproduce the specific bug trade: STO 5320C / BTO 5370C, ES at 5498."""
        scanner = _scanner_with_settlement(CALL_LEGS, settlement=5498.0)
        entries = _entries(1, EXP_DATE, short_strike=5320.0, long_strike=5370.0, right="C")
        result = scanner._compute_settlement_marks(entries, TZ)
        # STO 5320C: 178, BTO 5370C: 128 → mark = 50 (spread width = max loss)
        assert result.filter(pl.col("entry_id") == 1)["settlement_mark"][0] == pytest.approx(50.0)

    def test_does_not_call_roll_schedule(self):
        """_compute_settlement_marks reads from cache — does not call front_future_id."""
        scanner = _scanner_with_settlement(CALL_LEGS, settlement=5498.0)
        entries = _entries(1, EXP_DATE, short_strike=5320.0, long_strike=5370.0, right="C")
        scanner._compute_settlement_marks(entries, TZ)
        scanner.db.front_future_id.assert_not_called()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSettlementEdgeCases:
    def test_no_underlying_bar_returns_null(self):
        """When no settlement close is cached for the expiry date, settlement_mark is null."""
        scanner = _make_scanner(PUT_LEGS)
        scanner._opt_to_underlying = {SHORT_OPT_ID: UNDERLYING_ID}
        scanner._settlement_closes_by_key = {}  # no entry for (UNDERLYING_ID, EXP_DATE)
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5350.0, right="P")
        result = scanner._compute_settlement_marks(entries, TZ)
        assert result["settlement_mark"][0] is None

    def test_no_underlying_id_mapping_returns_null(self):
        """When no option→underlying mapping is cached, settlement_mark is null."""
        scanner = _make_scanner(PUT_LEGS)
        scanner._opt_to_underlying = {}
        scanner._settlement_closes_by_key = {}
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5350.0, right="P")
        result = scanner._compute_settlement_marks(entries, TZ)
        assert result["settlement_mark"][0] is None

    def test_no_expiry_close_time_returns_null(self):
        """When instrument has no expiry_close_time, settlement_mark is null."""
        scanner = _make_scanner(PUT_LEGS, expiry_close_time=None)
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5350.0, right="P")
        result = scanner._compute_settlement_marks(entries, TZ)
        assert result["settlement_mark"][0] is None

    def test_multiple_entries_different_expirations(self):
        """Each entry uses its own expiration date's settlement price."""
        exp1 = date(2026, 4, 9)
        exp2 = date(2026, 4, 16)
        opt_id_1, opt_id_2 = 2001, 2002
        underlying_1, underlying_2 = 991, 992

        scanner = _make_scanner(PUT_LEGS)
        scanner._opt_to_underlying = {opt_id_1: underlying_1, opt_id_2: underlying_2}
        scanner._settlement_closes_by_key = {
            (underlying_1, exp1): 5500.0,
            (underlying_2, exp2): 5300.0,
        }

        entries = pl.DataFrame(
            {
                "entry_id": pl.Series([1, 2], dtype=pl.UInt32),
                "leg_short_instrument_id": pl.Series([opt_id_1, opt_id_2], dtype=pl.Int64),
                "leg_short_strike_price": pl.Series([5400.0, 5400.0], dtype=pl.Float64),
                "leg_short_right": pl.Series(["P", "P"], dtype=pl.Utf8),
                "leg_short_expiration": pl.Series([exp1, exp2], dtype=pl.Date),
                "leg_long_instrument_id": pl.Series([opt_id_1 + 1, opt_id_2 + 1], dtype=pl.Int64),
                "leg_long_strike_price": pl.Series([5350.0, 5350.0], dtype=pl.Float64),
                "leg_long_right": pl.Series(["P", "P"], dtype=pl.Utf8),
                "leg_long_expiration": pl.Series([exp1, exp2], dtype=pl.Date),
            }
        )

        result = scanner._compute_settlement_marks(entries, TZ)
        # exp1: underlying 991, S=5500 → both OTM → mark=0
        assert result.filter(pl.col("entry_id") == 1)["settlement_mark"][0] == pytest.approx(0.0)
        # exp2: underlying 992, S=5300 → both ITM → mark=50
        assert result.filter(pl.col("entry_id") == 2)["settlement_mark"][0] == pytest.approx(50.0)

    def test_settlement_close_passed_through_from_cache(self):
        """Whatever settlement close is in the cache is used — no further DB calls."""
        scanner = _make_scanner(PUT_LEGS)
        # Cache populated with 5498 (simulates settlement_closes_for_underlyings result)
        scanner._opt_to_underlying = {SHORT_OPT_ID: UNDERLYING_ID}
        scanner._settlement_closes_by_key = {(UNDERLYING_ID, EXP_DATE): 5498.0}
        entries = _entries(1, EXP_DATE, short_strike=5400.0, long_strike=5350.0, right="P")
        result = scanner._compute_settlement_marks(entries, TZ)
        # S=5498 → both strikes OTM → mark=0
        assert result["settlement_mark"][0] == pytest.approx(0.0)
        # No DB queries should be made by _compute_settlement_marks
        scanner.db.underlying_bars.assert_not_called()
        scanner.db.underlying_ids_for_options.assert_not_called()

    def test_first_leg_opt_id_used_for_underlying_lookup(self):
        """The first-leg instrument ID is used to key into the opt-to-underlying cache."""
        scanner = _make_scanner(PUT_LEGS)
        # Only populate cache with a mapping for SHORT_OPT_ID (first leg)
        scanner._opt_to_underlying = {SHORT_OPT_ID: UNDERLYING_ID}
        scanner._settlement_closes_by_key = {(UNDERLYING_ID, EXP_DATE): 5450.0}
        entries = _entries(
            1, EXP_DATE, short_strike=5400.0, long_strike=5350.0,
            right="P", short_opt_id=SHORT_OPT_ID,
        )
        result = scanner._compute_settlement_marks(entries, TZ)
        # S=5450 → both OTM → mark=0
        assert result["settlement_mark"][0] == pytest.approx(0.0)
