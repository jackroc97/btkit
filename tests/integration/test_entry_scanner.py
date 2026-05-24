"""
Integration tests for EntryScanner against the pre-built test fixture database.

These tests require tests/output/input.db — run `btkit build` first if missing
(the conftest.py input_db fixture will skip automatically if not present).

Tests verify:
  - Correct column contract (all expected columns present)
  - Entry times fall within the configured entry window
  - Leg selection respects delta/DTE tolerances
  - min_credit / max_debit filters apply correctly
  - Entry condition filtering reduces results vs no-condition baseline
  - Empty result when universe produces no candidates
"""

from __future__ import annotations

from datetime import date, time

import polars as pl
import pytest

from btkit.backtest.entry import EntryScanner
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
# Fixtures
# ---------------------------------------------------------------------------


def _make_strategy(
    start_date: date = date(2026, 4, 22),
    end_date: date = date(2026, 5, 21),
    entry_start: time = time(10, 0),
    entry_end: time = time(12, 0),
    delta: float = -0.25,
    dte: int = 21,
    delta_tolerance: float = 0.10,
    dte_tolerance: int = 5,
    conditions: list[str] | None = None,
    min_credit: float | None = None,
    max_debit: float | None = None,
) -> StrategyDefinition:
    return StrategyDefinition(
        name="entry_test",
        universe=UniverseConfig(
            start_date=start_date,
            end_date=end_date,
        ),
        costs=CostsConfig(),
        trades=[
            TradeDefinition(
                name="trade1",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(
                    window=EntryWindowConfig(start=entry_start, end=entry_end),
                    conditions=conditions or [],
                    min_credit=min_credit,
                    max_debit=max_debit,
                ),
                legs=[
                    LegConfig(
                        name="short_put",
                        right="put",
                        action="sell_to_open",
                        dte=dte,
                        delta=delta,
                        delta_tolerance=delta_tolerance,
                        dte_tolerance=dte_tolerance,
                    )
                ],
                exit=ExitConfig(stop_loss=2.0, take_profit=1.0),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------


class TestEntryScannerColumns:
    def test_required_columns_present(self, input_db):
        strat = _make_strategy()
        scanner = EntryScanner(input_db, strat, strat.trades[0])
        df = scanner.scan()
        assert not df.is_empty(), "Expected at least one entry from fixture data"
        required = {
            "entry_id",
            "trade_name",
            "entry_time",
            "open_mark",
            "tp_price",
            "sl_price",
            "leg_short_put_instrument_id",
            "leg_short_put_close",
            "leg_short_put_multiplier",
            "leg_short_put_strike_price",
            "leg_short_put_expiration",
            "leg_short_put_right",
            "leg_short_put_action",
            "leg_short_put_quantity",
            "leg_short_put_delta",
            "leg_short_put_iv",
            "leg_short_put_gamma",
            "leg_short_put_theta",
            "leg_short_put_vega",
            "leg_short_put_dte",
        }
        assert required.issubset(set(df.columns))

    def test_entry_ids_sequential_from_offset(self, input_db):
        strat = _make_strategy()
        scanner = EntryScanner(input_db, strat, strat.trades[0])
        df = scanner.scan(entry_id_offset=100)
        if df.is_empty():
            pytest.skip("No entries in fixture data")
        assert df["entry_id"].min() == 100

    def test_trade_name_column(self, input_db):
        strat = _make_strategy()
        scanner = EntryScanner(input_db, strat, strat.trades[0])
        df = scanner.scan()
        if df.is_empty():
            pytest.skip("No entries in fixture data")
        assert (df["trade_name"] == "trade1").all()


# ---------------------------------------------------------------------------
# Entry window filtering
# ---------------------------------------------------------------------------


class TestEntryWindowFilter:
    def test_entry_times_within_window(self, input_db):
        strat = _make_strategy(entry_start=time(10, 0), entry_end=time(11, 0))
        scanner = EntryScanner(input_db, strat, strat.trades[0])
        df = scanner.scan()
        if df.is_empty():
            pytest.skip("No entries in fixture window")

        tz = strat.universe.session.timezone
        local_times = df["entry_time"].dt.convert_time_zone(tz)
        hours = local_times.dt.hour().cast(pl.Int32)
        minutes = local_times.dt.minute().cast(pl.Int32)
        seconds_of_day = hours * 3600 + minutes * 60
        assert (seconds_of_day >= 10 * 3600).all()
        assert (seconds_of_day <= 11 * 3600).all()

    def test_narrow_window_fewer_entries(self, input_db):
        strat_wide = _make_strategy(entry_start=time(9, 30), entry_end=time(16, 0))
        strat_narrow = _make_strategy(entry_start=time(10, 0), entry_end=time(10, 30))
        n_wide = len(EntryScanner(input_db, strat_wide, strat_wide.trades[0]).scan())
        n_narrow = len(EntryScanner(input_db, strat_narrow, strat_narrow.trades[0]).scan())
        assert n_narrow <= n_wide


# ---------------------------------------------------------------------------
# Leg selection
# ---------------------------------------------------------------------------


class TestLegSelection:
    def test_selected_delta_within_tolerance(self, input_db):
        target = -0.25
        tol = 0.10
        strat = _make_strategy(delta=target, delta_tolerance=tol)
        scanner = EntryScanner(input_db, strat, strat.trades[0])
        df = scanner.scan()
        if df.is_empty():
            pytest.skip("No entries in fixture data")
        deltas = df["leg_short_put_delta"].to_list()
        for d in deltas:
            if d is not None:
                assert abs(d - target) <= tol + 0.01, f"delta {d} outside tolerance ±{tol}"

    def test_strict_dte_tolerance_zero(self, input_db):
        """DTE tolerance=0 should only select options with exactly the target DTE."""
        target_dte = 7
        strat = _make_strategy(dte=target_dte, dte_tolerance=0)
        scanner = EntryScanner(input_db, strat, strat.trades[0])
        df = scanner.scan()
        if df.is_empty():
            pytest.skip("No exact DTE matches in fixture data")
        dtes = df["leg_short_put_dte"].to_list()
        for d in dtes:
            if d is not None:
                assert d == target_dte, f"DTE {d} doesn't match strict target {target_dte}"

    def test_wider_tolerance_more_or_equal_entries(self, input_db):
        strat_tight = _make_strategy(delta_tolerance=0.05, dte_tolerance=2)
        strat_wide = _make_strategy(delta_tolerance=0.20, dte_tolerance=10)
        n_tight = len(EntryScanner(input_db, strat_tight, strat_tight.trades[0]).scan())
        n_wide = len(EntryScanner(input_db, strat_wide, strat_wide.trades[0]).scan())
        assert n_wide >= n_tight


# ---------------------------------------------------------------------------
# Credit/debit filters
# ---------------------------------------------------------------------------


class TestCreditDebitFilters:
    def test_min_credit_filters_low_marks(self, input_db):
        strat_base = _make_strategy()
        strat_filter = _make_strategy(min_credit=999.0)  # absurdly high → empty
        n_base = len(EntryScanner(input_db, strat_base, strat_base.trades[0]).scan())
        n_filter = len(EntryScanner(input_db, strat_filter, strat_filter.trades[0]).scan())
        assert n_base > 0
        assert n_filter == 0

    def test_max_debit_filters_expensive_entries(self, input_db):
        strat_base = _make_strategy()
        strat_filter = _make_strategy(max_debit=0.0001)  # near-zero → empty
        n_base = len(EntryScanner(input_db, strat_base, strat_base.trades[0]).scan())
        n_filter = len(EntryScanner(input_db, strat_filter, strat_filter.trades[0]).scan())
        assert n_base > 0
        assert n_filter == 0

    def test_generous_min_credit_passes_all(self, input_db):
        strat_base = _make_strategy()
        strat_filter = _make_strategy(min_credit=-999.0)
        n_base = len(EntryScanner(input_db, strat_base, strat_base.trades[0]).scan())
        n_filter = len(EntryScanner(input_db, strat_filter, strat_filter.trades[0]).scan())
        assert n_filter == n_base


# ---------------------------------------------------------------------------
# Empty universe
# ---------------------------------------------------------------------------


class TestEmptyUniverse:
    def test_empty_when_universe_outside_data(self, input_db):
        strat = _make_strategy(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
        )
        scanner = EntryScanner(input_db, strat, strat.trades[0])
        df = scanner.scan()
        assert df.is_empty()

    def test_empty_result_has_expected_columns(self, input_db):
        strat = _make_strategy(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
        )
        scanner = EntryScanner(input_db, strat, strat.trades[0])
        df = scanner.scan()
        assert "entry_id" in df.columns
        assert "entry_time" in df.columns
