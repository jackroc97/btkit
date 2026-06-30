"""
Unit tests for EntryScanner indicator stitching across contract rolls.

Regression test for the bug where _get_indicators used only the first
front-month contract's underlying_id, causing null indicator values for
all bars after the first quarterly roll.
"""
from __future__ import annotations

import duckdb
import polars as pl
import pytest

from btkit.backtest.entry import EntryScanner
from btkit.db.input_db import InputDatabase
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
    SessionConfig,
)
from datetime import date, time


def _make_db_with_two_contracts() -> InputDatabase:
    """
    Seed an in-memory DB with two quarterly contracts (H and M) and
    one indicator (ves1d) defined for each, with non-overlapping bars.
    """
    con = duckdb.connect(":memory:")

    con.execute("""
        CREATE TABLE underlying_bars (
            instrument_id BIGINT,
            ts_event TIMESTAMPTZ,
            symbol VARCHAR,
            expiration DATE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT
        )
    """)
    con.execute("""
        CREATE TABLE option_bars (
            instrument_id BIGINT, ts_event TIMESTAMPTZ, symbol VARCHAR,
            expiration DATE, strike_price DOUBLE, "right" VARCHAR,
            multiplier DOUBLE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT
        )
    """)
    con.execute("""
        CREATE TABLE option_greeks (
            instrument_id BIGINT, ts_event TIMESTAMPTZ,
            dte DOUBLE, T DOUBLE, iv DOUBLE, delta DOUBLE,
            gamma DOUBLE, theta DOUBLE, vega DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE indicator_definition (
            id INTEGER PRIMARY KEY,
            underlying_id BIGINT,
            name VARCHAR,
            script TEXT
        )
    """)
    con.execute("""
        CREATE TABLE indicator_bars (
            indicator_id INTEGER,
            ts_event TIMESTAMPTZ,
            value DOUBLE
        )
    """)

    # Two contracts: ESH (id=1, expires 2024-03-15) and ESM (id=2, expires 2024-06-21)
    h_bars = [
        (1, "2024-01-02 14:30:00+00", "ESH24", "2024-03-15", 4800.0, 4810.0, 4790.0, 4805.0, 1000),
        (1, "2024-02-01 14:30:00+00", "ESH24", "2024-03-15", 4820.0, 4830.0, 4810.0, 4825.0, 1000),
    ]
    m_bars = [
        (2, "2024-04-01 14:30:00+00", "ESM24", "2024-06-21", 5000.0, 5010.0, 4990.0, 5005.0, 1000),
        (2, "2024-05-01 14:30:00+00", "ESM24", "2024-06-21", 5100.0, 5110.0, 5090.0, 5105.0, 1000),
    ]
    for row in h_bars + m_bars:
        con.execute("INSERT INTO underlying_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", list(row))

    # One indicator per contract
    con.execute("INSERT INTO indicator_definition VALUES (1, 1, 'ves1d', '')")
    con.execute("INSERT INTO indicator_definition VALUES (2, 2, 'ves1d', '')")

    # ves1d values for ESH bars
    con.execute("INSERT INTO indicator_bars VALUES (1, '2024-01-02 14:30:00+00', 22.0)")
    con.execute("INSERT INTO indicator_bars VALUES (1, '2024-02-01 14:30:00+00', 25.0)")
    # ves1d values for ESM bars
    con.execute("INSERT INTO indicator_bars VALUES (2, '2024-04-01 14:30:00+00', 18.0)")
    con.execute("INSERT INTO indicator_bars VALUES (2, '2024-05-01 14:30:00+00', 30.0)")

    db = object.__new__(InputDatabase)
    db._con = con
    return db


def _make_scanner(db: InputDatabase, conditions: list[str]) -> EntryScanner:
    universe = UniverseConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 5, 31),
        session=SessionConfig(
            timezone="America/New_York",
            start_time=time(9, 30),
            end_time=time(16, 0),
            weekdays_only=False,
        ),
    )
    trade = TradeDefinition(
        name="test",
        instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
        entry=EntryConfig(
            window=EntryWindowConfig(start=time(9, 0), end=time(16, 0)),
            conditions=conditions,
        ),
        legs=[
            LegConfig(name="short", right="put", action="sell_to_open", dte=30, delta=-0.16),
        ],
        exit=ExitConfig(),
    )
    strategy = StrategyDefinition(name="test_strategy", universe=universe, trades=[trade])
    return EntryScanner(db=db, strategy=strategy, trade=trade)


class TestIndicatorStitchingAcrossRolls:
    def setup_method(self):
        self.db = _make_db_with_two_contracts()

    def test_stitched_indicators_cover_both_contracts(self):
        """Both contracts' indicator bars should appear in the concatenated result."""
        from datetime import datetime, UTC
        schedule = pl.DataFrame({
            "date": pl.Series([
                date(2024, 1, 2), date(2024, 2, 1),
                date(2024, 4, 1), date(2024, 5, 1),
            ], dtype=pl.Date),
            "underlying_id": pl.Series([1, 1, 2, 2], dtype=pl.Int64),
        })
        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        end_dt   = datetime(2024, 5, 31, 23, 59, 59, tzinfo=UTC)

        scanner = _make_scanner(self.db, conditions=["ves1d > 20"])
        result = scanner._get_indicators_for_schedule(schedule, start_dt, end_dt)

        assert "ves1d" in result.columns
        assert len(result) == 4, f"expected 4 rows (2 per contract), got {len(result)}"

    def test_stitched_indicators_values_are_correct(self):
        """Values from both contracts should be present without corruption."""
        from datetime import datetime, UTC
        schedule = pl.DataFrame({
            "date": pl.Series([
                date(2024, 1, 2), date(2024, 2, 1),
                date(2024, 4, 1), date(2024, 5, 1),
            ], dtype=pl.Date),
            "underlying_id": pl.Series([1, 1, 2, 2], dtype=pl.Int64),
        })
        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        end_dt   = datetime(2024, 5, 31, 23, 59, 59, tzinfo=UTC)

        scanner = _make_scanner(self.db, conditions=["ves1d > 20"])
        result = scanner._get_indicators_for_schedule(schedule, start_dt, end_dt).sort("ts_event")

        values = result["ves1d"].to_list()
        assert values == pytest.approx([22.0, 25.0, 18.0, 30.0])

    def test_first_contract_only_would_miss_later_bars(self):
        """Confirm the old single-contract fetch only returns 2 of the 4 rows."""
        from datetime import datetime, UTC
        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        end_dt   = datetime(2024, 5, 31, 23, 59, 59, tzinfo=UTC)

        # Simulate the old broken path: query only ESH (id=1)
        old_result = self.db.indicators(1, start_dt, end_dt)
        assert len(old_result) == 2, "old path only gets ESH rows"

    def test_no_conditions_returns_empty(self):
        """No conditions means no indicator fetch needed."""
        from datetime import datetime, UTC
        schedule = pl.DataFrame({
            "date": pl.Series([date(2024, 1, 2)], dtype=pl.Date),
            "underlying_id": pl.Series([1], dtype=pl.Int64),
        })
        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        end_dt   = datetime(2024, 5, 31, 23, 59, 59, tzinfo=UTC)

        scanner = _make_scanner(self.db, conditions=[])
        result = scanner._get_indicators_for_schedule(schedule, start_dt, end_dt)
        assert result.is_empty()

    def test_preloaded_indicators_bypass_schedule(self):
        """Preloaded indicators should be returned as-is without any DB query."""
        from datetime import datetime, UTC
        preloaded = pl.DataFrame({
            "ts_event": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "ves1d": pl.Series([], dtype=pl.Float64),
        })
        schedule = pl.DataFrame({
            "date": pl.Series([date(2024, 1, 2)], dtype=pl.Date),
            "underlying_id": pl.Series([1], dtype=pl.Int64),
        })
        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        end_dt   = datetime(2024, 5, 31, 23, 59, 59, tzinfo=UTC)

        universe = UniverseConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 5, 31),
            session=SessionConfig(
                timezone="America/New_York",
                start_time=time(9, 30),
                end_time=time(16, 0),
                weekdays_only=False,
            ),
        )
        trade = TradeDefinition(
            name="test",
            instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
            entry=EntryConfig(
                window=EntryWindowConfig(start=time(9, 0), end=time(16, 0)),
                conditions=["ves1d > 20"],
            ),
            legs=[
                LegConfig(name="short", right="put", action="sell_to_open", dte=30, delta=-0.16),
            ],
            exit=ExitConfig(),
        )
        strategy = StrategyDefinition(name="test_strategy", universe=universe, trades=[trade])
        scanner = EntryScanner(db=self.db, strategy=strategy, trade=trade, indicators=preloaded)

        result = scanner._get_indicators_for_schedule(schedule, start_dt, end_dt)
        assert result is preloaded
