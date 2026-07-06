"""
Unit tests for IndicatorContext and the IndicatorRunner arity-detection logic.

Uses an in-memory DuckDB populated with a small fixture dataset so that
option_greeks() and option_bars() can be tested end-to-end without a real
input database file.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from btkit.pipeline.indicators import IndicatorContext, IndicatorRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def con():
    """In-memory DuckDB with minimal input schema tables."""
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE underlying_bars (
            ts_event      TIMESTAMPTZ NOT NULL,
            instrument_id INTEGER     NOT NULL,
            symbol        VARCHAR     NOT NULL,
            open          DOUBLE,
            high          DOUBLE,
            low           DOUBLE,
            close         DOUBLE,
            volume        BIGINT
        )
    """)
    c.execute("""
        CREATE TABLE option_greeks (
            ts_event      TIMESTAMPTZ NOT NULL,
            instrument_id INTEGER     NOT NULL,
            underlying_id INTEGER     NOT NULL,
            dte           INTEGER     NOT NULL,
            T             DOUBLE,
            iv            DOUBLE,
            delta         DOUBLE,
            gamma         DOUBLE,
            theta         DOUBLE,
            vega          DOUBLE
        )
    """)
    c.execute("""
        CREATE TABLE option_bars (
            ts_event      TIMESTAMPTZ NOT NULL,
            instrument_id INTEGER     NOT NULL,
            underlying_id INTEGER     NOT NULL,
            symbol        VARCHAR     NOT NULL,
            expiration    DATE        NOT NULL,
            strike_price  DOUBLE      NOT NULL,
            "right"       VARCHAR(1)  NOT NULL,
            multiplier    INTEGER     NOT NULL,
            open          DOUBLE,
            high          DOUBLE,
            low           DOUBLE,
            close         DOUBLE,
            volume        BIGINT
        )
    """)
    c.execute("""
        CREATE TABLE indicator_definition (
            id                INTEGER PRIMARY KEY,
            name              VARCHAR NOT NULL,
            underlying_id     INTEGER NOT NULL,
            underlying_symbol VARCHAR NOT NULL,
            params            JSON,
            script_source     TEXT    NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE indicator_bars (
            ts_event     TIMESTAMPTZ NOT NULL,
            indicator_id INTEGER     NOT NULL,
            value        DOUBLE
        )
    """)

    # Seed underlying bars (underlying_id=1)
    rows = [
        ("2024-01-02 14:30:00+00", 1, "ESH4", 4800.0, 4810.0, 4795.0, 4805.0, 1000),
        ("2024-01-02 14:31:00+00", 1, "ESH4", 4805.0, 4815.0, 4800.0, 4810.0, 1100),
        ("2024-01-02 14:32:00+00", 1, "ESH4", 4810.0, 4820.0, 4805.0, 4815.0, 900),
    ]
    c.executemany("INSERT INTO underlying_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    # Seed option_greeks — three instruments, two DTE buckets
    greeks_rows = [
        ("2024-01-02 14:30:00+00", 101, 1, 7, 0.019, 0.18, -0.20, 0.01, -0.05, 0.10),
        ("2024-01-02 14:30:00+00", 102, 1, 21, 0.058, 0.20, -0.30, 0.008, -0.03, 0.25),
        ("2024-01-02 14:30:00+00", 103, 1, 21, 0.058, 0.22, 0.30, 0.008, -0.03, 0.24),
        ("2024-01-02 14:31:00+00", 101, 1, 7, 0.019, 0.19, -0.21, 0.011, -0.05, 0.11),
        ("2024-01-02 14:31:00+00", 102, 1, 21, 0.058, 0.21, -0.31, 0.009, -0.03, 0.26),
    ]
    c.executemany("INSERT INTO option_greeks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", greeks_rows)

    # Seed option_bars — calls and puts across two expirations
    bar_rows = [
        (
            "2024-01-02 14:30:00+00",
            101,
            1,
            "ES240109P04750",
            "2024-01-09",
            4750.0,
            "P",
            50,
            10.0,
            11.0,
            9.5,
            10.5,
            200,
        ),
        (
            "2024-01-02 14:30:00+00",
            102,
            1,
            "ES240123P04725",
            "2024-01-23",
            4725.0,
            "P",
            50,
            20.0,
            21.0,
            19.0,
            20.5,
            300,
        ),
        (
            "2024-01-02 14:30:00+00",
            103,
            1,
            "ES240123C04850",
            "2024-01-23",
            4850.0,
            "C",
            50,
            15.0,
            16.0,
            14.5,
            15.5,
            250,
        ),
        (
            "2024-01-02 14:31:00+00",
            101,
            1,
            "ES240109P04750",
            "2024-01-09",
            4750.0,
            "P",
            50,
            10.5,
            11.5,
            10.0,
            11.0,
            180,
        ),
    ]
    c.executemany(
        "INSERT INTO option_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", bar_rows
    )

    yield c
    c.close()


def _make_ctx(con, underlying_id=1) -> IndicatorContext:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = datetime(2024, 1, 2, 14, 32, tzinfo=UTC)
    return IndicatorContext(con, underlying_id, start, end)


# ---------------------------------------------------------------------------
# IndicatorContext — attribute
# ---------------------------------------------------------------------------


class TestIndicatorContextAttributes:
    def test_underlying_id_exposed(self, con):
        ctx = _make_ctx(con, underlying_id=42)
        assert ctx.underlying_id == 42


# ---------------------------------------------------------------------------
# IndicatorContext.option_greeks()
# ---------------------------------------------------------------------------


class TestOptionGreeks:
    def test_returns_all_greeks_in_window(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_greeks()
        assert not df.is_empty()
        assert "ts_event" in df.columns
        assert "vega" in df.columns
        assert "delta" in df.columns

    def test_dte_max_filter(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_greeks(dte_max=10)
        # Only instrument 101 (dte=7) should survive
        assert set(df["instrument_id"].to_list()) == {101}

    def test_dte_min_filter(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_greeks(dte_min=15)
        # Only instruments 102 and 103 (dte=21) should survive
        assert set(df["instrument_id"].to_list()) == {102, 103}

    def test_dte_range_filter(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_greeks(dte_min=5, dte_max=10)
        assert set(df["instrument_id"].to_list()) == {101}

    def test_delta_max_filter(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_greeks(delta_max=0.0)
        # Only puts (negative delta) should survive
        assert all(d <= 0.0 for d in df["delta"].to_list())

    def test_delta_min_filter(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_greeks(delta_min=0.0)
        # Only calls (positive delta: instrument 103) should survive
        assert all(d >= 0.0 for d in df["delta"].to_list())

    def test_different_underlying_returns_empty(self, con):
        ctx = _make_ctx(con, underlying_id=999)
        df = ctx.option_greeks()
        assert df.is_empty()

    def test_columns_present(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_greeks()
        expected = {
            "ts_event",
            "instrument_id",
            "dte",
            "T",
            "iv",
            "delta",
            "gamma",
            "theta",
            "vega",
        }
        assert expected.issubset(set(df.columns))

    def test_no_underlying_id_in_output(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_greeks()
        assert "underlying_id" not in df.columns


# ---------------------------------------------------------------------------
# IndicatorContext.option_bars()
# ---------------------------------------------------------------------------


class TestOptionBars:
    def test_returns_bars_in_window(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_bars()
        assert not df.is_empty()
        assert "close" in df.columns
        assert "strike_price" in df.columns

    def test_right_filter_puts(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_bars(right="P")
        assert all(r == "P" for r in df["right"].to_list())

    def test_right_filter_calls(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_bars(right="C")
        assert all(r == "C" for r in df["right"].to_list())
        # Only instrument 103 is a call
        assert set(df["instrument_id"].to_list()) == {103}

    def test_dte_max_filter(self, con):
        ctx = _make_ctx(con)
        # expiration 2024-01-09 from 2024-01-02 = 7 days; keep only dte_max=10
        df = ctx.option_bars(dte_max=10)
        assert set(df["instrument_id"].to_list()) == {101}

    def test_dte_min_filter(self, con):
        ctx = _make_ctx(con)
        # expiration 2024-01-23 from 2024-01-02 = 21 days; keep only dte_min=15
        df = ctx.option_bars(dte_min=15)
        assert set(df["instrument_id"].to_list()) == {102, 103}

    def test_combined_right_and_dte(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_bars(right="P", dte_min=15)
        # Only instrument 102 (put, 21 DTE)
        assert set(df["instrument_id"].to_list()) == {102}

    def test_different_underlying_returns_empty(self, con):
        ctx = _make_ctx(con, underlying_id=999)
        df = ctx.option_bars()
        assert df.is_empty()

    def test_columns_present(self, con):
        ctx = _make_ctx(con)
        df = ctx.option_bars()
        expected = {
            "ts_event",
            "instrument_id",
            "symbol",
            "expiration",
            "strike_price",
            "right",
            "multiplier",
            "open",
            "close",
            "volume",
        }
        assert expected.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# IndicatorRunner — arity detection and context forwarding
# ---------------------------------------------------------------------------


class TestIndicatorRunnerArityDetection:
    def _write_script(self, tmp_path: Path, source: str) -> Path:
        p = tmp_path / "indicator.py"
        p.write_text(textwrap.dedent(source))
        return p

    def test_single_arg_script_runs_without_context(self, con, tmp_path):
        """A compute(df) script runs normally — no context passed."""
        script = self._write_script(
            tmp_path,
            """
            import polars as pl

            def compute(df):
                return df.with_columns(pl.lit(1.0).alias("my_ind"))
        """,
        )
        runner = IndicatorRunner(con, script)
        assert runner._wants_context is False

    def test_two_arg_script_detected(self, con, tmp_path):
        """A compute(df, ctx) script is detected as wanting context."""
        script = self._write_script(
            tmp_path,
            """
            def compute(df, ctx):
                return df.with_columns(__import__('polars').lit(0.0).alias("x"))
        """,
        )
        runner = IndicatorRunner(con, script)
        assert runner._wants_context is True

    def test_single_arg_script_writes_indicator(self, con, tmp_path):
        """Single-arg compute produces indicator rows in indicator_bars."""
        # Need at least one underlying_bars row for runner.run() to proceed
        script = self._write_script(
            tmp_path,
            """
            import polars as pl

            def compute(df):
                return df.with_columns(pl.lit(42.0).alias("test_ind"))
        """,
        )
        runner = IndicatorRunner(con, script)
        runner.run(underlying_id=1)

        rows = con.execute(
            "SELECT COUNT(*) FROM indicator_bars ib "
            "JOIN indicator_definition d ON ib.indicator_id = d.id "
            "WHERE d.name = 'test_ind'"
        ).fetchone()[0]
        assert rows == 3  # 3 underlying bars seeded

    def test_two_arg_script_receives_context_and_writes_indicator(self, con, tmp_path):
        """Two-arg compute receives a live IndicatorContext and can call option_greeks()."""
        script = self._write_script(
            tmp_path,
            """
            import polars as pl

            def compute(df, ctx):
                greeks = ctx.option_greeks(dte_max=10)
                # Count rows returned as a constant indicator value
                count = float(len(greeks))
                return df.with_columns(pl.lit(count).alias("greeks_count"))
        """,
        )
        runner = IndicatorRunner(con, script)
        runner.run(underlying_id=1)

        val = con.execute(
            "SELECT value FROM indicator_bars ib "
            "JOIN indicator_definition d ON ib.indicator_id = d.id "
            "WHERE d.name = 'greeks_count' LIMIT 1"
        ).fetchone()[0]
        # dte_max=10 → instrument 101 rows within window: 2 rows
        assert val == 2.0

    def test_two_arg_script_receives_correct_underlying_id(self, con, tmp_path):
        """ctx.underlying_id matches the underlying being processed."""
        script = self._write_script(
            tmp_path,
            """
            import polars as pl

            def compute(df, ctx):
                uid = float(ctx.underlying_id)
                return df.with_columns(pl.lit(uid).alias("uid_check"))
        """,
        )
        runner = IndicatorRunner(con, script)
        runner.run(underlying_id=1)

        val = con.execute(
            "SELECT value FROM indicator_bars ib "
            "JOIN indicator_definition d ON ib.indicator_id = d.id "
            "WHERE d.name = 'uid_check' LIMIT 1"
        ).fetchone()[0]
        assert val == 1.0

    def test_script_missing_compute_raises(self, con, tmp_path):
        """A script without a compute() function raises AttributeError at load time."""
        script = self._write_script(
            tmp_path,
            """
            def not_compute(df):
                return df
        """,
        )
        with pytest.raises(AttributeError, match="compute"):
            IndicatorRunner(con, script)

    def test_two_arg_script_option_bars(self, con, tmp_path):
        """ctx.option_bars() is accessible from a two-arg compute script."""
        script = self._write_script(
            tmp_path,
            """
            import polars as pl

            def compute(df, ctx):
                bars = ctx.option_bars(right='P')
                count = float(len(bars))
                return df.with_columns(pl.lit(count).alias("put_bar_count"))
        """,
        )
        runner = IndicatorRunner(con, script)
        runner.run(underlying_id=1)

        val = con.execute(
            "SELECT value FROM indicator_bars ib "
            "JOIN indicator_definition d ON ib.indicator_id = d.id "
            "WHERE d.name = 'put_bar_count' LIMIT 1"
        ).fetchone()[0]
        # Puts: instrument 101 (2 rows) + instrument 102 (1 row) = 3
        assert val == 3.0
