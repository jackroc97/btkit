"""
Unit tests for btkit.audit.*

Each test class builds a minimal in-memory DuckDB with the relevant tables
and exercises one phase (or utility) in isolation.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import duckdb
import numpy as np
import polars as pl
import pytest

from btkit.audit.schema import (
    FlagCode,
    FlagSeverity,
    HARD_FLAGS,
    SOFT_FLAGS,
    FLAG_SEVERITY,
    resolve_audit_filter,
)
from btkit.audit.rules import phase1_iv, phase2_delta, phase3_coverage, phase4_integrity
from btkit.audit.runner import AuditRunner


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TS = datetime(2024, 1, 10, 14, 0, 0, tzinfo=timezone.utc)
TS2 = datetime(2024, 1, 11, 14, 0, 0, tzinfo=timezone.utc)
EXP_NEAR = date(2024, 2, 16)   # 37 days from TS
EXP_FAR = date(2024, 3, 15)    # 65 days from TS


def _make_con() -> duckdb.DuckDBPyConnection:
    """Return a fresh in-memory DuckDB connection with the input schema."""
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE option_greeks (
            ts_event   TIMESTAMPTZ NOT NULL,
            instrument_id INTEGER   NOT NULL,
            underlying_id INTEGER   NOT NULL,
            dte        INTEGER      NOT NULL,
            T          DOUBLE       NOT NULL,
            iv         DOUBLE,
            delta      DOUBLE,
            gamma      DOUBLE,
            theta      DOUBLE,
            vega       DOUBLE
        );
        CREATE TABLE option_bars (
            ts_event   TIMESTAMPTZ NOT NULL,
            instrument_id INTEGER   NOT NULL,
            underlying_id INTEGER   NOT NULL,
            symbol     VARCHAR      NOT NULL,
            expiration DATE         NOT NULL,
            strike_price DOUBLE     NOT NULL,
            "right"    VARCHAR(1)   NOT NULL,
            multiplier INTEGER      NOT NULL,
            open       DOUBLE,
            high       DOUBLE,
            low        DOUBLE,
            close      DOUBLE,
            volume     BIGINT
        );
        CREATE TABLE underlying_bars (
            ts_event   TIMESTAMPTZ NOT NULL,
            instrument_id INTEGER   NOT NULL,
            symbol     VARCHAR      NOT NULL,
            expiration DATE,
            open       DOUBLE       NOT NULL,
            high       DOUBLE       NOT NULL,
            low        DOUBLE       NOT NULL,
            close      DOUBLE       NOT NULL,
            volume     BIGINT
        );
    """)
    return con


def _insert_option_greeks(con, rows: list[dict]) -> None:
    df = pl.DataFrame(rows).with_columns(pl.col("ts_event").cast(pl.Datetime("us", "UTC")))
    con.register("_g", df)
    con.execute("INSERT INTO option_greeks SELECT * FROM _g")
    con.unregister("_g")


def _insert_option_bars(con, rows: list[dict]) -> None:
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts_event").cast(pl.Datetime("us", "UTC")),
        pl.col("expiration").cast(pl.Date),
    )
    con.register("_b", df)
    con.execute("INSERT INTO option_bars SELECT * FROM _b")
    con.unregister("_b")


def _insert_underlying_bars(con, rows: list[dict]) -> None:
    df = pl.DataFrame(rows).with_columns(pl.col("ts_event").cast(pl.Datetime("us", "UTC")))
    con.register("_ub", df)
    con.execute("INSERT INTO underlying_bars SELECT * FROM _ub")
    con.unregister("_ub")


def _base_greek(
    ts=TS,
    iid=1,
    uid=100,
    dte=37,
    T=0.10,
    iv=0.20,
    delta=-0.25,
    gamma=0.01,
    theta=-0.05,
    vega=0.30,
):
    return dict(
        ts_event=ts, instrument_id=iid, underlying_id=uid,
        dte=dte, T=T, iv=iv, delta=delta,
        gamma=gamma, theta=theta, vega=vega,
    )


def _base_bar(
    ts=TS,
    iid=1,
    uid=100,
    symbol="EW1Q4 P5000",
    expiration=EXP_NEAR,
    strike=5000.0,
    right="P",
    multiplier=50,
    close=10.0,
):
    return dict(
        ts_event=ts, instrument_id=iid, underlying_id=uid,
        symbol=symbol, expiration=expiration,
        strike_price=strike, right=right, multiplier=multiplier,
        open=close, high=close, low=close, close=close, volume=100,
    )


# ---------------------------------------------------------------------------
# Schema / utility tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_hard_soft_disjoint(self):
        assert HARD_FLAGS.isdisjoint(SOFT_FLAGS)

    def test_all_codes_have_severity(self):
        for code in FlagCode:
            assert code in FLAG_SEVERITY

    def test_hard_flags_are_hard(self):
        for code in HARD_FLAGS:
            assert FLAG_SEVERITY[code] == FlagSeverity.HARD

    def test_soft_flags_are_soft(self):
        for code in SOFT_FLAGS:
            assert FLAG_SEVERITY[code] == FlagSeverity.SOFT


class TestResolveAuditFilter:
    def test_preset_none(self):
        assert resolve_audit_filter("none") == frozenset()

    def test_preset_hard_only(self):
        codes = resolve_audit_filter("hard_errors_only")
        assert FlagCode.BARS_TRUNCATED.value in codes
        assert FlagCode.IV_NAN.value not in codes

    def test_preset_strict(self):
        codes = resolve_audit_filter("strict")
        assert FlagCode.BARS_TRUNCATED.value in codes
        assert FlagCode.IV_NAN.value in codes

    def test_explicit_list(self):
        codes = resolve_audit_filter(["BARS_TRUNCATED", "IV_NAN"])
        assert codes == frozenset({"BARS_TRUNCATED", "IV_NAN"})

    def test_unknown_preset_returns_empty(self):
        assert resolve_audit_filter("bogus") == frozenset()


# ---------------------------------------------------------------------------
# Phase 1 — IV flags
# ---------------------------------------------------------------------------

class TestPhase1IV:
    def test_iv_nan_flagged(self):
        con = _make_con()
        _insert_option_greeks(con, [
            _base_greek(iid=1, iv=float("nan")),
            _base_greek(iid=2, iv=0.20),  # clean
        ])
        result = phase1_iv.run(con)
        assert set(result["instrument_id"].to_list()) == {1}
        assert result["flag_code"][0] == "IV_NAN"
        assert result["flag_severity"][0] == "soft"

    def test_iv_sentinel_flagged(self):
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=5, iv=10.0)])
        result = phase1_iv.run(con)
        assert result["flag_code"][0] == "IV_SENTINEL"
        assert result["threshold"][0] == 10.0

    def test_iv_high_flagged(self):
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=7, iv=2.5)])
        result = phase1_iv.run(con)
        assert result["flag_code"][0] == "IV_HIGH"
        assert result["threshold"][0] == 2.0
        assert result["flag_value"][0] == pytest.approx(2.5)

    def test_iv_high_excludes_sentinel(self):
        # iv = 10.0 should be IV_SENTINEL, not IV_HIGH
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=8, iv=10.0)])
        result = phase1_iv.run(con)
        assert result["flag_code"][0] == "IV_SENTINEL"

    def test_clean_returns_empty(self):
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=1, iv=0.20)])
        result = phase1_iv.run(con)
        assert result.is_empty()

    def test_multiple_flags_per_instrument(self):
        # Two rows for same instrument: one NaN, one sentinel
        con = _make_con()
        _insert_option_greeks(con, [
            _base_greek(iid=3, ts=TS, iv=float("nan")),
            _base_greek(iid=3, ts=TS2, iv=10.0),
        ])
        result = phase1_iv.run(con)
        assert len(result) == 2
        codes = set(result["flag_code"].to_list())
        assert codes == {"IV_NAN", "IV_SENTINEL"}


# ---------------------------------------------------------------------------
# Phase 2 — Delta consistency
# ---------------------------------------------------------------------------

class TestPhase2Delta:
    def _setup(self, reported_delta: float, right: str = "P"):
        """
        Build a minimal DB with one option row where reported delta = reported_delta
        but underlying/strike/T/iv are set so Black-76 produces a predictably different value.
        """
        con = _make_con()
        # strike 5000, underlying 5200 (OTM put), iv=0.20, T=0.10 → delta ≈ -0.12 (approx)
        _insert_option_greeks(con, [
            _base_greek(iid=1, uid=100, iv=0.20, T=0.10, delta=reported_delta),
        ])
        _insert_option_bars(con, [
            _base_bar(iid=1, uid=100, strike=5000.0, right=right),
        ])
        _insert_underlying_bars(con, [{
            "ts_event": TS, "instrument_id": 100,
            "symbol": "ES", "expiration": None,
            "open": 5200.0, "high": 5200.0, "low": 5200.0, "close": 5200.0,
            "volume": 1000,
        }])
        return con

    def test_consistent_delta_not_flagged(self):
        # Compute Black-76 delta first with known params then use it as reported
        from btkit.pipeline.greeks import _greeks
        import numpy as np
        F, K, T, r, iv = 5200.0, 5000.0, 0.10, 0.01, 0.20
        theoretical, _, _, _ = _greeks(
            np.array([F]), np.array([K]), np.array([T]),
            np.array([r]), np.array([iv]), np.array([0], dtype=np.int64),
        )
        con = self._setup(reported_delta=float(theoretical[0]))
        result = phase2_delta.run(con)
        assert result.is_empty()

    def test_wildly_inconsistent_delta_flagged(self):
        # reported = 0.0 for an OTM put, theoretical ≈ -0.12 → diff > 0.10
        con = self._setup(reported_delta=0.0)
        result = phase2_delta.run(con)
        assert len(result) == 1
        assert result["flag_code"][0] == "DELTA_INCONSISTENT"
        assert result["flag_severity"][0] == "soft"
        assert result["threshold"][0] == pytest.approx(0.10)
        assert result["flag_value"][0] > 0.10

    def test_nan_delta_skipped(self):
        con = self._setup(reported_delta=float("nan"))
        result = phase2_delta.run(con)
        assert result.is_empty()

    def test_progress_callback_fires(self):
        con = self._setup(reported_delta=0.0)
        ticks: list[tuple[int, int]] = []
        phase2_delta.run(con, progress_cb=lambda d, t: ticks.append((d, t)))
        assert len(ticks) > 0
        # Last tick should have done == total
        assert ticks[-1][0] == ticks[-1][1]


# ---------------------------------------------------------------------------
# Phase 3 — Coverage flags
# ---------------------------------------------------------------------------

class TestPhase3Coverage:
    def _insert_bars_for_instrument(
        self,
        con,
        iid: int,
        days: list[date],
        expiration: date,
        bars_per_day: int = 50,
    ) -> None:
        rows = []
        for d in days:
            for h in range(bars_per_day):
                ts = datetime(d.year, d.month, d.day, 9, 30 + h, 0, tzinfo=timezone.utc)
                rows.append(_base_bar(
                    ts=ts, iid=iid, expiration=expiration, close=5.0,
                ))
        _insert_option_bars(con, rows)

    def test_bars_truncated_flagged(self):
        # observable_life = 40 days, last bar is 20 days before expiry → ratio = 0.50 > 0.15
        con = _make_con()
        first = date(2024, 1, 10)
        last = date(2024, 2, 5)    # 20 days before expiration
        expiration = date(2024, 2, 25)
        # Only insert first and last day (for simplicity)
        for d in [first, last]:
            ts = datetime(d.year, d.month, d.day, 14, 0, 0, tzinfo=timezone.utc)
            _insert_option_bars(con, [_base_bar(ts=ts, iid=1, expiration=expiration)])
        result = phase3_coverage.run(con)
        truncated = result.filter(pl.col("flag_code") == "BARS_TRUNCATED")
        assert len(truncated) == 1
        assert truncated["instrument_id"][0] == 1
        assert truncated["flag_value"][0] == pytest.approx(20 / 46, rel=0.01)

    def test_bars_not_truncated_within_threshold(self):
        # last bar 2 days before expiry, observable life 40 → ratio = 0.05 < 0.15
        con = _make_con()
        expiration = date(2024, 2, 25)
        for d in [date(2024, 1, 16), date(2024, 2, 23)]:
            ts = datetime(d.year, d.month, d.day, 14, 0, 0, tzinfo=timezone.utc)
            _insert_option_bars(con, [_base_bar(ts=ts, iid=1, expiration=expiration)])
        result = phase3_coverage.run(con)
        truncated = result.filter(pl.col("flag_code") == "BARS_TRUNCATED")
        assert truncated.is_empty()

    def test_bars_sparse_flagged(self):
        # 2 bars total, 2 trading days → 1 bar/day < 10
        con = _make_con()
        expiration = date(2024, 3, 15)
        for d in [date(2024, 1, 10), date(2024, 1, 11)]:
            ts = datetime(d.year, d.month, d.day, 10, 0, 0, tzinfo=timezone.utc)
            _insert_option_bars(con, [_base_bar(ts=ts, iid=2, expiration=expiration)])
        result = phase3_coverage.run(con)
        sparse = result.filter(pl.col("flag_code") == "BARS_SPARSE")
        assert len(sparse) == 1
        assert sparse["flag_value"][0] == pytest.approx(1.0)

    def test_no_expiry_bars_flagged(self):
        # Bars exist up to the day before expiry but not on expiry day itself
        con = _make_con()
        expiration = date(2024, 2, 16)
        for d in [date(2024, 1, 10), date(2024, 2, 15)]:
            ts = datetime(d.year, d.month, d.day, 14, 0, 0, tzinfo=timezone.utc)
            _insert_option_bars(con, [_base_bar(ts=ts, iid=3, expiration=expiration)])
        result = phase3_coverage.run(con)
        no_exp = result.filter(pl.col("flag_code") == "NO_EXPIRY_BARS")
        assert len(no_exp) == 1
        assert no_exp["instrument_id"][0] == 3

    def test_no_expiry_bars_not_flagged_when_expiry_bar_present(self):
        con = _make_con()
        expiration = date(2024, 2, 16)
        for d in [date(2024, 1, 10), expiration]:
            ts = datetime(d.year, d.month, d.day, 14, 0, 0, tzinfo=timezone.utc)
            _insert_option_bars(con, [_base_bar(ts=ts, iid=4, expiration=expiration)])
        result = phase3_coverage.run(con)
        no_exp = result.filter(pl.col("flag_code") == "NO_EXPIRY_BARS")
        assert no_exp.is_empty()


# ---------------------------------------------------------------------------
# Phase 4 — Integrity flags
# ---------------------------------------------------------------------------

class TestPhase4Integrity:
    def test_negative_close_flagged(self):
        con = _make_con()
        _insert_option_bars(con, [_base_bar(iid=1, close=-1.0)])
        result = phase4_integrity.run(con)
        assert result["flag_code"][0] == "NEGATIVE_CLOSE"
        assert result["flag_severity"][0] == "hard"
        assert result["flag_value"][0] == pytest.approx(-1.0)

    def test_positive_close_not_flagged(self):
        con = _make_con()
        _insert_option_bars(con, [_base_bar(iid=1, close=5.0)])
        result = phase4_integrity.run(con)
        assert result.is_empty()

    def test_negative_dte_flagged(self):
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=2, dte=-1)])
        result = phase4_integrity.run(con)
        neg_dte = result.filter(pl.col("flag_code") == "NEGATIVE_DTE")
        assert len(neg_dte) == 1
        assert neg_dte["flag_severity"][0] == "hard"

    def test_zombie_bar_flagged(self):
        # ts_event is 1 day after expiration
        con = _make_con()
        expired = date(2024, 1, 9)   # expiry in the past relative to TS
        _insert_option_bars(con, [
            _base_bar(iid=3, ts=TS, expiration=expired)
        ])
        result = phase4_integrity.run(con)
        zombie = result.filter(pl.col("flag_code") == "ZOMBIE_BAR")
        assert len(zombie) == 1
        assert zombie["flag_severity"][0] == "hard"
        assert zombie["flag_value"][0] > 0  # days after expiry

    def test_delta_sign_error_put_positive_delta(self):
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=10, delta=0.10)])  # put with positive delta
        _insert_option_bars(con, [_base_bar(iid=10, right="P")])
        result = phase4_integrity.run(con)
        sign_err = result.filter(pl.col("flag_code") == "DELTA_SIGN_ERROR")
        assert len(sign_err) == 1
        assert sign_err["flag_severity"][0] == "hard"

    def test_delta_sign_error_call_negative_delta(self):
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=11, delta=-0.05)])  # call with negative delta
        _insert_option_bars(con, [_base_bar(iid=11, right="C")])
        result = phase4_integrity.run(con)
        sign_err = result.filter(pl.col("flag_code") == "DELTA_SIGN_ERROR")
        assert len(sign_err) == 1

    def test_delta_sign_ok_put_negative(self):
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=12, delta=-0.25)])
        _insert_option_bars(con, [_base_bar(iid=12, right="P")])
        result = phase4_integrity.run(con)
        sign_err = result.filter(pl.col("flag_code") == "DELTA_SIGN_ERROR")
        assert sign_err.is_empty()

    def test_delta_magnitude_error_flagged(self):
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=13, delta=-1.5)])
        _insert_option_bars(con, [_base_bar(iid=13, right="P")])
        result = phase4_integrity.run(con)
        mag_err = result.filter(pl.col("flag_code") == "DELTA_MAGNITUDE_ERROR")
        assert len(mag_err) == 1
        assert mag_err["flag_value"][0] == pytest.approx(1.5)
        assert mag_err["threshold"][0] == pytest.approx(1.0)

    def test_clean_instrument_returns_empty(self):
        con = _make_con()
        _insert_option_greeks(con, [_base_greek(iid=1, dte=37, delta=-0.25)])
        _insert_option_bars(con, [_base_bar(iid=1, close=5.0, right="P")])
        result = phase4_integrity.run(con)
        assert result.is_empty()


# ---------------------------------------------------------------------------
# AuditRunner integration
# ---------------------------------------------------------------------------

class TestAuditRunner:
    def _build_db(self, tmp_path) -> str:
        """Write a minimal DuckDB file with one flagged and one clean instrument."""
        db_path = str(tmp_path / "test_input.db")
        con = duckdb.connect(db_path)
        con.execute("""
            CREATE TABLE option_greeks (
                ts_event TIMESTAMPTZ NOT NULL, instrument_id INTEGER NOT NULL,
                underlying_id INTEGER NOT NULL, dte INTEGER NOT NULL,
                T DOUBLE NOT NULL, iv DOUBLE, delta DOUBLE,
                gamma DOUBLE, theta DOUBLE, vega DOUBLE
            );
            CREATE TABLE option_bars (
                ts_event TIMESTAMPTZ NOT NULL, instrument_id INTEGER NOT NULL,
                underlying_id INTEGER NOT NULL, symbol VARCHAR NOT NULL,
                expiration DATE NOT NULL, strike_price DOUBLE NOT NULL,
                "right" VARCHAR(1) NOT NULL, multiplier INTEGER NOT NULL,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT
            );
            CREATE TABLE underlying_bars (
                ts_event TIMESTAMPTZ NOT NULL, instrument_id INTEGER NOT NULL,
                symbol VARCHAR NOT NULL, expiration DATE,
                open DOUBLE NOT NULL, high DOUBLE NOT NULL,
                low DOUBLE NOT NULL, close DOUBLE NOT NULL, volume BIGINT
            );
        """)
        # Instrument 1: negative close (hard flag)
        con.execute("""
            INSERT INTO option_bars VALUES (
                '2024-01-10 14:00:00+00'::TIMESTAMPTZ, 1, 100, 'EW P5000',
                '2024-02-16', 5000.0, 'P', 50, -1.0, -1.0, -1.0, -1.0, 100
            )
        """)
        # Instrument 2: clean
        con.execute("""
            INSERT INTO option_greeks VALUES (
                '2024-01-10 14:00:00+00'::TIMESTAMPTZ, 2, 100, 37, 0.10,
                0.20, -0.25, 0.01, -0.05, 0.30
            );
            INSERT INTO option_bars VALUES (
                '2024-01-10 14:00:00+00'::TIMESTAMPTZ, 2, 100, 'EW P4900',
                '2024-02-16', 4900.0, 'P', 50, 5.0, 5.1, 4.9, 5.0, 100
            )
        """)
        con.close()
        return db_path

    def test_run_writes_audit_table(self, tmp_path):
        db_path = self._build_db(tmp_path)
        runner = AuditRunner(db_path, skip_phase2=True)
        result = runner.run(verbose=False)

        # Verify table was written
        con = duckdb.connect(db_path, read_only=True)
        count = con.execute("SELECT COUNT(*) FROM option_audit").fetchone()[0]
        con.close()
        assert count > 0
        assert result.total_flagged_instruments >= 1

    def test_dry_run_does_not_write(self, tmp_path):
        db_path = self._build_db(tmp_path)
        runner = AuditRunner(db_path, dry_run=True, skip_phase2=True)
        runner.run(verbose=False)

        con = duckdb.connect(db_path, read_only=True)
        table_exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'option_audit'"
        ).fetchone()[0]
        con.close()
        assert table_exists == 0

    def test_result_contains_negative_close_flag(self, tmp_path):
        db_path = self._build_db(tmp_path)
        runner = AuditRunner(db_path, skip_phase2=True)
        result = runner.run(verbose=False)
        codes = {fc.flag_code for fc in result.flag_counts}
        assert "NEGATIVE_CLOSE" in codes

    def test_run_twice_is_idempotent(self, tmp_path):
        db_path = self._build_db(tmp_path)
        runner = AuditRunner(db_path, skip_phase2=True)
        r1 = runner.run(verbose=False)
        r2 = runner.run(verbose=False)
        assert r1.total_flagged_instruments == r2.total_flagged_instruments
        assert len(r1.flag_counts) == len(r2.flag_counts)
