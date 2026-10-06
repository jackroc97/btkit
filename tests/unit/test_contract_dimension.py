"""
Tests for the contract dimension written during ingest.

Covers _build_segments_frame assigning contract_id per segment and
DatabaseBuilder._write_contracts upserting the dimension table, including the two
properties that matter for a recycled instrument_id:

  * one instrument_id carrying several contracts yields several contract rows;
  * the same contract reached through different instrument_ids collapses to one
    contract_id, because the key is (symbol, expiration).

See docs/data_duplication_fix.md §7.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import polars as pl

from btkit.db.contract_key import contract_id
from btkit.db.input_db import INPUT_SCHEMA_SQL
from btkit.pipeline.builder import (
    DatabaseBuilder,
    _build_segments_frame,
    _InstrumentInfo,
    _order_segments,
)


def _info(symbol, exp, *, cls="P", strike=5000.0, right="P", uid=100, pub=None):
    return _InstrumentInfo(
        instrument_class=cls,
        raw_symbol=symbol,
        activation=date(2024, 7, 1),
        expiration=exp,
        strike_price=strike,
        right=right,
        multiplier=50,
        underlying_id=uid,
        pub_from=pub or datetime(2024, 7, 1, tzinfo=UTC),
    )


def _builder(tmp_path) -> DatabaseBuilder:
    """A builder with an open connection but no ingest run."""
    db_path = tmp_path / "in.db"
    builder = DatabaseBuilder(raw_data_path=tmp_path, db_path=db_path)
    builder._con = duckdb.connect(str(db_path))
    builder._con.execute(INPUT_SCHEMA_SQL)
    return builder


class TestSegmentContractId:
    def test_segment_carries_contract_id(self):
        frame = _build_segments_frame(
            {1: _order_segments([_info("EW1Q4 P5000", date(2024, 8, 16))])}
        )
        assert frame.row(0, named=True)["contract_id"] == contract_id(
            "EW1Q4 P5000", date(2024, 8, 16)
        )

    def test_recycled_id_gets_distinct_contract_ids(self):
        segs = _order_segments(
            [
                _info("EW1Q4 P5000", date(2024, 8, 16), pub=datetime(2024, 7, 1, tzinfo=UTC)),
                _info("EW3U4 C5600", date(2024, 9, 20), pub=datetime(2024, 8, 20, tzinfo=UTC)),
            ]
        )
        frame = _build_segments_frame({1: segs})
        assert frame["contract_id"].n_unique() == 2

    def test_same_contract_under_two_instrument_ids_shares_contract_id(self):
        """The point of the surrogate key: identity follows the contract, not the
        publisher's recycled numeric id."""
        frame = _build_segments_frame(
            {
                11: _order_segments([_info("EW1Q4 P5000", date(2024, 8, 16))]),
                22: _order_segments([_info("EW1Q4 P5000", date(2024, 8, 16))]),
            }
        )
        assert frame["contract_id"].n_unique() == 1
        assert frame["instrument_id"].n_unique() == 2

    def test_same_symbol_different_expiration_differs(self):
        """CME's 10-year symbol cycle."""
        frame = _build_segments_frame(
            {
                1: _order_segments([_info("EW2H5 P2100", date(2015, 3, 13))]),
                2: _order_segments([_info("EW2H5 P2100", date(2025, 3, 14))]),
            }
        )
        assert frame["contract_id"].n_unique() == 2


class TestWriteContracts:
    def test_writes_one_row_per_contract(self, tmp_path):
        builder = _builder(tmp_path)
        frame = _build_segments_frame(
            {
                1: _order_segments([_info("EW1Q4 P5000", date(2024, 8, 16))]),
                2: _order_segments([_info("EW1Q4 C5000", date(2024, 8, 16), right="C", cls="C")]),
            }
        )
        builder._write_contracts(frame)
        n = builder._con.execute("SELECT count(*) FROM contract").fetchone()[0]
        assert n == 2

    def test_fields_are_persisted(self, tmp_path):
        builder = _builder(tmp_path)
        frame = _build_segments_frame(
            {1: _order_segments([_info("EW1Q4 P5000", date(2024, 8, 16))])}
        )
        builder._write_contracts(frame)
        row = builder._con.execute(
            'SELECT symbol, expiration, strike_price, "right", multiplier, instrument_class '
            "FROM contract"
        ).fetchone()
        assert row == ("EW1Q4 P5000", date(2024, 8, 16), 5000.0, "P", 50, "P")

    def test_is_idempotent(self, tmp_path):
        """--append re-derives the same contract_ids, so a second write must be a
        no-op rather than a primary-key violation or a duplicate."""
        builder = _builder(tmp_path)
        frame = _build_segments_frame(
            {1: _order_segments([_info("EW1Q4 P5000", date(2024, 8, 16))])}
        )
        builder._write_contracts(frame)
        builder._write_contracts(frame)
        n = builder._con.execute("SELECT count(*) FROM contract").fetchone()[0]
        assert n == 1

    def test_second_batch_adds_only_new_contracts(self, tmp_path):
        builder = _builder(tmp_path)
        builder._write_contracts(
            _build_segments_frame({1: _order_segments([_info("A P1", date(2024, 8, 16))])})
        )
        builder._write_contracts(
            _build_segments_frame(
                {
                    1: _order_segments([_info("A P1", date(2024, 8, 16))]),
                    2: _order_segments([_info("B P2", date(2024, 9, 20))]),
                }
            )
        )
        symbols = {r[0] for r in builder._con.execute("SELECT symbol FROM contract").fetchall()}
        assert symbols == {"A P1", "B P2"}

    def test_duplicate_contract_within_one_frame_collapses(self, tmp_path):
        """Two instrument_ids for the same contract must not both try to insert it."""
        builder = _builder(tmp_path)
        frame = _build_segments_frame(
            {
                11: _order_segments([_info("EW1Q4 P5000", date(2024, 8, 16))]),
                22: _order_segments([_info("EW1Q4 P5000", date(2024, 8, 16))]),
            }
        )
        builder._write_contracts(frame)
        n = builder._con.execute("SELECT count(*) FROM contract").fetchone()[0]
        assert n == 1

    def test_empty_frame_is_a_noop(self, tmp_path):
        builder = _builder(tmp_path)
        builder._write_contracts(_build_segments_frame({}))
        n = builder._con.execute("SELECT count(*) FROM contract").fetchone()[0]
        assert n == 0


class TestOptionBarsCarryContractId:
    def test_contract_id_column_exists_and_joins(self, tmp_path):
        """option_bars.contract_id must resolve against the dimension."""
        builder = _builder(tmp_path)
        frame = _build_segments_frame(
            {1: _order_segments([_info("EW1Q4 P5000", date(2024, 8, 16))])}
        )
        builder._write_contracts(frame)
        cid = contract_id("EW1Q4 P5000", date(2024, 8, 16))
        bars = pl.DataFrame(
            {
                "ts_event": pl.Series(
                    [datetime(2024, 8, 1, 14, 30, tzinfo=UTC)], dtype=pl.Datetime("us", "UTC")
                ),
                "instrument_id": pl.Series([1], dtype=pl.Int64),
                "contract_id": pl.Series([cid], dtype=pl.Int64),
                "underlying_id": pl.Series([100], dtype=pl.Int64),
                "symbol": ["EW1Q4 P5000"],
                "expiration": pl.Series([date(2024, 8, 16)], dtype=pl.Date),
                "strike_price": [5000.0],
                "right": ["P"],
                "multiplier": pl.Series([50], dtype=pl.Int64),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": pl.Series([5], dtype=pl.Int64),
            }
        )
        builder._con.register("_b", bars)
        builder._con.execute("INSERT INTO option_bars SELECT * FROM _b")
        joined = builder._con.execute(
            "SELECT c.symbol FROM option_bars b JOIN contract c USING (contract_id)"
        ).fetchall()
        assert joined == [("EW1Q4 P5000",)]
