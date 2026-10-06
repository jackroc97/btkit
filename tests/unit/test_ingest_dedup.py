"""
Regression tests for duplicate-row insertion during OHLCV ingest.

The defect these cover put 4.84M duplicate option bars and 192k duplicate
underlying bars into the ES input database (docs/data_duplication_fix.md §2):

  * `_write_df`'s fast path skips the NOT EXISTS existence check for rows newer
    than `cutoff_ts`. The cutoff was computed once before the ingest loop, so when
    a later batch re-supplied bars from an overlapping source file those rows still
    compared as "newer" and were inserted a second time. `_advance_cutoff` keeps the
    cutoff in step with what has actually been written.

  * A plain INSERT (the old full-build path) has no check at all, so overlapping
    source files duplicated rows on a full build too. Both paths now dedup.

  * NOT EXISTS cannot deduplicate within the frame being inserted — its subquery
    sees only the pre-insert table — which is why the caller must hand over a frame
    already unique on (ts_event, instrument_id).

  * `_work_unit_sort_key` orders work units by the date range in the filename so
    the cutoff advances monotonically and the fast path stays effective. Zip-name
    order does not imply date order: a full-history re-download sorts after the
    incremental zips whose dates it already covers, which is exactly the ES layout.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl
import pytest

from btkit.db.input_db import INPUT_SCHEMA_SQL
from btkit.pipeline import greeks as greeks_mod
from btkit.pipeline.builder import _advance_cutoff, _work_unit_sort_key, _write_df
from btkit.pipeline.greeks import GreeksCalculator


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(INPUT_SCHEMA_SQL)
    return con


def _ts(day: int, hour: int = 14) -> datetime:
    return datetime(2026, 1, day, hour, 30, tzinfo=UTC)


def _bars(timestamps: list[datetime], iid: int = 1) -> pl.DataFrame:
    """Minimal underlying_bars frame matching the table's column order."""
    return pl.DataFrame(
        {
            "ts_event": timestamps,
            "instrument_id": [iid] * len(timestamps),
            "symbol": ["ESH6"] * len(timestamps),
            "expiration": [None] * len(timestamps),
            "open": [100.0] * len(timestamps),
            "high": [101.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [100.5] * len(timestamps),
            "volume": [10] * len(timestamps),
        },
        schema={
            "ts_event": pl.Datetime("us", "UTC"),
            "instrument_id": pl.Int64,
            "symbol": pl.Utf8,
            "expiration": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        },
    )


def _count(con, ts: datetime | None = None) -> int:
    if ts is None:
        return con.execute("SELECT count(*) FROM underlying_bars").fetchone()[0]
    return con.execute("SELECT count(*) FROM underlying_bars WHERE ts_event = ?", [ts]).fetchone()[
        0
    ]


# ---------------------------------------------------------------------------
# The core defect: a stale cutoff duplicates re-supplied rows
# ---------------------------------------------------------------------------


class TestStaleCutoffDuplication:
    def test_stale_cutoff_duplicates_resupplied_rows(self):
        """Demonstrates the original defect: holding the cutoff at its pre-loop
        value lets a second batch re-insert bars that are already present."""
        con = _con()
        stale = datetime.min.replace(tzinfo=UTC)

        batch = _bars([_ts(5), _ts(6)])
        _write_df(con, "underlying_bars", batch, skip_existing=True, cutoff_ts=stale)
        # Same bars again, cutoff NOT advanced — the bug.
        _write_df(con, "underlying_bars", batch, skip_existing=True, cutoff_ts=stale)

        assert _count(con) == 4  # duplicated

    def test_advanced_cutoff_skips_resupplied_rows(self):
        """With the cutoff advanced after each write, the re-supplied batch is
        checked against the table and skipped."""
        con = _con()
        cutoff = datetime.min.replace(tzinfo=UTC)

        batch = _bars([_ts(5), _ts(6)])
        _write_df(con, "underlying_bars", batch, skip_existing=True, cutoff_ts=cutoff)
        cutoff = _advance_cutoff(cutoff, batch)
        _write_df(con, "underlying_bars", batch, skip_existing=True, cutoff_ts=cutoff)

        assert _count(con) == 2
        assert _count(con, _ts(5)) == 1
        assert _count(con, _ts(6)) == 1

    def test_overlapping_sources_out_of_order(self):
        """The ES layout: an incremental file, then a history file that re-covers
        the same dates and extends further. Only the new dates should be added."""
        con = _con()
        cutoff = datetime.min.replace(tzinfo=UTC)

        incremental = _bars([_ts(10), _ts(11)])
        _write_df(con, "underlying_bars", incremental, skip_existing=True, cutoff_ts=cutoff)
        cutoff = _advance_cutoff(cutoff, incremental)

        history = _bars([_ts(9), _ts(10), _ts(11), _ts(12)])
        _write_df(con, "underlying_bars", history, skip_existing=True, cutoff_ts=cutoff)
        cutoff = _advance_cutoff(cutoff, history)

        assert _count(con) == 4
        for day in (9, 10, 11, 12):
            assert _count(con, _ts(day)) == 1

    def test_genuinely_new_rows_still_inserted(self):
        con = _con()
        cutoff = datetime.min.replace(tzinfo=UTC)
        first = _bars([_ts(5)])
        _write_df(con, "underlying_bars", first, skip_existing=True, cutoff_ts=cutoff)
        cutoff = _advance_cutoff(cutoff, first)

        later = _bars([_ts(6), _ts(7)])
        _write_df(con, "underlying_bars", later, skip_existing=True, cutoff_ts=cutoff)

        assert _count(con) == 3

    def test_same_timestamp_different_instruments_both_kept(self):
        con = _con()
        cutoff = datetime.min.replace(tzinfo=UTC)
        both = pl.concat([_bars([_ts(5)], iid=1), _bars([_ts(5)], iid=2)])
        _write_df(con, "underlying_bars", both, skip_existing=True, cutoff_ts=cutoff)
        assert _count(con) == 2


# ---------------------------------------------------------------------------
# NOT EXISTS does not dedup within the inserted frame
# ---------------------------------------------------------------------------


class TestWithinFrameDuplicates:
    def test_not_exists_does_not_dedup_within_frame(self):
        """Documents why the caller must pre-deduplicate: the subquery is evaluated
        against the pre-insert table, so both copies inside one frame pass."""
        con = _con()
        dup_frame = _bars([_ts(5), _ts(5)])
        _write_df(con, "underlying_bars", dup_frame, skip_existing=True, cutoff_ts=None)
        assert _count(con, _ts(5)) == 2

    def test_caller_side_unique_prevents_it(self):
        con = _con()
        dup_frame = _bars([_ts(5), _ts(5)]).unique(
            subset=["ts_event", "instrument_id"], keep="first"
        )
        _write_df(con, "underlying_bars", dup_frame, skip_existing=True, cutoff_ts=None)
        assert _count(con, _ts(5)) == 1


# ---------------------------------------------------------------------------
# _advance_cutoff
# ---------------------------------------------------------------------------


class TestAdvanceCutoff:
    def test_advances_to_frame_max(self):
        start = datetime.min.replace(tzinfo=UTC)
        assert _advance_cutoff(start, _bars([_ts(5), _ts(9)])) == _ts(9)

    def test_never_moves_backwards(self):
        assert _advance_cutoff(_ts(20), _bars([_ts(5)])) == _ts(20)

    def test_none_cutoff_stays_none(self):
        """None means 'no fast path'; an append that never had a cutoff must keep
        checking every row."""
        assert _advance_cutoff(None, _bars([_ts(5)])) is None

    def test_empty_frame_leaves_cutoff_unchanged(self):
        assert _advance_cutoff(_ts(5), _bars([])) == _ts(5)


# ---------------------------------------------------------------------------
# Chronological work-unit ordering
# ---------------------------------------------------------------------------


class TestWorkUnitSortKey:
    @staticmethod
    def _unit(name: str, zip_name: str = "GLBX-20260603-ZZZ.zip"):
        return (Path(zip_name), name)

    def test_history_file_sorts_before_later_incrementals(self):
        """The ES case: the history zip name sorts last but its data comes first."""
        history = self._unit(
            "glbx-mdp3-20260101-20260131.ohlcv-1m.dbn.zst", "GLBX-20260603-HIST.zip"
        )
        incremental = self._unit(
            "glbx-mdp3-20260422-20260521.ohlcv-1m.dbn.zst", "GLBX-20260522-INC.zip"
        )
        assert sorted([incremental, history], key=_work_unit_sort_key) == [history, incremental]

    def test_single_date_filename_parsed(self):
        unit = self._unit("glbx-mdp3-20260422.ohlcv-1m.dbn.zst")
        assert _work_unit_sort_key(unit)[:2] == ("20260422", "20260422")

    def test_date_range_filename_parsed(self):
        unit = self._unit("glbx-mdp3-20260101-20260131.ohlcv-1m.dbn.zst")
        assert _work_unit_sort_key(unit)[:2] == ("20260101", "20260131")

    def test_full_ordering_is_chronological(self):
        names = [
            "glbx-mdp3-20260615-20260930.ohlcv-1m.dbn.zst",
            "glbx-mdp3-20100606-20100630.ohlcv-1m.dbn.zst",
            "glbx-mdp3-20260422-20260521.ohlcv-1m.dbn.zst",
            "glbx-mdp3-20260101-20260421.ohlcv-1m.dbn.zst",
        ]
        ordered = sorted((self._unit(n) for n in names), key=_work_unit_sort_key)
        assert [_work_unit_sort_key(u)[0] for u in ordered] == [
            "20100606",
            "20260101",
            "20260422",
            "20260615",
        ]

    def test_unparseable_name_sorts_last_without_raising(self):
        dated = self._unit("glbx-mdp3-20260101.ohlcv-1m.dbn.zst")
        odd = self._unit("mystery.ohlcv-1m.dbn.zst")
        assert sorted([odd, dated], key=_work_unit_sort_key) == [dated, odd]

    def test_zip_date_is_not_used(self):
        """Only the payload's own date range matters; the zip's download date in the
        archive name must not influence ordering."""
        early_data_late_zip = self._unit(
            "glbx-mdp3-20200101.ohlcv-1m.dbn.zst", "GLBX-20261002-A.zip"
        )
        late_data_early_zip = self._unit(
            "glbx-mdp3-20250101.ohlcv-1m.dbn.zst", "GLBX-20260101-B.zip"
        )
        ordered = sorted([late_data_early_zip, early_data_late_zip], key=_work_unit_sort_key)
        assert ordered == [early_data_late_zip, late_data_early_zip]


# ---------------------------------------------------------------------------
# Greeks: dedup only when the database actually has duplicate bars
# ---------------------------------------------------------------------------


class TestGreeksConditionalDedup:
    """Duplicate option bars would yield duplicate greeks, because the NOT EXISTS
    filter only excludes rows already in option_greeks and neither copy is there
    yet. But deduplicating unconditionally is not affordable: a
    QUALIFY ROW_NUMBER() over the pending set is a window function over ~70M rows,
    which cannot stream and exhausted a 12.7 GiB limit on the full ES rebuild. So
    the dedup is applied only when a probe finds duplicates.
    """

    @staticmethod
    def _seed(con, *, copies: int) -> None:
        ts = _ts(5)
        con.execute(
            "INSERT INTO underlying_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [ts, 100, "ESH6", None, 5000.0, 5000.0, 5000.0, 5000.0, 10],
        )
        for _ in range(copies):
            con.execute(
                "INSERT INTO option_bars "
                "(ts_event, instrument_id, underlying_id, symbol, expiration, "
                'strike_price, "right", multiplier, open, high, low, close, volume) '
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    ts,
                    1,
                    100,
                    "EW1F6 P4900",
                    date(2026, 1, 30),
                    4900.0,
                    "P",
                    50,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    5,
                ],
            )

    def test_probe_false_on_clean_database(self):
        con = _con()
        self._seed(con, copies=1)
        assert GreeksCalculator(con)._has_duplicate_bars() is False

    def test_probe_true_on_duplicated_database(self):
        con = _con()
        self._seed(con, copies=2)
        assert GreeksCalculator(con)._has_duplicate_bars() is True

    def test_probe_false_on_empty_database(self):
        assert GreeksCalculator(_con())._has_duplicate_bars() is False

    def test_clean_database_yields_one_greeks_row_per_bar(self):
        con = _con()
        self._seed(con, copies=1)
        GreeksCalculator(con).run(skip_existing=True)
        assert con.execute("SELECT count(*) FROM option_greeks").fetchone()[0] == 1

    def test_duplicated_bars_still_yield_one_greeks_row(self):
        con = _con()
        self._seed(con, copies=2)
        GreeksCalculator(con).run(skip_existing=True)
        assert con.execute("SELECT count(*) FROM option_greeks").fetchone()[0] == 1

    def test_three_copies_still_yield_one_greeks_row(self):
        con = _con()
        self._seed(con, copies=3)
        GreeksCalculator(con).run(skip_existing=True)
        assert con.execute("SELECT count(*) FROM option_greeks").fetchone()[0] == 1

    def test_distinct_bars_are_not_collapsed(self):
        """The dedup keys on (instrument_id, ts_event); two genuinely different
        bars must both get greeks."""
        con = _con()
        self._seed(con, copies=2)  # duplicates, so the dedup plan is active
        con.execute(
            "INSERT INTO underlying_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [_ts(6), 100, "ESH6", None, 5000.0, 5000.0, 5000.0, 5000.0, 10],
        )
        con.execute(
            "INSERT INTO option_bars "
            "(ts_event, instrument_id, underlying_id, symbol, expiration, "
            'strike_price, "right", multiplier, open, high, low, close, volume) '
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                _ts(6),
                1,
                100,
                "EW1F6 P4900",
                date(2026, 1, 30),
                4900.0,
                "P",
                50,
                1.0,
                1.0,
                1.0,
                1.0,
                5,
            ],
        )
        GreeksCalculator(con).run(skip_existing=True)
        assert con.execute("SELECT count(*) FROM option_greeks").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# Greeks: every pending bar must produce a greeks row
# ---------------------------------------------------------------------------


class TestGreeksCompleteness:
    """Regression cover for a silent-truncation bug.

    Greeks are computed in chunks pulled from a temp table while bulk INSERTs run
    on the same connection. An earlier version streamed those chunks with
    to_arrow_reader(); DuckDB allows one active result per connection, so the first
    INSERT invalidated the reader and iteration stopped with no error, writing
    8,750,000 of 68,936,444 rows on a full rebuild. Chunks are now addressed by
    rowid range, and run() asserts the output row count matches the input.
    """

    @staticmethod
    def _seed(con, n_bars: int) -> None:
        """n_bars distinct option bars, each with a same-minute underlying bar."""
        base = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
        und = [
            (base + timedelta(minutes=i), 100, "ESH6", None, 5000.0, 5000.0, 5000.0, 5000.0, 10)
            for i in range(n_bars)
        ]
        opt = [
            (
                base + timedelta(minutes=i),
                1,
                100,
                "EW1F6 P4900",
                date(2026, 1, 30),
                4900.0,
                "P",
                50,
                1.0,
                1.0,
                1.0,
                1.0,
                5,
            )
            for i in range(n_bars)
        ]
        con.executemany("INSERT INTO underlying_bars VALUES (?,?,?,?,?,?,?,?,?)", und)
        con.executemany(
            "INSERT INTO option_bars "
            "(ts_event, instrument_id, underlying_id, symbol, expiration, strike_price, "
            '"right", multiplier, open, high, low, close, volume) '
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            opt,
        )

    def test_all_bars_get_greeks_across_many_chunks(self, monkeypatch):
        """Forces multiple chunks and multiple flushes, the configuration that
        triggered the truncation."""
        monkeypatch.setattr(greeks_mod, "_GREEKS_CHUNK_ROWS", 10)
        monkeypatch.setattr(greeks_mod, "_GREEKS_FLUSH_ROWS", 20)
        con = _con()
        self._seed(con, 250)
        GreeksCalculator(con).run(skip_existing=True)
        assert con.execute("SELECT count(*) FROM option_greeks").fetchone()[0] == 250

    def test_chunk_size_does_not_change_the_result(self, monkeypatch):
        counts = []
        for chunk in (3, 10, 1000):
            monkeypatch.setattr(greeks_mod, "_GREEKS_CHUNK_ROWS", chunk)
            con = _con()
            self._seed(con, 120)
            GreeksCalculator(con).run(skip_existing=True)
            counts.append(con.execute("SELECT count(*) FROM option_greeks").fetchone()[0])
        assert counts == [120, 120, 120]

    def test_single_bar_still_works(self, monkeypatch):
        monkeypatch.setattr(greeks_mod, "_GREEKS_CHUNK_ROWS", 10)
        con = _con()
        self._seed(con, 1)
        GreeksCalculator(con).run(skip_existing=True)
        assert con.execute("SELECT count(*) FROM option_greeks").fetchone()[0] == 1

    def test_incomplete_write_raises(self, monkeypatch):
        """The completeness guard must fail loudly, not return quietly."""
        monkeypatch.setattr(greeks_mod, "_GREEKS_CHUNK_ROWS", 10)
        con = _con()
        self._seed(con, 50)

        calc = GreeksCalculator(con)
        real = calc._compute_batch
        state = {"n": 0}

        def _drop_after_two(df):
            state["n"] += 1
            # Simulate rows vanishing mid-run, as the invalidated reader did.
            return real(df).head(0) if state["n"] > 2 else real(df)

        monkeypatch.setattr(calc, "_compute_batch", _drop_after_two)
        with pytest.raises(RuntimeError, match="greeks incomplete"):
            calc.run(skip_existing=True)

    def test_insert_size_is_bounded_by_rows_not_chunks(self, monkeypatch):
        """The flush threshold must be in rows, so a larger chunk size cannot
        enlarge the INSERT. Chunk-count batching turned a 250k chunk size into
        5,000,000-row INSERTs and exhausted DuckDB's memory limit."""
        monkeypatch.setattr(greeks_mod, "_GREEKS_CHUNK_ROWS", 50)
        monkeypatch.setattr(greeks_mod, "_GREEKS_FLUSH_ROWS", 60)
        con = _con()
        self._seed(con, 300)

        insert_sizes: list[int] = []

        class _Spy:
            """Delegating proxy: DuckDBPyConnection.execute is read-only, so the
            size of each bulk INSERT is observed through a wrapper instead."""

            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *a, **kw):
                if "INSERT INTO option_greeks" in sql:
                    insert_sizes.append(
                        self._inner.execute("SELECT count(*) FROM _greeks_batch").fetchone()[0]
                    )
                return self._inner.execute(sql, *a, **kw)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        GreeksCalculator(_Spy(con)).run(skip_existing=True)

        assert insert_sizes, "expected at least one bulk INSERT"
        # Each flush fires as soon as the row threshold is met, so an INSERT can
        # overshoot by at most one chunk.
        assert max(insert_sizes) <= 60 + 50
        assert con.execute("SELECT count(*) FROM option_greeks").fetchone()[0] == 300
