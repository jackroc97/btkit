"""
Unit tests for time-aware OHLCV bar routing in DatabaseBuilder.

Regression coverage for the recycled-instrument-ID bug: Databento reassigns a
numeric instrument_id from an expired instrument to a new one, so a single ID can
be a future in one window and an option in another. `_route_bars_to_segments`
must classify each bar by the definition whose [activation, expiration] window
contains it — not by a single "winner" definition for all time.

These tests exercise the pure routing/parse helpers directly (no DBN files or DB
needed), and assert the report's regression invariants: correct reclassification,
row conservation, no ID in two classes at the same ts_event, tightest-window on
overlap, nearest-segment (not drop) on gap, and no collateral change to
non-recycled IDs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from btkit.pipeline.builder import (
    _InstrumentInfo,
    _ns_to_date,
    _route_bars_to_segments,
    _segment_sort_key,
)

_SEG_SCHEMA = {
    "instrument_id": pl.Int64,
    "instrument_class": pl.Utf8,
    "symbol": pl.Utf8,
    "seg_expiration": pl.Date,
    "underlying_id": pl.Int64,
    "strike_price": pl.Float64,
    "right": pl.Utf8,
    "multiplier": pl.Int64,
    "_act": pl.Date,
    "_exp_route": pl.Date,
}


def _seg(iid, cls, symbol, act, exp, *, underlying_id=0, strike=None, right=None):
    return {
        "instrument_id": iid,
        "instrument_class": cls,
        "symbol": symbol,
        "seg_expiration": exp,
        "underlying_id": underlying_id,
        "strike_price": strike,
        "right": right,
        "multiplier": 50,
        "_act": act,
        "_exp_route": exp,
    }


def _segments(rows):
    return pl.DataFrame(rows, schema=_SEG_SCHEMA)


def _bars(rows):
    """rows: list of (ts_event, instrument_id)."""
    return pl.DataFrame(
        {
            "ts_event": pl.Series([r[0] for r in rows], dtype=pl.Datetime("us", "UTC")),
            "instrument_id": pl.Series([r[1] for r in rows], dtype=pl.Int64),
            "open": pl.Series([1.0] * len(rows), dtype=pl.Float64),
            "high": pl.Series([1.0] * len(rows), dtype=pl.Float64),
            "low": pl.Series([1.0] * len(rows), dtype=pl.Float64),
            "close": pl.Series([1.0] * len(rows), dtype=pl.Float64),
            "volume": pl.Series([1] * len(rows), dtype=pl.Int64),
        }
    )


def _dt(y, m, d):
    return datetime(y, m, d, 14, 30, tzinfo=UTC)


# ESM0 future (2020) recycled as a 2023 call — the worked example from the report.
_RECYCLED = _segments(
    [
        _seg(21336, "F", "ESM0", date(2019, 3, 1), date(2020, 6, 19), underlying_id=21336),
        _seg(21336, "C", "ES C3855", date(2022, 6, 1), date(2023, 2, 17), strike=3855.0, right="C"),
    ]
)


class TestRecycledRouting:
    def test_future_and_option_bars_route_to_their_own_window(self):
        bars = _bars([(_dt(2020, 3, 10), 21336), (_dt(2023, 1, 10), 21336)])
        resolved, _ = _route_bars_to_segments(bars, _RECYCLED)
        by_ts = {r["ts_event"]: r for r in resolved.iter_rows(named=True)}
        # 2020 bar → the future; 2023 bar → the call
        assert by_ts[_dt(2020, 3, 10)]["instrument_class"] == "F"
        assert by_ts[_dt(2020, 3, 10)]["symbol"] == "ESM0"
        assert by_ts[_dt(2020, 3, 10)]["_contained"] is True
        assert by_ts[_dt(2023, 1, 10)]["instrument_class"] == "C"
        assert by_ts[_dt(2023, 1, 10)]["strike_price"] == 3855.0
        assert by_ts[_dt(2023, 1, 10)]["_contained"] is True

    def test_no_id_in_two_classes_at_same_ts(self):
        # Each (ts_event, instrument_id) resolves to exactly one class → a bar can
        # never land in both underlying_bars and option_bars.
        bars = _bars([(_dt(2020, 3, 10), 21336), (_dt(2023, 1, 10), 21336)])
        resolved, _ = _route_bars_to_segments(bars, _RECYCLED)
        assert resolved.select(["ts_event", "instrument_id"]).is_unique().all()

    def test_row_conservation(self):
        bars = _bars(
            [(_dt(2020, 3, 10), 21336), (_dt(2020, 4, 1), 21336), (_dt(2023, 1, 10), 21336)]
        )
        resolved, _ = _route_bars_to_segments(bars, _RECYCLED)
        assert resolved.height == bars.height  # reclassify, never add/drop

    def test_gap_bar_falls_back_to_nearest_not_dropped(self):
        # A bar between the future's expiry and the call's activation (2021) is in
        # no window; it must survive (nearest segment = the future), flagged as a
        # non-containment fallback.
        bars = _bars([(_dt(2021, 1, 1), 21336)])
        resolved, _ = _route_bars_to_segments(bars, _RECYCLED)
        assert resolved.height == 1
        row = resolved.row(0, named=True)
        assert row["instrument_class"] == "F"  # nearest by date distance
        assert row["_contained"] is False


class TestOverlapAndMissing:
    def test_overlap_tightest_window_wins_and_is_counted(self):
        segs = _segments(
            [
                _seg(7, "C", "wide", date(2020, 1, 1), date(2020, 12, 31), strike=10.0, right="C"),
                _seg(7, "C", "tight", date(2020, 3, 1), date(2020, 4, 30), strike=20.0, right="C"),
            ]
        )
        bars = _bars([(_dt(2020, 3, 15), 7)])  # inside both windows
        resolved, n_overlap = _route_bars_to_segments(bars, segs)
        assert n_overlap == 1
        assert resolved.row(0, named=True)["symbol"] == "tight"  # tightest window kept

    def test_bar_without_definition_is_dropped(self):
        segs = _segments([_seg(1, "F", "ESM0", date(2020, 1, 1), date(2020, 6, 19))])
        bars = _bars([(_dt(2020, 3, 1), 1), (_dt(2020, 3, 1), 999)])  # 999 has no segment
        resolved, _ = _route_bars_to_segments(bars, segs)
        assert set(resolved["instrument_id"].to_list()) == {1}

    def test_non_recycled_id_unchanged(self):
        segs = _segments(
            [
                _seg(
                    500,
                    "P",
                    "ESM0 P2400",
                    date(2020, 1, 1),
                    date(2020, 6, 19),
                    strike=2400.0,
                    right="P",
                )
            ]
        )
        bars = _bars([(_dt(2020, 2, 1), 500), (_dt(2020, 5, 1), 500)])
        resolved, n_overlap = _route_bars_to_segments(bars, segs)
        assert resolved.height == 2
        assert n_overlap == 0
        assert resolved["instrument_class"].unique().to_list() == ["P"]
        assert resolved["_contained"].all()


class TestParseHelpers:
    def test_ns_to_date_valid(self):
        ns = int(datetime(2020, 6, 19, tzinfo=UTC).timestamp() * 1_000_000_000)
        assert _ns_to_date(ns) == date(2020, 6, 19)

    def test_ns_to_date_sentinels(self):
        assert _ns_to_date(0) is None
        assert _ns_to_date(2**64 - 1) is None  # UINT64_MAX undefined marker

    def test_segment_sort_key_orders_by_activation_then_expiration(self):
        a = _InstrumentInfo("F", "a", date(2019, 1, 1), date(2020, 6, 19), None, None, 50, 1)
        b = _InstrumentInfo("C", "b", date(2022, 1, 1), date(2023, 2, 17), 1.0, "C", 50, 2)
        assert sorted([b, a], key=_segment_sort_key) == [a, b]
