"""
Unit tests for time-aware OHLCV bar routing in DatabaseBuilder.

Regression coverage for the recycled-instrument-ID bug: Databento reassigns a
numeric instrument_id from an expired instrument to a new one, so a single ID can
be a future in one window and an option in another. `_route_bars_to_segments`
must classify each bar by the definition whose [activation, expiration] window
contains it — not by a single "winner" definition for all time.

Routing is driven by each definition's PUBLICATION time (the definition record's
own ts_event), not by its [activation, expiration] window. `activation` is when the
contract was listed, which can precede the period the id referred to it, so a
recycled id's activation windows overlap — and because CME lists options with fixed
tenors, the two windows are usually the same length, making the old tightest-window
tie-break a coin flip. See docs/data_duplication_fix.md §5.

These tests exercise the pure routing/parse helpers directly (no DBN files or DB
needed), and assert the regression invariants: correct reclassification, row
conservation, no ID in two classes at the same ts_event, publication-window
precedence, deterministic resolution of activation-window ties, nearest-segment
(not drop) when only the fallback applies, and no collateral change to
non-recycled IDs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from btkit.pipeline.builder import (
    _PUB_MAX,
    _PUB_MIN,
    _build_segments_frame,
    _InstrumentInfo,
    _ns_to_date,
    _ns_to_datetime,
    _order_segments,
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
    "_pub_from": pl.Datetime("us", "UTC"),
    "_pub_to": pl.Datetime("us", "UTC"),
    "_seg_rank": pl.Int64,
}


def _seg(
    iid,
    cls,
    symbol,
    act,
    exp,
    *,
    underlying_id=0,
    strike=None,
    right=None,
    pub_from=None,
    pub_to=None,
    seg_rank=0,
):
    """One routing segment.

    pub_from/pub_to default to the widest possible publication window, which makes
    every segment a published match — the shape _build_segments_frame produces for a
    non-recycled id. Tests that care about recycling set them explicitly.
    """
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
        "_pub_from": pub_from if pub_from is not None else _PUB_MIN,
        "_pub_to": pub_to if pub_to is not None else _PUB_MAX,
        "_seg_rank": seg_rank,
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
# The future's mapping is published from the beginning of time (first segment for
# the id) until the call's mapping is published in mid-2022.
_RECYCLE_HANDOVER = datetime(2022, 6, 1, 23, 0, tzinfo=UTC)
_RECYCLED = _segments(
    [
        _seg(
            21336,
            "F",
            "ESM0",
            date(2019, 3, 1),
            date(2020, 6, 19),
            underlying_id=21336,
            pub_from=_PUB_MIN,
            pub_to=_RECYCLE_HANDOVER,
            seg_rank=0,
        ),
        _seg(
            21336,
            "C",
            "ES C3855",
            date(2022, 6, 1),
            date(2023, 2, 17),
            strike=3855.0,
            right="C",
            pub_from=_RECYCLE_HANDOVER,
            pub_to=_PUB_MAX,
            seg_rank=1,
        ),
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

    def test_bar_after_expiry_but_before_reassignment_stays_with_the_future(self):
        """A 2021 bar sits past the future's expiration but before the id was
        reassigned, so the published mapping still resolves it to the future."""
        bars = _bars([(_dt(2021, 1, 1), 21336)])
        resolved, _ = _route_bars_to_segments(bars, _RECYCLED)
        assert resolved.height == 1
        row = resolved.row(0, named=True)
        assert row["instrument_class"] == "F"
        assert row["_in_pub"] is True


class TestOverlapAndMissing:
    def test_publication_window_wins_over_activation_overlap(self):
        """The real ES defect: two contracts on one id whose activation windows both
        contain the bar AND have identical length, so the old tightest-window
        tie-break was arbitrary. The publication window decides it."""
        handover = datetime(2022, 9, 8, 23, 0, tzinfo=UTC)
        segs = _segments(
            [
                # Both windows are 35 days long, as CME's fixed tenors make them.
                _seg(
                    116864,
                    "P",
                    "E1CU2 P4060",
                    date(2022, 8, 3),
                    date(2022, 9, 7),
                    strike=4060.0,
                    right="P",
                    pub_from=_PUB_MIN,
                    pub_to=handover,
                    seg_rank=0,
                ),
                _seg(
                    116864,
                    "P",
                    "E1AV2 P2880",
                    date(2022, 8, 29),
                    date(2022, 10, 3),
                    strike=2880.0,
                    right="P",
                    pub_from=handover,
                    pub_to=_PUB_MAX,
                    seg_rank=1,
                ),
            ]
        )
        bars = _bars([(_dt(2022, 9, 1), 116864)])  # inside BOTH activation windows
        resolved, n_overlap = _route_bars_to_segments(bars, segs)
        assert n_overlap == 0  # publication windows do not overlap
        assert resolved.row(0, named=True)["symbol"] == "E1CU2 P4060"

    def test_same_day_bars_resolve_consistently(self):
        """Databento only guarantees instrument_id uniqueness per day, so every bar
        on one day must resolve to the same contract."""
        segs = _segments(
            [
                _seg(
                    116864,
                    "P",
                    "early",
                    date(2022, 8, 3),
                    date(2022, 9, 7),
                    strike=4060.0,
                    right="P",
                    pub_from=_PUB_MIN,
                    pub_to=datetime(2022, 9, 8, 23, 0, tzinfo=UTC),
                    seg_rank=0,
                ),
                _seg(
                    116864,
                    "P",
                    "late",
                    date(2022, 8, 29),
                    date(2022, 10, 3),
                    strike=2880.0,
                    right="P",
                    pub_from=datetime(2022, 9, 8, 23, 0, tzinfo=UTC),
                    pub_to=_PUB_MAX,
                    seg_rank=1,
                ),
            ]
        )
        same_day = [(datetime(2022, 9, 1, h, 11, tzinfo=UTC), 116864) for h in (11, 15, 19, 23)]
        resolved, _ = _route_bars_to_segments(_bars(same_day), segs)
        assert resolved["symbol"].n_unique() == 1

    def test_activation_tie_is_resolved_deterministically(self):
        """Fallback path only: with no publication information the two candidates tie
        on (_dist, _win), so _seg_rank must settle it the same way every time."""
        segs = _segments(
            [
                _seg(
                    7,
                    "C",
                    "first",
                    date(2020, 1, 1),
                    date(2020, 2, 5),
                    strike=10.0,
                    right="C",
                    seg_rank=0,
                ),
                _seg(
                    7,
                    "C",
                    "second",
                    date(2020, 1, 2),
                    date(2020, 2, 6),
                    strike=20.0,
                    right="C",
                    seg_rank=1,
                ),
            ]
        )
        bars = _bars([(_dt(2020, 1, 15), 7)])
        winners = set()
        for _ in range(8):
            resolved, _n = _route_bars_to_segments(bars, segs)
            winners.add(resolved.row(0, named=True)["symbol"])
        assert winners == {"first"}

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

    def test_ns_to_datetime_keeps_time_of_day(self):
        ts = datetime(2022, 9, 8, 23, 1, 36, tzinfo=UTC)
        assert _ns_to_datetime(int(ts.timestamp() * 1_000_000_000)) == ts

    def test_ns_to_datetime_sentinels(self):
        assert _ns_to_datetime(0) is None
        assert _ns_to_datetime(2**64 - 1) is None


class TestSegmentWindows:
    """_order_segments + _build_segments_frame turn publication timestamps into
    back-to-back validity windows."""

    @staticmethod
    def _info(symbol, act, exp, pub_from):
        return _InstrumentInfo("P", symbol, act, exp, 1.0, "P", 50, 9, pub_from=pub_from)

    def test_order_segments_sorts_by_publication_time(self):
        late = self._info(
            "late", date(2022, 8, 29), date(2022, 10, 3), datetime(2022, 10, 2, tzinfo=UTC)
        )
        early = self._info(
            "early", date(2022, 8, 3), date(2022, 9, 7), datetime(2022, 8, 3, tzinfo=UTC)
        )
        assert [s.raw_symbol for s in _order_segments([late, early])] == ["early", "late"]

    def test_windows_are_back_to_back(self):
        early = self._info(
            "early", date(2022, 8, 3), date(2022, 9, 7), datetime(2022, 8, 3, tzinfo=UTC)
        )
        late = self._info(
            "late", date(2022, 8, 29), date(2022, 10, 3), datetime(2022, 10, 2, tzinfo=UTC)
        )
        frame = _build_segments_frame({116864: _order_segments([early, late])})
        rows = frame.sort("_seg_rank").to_dicts()
        assert rows[0]["_pub_to"] == rows[1]["_pub_from"] == datetime(2022, 10, 2, tzinfo=UTC)

    def test_first_segment_window_opens_at_the_beginning_of_time(self):
        """Bars predating our definition coverage must still classify."""
        only = self._info(
            "only", date(2022, 8, 3), date(2022, 9, 7), datetime(2022, 8, 3, tzinfo=UTC)
        )
        frame = _build_segments_frame({1: _order_segments([only])})
        assert frame.row(0, named=True)["_pub_from"] == _PUB_MIN

    def test_last_segment_window_runs_to_the_end_of_time(self):
        only = self._info(
            "only", date(2022, 8, 3), date(2022, 9, 7), datetime(2022, 8, 3, tzinfo=UTC)
        )
        frame = _build_segments_frame({1: _order_segments([only])})
        assert frame.row(0, named=True)["_pub_to"] == _PUB_MAX

    def test_windows_cover_the_timeline_without_gaps(self):
        infos = [
            self._info("a", date(2019, 1, 1), date(2019, 3, 1), datetime(2019, 1, 1, tzinfo=UTC)),
            self._info("b", date(2021, 1, 1), date(2021, 3, 1), datetime(2021, 1, 1, tzinfo=UTC)),
            self._info("c", date(2023, 1, 1), date(2023, 3, 1), datetime(2023, 1, 1, tzinfo=UTC)),
        ]
        frame = _build_segments_frame({5: _order_segments(infos)}).sort("_seg_rank")
        rows = frame.to_dicts()
        assert rows[0]["_pub_from"] == _PUB_MIN
        for prev, nxt in zip(rows, rows[1:], strict=False):
            assert prev["_pub_to"] == nxt["_pub_from"]
        assert rows[-1]["_pub_to"] == _PUB_MAX

    def test_seg_rank_is_publication_order(self):
        infos = [
            self._info("b", date(2021, 1, 1), date(2021, 3, 1), datetime(2021, 1, 1, tzinfo=UTC)),
            self._info("a", date(2019, 1, 1), date(2019, 3, 1), datetime(2019, 1, 1, tzinfo=UTC)),
        ]
        frame = _build_segments_frame({5: _order_segments(infos)}).sort("_seg_rank")
        assert frame["symbol"].to_list() == ["a", "b"]
