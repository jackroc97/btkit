"""
Unit tests for the Chart Explorer API (btkit/analysis/api/routes/explore.py).

Helper-level tests (aggregation, placement, root parsing) run standalone.
Endpoint tests run against the fixture input DB when present, calling the route
functions directly with explicit args (FastAPI's Query defaults are only
resolved over HTTP, so every parameter is passed).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btkit.analysis.api.routes import explore as E  # noqa: N812

_FIXTURE = Path(__file__).parents[1] / "output" / "input.db"


def _body(resp) -> dict:
    return json.loads(resp.body)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_root_of(self):
        assert E._root_of("ESM6") == "ES"
        assert E._root_of("NQZ24") == "NQ"
        assert E._root_of("ES C4000") is None  # not a plain future symbol

    def test_placement(self):
        assert E._placement("sma_5") == "overlay"
        assert E._placement("vwap") == "overlay"
        assert E._placement("rsi_14") == "panel"
        assert E._placement("atr_14") == "panel"
        assert E._placement("ves1d_zscore") == "panel"

    def _minute_bars(self, rows):
        # rows: list of (open, high, low, close, volume) at consecutive minutes
        t0 = datetime(2020, 1, 2, 14, 30, tzinfo=UTC)
        return pl.DataFrame(
            {
                "ts_event": pl.Series(
                    [t0 + timedelta(minutes=i) for i in range(len(rows))],
                    dtype=pl.Datetime("us", "UTC"),
                ),
                "open": [r[0] for r in rows],
                "high": [r[1] for r in rows],
                "low": [r[2] for r in rows],
                "close": [r[3] for r in rows],
                "volume": [r[4] for r in rows],
            }
        )

    def test_aggregate_ohlcv_5m(self):
        # Five 1-min bars → one 5m candle: open=first, high=max, low=min,
        # close=last, volume=sum; color from up/down.
        bars = self._minute_bars(
            [
                (100, 102, 99, 101, 10),
                (101, 105, 100, 104, 20),
                (104, 104, 98, 100, 30),
                (100, 103, 97, 102, 40),
                (102, 106, 101, 103, 50),
            ]
        )
        candles, volume = E._aggregate_ohlcv(bars, "5m")
        assert len(candles) == 1
        c = candles[0]
        assert (c["open"], c["high"], c["low"], c["close"]) == (100, 106, 97, 103)
        assert volume[0]["value"] == 150
        assert volume[0]["color"] == E._UP  # close 103 >= open 100

    def test_aggregate_ohlcv_1m_passthrough(self):
        bars = self._minute_bars([(100, 101, 99, 100, 5), (100, 102, 99, 98, 6)])
        candles, _ = E._aggregate_ohlcv(bars, "1m")
        assert len(candles) == 2
        assert candles[1]["close"] == 98

    def test_bucket_last(self):
        df = pl.DataFrame(
            {
                "ts_event": pl.Series(
                    [
                        datetime(2020, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=i)
                        for i in range(5)
                    ],
                    dtype=pl.Datetime("us", "UTC"),
                ),
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        out = E._bucket_last(df, "5m")
        assert len(out) == 1 and out[0]["value"] == 5.0  # last-in-bucket


# ---------------------------------------------------------------------------
# Empty / unconfigured DB — graceful degradation
# ---------------------------------------------------------------------------


class TestNoInputDb:
    def test_endpoints_return_empty_without_db(self, monkeypatch):
        monkeypatch.delenv("BTKIT_INPUT_DB", raising=False)
        assert _body(E.list_contracts()) == {"contracts": []}
        assert (
            _body(E.get_bars(timeframe="1D", instrument_id=1, root=None, start=None, end=None))[
                "has_data"
            ]
            is False
        )
        assert _body(E.list_explore_indicators(instrument_id=1, root=None)) == {"indicators": []}


# ---------------------------------------------------------------------------
# Fixture-backed endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture input.db not built")
class TestAgainstFixture:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.setenv("BTKIT_INPUT_DB", str(_FIXTURE))

    def test_contracts_has_continuous_and_individual(self):
        contracts = _body(E.list_contracts())["contracts"]
        kinds = {c["kind"] for c in contracts}
        assert "continuous" in kinds and "contract" in kinds
        # continuous entries sort first
        assert contracts[0]["kind"] == "continuous"

    def test_individual_bars_aggregate(self):
        contracts = _body(E.list_contracts())["contracts"]
        esm = next(c for c in contracts if c["symbol"] == "ESM6" and c["kind"] == "contract")
        b1m = _body(
            E.get_bars(
                timeframe="1m", instrument_id=esm["instrument_id"], root=None, start=None, end=None
            )
        )
        b1h = _body(
            E.get_bars(
                timeframe="1H", instrument_id=esm["instrument_id"], root=None, start=None, end=None
            )
        )
        assert b1m["has_data"] and b1h["has_data"]
        # coarser timeframe yields fewer candles
        assert len(b1h["candles"]) < len(b1m["candles"])
        c = b1h["candles"][0]
        assert c["high"] >= max(c["open"], c["close"]) >= min(c["open"], c["close"]) >= c["low"]

    def test_continuous_bars_and_indicator(self):
        contracts = _body(E.list_contracts())["contracts"]
        cont = next(c for c in contracts if c["kind"] == "continuous")
        b = _body(
            E.get_bars(timeframe="1D", root=cont["root"], instrument_id=None, start=None, end=None)
        )
        assert b["has_data"] and len(b["candles"]) > 0

        inds = _body(E.list_explore_indicators(root=cont["root"], instrument_id=None))["indicators"]
        assert inds and all(i["id"] is None for i in inds)  # continuous → keyed by name
        s = _body(
            E.get_indicator(
                timeframe="1D",
                name=inds[0]["name"],
                root=cont["root"],
                indicator_id=None,
                start=None,
                end=None,
            )
        )
        assert len(s["data"]) > 0
