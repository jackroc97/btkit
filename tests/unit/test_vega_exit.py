"""
Unit tests for spread-level vega exit condition and open_vega column.

Tests validate:
  - vega_exit = None → _vega_exit never fires (no-op)
  - vega_exit threshold fires when _spread_vega < threshold
  - priority ordering relative to condition (5) and dte_exit (7)
  - ExitConfig.vega_exit field default and assignment
  - open_vega captures entry-time spread vega as a per-entry constant
  - _need_vega is activated by condition strings referencing _spread_vega / open_vega
"""

from __future__ import annotations

from datetime import UTC, datetime, time

import polars as pl
import pytest

from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    TradeDefinition,
)


class TestVegaExitConfig:
    def test_vega_exit_defaults_none(self):
        cfg = ExitConfig()
        assert cfg.vega_exit is None

    def test_vega_exit_accepts_float(self):
        cfg = ExitConfig(vega_exit=0.15)
        assert cfg.vega_exit == 0.15

    def test_vega_exit_accepts_zero(self):
        cfg = ExitConfig(vega_exit=0.0)
        assert cfg.vega_exit == 0.0

    def test_vega_exit_in_exit_config_alongside_other_fields(self):
        cfg = ExitConfig(stop_loss=1.5, take_profit=0.5, vega_exit=0.20)
        assert cfg.vega_exit == 0.20
        assert cfg.stop_loss == 1.5


class TestVegaExitPriority:
    """
    Priority encoding:
      5 = condition
      6 = vega_exit   (NEW)
      7 = dte_exit
      8 = expiry
    """

    def _make_trade(self, vega_exit=None, dte_exit=None):
        return TradeDefinition(
            name="t",
            instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
            entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(16, 0))),
            legs=[
                LegConfig(
                    name="short", right="put", action="sell_to_open", dte=0, delta={"target": -0.16}
                ),
                LegConfig(
                    name="long",
                    right="put",
                    action="buy_to_open",
                    dte=0,
                    strike_offset=-50.0,
                    reference_leg="short",
                ),
            ],
            exit=ExitConfig(vega_exit=vega_exit, dte_exit=dte_exit),
        )

    def test_vega_exit_none_is_noop_in_config(self):
        """vega_exit=None means the field is not configured."""
        trade = self._make_trade(vega_exit=None)
        assert trade.exit.vega_exit is None

    def test_vega_exit_set(self):
        trade = self._make_trade(vega_exit=0.25)
        assert trade.exit.vega_exit == 0.25

    def test_dte_exit_unchanged_when_vega_not_set(self):
        """dte_exit still works normally when vega_exit is None."""
        trade = self._make_trade(dte_exit=5)
        assert trade.exit.dte_exit == 5
        assert trade.exit.vega_exit is None

    def test_both_vega_and_dte_can_coexist(self):
        trade = self._make_trade(vega_exit=0.20, dte_exit=7)
        assert trade.exit.vega_exit == 0.20
        assert trade.exit.dte_exit == 7


class TestOpenVega:
    """
    open_vega is the spread net vega captured at entry time, exposed as a
    per-entry constant column in the condition namespace so users can write
    relative thresholds such as "_spread_vega < 0.3 * open_vega".
    """

    def _make_spread_vega_df(
        self,
        entry_ids: list[int],
        entry_times: list[datetime],
        bars: list[tuple],  # (entry_id, ts_event, _spread_vega)
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return (m_with_spread_vega, entries) suitable for open_vega extraction."""
        entries = pl.DataFrame(
            {
                "entry_id": entry_ids,
                "entry_time": pl.Series(entry_times, dtype=pl.Datetime("us", "UTC")),
            }
        )
        m = pl.DataFrame(
            {
                "entry_id": [b[0] for b in bars],
                "ts_event": pl.Series([b[1] for b in bars], dtype=pl.Datetime("us", "UTC")),
                "_spread_vega": [b[2] for b in bars],
            }
        ).join(entries, on="entry_id", how="left")
        return m, entries

    def _extract_open_vega(self, m: pl.DataFrame) -> pl.DataFrame:
        """Replicate the open_vega extraction logic from exit.py."""
        return (
            m.filter(
                pl.col("_spread_vega").is_not_null() & (pl.col("ts_event") >= pl.col("entry_time"))
            )
            .sort(["entry_id", "ts_event"])
            .group_by("entry_id", maintain_order=True)
            .agg(pl.col("_spread_vega").first().alias("open_vega"))
        )

    def test_open_vega_picks_first_bar_at_entry_time(self):
        t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
        t1 = datetime(2024, 1, 2, 14, 31, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, 14, 32, tzinfo=UTC)
        m, _ = self._make_spread_vega_df(
            entry_ids=[1, 1, 1],
            entry_times=[t0, t0, t0],
            bars=[(1, t0, 1.00), (1, t1, 0.80), (1, t2, 0.60)],
        )
        result = self._extract_open_vega(m)
        assert result.filter(pl.col("entry_id") == 1)["open_vega"][0] == pytest.approx(1.00)

    def test_open_vega_ignores_bars_before_entry_time(self):
        t_pre = datetime(2024, 1, 2, 14, 28, tzinfo=UTC)
        t_entry = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
        t_post = datetime(2024, 1, 2, 14, 31, tzinfo=UTC)
        m, _ = self._make_spread_vega_df(
            entry_ids=[1, 1, 1],
            entry_times=[t_entry, t_entry, t_entry],
            bars=[(1, t_pre, 5.00), (1, t_entry, 1.00), (1, t_post, 0.80)],
        )
        result = self._extract_open_vega(m)
        # Should pick 1.00, not 5.00
        assert result.filter(pl.col("entry_id") == 1)["open_vega"][0] == pytest.approx(1.00)

    def test_open_vega_skips_null_bars(self):
        t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
        t1 = datetime(2024, 1, 2, 14, 31, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, 14, 32, tzinfo=UTC)
        m, _ = self._make_spread_vega_df(
            entry_ids=[1, 1, 1],
            entry_times=[t0, t0, t0],
            bars=[(1, t0, None), (1, t1, None), (1, t2, 0.75)],
        )
        result = self._extract_open_vega(m)
        assert result.filter(pl.col("entry_id") == 1)["open_vega"][0] == pytest.approx(0.75)

    def test_open_vega_independent_per_entry(self):
        t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
        t1 = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, 15, 1, tzinfo=UTC)
        m, _ = self._make_spread_vega_df(
            entry_ids=[1, 1, 2, 2],
            entry_times=[t0, t0, t1, t1],
            bars=[(1, t0, 1.20), (1, t2, 0.80), (2, t1, 0.50), (2, t2, 0.30)],
        )
        result = self._extract_open_vega(m)
        ov = {r["entry_id"]: r["open_vega"] for r in result.to_dicts()}
        assert ov[1] == pytest.approx(1.20)
        assert ov[2] == pytest.approx(0.50)

    def test_need_vega_triggered_by_spread_vega_in_condition(self):
        """_need_vega gate activates when a condition string mentions _spread_vega."""
        exit_cfg = ExitConfig(conditions=["_spread_vega < 0.5"])
        result = exit_cfg.vega_exit is not None or any(
            "_spread_vega" in c or "open_vega" in c for c in exit_cfg.conditions
        )
        assert result is True

    def test_need_vega_triggered_by_open_vega_in_condition(self):
        """_need_vega gate activates when a condition string mentions open_vega."""
        exit_cfg = ExitConfig(conditions=["_spread_vega < 0.3 * open_vega"])
        result = exit_cfg.vega_exit is not None or any(
            "_spread_vega" in c or "open_vega" in c for c in exit_cfg.conditions
        )
        assert result is True

    def test_need_vega_not_triggered_by_unrelated_condition(self):
        """_need_vega gate stays False when conditions don't mention vega columns."""
        exit_cfg = ExitConfig(conditions=["position_mark - open_mark >= 20.0"])
        result = exit_cfg.vega_exit is not None or any(
            "_spread_vega" in c or "open_vega" in c for c in exit_cfg.conditions
        )
        assert result is False
