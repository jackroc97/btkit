"""
Regression tests for the `session.weekdays_only` off-by-one bug.

Polars' `.dt.weekday()` is ISO (Mon=1..Sun=7), so a `< 5` threshold silently
drops Friday (weekday=5) along with Sat/Sun, instead of keeping Mon-Fri and
dropping only Sat/Sun. Fixed to `<= 5` in:
  - EntryScanner._apply_window_filters (btkit/backtest/entry.py)
  - ExitScanner._find_first_hit's session gate (btkit/backtest/exit.py)
"""

from __future__ import annotations

from datetime import date, time

import polars as pl

from btkit.backtest.entry import EntryScanner
from btkit.backtest.exit import ExitScanner
from btkit.strategy.definition import (
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
# EntryScanner._apply_window_filters
# ---------------------------------------------------------------------------


def _entry_scanner(weekdays_only: bool = True) -> EntryScanner:
    trade = TradeDefinition(
        name="t",
        instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
        entry=EntryConfig(window=EntryWindowConfig(start=time(0, 0), end=time(23, 59))),
        legs=[
            LegConfig(
                name="sp",
                right="put",
                action="sell_to_open",
                dte=0,
                delta={"target": -0.10},
            )
        ],
        exit=ExitConfig(stop_loss=2.0),
    )
    strategy = StrategyDefinition(
        name="test",
        universe=UniverseConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
        ),
        trades=[trade],
    )
    strategy.universe.session.weekdays_only = weekdays_only
    return EntryScanner(db=None, strategy=strategy, trade=trade)


class TestEntryWeekdaysOnlyFilter:
    def _week_bars(self) -> pl.DataFrame:
        # 2024-01-01 (Mon) .. 2024-01-07 (Sun), one noon-UTC bar per day.
        ts = [f"2024-01-0{d} 12:00:00" for d in range(1, 8)]
        return pl.DataFrame(
            {"ts_event": pl.Series(ts, dtype=pl.Datetime("us", "UTC"))}
        )

    def test_friday_kept_saturday_sunday_dropped(self):
        scanner = _entry_scanner(weekdays_only=True)
        result = scanner._apply_window_filters(self._week_bars())
        kept_weekdays = {t.weekday() for t in result["ts_event"].to_list()}
        # python date.weekday(): Mon=0..Sun=6. Fri=4 must be present; Sat=5/Sun=6 absent.
        assert kept_weekdays == {0, 1, 2, 3, 4}, (
            f"expected Mon-Fri only, got weekdays {kept_weekdays}"
        )

    def test_weekdays_only_false_keeps_everything(self):
        scanner = _entry_scanner(weekdays_only=False)
        result = scanner._apply_window_filters(self._week_bars())
        assert len(result) == 7


# ---------------------------------------------------------------------------
# ExitScanner._find_first_hit session gate
# ---------------------------------------------------------------------------


def _exit_scanner() -> ExitScanner:
    strat = StrategyDefinition(
        name="test",
        universe=UniverseConfig(start_date=date(2026, 1, 1), end_date=date(2026, 3, 31)),
        trades=[
            TradeDefinition(
                name="trade1",
                instrument=InstrumentConfig(root_symbol="ES", asset_class="future"),
                entry=EntryConfig(window=EntryWindowConfig(start=time(9, 30), end=time(16, 0))),
                legs=[
                    LegConfig(
                        name="short_put",
                        right="put",
                        action="sell_to_open",
                        delta={"target": -0.25},
                        dte=0,
                    )
                ],
                exit=ExitConfig(stop_loss=1.0, expiry_exit=False),
            )
        ],
    )
    return ExitScanner(db=None, strategy=strat, trade=strat.trades[0])


def _pm_entries_for_sl(
    ts_event_utc: str, position_mark: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    pm_df = pl.DataFrame(
        {
            "entry_id": [0],
            "ts_event": pl.Series([ts_event_utc], dtype=pl.Datetime("us", "UTC")),
            "position_mark": [position_mark],
            "spread_open_mark": [position_mark],
        }
    )
    entries_df = pl.DataFrame(
        {
            "entry_id": [0],
            "entry_time": pl.Series(["2026-01-12 16:00:00"], dtype=pl.Datetime("us", "UTC")),
            "open_mark": pl.Series([1.00], dtype=pl.Float64),
            "tp_price": pl.Series([None], dtype=pl.Float64),
            "sl_price": pl.Series([2.00], dtype=pl.Float64),
            "dte_exit": pl.Series([None], dtype=pl.Int32),
            "leg_short_put_expiration": pl.Series([date(2026, 2, 20)], dtype=pl.Date),
        }
    )
    return pm_df, entries_df


class TestExitWeekdaysOnlySessionGate:
    # 2026-01-16 is a Friday, 11:00 ET = 16:00 UTC — within the 9:30-16:00 session.
    def test_stop_loss_fires_on_friday(self):
        scanner = _exit_scanner()
        pm_df, entries_df = _pm_entries_for_sl("2026-01-16 16:00:00", position_mark=2.50)
        result = scanner._find_first_hit(pm_df, pl.DataFrame(), entries_df)
        assert not result.is_empty(), (
            "stop_loss should fire on a Friday bar within session hours — "
            "the weekdays_only session gate must not exclude Friday"
        )
        assert result["exit_reason"][0] == "stop_loss"

    def test_stop_loss_does_not_fire_when_weekdays_only_off_is_unaffected(self):
        # Sanity check: same Friday bar with weekdays_only explicitly False —
        # should behave identically (Friday was never the issue when the
        # gate is disabled outright).
        scanner = _exit_scanner()
        scanner.strategy.universe.session.weekdays_only = False
        pm_df, entries_df = _pm_entries_for_sl("2026-01-16 16:00:00", position_mark=2.50)
        result = scanner._find_first_hit(pm_df, pl.DataFrame(), entries_df)
        assert not result.is_empty()
        assert result["exit_reason"][0] == "stop_loss"
