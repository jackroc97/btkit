"""Unit tests for btkit.db._dte_calendar — pure formula correctness, no DB."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from btkit.db._dte_calendar import business_dte, worst_case_slack


def _walk_forward_business_dte(target: int, weekday: int) -> int:
    """
    Reference implementation: literally walk forward day-by-day from a date
    with the given ISO weekday (1=Mon..7=Sun), skipping Sat/Sun, until
    `target` business days have been counted. Returns the calendar-day count.
    """
    # 2024-01-01 is a Monday (ISO weekday 1); offset to the requested weekday.
    start = date(2024, 1, 1) + timedelta(days=weekday - 1)
    d = start
    counted = 0
    calendar_days = 0
    while counted < target:
        d += timedelta(days=1)
        calendar_days += 1
        if d.weekday() < 5:  # Mon-Fri
            counted += 1
    return calendar_days


@pytest.mark.parametrize("weekday", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("target", list(range(0, 31)))
def test_business_dte_scalar_matches_walk_forward(target: int, weekday: int) -> None:
    assert business_dte(target, weekday) == _walk_forward_business_dte(target, weekday)


def test_thursday_n2_tolerance0_reported_bug() -> None:
    # The exact reported bug: dte=2, tolerance=0 on Thursday/Friday should
    # resolve to calendar dte=4 (2nd trading day out), not 2 (which lands on
    # a weekend and matches zero real contracts).
    assert business_dte(2, 4) == 4  # Thursday
    assert business_dte(2, 5) == 4  # Friday
    assert business_dte(2, 1) == 2  # Monday — unaffected, no weekend crossed


def test_business_dte_expr_matches_scalar() -> None:
    df = pl.DataFrame({"weekday": [1, 2, 3, 4, 5]})
    target = 7
    result = df.select(business_dte(target, pl.col("weekday")).alias("eff"))["eff"].to_list()
    expected = [business_dte(target, w) for w in [1, 2, 3, 4, 5]]
    assert result == expected


def test_business_dte_expr_target_also_expr() -> None:
    df = pl.DataFrame({"weekday": [1, 3, 5], "target": [1, 5, 10]})
    result = df.select(
        business_dte(pl.col("target"), pl.col("weekday")).alias("eff")
    )["eff"].to_list()
    expected = [business_dte(t, w) for t, w in [(1, 1), (5, 3), (10, 5)]]
    assert result == expected


@pytest.mark.parametrize("target", list(range(0, 31)))
def test_worst_case_slack_is_friday_argmax(target: int) -> None:
    expected = max(business_dte(target, w) for w in range(1, 6)) - target
    assert worst_case_slack(target) == expected


def test_zero_target_is_noop_for_every_weekday() -> None:
    for weekday in range(1, 6):
        assert business_dte(0, weekday) == 0
