"""
Business-day-aware DTE arithmetic for LegConfig.dte_skip_weekends.

Options only expire on trading days, so a pure calendar-day DTE target is
weekday-dependent in how many real contracts it can match (e.g. target=2,
tolerance=0 on a Thursday lands on Saturday — no contract expires there).
`business_dte` reinterprets `target` as a business-day count N and returns
the calendar-day DTE that is N trading days out from an entry candidate on
the given weekday.

Derivation: for entry weekday w (Polars .dt.weekday() convention, Mon=1..
Sun=7) let w0 = w - 1 (0=Mon..6=Sun). Walking N business days forward from a
Mon-Fri start crosses exactly floor((N + w0) / 5) weekends, each of which
adds 2 calendar days beyond the N business days already counted:

    eff = N + 2 * floor((N + w0) / 5)

Verified against a literal day-by-day walk-forward for all w0 in 0..4 and
N in 0..30 (see tests/unit/test_dte_calendar.py). Valid only for entry
weekdays Mon-Fri (w0 in 0..4) — SessionConfig.weekdays_only already
restricts entries to weekdays by default, and option market data does not
exist for Sat/Sun regardless.
"""

from __future__ import annotations

import polars as pl

# Polars .dt.weekday() convention: Monday=1 .. Sunday=7.
_FRIDAY = 5


def business_dte(target: int | pl.Expr, weekday: int | pl.Expr) -> int | pl.Expr:
    """
    Calendar-day DTE equivalent to `target` business days out from an entry
    candidate on ISO `weekday` (1=Mon..7=Sun).

    Works as a plain-int computation when both arguments are python ints
    (returns int), and as a vectorized Polars expression when either
    argument is a pl.Expr (returns pl.Expr) — Polars' operator overloading
    on +, -, *, // handles both cases through this single implementation.
    """
    w0 = weekday - 1
    return target + 2 * ((target + w0) // 5)


def worst_case_slack(target: int) -> int:
    """
    Max extra calendar days `business_dte` can add over `target`, across all
    Mon-Fri entry weekdays — i.e. how much a leg's contribution to a global
    dte upper bound must widen when dte_skip_weekends is set. Always occurs
    at Friday, since skipping a weekend can only ever push the effective
    target later, and Friday is the weekday closest to the next weekend.
    """
    return business_dte(target, _FRIDAY) - target
