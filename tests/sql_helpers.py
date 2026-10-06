"""
Column-explicit INSERT statements for the input-database fact tables.

Tests used to insert positionally (`INSERT INTO option_bars VALUES (?, ?, ...)`),
which breaks every time a column is added — as it did when option_bars and
option_greeks gained `contract_id`. Naming the columns keeps fixtures working
across schema additions and documents what each row of literals means.
"""

from __future__ import annotations

OPTION_BARS_INSERT = (
    "INSERT INTO option_bars "
    "(ts_event, instrument_id, underlying_id, symbol, expiration, strike_price, "
    '"right", multiplier, open, high, low, close, volume) '
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

OPTION_GREEKS_INSERT = (
    "INSERT INTO option_greeks "
    "(ts_event, instrument_id, underlying_id, dte, T, iv, delta, gamma, theta, vega) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

OPTION_GREEKS_INSERT_WITH_LAG = (
    "INSERT INTO option_greeks "
    "(ts_event, instrument_id, underlying_id, dte, T, iv, delta, gamma, theta, vega, "
    "underlying_lag_s) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
