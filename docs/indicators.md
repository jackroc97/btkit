# Indicator Scripts

Indicators are user-supplied Python scripts that produce derived time-series values
from market data. Each script is run during the pipeline build phase and its outputs
are stored in the `indicator_bars` table, where they are later joined to entries and
exits during the backtest.

## Declaring indicators in a strategy

```yaml
indicators:
  - indicators/iv_rank.py
  - indicators/spread_vega_score.py
```

Paths are relative to the working directory when `btk run` or `btk build` is invoked,
or may be absolute.

---

## Writing an indicator script

Every script must expose a top-level `compute` function. The engine calls this
function once per underlying instrument.

### Minimal form — underlying bars only

```python
import polars as pl

def compute(df: pl.DataFrame) -> pl.DataFrame:
    """
    df: underlying_bars for one instrument, sorted by ts_event.
    Columns: ts_event, instrument_id, symbol, open, high, low, close, volume

    Return df with one or more indicator columns appended.
    """
    return df.with_columns(
        pl.col("close").rolling_mean(window_size=20).alias("sma_20")
    )
```

Each new column in the returned DataFrame becomes an independent indicator series.
The engine stores each column separately in `indicator_bars` and makes it available
in entry/exit conditions under its column name.

> **Cadence is up to you.** An indicator column may be emitted at any frequency — one
> value per session, per 5 minutes, or per bar. At backtest time each column is merged
> onto candidates with a **session-scoped backward as-of join**: a candidate receives the
> latest value at or before its timestamp within the session, never filled across the
> session boundary. A daily signal therefore gates every intraday entry in its session,
> so you do **not** need to forward-fill coarse indicators to the 1-minute grid yourself.
> See [Indicator Alignment](strategy.md#indicator-alignment-session-scoped-as-of-join) in
> the strategy reference.

### Extended form — with option data access

When your indicator needs option chain data, declare a second `ctx` parameter:

```python
import polars as pl

def compute(df: pl.DataFrame, ctx) -> pl.DataFrame:
    # ctx is an IndicatorContext — see API below
    greeks = ctx.option_greeks(dte_min=0, dte_max=30)
    ...
    return df.with_columns(...)
```

The runner detects the arity of `compute` at import time and automatically passes
an `IndicatorContext` when the second parameter is present. **Scripts that only
declare `compute(df)` are completely unaffected** — no changes required.

---

## IndicatorContext API

`IndicatorContext` provides lazy, filtered access to option data scoped to the same
underlying and time window as the `df` argument.

### Properties

| Property | Type | Description |
|---|---|---|
| `underlying_id` | `int` | instrument_id of the underlying being processed |

### `ctx.option_greeks(...)` → `pl.DataFrame`

Fetches per-minute option greeks from `option_greeks` for this underlying.

**Columns returned:** `ts_event`, `instrument_id`, `dte`, `T`, `iv`, `delta`,
`gamma`, `theta`, `vega`

```python
# No filters — entire chain
greeks = ctx.option_greeks()

# 0–30 DTE, puts only by post-filtering
greeks_30d = ctx.option_greeks(dte_max=30)

# Delta-selected slice (e.g. -0.50 to 0.0 for put wing)
puts = ctx.option_greeks(delta_min=-0.50, delta_max=0.0)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `dte_min` | `int \| None` | Minimum DTE (inclusive). `None` = no lower bound. |
| `dte_max` | `int \| None` | Maximum DTE (inclusive). Recommended for large datasets. |
| `delta_min` | `float \| None` | Minimum delta (inclusive). Puts have negative delta. |
| `delta_max` | `float \| None` | Maximum delta (inclusive). |

All filters are combined with AND logic. All parameters default to `None` (no filter).

### `ctx.option_bars(...)` → `pl.DataFrame`

Fetches per-minute OHLCV bars from `option_bars` for this underlying.

**Columns returned:** `ts_event`, `instrument_id`, `symbol`, `expiration`,
`strike_price`, `right`, `multiplier`, `open`, `high`, `low`, `close`, `volume`

```python
# All options, no filter
bars = ctx.option_bars()

# Near-term calls only
calls = ctx.option_bars(right="C", dte_max=21)

# Puts with at least 14 days to expiry
puts = ctx.option_bars(right="P", dte_min=14)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `dte_min` | `int \| None` | Keep options where `DATEDIFF(ts_event, expiration) >= dte_min`. |
| `dte_max` | `int \| None` | Keep options where `DATEDIFF(ts_event, expiration) <= dte_max`. |
| `right` | `str \| None` | `"C"` for calls, `"P"` for puts. `None` returns both. |

Note: `option_bars` does not have a `dte` column — DTE is computed on the fly
from `DATEDIFF(ts_event::DATE, expiration)`. Use `ctx.option_greeks()` for
pre-computed DTE values.

---

## Full example — IV rank indicator

```python
import polars as pl

def compute(df: pl.DataFrame, ctx) -> pl.DataFrame:
    """Compute 30-day IV rank using ATM options."""
    # Fetch near-term greeks, both puts and calls
    greeks = ctx.option_greeks(dte_max=30)

    if greeks.is_empty():
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias("iv_rank_30d"))

    # Median IV across all near-term options per minute
    iv_by_ts = (
        greeks.group_by("ts_event")
        .agg(pl.col("iv").median().alias("median_iv"))
        .sort("ts_event")
    )

    # Rolling 252-day min/max for rank normalization
    iv_by_ts = iv_by_ts.with_columns([
        pl.col("median_iv").rolling_min(window_size=252 * 390).alias("iv_252d_min"),
        pl.col("median_iv").rolling_max(window_size=252 * 390).alias("iv_252d_max"),
    ]).with_columns(
        (
            (pl.col("median_iv") - pl.col("iv_252d_min"))
            / (pl.col("iv_252d_max") - pl.col("iv_252d_min"))
        ).alias("iv_rank_30d")
    )

    # Join back to underlying df
    return df.join(
        iv_by_ts.select(["ts_event", "iv_rank_30d"]),
        on="ts_event",
        how="left",
    )
```

---

## Performance guidance

Option datasets can be large. Follow these practices to avoid loading unnecessary data:

- **Always supply `dte_max`** when you only care about near-term options. A full
  option chain for a 4-year backtest can be many GB; filtering to `dte_max=45`
  reduces this to a small fraction.
- **Use `right=` when you only need puts or calls.** Halves the result set.
- **Use `delta_min`/`delta_max` on `option_greeks`** to focus on a specific
  moneyness range instead of loading the entire chain and filtering in Python.
- Data is fetched lazily — methods are only called when your script actually
  invokes them. Scripts that never call `ctx.option_greeks()` or `ctx.option_bars()`
  incur no overhead at all.

---

## Backward compatibility

All existing indicator scripts that declare `compute(df)` continue to work without
modification. The `ctx` parameter is entirely opt-in — the runner detects arity
using `inspect.signature` at load time and only builds an `IndicatorContext` when
the second parameter is present.
