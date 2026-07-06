# Expiry Exit — Settlement-Based Mark

## Why bar prices are wrong at expiry

The exit scanner computes position marks by taking each leg's bar close price and
forward-filling gaps (a leg that didn't trade at minute T carries its most recent
traded price forward). This is correct for intraday TP/SL detection, where both
legs are usually liquid and prices are recent.

At expiry it breaks down. On a 0-DTE options day the short leg of a credit spread
frequently goes illiquid well before the options close (16:00 ET for ES). The last
real short-leg bar might be at 15:12; the long leg keeps trading and has a bar at
15:51. If the underlying rallied 35 points in between:

```
position_mark at 15:51:
  STO 5320C  forward-filled from 15:12  → close = 96.25
  BTO 5370C  fresh bar at 15:51         → close = 131.50
  mark = 96.25 − 131.50 = −35.25   ← physically impossible
```

A lower-strike call *cannot* be worth less than a higher-strike call of the same
expiry — this is a no-arbitrage constraint. The negative mark is pure data artifact
from mixing stale and fresh bar prices. The PnL formula then computes:

```
net_pnl = (open_mark − exit_mark) × multiplier
        = (0.90 − (−35.25)) × 50
        = +$1,807   ← phantom profit, true result is ~−$2,455
```

The bug report that surfaced this (`bug_report_pnl_sign_inversion.md`) attributed
the error to a sign convention mismatch between `open_mark` and `exit_mark`. The
sign conventions are in fact consistent; the root cause is the stale-fresh mix.

---

## The fix — intrinsic value from underlying settlement

At expiry, an option's fair value is exactly its intrinsic value. No time value
remains; no bid-ask spread needs to be crossed (cash settlement):

```
call leg:  max(0, S − K)
put leg:   max(0, K − S)
```

where `S` is the underlying's closing price at `expiry_close_time` and `K` is the
leg's strike.

The spread position mark at settlement is:

```
settlement_mark = Σ  intrinsic_i × signed_qty_i
```

| Leg type | signed_qty |
|---|---|
| sell_to_open | +qty |
| buy_to_open  | −qty |

For the specific bug trade (STO 5320C / BTO 5370C, ES settles at 5498):

```
STO 5320C intrinsic = max(0, 5498 − 5320) = 178
BTO 5370C intrinsic = max(0, 5498 − 5370) = 128
settlement_mark = 178 × (+1) + 128 × (−1) = 50   (spread width = max loss)
```

Correct PnL = (0.90 − 50) × 50 − fees ≈ −$2,455.

---

## Properties of settlement-based marks

**Always non-negative for credit spreads.** `settlement_mark ≥ 0` for any
credit spread at any underlying price. A negative mark is a no-arbitrage
violation; intrinsic values eliminate this by construction.

**Ranges.**

| Outcome | settlement_mark |
|---|---|
| Spread expires worthless (OTM) | 0 |
| Short leg ITM, long leg OTM | STO intrinsic (partial loss) |
| Both legs ITM (max loss) | spread width |

**Zero slippage.** Options are cash-settled at expiry; no market fill is
required. The slippage model (flat percentage or OHLC-spread) is therefore
*not* applied to settlement-based exit marks. The correct slippage is $0.

**`expiration_fee_per_contract` still applies.** Clearing fees are charged at
expiry regardless of moneyness.

**`worst_mark` is unaffected.** The running maximum adverse excursion is
computed from intraday bar prices and is not touched by this change.

---

## Implementation — `ExitScanner._compute_settlement_marks`

`ExitScanner._compute_settlement_marks(entries, tz_str)` is called once per
cohort bucket at the start of `_find_first_hit`. It returns a DataFrame with
columns `[entry_id, settlement_mark]`.

**Data source.** Settlement prices are pre-loaded once per `scan()` call (not
per cohort) using two queries:
- `InputDatabase.underlying_ids_for_options(instrument_ids)` — a single batch
  query against `option_bars.underlying_id` that returns the exact futures
  contract each option instrument settles against, as recorded by the data
  vendor. This is more reliable than a roll-schedule heuristic: near a roll
  date the front-month can switch while options expiring before the roll still
  settle against the old contract.
- `InputDatabase.settlement_closes_for_underlyings(underlying_ids, start, end,
  tz_str, close_time)` — a single query fetching all underlying bars for the
  full scan window, then filtering in Polars to the last bar at or before
  `expiry_close_time` local time on each trading day. Returns
  `[underlying_id, exp_date, settlement_close]`.

Both results are cached on `self._opt_to_underlying` and
`self._settlement_closes_by_key`. `_compute_settlement_marks` then does pure
dict lookups — no DB access per cohort.

**Vectorised intrinsic computation.** Once settlement prices are fetched, the
per-leg intrinsic sum is computed in Polars:

```python
call_intr = (pl.col("_settlement") - pl.col(strike_col)).clip(lower_bound=0.0)
put_intr  = (pl.col(strike_col) - pl.col("_settlement")).clip(lower_bound=0.0)
leg_intr  = (
    pl.when(pl.col(right_col).str.to_uppercase().str.starts_with("C"))
    .then(call_intr)
    .otherwise(put_intr)
) * signed_qty
```

**Null sentinel.** If `underlying_ids_for_options` has no mapping for a given
option instrument ID, or if `underlying_bars` returns an empty DataFrame,
`settlement_mark` is set to `null` and the exit scanner falls back to
bar-price logic for that entry.

---

## How settlement_mark flows through the exit scan

`settlement_mark` is joined onto `entry_meta` and therefore onto every bar row in
`m` (position_marks ⋈ entry_meta). It is a constant per entry — the same value
appears on every minute-bar row for that trade.

**Expiry exit (priority 7).** The `_expiry` flag fires at the first bar where
`ts_event.date() ≥ trade_expiration` AND `_local_sec ≥ expiry_close_sec`. The
fill price uses `settlement_mark` when not null:

```python
raw_exit_mark = (
    pl.when(pl.col("_priority").is_in([1, 2]))   # gap exits
    .then(pl.col("spread_open_mark") + slippage)
    .when(
        (pl.col("_priority") == 7) & pl.col("settlement_mark").is_not_null()
    )
    .then(pl.col("settlement_mark"))              # intrinsic, zero slippage
    .otherwise(_bar_mark_clipped)                 # no-arb clipped bar close
)
```

**Fallback expiry.** If no bar exists at or after `expiry_close_time` (common:
0-DTE option bars stop before 16:00), the fallback path fires instead. It also
uses `settlement_mark` when available. When unavailable, the fallback prefers the
last bar where all legs had fresh prices (`_max_leg_stale_minutes ≤ threshold`),
further falling back to the last available bar. All bar-price fallback marks are
clipped to the no-arbitrage bound (see below).

**Tick rounding.** `settlement_mark` is rounded to the nearest `tick_size` before
being stored, matching the rounding applied to all other exit marks.

---

## Graceful degradation

When settlement data is unavailable:

| Root cause | Behaviour |
|---|---|
| Option instrument ID not found in `option_bars` | `settlement_mark = null`; fallback uses last fresh bar |
| Underlying DB has no bars on the expiry day | `settlement_mark = null`; fallback uses last fresh bar |
| `expiry_close_time` not configured on instrument | `settlement_mark = null`; full bar-price path (no fix) |

In the degenerate case — no settlement AND no fresh bar — the fallback uses the
last available bar. The bar-price mark is then clipped to the no-arbitrage bound
(≥ 0 for credit spreads, ≤ 0 for debit spreads), preventing phantom profits from
stale forward-fills even in this last-resort path.

---

## Affected data

`es_spreads_full_sweep_v3.db` was produced before this fix. It contains
**16,656 expiry exits with negative `exit_mark`** totalling approximately
**$1.3 M in phantom P&L**. The database should be re-run against the fixed engine.

`es_spreads_full_sweep.db` (the earlier database) also needs a re-run to address
the separate instrument-ID reuse bug (see `docs/` index).
