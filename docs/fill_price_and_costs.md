# Fill Price, Slippage, and Fees

## Overview

btkit 2.0 uses 1-minute OHLCV bars. Exit events (take-profit and stop-loss) are detected by
scanning bars forward from the trade open. This document defines how fill prices are determined
and how transaction costs are modeled.

---

## Exit Detection

For each open trade, the backtest scans 1-min bars forward from `open_time` and checks whether
a TP or SL condition has been met.

### Single-Leg Positions

The bar `high` and `low` are used to detect whether the threshold was crossed within the bar.

```
if bar.low <= tp_price:   → TP hit
if bar.high >= sl_price:  → SL hit
```

### Multi-Leg Spreads

Individual leg highs/lows cannot be used reliably because the legs do not reach their intrabar
extremes at the same time. Instead, the **spread mark** is computed from each bar's close prices
and checked against thresholds.

```
spread_mark = sum(leg_close * signed_quantity for each leg)

if spread_mark <= tp_price:   → TP hit
if spread_mark >= sl_price:   → SL hit
```

This is slightly less sensitive than per-leg high/low detection but avoids fabricating
simultaneous extremes across legs. The 1-min resolution keeps the resulting error small.

---

## Fill Price Rules

### Standard Case (threshold crossed mid-bar)

Fill at the **exact TP or SL threshold price** — not the bar close, not the bar extreme.

- **TP** is a limit order. The threshold is the limit price; filling there is correct.
- **SL** is a stop order. Filling at the threshold is slightly optimistic (real stops can
  experience slippage past the trigger price). Slippage (see below) corrects for this.

### Gap Open Case

If the bar **opens** beyond the threshold (e.g., overnight gap, news event), the threshold was
never available for fill. Fill at the **bar open mark** instead.

```
For single-leg:
    if bar.open <= tp_price:   → fill at bar.open (gapped through TP)
    if bar.open >= sl_price:   → fill at bar.open (gapped through SL)

For spreads:
    spread_open_mark = sum(leg_open * signed_quantity for each leg)
    apply same logic against spread_open_mark
```

Gap opens should be detected before the mid-bar check on each bar.

### Both TP and SL Crossed in the Same Bar

When a single bar's range spans both the TP and SL threshold (wide bar, volatile bar), it is
unknowable from bar data which was hit first. **Assume the stop loss was hit first.** This is
the pessimistic/conservative convention and produces more realistic results.

---

## Fill Price Summary Table

| Scenario | Detection | Fill Price |
|---|---|---|
| Single-leg, threshold crossed mid-bar | `bar.high` / `bar.low` | TP or SL threshold price |
| Single-leg, gap open past threshold | `bar.open` | `bar.open` |
| Multi-leg spread, threshold crossed | `spread_mark` (close-based) | TP or SL threshold price |
| Multi-leg spread, gap open past threshold | `spread_open_mark` | `spread_open_mark` |
| Both TP and SL crossed in same bar | — | SL threshold price (SL takes priority) |

---

## Transaction Costs

Both slippage and fees are **per-trade parameters**, applied at exit. They are defined at the
strategy level and can be overridden per backtest run.

### Slippage

Slippage captures two real-world effects:
1. Stop orders filling at worse-than-trigger prices during fast markets
2. Bid/ask spread on options (OHLCV close is last traded price, not mid; closing a position
   means crossing the spread)

Slippage is modeled as a **percentage of the exit spread mark**, applied as an additional cost
at exit.

```
effective_exit_price = exit_price * (1 + slippage_pct)   # for debit-to-close positions
```

**Default:** `slippage_pct = 0.0` (disabled). A value of `0.01` (1%) is a reasonable starting
point for liquid index options.

### Fees

Fees are modeled per contract per leg, with separate rates for the three lifecycle events:

```
fee_cost = (entry_fee_per_contract + exit_or_expiration_fee_per_contract)
           × total_contracts_in_position
```

`total_contracts_in_position` is the sum of `quantity` across all legs.
Expiry exits use `expiration_fee_per_contract`; all other exits (TP, SL, condition, DTE)
use `exit_fee_per_contract`.

#### Structured fees (recommended)

```yaml
costs:
  slippage_pct: 0.01
  fees:
    entry_fee_per_contract:      0.65   # charged at open, per leg
    exit_fee_per_contract:       0.65   # charged on TP/SL/condition/DTE exit, per leg
    expiration_fee_per_contract: 0.00   # charged at expiry (IBKR waives for worthless expiry)
```

For a 2-leg spread (e.g. a put spread) with the values above:
- Active exit (TP/SL): `(0.65 + 0.65) × 2 = $2.60`
- Expiry (worthless): `(0.65 + 0.00) × 2 = $1.30`

#### Legacy form

`fee_per_contract` is still accepted and is split evenly across entry and exit:

```yaml
costs:
  fee_per_contract: 0.65    # equivalent to entry=0.325 + exit=0.325 per leg
```

The `fees:` block and `fee_per_contract` are mutually exclusive; using both is a
validation error at load time.

**Defaults:** All fee fields default to `0.0`.

### Application Order

```
1. Determine raw fill price (per rules above)
2. Apply slippage to fill price
3. Compute gross PnL from slippage-adjusted fill
4. Subtract fee_cost from gross PnL → net PnL
```

---

## Tick Size Rounding

Options trade at discrete price increments. When `instrument.tick_size` is non-zero,
all fill prices and the MAE high-water mark are rounded to the nearest tick before
being written to the output database.

```yaml
instrument:
  root_symbol: ES
  asset_class: future
  tick_size: 0.05          # $0.05 per point — standard for ES options
```

Prices rounded:

| Price | When | Notes |
|---|---|---|
| `open_mark` | Entry fill | Signed sum of leg closes, then rounded |
| `tp_price` | Threshold derivation | Derived from rounded `open_mark`, then rounded |
| `sl_price` | Threshold derivation | Derived from rounded `open_mark`, then rounded |
| `exit_mark` | All exit types | Gap fills, condition/DTE/expiry bar-close prices |
| `worst_mark` | MAE tracking | Running max of position marks, rounded at output |

**TP and SL threshold prices** are already on-tick (rounded at entry time) so their
fill prices are not double-rounded — only the bar-level gap/condition/expiry marks
need rounding at exit time.

**Default:** `tick_size = 0.0` disables all rounding, reproducing the original
continuous-price behaviour exactly.

---

## Known Limitations

- **1-min bar resolution**: TP/SL events that occur and reverse within a single 1-min bar are
  not detected. With 1-min bars this error is small but non-zero. Tick data would eliminate it.
- **Spread mark uses closes**: the spread mark is computed from bar closes, not intrabar
  extremes. A TP/SL triggered mid-bar but reversed by close will be missed until the following
  bar.
- **Slippage is symmetric**: the model applies a flat percentage regardless of market
  conditions. In practice, slippage is larger during high-volatility periods.
- **Fees are structured but not exchange-specific**: the model supports entry, exit, and
  expiration fee tiers per contract per leg. Exchange pass-through fees, regulatory fees, and
  tiered volume rebates are not modelled.
