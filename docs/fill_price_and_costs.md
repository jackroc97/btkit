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

Fees are modeled as a **flat dollar amount per contract** applied at both open and close.

```
trade_cost = fee_per_contract * total_contracts_in_position * 2   # open + close
```

`total_contracts_in_position` is the sum of absolute contract counts across all legs.

**Default:** `fee_per_contract = 0.0`.

### Application Order

Costs are applied after fill price is determined:

```
1. Determine raw fill price (per rules above)
2. Apply slippage to fill price
3. Compute gross PnL from slippage-adjusted fill
4. Subtract flat fee cost from gross PnL → net PnL
```

### Strategy-Level Parameters

```yaml
costs:
  slippage_pct: 0.01        # 1% of exit spread mark
  fee_per_contract: 0.65    # dollars per contract per side
```

---

## Known Limitations

- **1-min bar resolution**: TP/SL events that occur and reverse within a single 1-min bar are
  not detected. With 1-min bars this error is small but non-zero. Tick data would eliminate it.
- **Spread mark uses closes**: the spread mark is computed from bar closes, not intrabar
  extremes. A TP/SL triggered mid-bar but reversed by close will be missed until the following
  bar.
- **Slippage is symmetric**: the model applies a flat percentage regardless of market
  conditions. In practice, slippage is larger during high-volatility periods.
- **Fees are flat**: exchange fees, regulatory fees, and per-share/per-contract structures
  vary by broker. The flat-per-contract model is an approximation.
