# Exit Liquidity Fix — Stale Forward-Fill Staleness Gate

## Background

`btkit` v2.0 was found to exit positions via take-profit at systematically higher
rates than a live IB execution system running the same strategy parameters:

| Strategy | Backtest TP rate | Live TP rate |
|---|---|---|
| BT8 — 1-DTE put spread | ~92% | ~12% |
| BT28 — 0-DTE IC put wing | ~78% | ~54% |
| BT40 — 0-DTE IC call wing | ~85% | ~28% |

An initial hypothesis attributed the gap to "monitoring intensity" — the backtest
evaluates every 1-minute bar whereas live limit orders only fill on real quote changes.
This is partially true but is insufficient to explain gaps of 3–8×.

---

## Root Cause

### Forward-fill creates artificial spread compression

`ExitScanner` computes the position mark by joining each leg's 1-minute option bars on
`(entry_id, ts_event)` using a **full outer join**, then **forward-filling** stale leg
prices from the last real bar:

```python
# exit.py — _compute_position_marks()
*[pl.col(c).forward_fill().over("entry_id") for c in close_cols],
*[pl.col(c).forward_fill().over("entry_id") for c in open_cols],
```

Forward-fill is necessary to produce a position mark at every bar even when only one
leg traded. But it introduces a critical correctness problem: when leg prices diverge
in staleness, the computed spread is no longer a valid market observation.

### The divergence mechanism

For a short put vertical spread:

- **Short leg** (higher strike, closer to the money): liquid, generates a bar roughly
  every 1–3 minutes.
- **Long leg** (lower strike, further OTM): illiquid, may go 10–60+ minutes between
  trades.

When the underlying rallies (favourable for short puts):

1. Short put premium drops rapidly — fresh bars arrive, each with a new lower price.
2. Long put premium also drops, but **no new bars are recorded** — the long leg's
   price is still the stale, higher value from the last real trade.
3. Computed spread = `short(fresh, low) − long(stale, high)` → **artificially small**.
4. The spread falls below the TP threshold → take-profit fires spuriously.

In the live system, the broker's limit order is priced on a real-time spread mark.
The order does not fill until **both legs** can be executed at the target price.
The backtest's forward-fill provides no such synchronisation.

### Gap TP amplifies the effect

91% of BT8 TP exits are `gap_tp` (priority 2), fired on the **bar open mark** rather
than the bar close:

```
spread_open_mark = Σ(leg_open × signed_qty)
```

This means the bar's open is computed from the long leg's stale close-of-prior-bar
price. By the time that bar opens, the short leg may have already traded at a much
lower price. The stale long-leg price was never updated to match, so the gap-open
spread already appears below TP before the bar even trades.

---

## Quantitative Evidence

Analysis was performed across BT8 (1-DTE put spread), BT28 (0-DTE IC put wing), and
BT40 (0-DTE IC call wing).

### Bar frequency asymmetry

| Strategy | Short leg bars/day | Long leg bars/day | Ratio |
|---|---|---|---|
| BT8 — 1-DTE put spread | ~180 | ~84 | 2.2× |
| BT28 — 0-DTE IC put wing | ~210 | ~38 | 5.5× |
| BT40 — 0-DTE IC call wing | ~195 | ~26 | 7.6× |

The more OTM the protective long leg, the worse the staleness problem.

### Staleness at TP exit time (BT8)

| Metric | Value |
|---|---|
| TP exits with long leg stale > 5 min at exit | **81%** |
| Mean long-leg staleness at TP exit | **35.8 min** |
| TP exits fired as gap_tp (bar open) | **91%** |

### Freshness-gate simulation (BT8)

Simulating a rule "TP/SL only fires when all legs have had a bar within the last
5 minutes" categorises existing TP exits as:

| Category | Share |
|---|---|
| Always-fresh (both legs within 5 min at exit) | 16% |
| Stale-only (TP would never fire with fresh requirement) | **72%** |
| Borderline | 12% |

72% of TP exits are spurious — they fired only because the long leg's price was stale.

### Last verified spread vs TP threshold (BT8)

Looking backwards from each stale-only TP exit to the last bar where **both legs** had
traded within 5 minutes:

- Mean last-fresh spread / TP threshold: **261%**

The position was 2.6× above the TP threshold the last time we had synchronised prices.
It had not approached TP legitimately before the spurious trigger.

### Time-of-day distribution

Staleness occurs throughout the trading day (no concentration near open/close). This
rules out session-boundary effects as the cause and confirms it is a persistent
structural issue with the long-leg's liquidity.

---

## Why Existing Mitigations Fall Short

### `min_exit_volume`

Filters low-volume bars by requiring a minimum cumulative contracts traded across legs
over a rolling window. Staleness can occur even when individual bars have volume —
the long leg simply trades less frequently. A stale forward-filled bar inherits the
last real bar's price but contributes **zero volume** by construction (`fill_null(0)`).
The volume gate helps but doesn't block exits where the long leg did trade recently
but less recently than the short leg.

### `confirmation_bars`

Requires TP to hold for N consecutive bars before firing. A stale long-leg price
persists across all N bars (no new real price to update it), so the spread remains
artificially compressed for the full confirmation window. The gate delays the exit
but doesn't prevent it.

### `pre_expiry_lock_minutes`

Suppresses price exits in the final N minutes of expiry day. Options are most illiquid
near expiry, so this has the largest individual impact. But staleness also occurs
throughout the day — the lock only covers the final 15–30 minutes.

None of these mitigations address the structural cause: the spread mark used for
TP/SL evaluation is not a valid market observation when one leg's price is stale.

---

## The Fix — `max_leg_stale_minutes`

### Concept

Add a `max_leg_stale_minutes` parameter to `LiquidityConfig`. When set, the exit
scanner tracks the elapsed time since each leg's last real bar and **suppresses all
price-triggered exits (TP, SL, gap-TP, gap-SL) whenever any leg's price is stale
beyond the threshold**.

This is analogous to `pre_expiry_lock_minutes` but data-driven: instead of locking
by time-of-day, it locks based on whether the spread mark is trustworthy.

Expiry and DTE exits are unaffected — they fire on calendar conditions, not price.

### Implementation

**`LiquidityConfig`** gains a new field:

```python
max_leg_stale_minutes: Optional[int] = None
```

**`_compute_position_marks()`** tracks the timestamp of each leg's last real bar:

```python
# Before forward-filling prices, record when each leg last had a real bar.
# After forward-fill, compute staleness in minutes per leg; take the max.
for leg in self.trade.legs:
    close_col = f"_leg_{leg.name}_mark_close"
    result = result.with_columns(
        pl.when(pl.col(close_col).is_not_null())
        .then(pl.col("ts_event"))
        .otherwise(pl.lit(None))
        .cast(pl.Datetime("us", "UTC"))
        .forward_fill().over("entry_id")
        .alias(f"_leg_{leg.name}_last_bar_ts")
    )
stale_mins = [
    (
        (pl.col("ts_event") - pl.col(f"_leg_{leg.name}_last_bar_ts"))
        .dt.total_minutes()
    )
    for leg in self.trade.legs
]
result = result.with_columns(
    pl.max_horizontal(stale_mins).alias("_max_leg_stale_minutes")
)
```

**`_find_first_hit()`** adds a staleness gate to `_price_ok`:

```python
if liq.max_leg_stale_minutes is not None and "_max_leg_stale_minutes" in m.columns:
    _fresh = pl.col("_max_leg_stale_minutes") <= pl.lit(float(liq.max_leg_stale_minutes))
    _price_ok = _price_ok & _fresh
```

### YAML usage

```yaml
exit:
  take_profit:
    pct: 0.70
  liquidity:
    pre_expiry_lock_minutes: 30
    max_leg_stale_minutes:   5    # suppress TP/SL when any leg's last bar is older than this
    slippage_model: spread
```

### Recommended value

**3–5 minutes** based on the bar frequency data:

- 3 minutes: conservative; matches ES front-month liquidity profile for liquid short legs.
- 5 minutes: permissive; allows small gaps in long-leg coverage without blocking exits.

Values above 10 minutes make the gate ineffective — mean staleness at spurious exits
is 35+ minutes, so the gate must be well below that to intercept them.

### Expected impact

Based on freshness-gate simulation:

| Exit category | Before fix | After fix (5 min gate) |
|---|---|---|
| Always-fresh | 16% | ~100% of survivors |
| Stale-only (spurious) | 72% | eliminated |
| Borderline | 12% | partial — depends on exact staleness |

Net effect: TP exit rate expected to drop from ~92% toward the 20–35% range seen in
live trading for BT8. Positions that previously exited spuriously at 70% TP will now
hold and more frequently expire worthless (full credit retained), which matches
observed live behaviour.

### Trade-offs

**Positions held through stale windows often expire at full credit.** This is the
correct outcome — the position was profitable enough to expire, but the spurious
backtest exit was claiming 70% TP instead of 100% credit. The fix improves P&L
accuracy in both directions: fewer false TP credits, more legitimate expiry credits.

**Legitimate TP exits are not blocked.** A TP exit with a fresh long-leg bar (both
legs current) still fires immediately. Only exits where the computed spread is
unreliable are deferred.

**Gap exits are also gated.** `gap_tp` (bar open mark) is gated by the same staleness
check. Since 91% of TP exits are gap_tp, this is the primary benefit.

**Staleness tracking has negligible compute cost.** The per-leg last-bar-ts columns
are computed in a single `.with_columns()` pass alongside the existing forward-fill.
No additional DB queries are made.

---

## Files Changed

| File | Change |
|---|---|
| `btkit/strategy/definition.py` | Added `max_leg_stale_minutes` to `LiquidityConfig`; added `needs_staleness` property; updated `is_default` |
| `btkit/backtest/exit.py` | Track per-leg last-bar timestamp in `_compute_position_marks()`; add staleness gate to `_find_first_hit()` |
| `docs/strategy.md` | Document `max_leg_stale_minutes` in `liquidity` schema section |
| `docs/exit_liquidity_fix.md` | This file |
