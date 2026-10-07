# Exit System Redesign (draft)

Status: design draft, not implemented. Covers the four live-vs-backtest exit gaps and
replaces the bolt-on features (`leg_out`, `on_sl_long_continuation`,
`allow_after_hours_exits`) with one model.

---

## 1. The current model and why it can't express the gaps

Today an exit is one event per position, found on a single spread-level time series:

1. `_compute_position_marks` reduces every leg to one `position_mark` per bar
   (Σ signed forward-filled closes). Per-leg prices are dropped.
2. `tp_price` / `sl_price` are computed at entry, always relative to `open_mark`
   (`entry.py:_compute_open_mark`).
3. `_find_first_hit` picks the first bar where any flag fires, and every leg exits on that
   bar at the bar-close spread mark.
4. Two post-processors patch the result:
   - `_adjust_leg_out_exits` re-prices long legs to the next real bar.
   - `scan_long_continuation` tracks the long after an SL, with a trailing stop.
     `engine._build_gate_overrides` then holds re-entry until the continuation closes.
5. `PnLCalculator` computes P&L from `open_mark − exit_mark`. `position_leg.exit_price`
   is always written as NULL.

| Gap | Why it can't be expressed |
|---|---|
| 1. TP is a limit on the short leg | Triggers only watch `position_mark`. Levels are only relative to `open_mark`, with no rounding. Fills are always the bar close (see §8.1). |
| 2. SL is a fixed short-leg price | Same: there's no per-leg watch and no absolute level. |
| 3. Long leg held to expiry, no re-entry block | Legs can only exit on the trigger bar (default) or the next bar (`leg_out`). The continuation feature holds the long only after SL, uses a trailing stop, and blocks re-entry. |
| 4. Sunday-evening exits | The session mask is `weekday <= 5` plus one daily window. There's no way to describe a weekly session such as Sun 18:00 to Fri 17:00. |

The common cause is that **the exit is modelled for the position, not for each leg.**
The redesign makes legs first-class at exit.

---

## 2. Core concepts

**Order.** A price-triggered exit rule. There are two: `stop_loss` and `take_profit`.
They act as OCO siblings: the first to fire cancels the other. Each order has:

| Field | Meaning |
|---|---|
| `watch` | What the trigger observes: `spread` (net position mark, as today) or a leg name. |
| `level` | The trigger price, computed once per position at entry (§3). |
| `trigger` | How a bar is tested against the level: `bar_close` or `touch` (§4). |
| `fill` | The price the watched instrument fills at (§4). |
| `close` | Which legs close when the order fires, and what the rest do (§5). |

**Whole-position exits.** `conditions`, `roll`, `vega_exit`, `dte_exit` and `expiry` keep
their current behaviour: every leg closes on that bar. They're evaluated only while the
primary legs are open.

**Primary exit.** The first exit event for a position. It sets `position.exit_time` and
`exit_reason`, ends the MAE (`worst_mark`) window, and by default releases the re-entry
gate.

**Residual legs.** Legs that the primary exit doesn't close on its own bar. Each resolves
independently afterwards (next bar, or expiry), and the outcome is written to its
`position_leg` row.

---

## 3. Levels

A level is a price **in the units of the watched thing**: the net spread mark when
`watch: spread`, or that leg's own option price when watching a leg. It's computed per
position at entry.

```yaml
level:
  # exactly one of:
  at: 10.00        # absolute price
  points: 2.0      # reference ± points (SL: +, TP: −)
  pct: 0.70        # TP: reference × (1 − pct), SL: reference × (1 + pct)
  # options:
  reference: open_mark   # open_mark (net credit/debit, default) | leg_open (watched leg's entry price)
  round_to: 0.25         # round to this increment after computing; default = instrument tick_size
  round: nearest         # nearest | favorable | adverse
```

All numeric fields are sweepable (`NumericSweep`).

**Rounding floor.** A rounded level is never allowed to reach 0. If rounding would give
≤ 0, the level becomes one `round_to` increment instead. For example, 0.30 × a 0.30
credit is 0.09, which becomes 0.25. Any position where the floor applied is counted in a
backtest warning, so it doesn't happen silently.

The `reference` field is what makes gap 1 expressible: the order **watches** the short
leg, but its level is **derived from** the net credit (`0.30 × open_mark`, rounded to 0.25).

Existing forms keep working as shorthands:

| Today | Equivalent |
|---|---|
| `stop_loss: 2.0` / `stop_loss: {price: 2.0}` | `{watch: spread, level: {points: 2.0}}` |
| `take_profit: {pct: 0.70}` | `{watch: spread, level: {pct: 0.70}}` |
| `take_profit: {price: 1.0}` / bare `take_profit: 1.0` | `{watch: spread, level: {points: 1.0}}` |
| `take_profit_pct: 0.70` (legacy top-level) | **removed**; the validator gives the replacement |

Level computation moves out of `EntryScanner._compute_open_mark` into the exit module,
which keeps exit config in one place. Entries still carry `open_mark` and each leg's
`leg_<name>_close`, which is everything a level needs.

---

## 4. Trigger and fill models

### Trigger

| `trigger` | Test (SL / TP) | Valid for |
|---|---|---|
| `bar_close` (default) | bar close ≥ level / ≤ level | spread or leg |
| `touch` | bar high ≥ level / bar low ≤ level | **leg only** |

`touch` counts a bar whose high/low *equals* the level as a hit (`touch_rule: at`, the
default). `touch_rule: through` requires the price to trade strictly beyond the level,
which models losing queue priority on a resting limit. Comparing the two in a backtest
shows how sensitive results are to queue position.

`touch` is rejected for `watch: spread`. Leg extremes don't happen at the same moment, so
a spread high/low would be made up. This is the rule already stated in
`docs/fill_price_and_costs.md`. With `touch`, the trigger can only fire on a real bar of
the watched leg; forward-filled prices never trigger it.

`confirmation_bars` and the `condition` AND-gate stay as they are, on both orders.
`confirmation_bars` is only valid with `trigger: bar_close`.

Gap handling becomes part of the fill model instead of separate priorities. If a bar
**opens** beyond the level, the order fires on that bar and fills at the open. This
replaces `_gap_sl` / `_gap_tp`.

Tie-break: if SL and TP both fire on the same bar, SL wins (unchanged, conservative).

### Fill

| `fill` | Watched leg / spread fills at | Typical use |
|---|---|---|
| `bar_close` (default) | bar close (+ spread slippage when `slippage_model: spread`) | today's behaviour |
| `level` | the level, or the bar open if the bar gapped through it | resting limit (TP), idealised stop |
| `next_trade` | close of the first real bar strictly after the trigger bar | stop-market with latency |
| `stop` | worst of (level, bar close), or the bar open if the bar gapped through the level | stop-market, conservative default |
| `stop_high` | worst of (level, bar high), or the bar open if gapped | pessimistic bound, for sensitivity checks |

Live gap 1 maps to `trigger: touch, fill: level`.
Live gap 2 maps to `trigger: touch, fill: stop`.

The `stop` model is **never better than the stop price**. It costs more when the move
continues through the bar close. It's **optimistic** on spike-and-revert bars: the real
fill happens right after the touch, near the spike, while the model fills at the stop
because the bar closed back below it. It's **pessimistic** when the touch comes early and
the price runs for the rest of the minute.

To check how much the stop model matters, run `level`, `stop` and `stop_high` as three
values of one sweep. They give a lower bound, the default and an upper bound for stop cost.

---

## 5. How the legs close: `close`

`close` names the exit style: which legs close when the order fires, and what the rest
do. The common cases are named presets:

| `close` | Short (`sell_to_open`) legs | Long (`buy_to_open`) legs | Replaces |
|---|---|---|---|
| `together` (default) | trigger bar | trigger bar, at the last traded price | today's default |
| `short_first` | trigger bar | first real bar after the trigger; 0 if none | `leg_out: true` |
| `short_only` | trigger bar | held to their own expiration, settled at intrinsic | (gap 3) |

```yaml
exit:
  close: short_first              # leg out: shorts first, longs at their next trade
```

For anything the presets don't cover, use a per-leg map. Any leg not listed closes with
the order:

```yaml
close: {long_put: hold_to_expiry}
close: {long_put: next_trade, long_call: hold_to_expiry}
```

| Per-leg value | Leg exits at | Exit time |
|---|---|---|
| `with_order` | its last traded price on the trigger bar | trigger bar |
| `next_trade` | close of its first real bar strictly after the trigger; 0 if none | that bar |
| `hold_to_expiry` | intrinsic at its **own** expiration settlement; last bar close if no settlement price | expiry close time |

Rules:

- **Default and override.** `close` can be set at `exit:` level as a default and overridden
  on each order, so TP and SL can differ (for example, `short_first` after TP and
  `short_only` after SL).
- **The watched leg always closes with the order.** It fills by the order's `fill` model
  (§4). Listing it in the map, or watching a long leg with `short_first` / `short_only`,
  is a validation error. Long-first legging isn't supported.
- **Spread watch.** A spread-level `fill` (`level`, `stop`) can't be split into
  per-leg prices. So with `watch: spread`, any `close` other than `together` requires
  `fill: bar_close`, and each closing leg fills at its own bar close (today's `leg_out` maths).
- **Whole-position exits** (§2) always close every leg on their bar; `close` doesn't apply
  to them.

### Liquidity gates follow the watch

| Gate | Evaluated on |
|---|---|
| Volume, staleness | **Leg watch:** the watched leg only. **Spread watch:** every leg that closes on the trigger bar. This generalises today's `leg_out` special case, which only checks short-leg volume. |
| No-arbitrage guard | spread watch only (it's a property of the net mark) |
| Pre-expiry lock | unchanged |

---

## 6. Re-entry gating

```yaml
exit:
  reentry_after: primary_exit   # default; | all_legs_closed
```

- **`primary_exit`**: a new position may open from the primary exit's bar onwards. This
  is today's default and the `leg_out` behaviour, and it's what gap 3 needs: the long is
  held to expiry without blocking the next spread.
- **`all_legs_closed`**: the gate opens when the last residual leg resolves.

Both are just a choice of which timestamp feeds `_enforce_entries`. `_build_gate_overrides`
and `_next_trading_day_open` are deleted. The "wait for next session" rule existed only
for continuation.

btkit has no capital or margin model (`initial_equity` only seeds the equity curve), so
overlapping residual legs from earlier positions need no further accounting.

---

## 7. Exit hours (gap 4)

`exit.hours` replaces `allow_after_hours_exits`. It gates **price-triggered** exits only.
DTE, expiry and residual-leg resolution are unaffected, as today.

```yaml
exit:
  hours: session        # default: universe.session window + weekdays_only (today's behaviour)
  # or a preset:
  hours: cme_globex     # Sun 18:00 → Fri 17:00, daily 17:00–18:00 halt, session timezone
  # or explicit weekly windows:
  hours:
    - {days: [sun, mon, tue, wed, thu], start: "18:00", end: "23:59"}
    - {days: [mon, tue, wed, thu, fri], start: "00:00", end: "17:00"}
```

The mask is built from local weekday plus `_local_sec`, which is already materialised in
`_find_first_hit`. Migration: `allow_after_hours_exits: true` becomes Mon–Fri 00:00–23:59,
which is its exact current meaning. `skip_dates` stays entry-only, as today. Entry-side
Sunday sessions are out of scope.

---

## 8. Fixing existing behaviour along the way

### 8.1 Doc/code mismatch on TP/SL fills

`docs/fill_price_and_costs.md` says close-triggered spread TP/SL fill **at the threshold**.
The code fills at the **bar close** mark (`raw_exit_mark` → `_bar_mark_clipped`). The
redesign makes this explicit as `fill: bar_close` (default, which preserves current results)
versus `fill: level`. The doc gets updated to match.

### 8.2 Per-leg exits are recorded

`position_leg.exit_price` is filled in for every leg, and the table gains `exit_time` and
`exit_reason`. That's needed for residual legs and also helps the trade-detail chart, which
already reads `exit_price` (`chart.py:316`).

---

## 9. Output and P&L

### Schema

| Table | Change |
|---|---|
| `position` | `exit_time` / `exit_reason` = primary exit (unchanged meaning). `exit_mark` = Σ signed leg exit prices (includes residual legs). New `final_exit_time` = last leg exit. `worst_mark` = MAE up to the primary exit. |
| `position_leg` | `exit_price` populated; add `exit_time` and `exit_reason` (`take_profit`, `stop_loss`, `next_trade`, `expiry`, `condition`, …). |
| `position_continuation` | **dropped**, along with the reader in `detail.py`. Old DBs keep the table; nothing reads it. |

### P&L

| Component | Rule |
|---|---|
| `gross_pnl` | `(open_mark − exit_mark) × multiplier`. Unchanged formula; `exit_mark` now includes residuals. |
| Fees | Per leg: legs held to expiry pay `expiration_fee_per_contract`, all others `exit_fee_per_contract`. |
| `slippage_pct` | Applied per **exit event** (`\|Σ signed prices of legs closed together\|`). Positions where every leg closes on one bar get exactly today's result. Legs closed by `hold_to_expiry` carry **no** slippage (cash-settled, the same as today's expiry exits). |
| Equity curve / metrics | Booked per **exit event**, at the time it happens (see below). |

### Realization events

A residual leg's value isn't known until it resolves, and it may not expire worthless.
Booking it at the primary exit would show P&L in the equity curve before it was realized.
That would affect drawdown, daily P&L and any time-sliced metric.

So P&L is booked as **events**: one for the primary exit (entry fees + the P&L of the legs
it closed), plus one for each residual leg when it resolves. They're stored in a new
`position_pnl_event` table (`position_id, event_time, leg_ids, realized_pnl, fee, slippage`).

- **Uses the events:** the equity curve, daily P&L, drawdown, and the explore overlay
  (`metrics.py`, `explore.py`, the `detail.py` cumulative queries).
- **Unchanged:** position-level stats (win rate, average P&L, `net_pnl`) still use the
  position total.
- **Single-event positions:** where every leg closes on one bar, the result is identical
  to today.

---

## 10. Open questions

### Decided

1. **Stop-loss fill:** `fill: stop`, worst of (level, bar close), with gap-at-open (§4).
   `stop_high` is the pessimistic bound. Check against the Aug 17 bars once that trade's
   strike and expiration are known.
2. **Touch strictness:** `touch_rule: at` is the default; `through` is available (§4).
3. **Small credits:** the level floors at one `round_to` increment, with a warning (§3).
4. **Slippage on held legs:** none for legs held to expiry (§9).
5. **P&L timing:** booked per realization event (§9).
6. **Removed fields** (`leg_out`, `allow_after_hours_exits`, `on_sl_long_continuation`,
   `long_trailing_stop_pct`, `take_profit_pct`): hard validation error naming the
   replacement. `vega_exit` stays as is.

---

## 11. Examples

**Default: close the whole spread** (unchanged YAML, unchanged results)

```yaml
exit:
  stop_loss: 2.0
  take_profit: {pct: 0.70}
```

**Basic leg-out** (replaces `leg_out: true`)

```yaml
exit:
  stop_loss: 2.0
  take_profit: {pct: 0.70}
  close: short_first
```

**Live Period 2 strategy** (gaps 1–4)

```yaml
exit:
  take_profit:
    watch: short_put
    level: {pct: 0.70, reference: open_mark, round_to: 0.25}   # limit = 0.30 × credit
    trigger: touch
    fill: level
  stop_loss:
    watch: short_put
    level: {at: 10.00}
    trigger: touch
    fill: stop
  close: short_only
  reentry_after: primary_exit
  hours: cme_globex
```

---

## 12. Removed

| Item | Location |
|---|---|
| `on_sl_long_continuation`, `long_trailing_stop_pct` | `definition.py` |
| `scan_long_continuation`, `_empty_continuation_df`, `continuation_exits` | `exit.py` |
| `_build_gate_overrides`, `_next_trading_day_open`, `gate_overrides` plumbing | `engine.py` |
| continuation handling | `pnl.py`, `output_db.py`, `study/merger.py`, `detail.py`, frontend |
| `leg_out`, `_adjust_leg_out_exits` | absorbed into `close: short_first` |
| `allow_after_hours_exits` | absorbed into `hours` |
| `take_profit_pct` (legacy top-level) | `definition.py` |
| `_gap_sl` / `_gap_tp` priorities | absorbed into the fill model |
| docs: Leg-Out, Long-Leg Continuation, After-Hours Exits sections | `docs/strategy.md` |

---

## 13. Implementation plan

Each phase leaves the suite green. Phase 1 must reproduce current results exactly: compare
positions before and after on the fixture strategies plus a real study.

1. **Per-leg exit rows, no behaviour change.**
   - `_compute_position_marks` keeps per-leg `close/open/high/low/volume` columns
     alongside `position_mark`.
   - `_find_first_hit` emits per-leg exit rows.
   - `PnLCalculator` sums them; `position_leg.exit_price`, `exit_time` and `exit_reason`
     get populated.
   - Settlement becomes per-leg: generalise `_compute_settlement_marks` and the per-leg
     lookup already in `scan_long_continuation`, keyed by each leg's own expiration.
2. **Remove continuation.** Delete the code paths, the table writer and the UI. Add a
   validation error for the removed fields.
3. **Config models + levels.**
   - `OrderConfig` with `watch / level / trigger / fill / close`, shorthand normalisation and
     validators.
   - `_compute_levels` in the exit module; the TP/SL derivation is removed from
     `entry.py`.
4. **Triggers and fills.**
   - Leg-watch `bar_close` / `touch`, gap-at-open, and the `level` / `next_trade` fills.
   - Liquidity gates follow the watch.
5. **Dispositions.**
   - `with_order` / `next_trade` / `hold_to_expiry` resolution replaces `_adjust_leg_out_exits`.
   - `reentry_after` in `_enforce_entries`; `final_exit_time` written to the output.
   - Write `position_pnl_event`, and move the equity curve and daily metrics onto it.
6. **`exit.hours`.** Presets plus weekly windows; `allow_after_hours_exits` removed.
7. **Docs and UI.**
   - Rewrite the exit sections of `strategy.md` and fix `fill_price_and_costs.md`.
   - Trade detail shows each leg's exit.
8. **Validation against live.** Run the Period 2 config and compare TP fill prices
   (should land exactly on 0.25 / 0.50 / 0.75), the Aug 17 SL, and the Sunday-night TPs.
