# btkit 2.0 — Strategy Definition Schema

Strategies are defined in YAML and loaded into a Pydantic `StrategyDefinition` model,
which validates structure and parses indicator condition expressions at load time.
A strategy may define a single design point or a matrix of combinations via sweep
parameters or explicit combination lists.

---

## Full YAML Schema

A strategy contains one or more `trades`. Each trade is an independent position
structure with its own instrument, entry rules, legs, and exit rules. `universe`,
`costs`, and `matrix` are shared across all trades and defined at the strategy root.

The example below shows a single-trade parameterized strategy using sweep syntax.
Scalar values (single backtests) follow the same structure with plain numbers instead
of lists.

```yaml
strategy:
  name: short_put_spread
  version: "1.0"

  # Backtest date/time universe — shared across all trades
  universe:
    start_date: "2022-01-03"
    end_date:   "2024-12-31"
    session:
      timezone:       "America/New_York"
      start_time:     "09:30"
      end_time:       "16:00"
      weekdays_only:  true
      skip_dates:     []

  # Transaction costs — shared across all trades
  costs:
    slippage_pct: 0.01
    fees:
      entry_fee_per_contract:      0.65   # per leg, charged at open
      exit_fee_per_contract:       0.65   # per leg, charged on TP/SL/condition/DTE exit
      expiration_fee_per_contract: 0.00   # per leg, charged at expiry (often $0 at IBKR)

  # Matrix expansion settings (only relevant for parameterized strategies)
  matrix:
    max_combinations: 100               # error before running if expansion exceeds this

  # One or more independent trade definitions
  trades:
    - name: put_spread

      instrument:
        root_symbol: ES
        asset_class: future         # future | equity | etf
        tick_size: 0.05             # minimum price increment; 0.0 = no rounding (default)

      entry:
        window:
          start: "09:30"            # no entries before this time
          end:   "14:00"            # no entries after this time
        conditions:
          - "rsi_14 < 40"
          - "vix_close < 30"
        min_credit:  0.50           # skip entry if open_mark < this (optional)
        max_debit:   2.00           # skip entry if open_mark > this (optional)
        max_entries_per_day: 1      # cap total positions opened per calendar day (optional)

      # Leg definitions — numeric fields accept scalar, list, or range (see Sweep Parameters)
      legs:
        - name:     short_put
          right:    put
          action:   sell_to_open
          delta:    [-0.20, -0.25, -0.30]   # list sweep
          dte:
            start: 30
            stop:  60
            step:  15                        # range sweep → [30, 45, 60]
          quantity: 1

        - name:          long_put
          right:         put
          action:        buy_to_open
          reference_leg: short_put
          strike_offset: -25.0              # 25 points below short_put — inherits its expiry
          quantity:      1

      exit:
        stop_loss:   [1.50, 2.00, 2.50]     # list sweep
        take_profit: 0.50                    # scalar
        dte_exit:    21
        expiry_exit: true
        conditions:                          # exit if any condition is true (OR logic)
          - "rsi_14 > 70"
          - "vix_close > 40"
        # liquidity:                         # optional — omit to use naive-fill defaults
        #   min_exit_volume:  100
        #   lookback_minutes: 3
        #   pre_expiry_lock_minutes: 15
        #   slippage_model: spread
```

### Multi-trade example: iron condor with independently managed wings

Each wing is its own trade with independent entry signals and exit rules. Both trades
reference the same underlying — all trades in a strategy must share a single underlying.

```yaml
strategy:
  name: iron_condor_independent
  version: "1.0"

  universe:
    start_date: "2022-01-03"
    end_date:   "2024-12-31"
    session:
      timezone:      "America/New_York"
      start_time:    "09:30"
      end_time:      "16:00"
      weekdays_only: true

  costs:
    slippage_pct:     0.01
    fee_per_contract: 0.65

  trades:
    - name: put_spread
      instrument:
        root_symbol: ES
        asset_class: future
      entry:
        window: {start: "09:30", end: "14:00"}
        conditions: ["vix_close < 30", "rsi_14 < 40"]
      legs:
        - {name: short_put, right: put,  action: sell_to_open, delta: -0.20, dte: 45, quantity: 1}
        - {name: long_put,  right: put,  action: buy_to_open,  reference_leg: short_put, strike_offset: -25.0, quantity: 1}
      exit:
        stop_loss:   2.0
        take_profit: 0.50
        dte_exit:    21

    - name: call_spread
      instrument:
        root_symbol: ES
        asset_class: future
      entry:
        window: {start: "09:30", end: "14:00"}
        conditions: ["vix_close < 30", "rsi_14 > 60"]
      legs:
        - {name: short_call, right: call, action: sell_to_open, delta: 0.20, dte: 45, quantity: 1}
        - {name: long_call,  right: call, action: buy_to_open,  reference_leg: short_call, strike_offset: 25.0, quantity: 1}
      exit:
        stop_loss:   2.0
        take_profit: 0.50
        dte_exit:    21
```

---

## Sweep Parameters

Any numeric parameter on `legs` or `exit` can be expressed as a scalar, a list, or a
range. The cartesian product of all non-scalar parameters forms the matrix.

```yaml
# Scalar — single value, fixed across all combinations
delta: -0.30

# List — one combination per value
delta: [-0.20, -0.25, -0.30]

# Range — generates values from start to stop inclusive at the given step
delta:
  start: -0.20
  stop:  -0.35
  step:  -0.05              # generates [-0.20, -0.25, -0.30, -0.35]

# List with null — include "exit disabled" as one combination
# (valid for exit fields only; null is not meaningful for delta or dte)
stop_loss: [null, 1.5, 2.0]        # 3 combos: no SL, 1.5, 2.0
take_profit: [null, 0.50]           # 2 combos: no TP, 0.50 per point
take_profit_pct: [null, 0.70]       # 2 combos: no TP, 70% of open mark
```

**Sweepable parameters:**

| Parameter | Location |
|---|---|
| `delta` | `legs[n]` |
| `dte` | `legs[n]` |
| `take_profit` | `exit` |
| `stop_loss` | `exit` |
| `dte_exit` | `exit` |

A strategy with `delta: [0.20, 0.25, 0.30]` and `dte: [30, 45]` expands to
3 × 2 = 6 combinations. The `matrix.max_combinations` guard raises an error before
any backtest runs if the expansion exceeds the configured limit, forcing the user to
either reduce the parameter space or explicitly raise the cap.

**Design note:** Full-factorial expansion is provided as a convenience for small
parameter spaces. For larger designs, use explicit combinations (see below) to specify
only the points of interest. Sweep parameters and explicit combinations are mutually
exclusive — using both in the same YAML is a validation error at load time.

---

## Explicit Combinations

When full-factorial expansion is not desired, combinations can be listed explicitly.
Each combination specifies only the parameters that vary from the base definition;
all other parameters inherit their scalar values from the base.

### Structured Mode

Uses leg names and section names as keys, mirroring the structure of the base
definition. Only the parameters that differ need to be specified per combination.

```yaml
  combinations:
    - short_put:
        delta: -0.20
        dte:   30
      exit:
        stop_loss: 1.50

    - short_put:
        delta: -0.25
        dte:   45
      exit:
        stop_loss: 2.00

    - short_put:
        delta: -0.30
        dte:   45
      exit:
        stop_loss: 2.50
        take_profit: 0.40   # this combination also overrides take_profit
```

Multi-leg combinations are written naturally by including multiple leg names:

```yaml
    - short_put: {delta: -0.30, dte: 45}
      long_put:  {delta: -0.15, dte: 45}
    - short_put: {delta: -0.25, dte: 30}
      long_put:  {delta: -0.10, dte: 30}
```

### Table Mode

For larger designs where the structured format becomes repetitive, a compact table
syntax is available. Column headers use dot-path notation (`leg_name.param` or
`section.param`); values are plain numbers.

```yaml
  combinations:
    mode: table
    columns: [short_put.delta, short_put.dte, exit.stop_loss]
    rows:
      - [-0.20, 30, 1.50]
      - [-0.25, 45, 2.00]
      - [-0.30, 45, 2.50]
```

Both modes parse to the same internal representation in `StrategyMatrix`. Table mode
is syntactic sugar — the loader converts it to structured form before processing.

**Design note:** The structured mode uses the leg names the user already defined
(e.g. `short_put`, `exit`) as keys, so no new path syntax needs to be learned. Table
mode is offered for designs with many combinations where repeating the key structure
on every row would obscure the parameter values. The dot-path notation in table column
headers is acceptable in that context because it appears only once.

---

## Pydantic Models

```python
from __future__ import annotations
from pydantic import BaseModel, model_validator
from typing import Literal, Any
from datetime import date, time
import numpy as np


# ---------------------------------------------------------------------------
# Sweep parameter types
# ---------------------------------------------------------------------------

class SweepRange(BaseModel):
    start: float
    stop:  float
    step:  float

    def values(self) -> list[float]:
        """
        Generates values from start to stop inclusive.
        Handles both positive and negative step directions.
        """
        result = []
        v = self.start
        while (self.step > 0 and v <= self.stop + abs(self.step) * 1e-9) or \
              (self.step < 0 and v >= self.stop - abs(self.step) * 1e-9):
            result.append(round(v, 10))
            v += self.step
        return result


# Scalar, list, or range — valid for any numeric sweep field
NumericSweep = float | list[float] | SweepRange
IntSweep     = int   | list[int]   | SweepRange


# ---------------------------------------------------------------------------
# Base config models
# ---------------------------------------------------------------------------

class SessionConfig(BaseModel):
    timezone:       str = "America/New_York"
    start_time:     time = time(9, 30)
    end_time:       time = time(16, 0)
    weekdays_only:  bool = True
    skip_dates:     list[date] = []


class UniverseConfig(BaseModel):
    start_date: date
    end_date:   date
    session:    SessionConfig = SessionConfig()


class InstrumentConfig(BaseModel):
    root_symbol: str
    asset_class: Literal["future", "equity", "etf"]
    tick_size: float = 0.0   # minimum price increment; 0.0 = continuous (no rounding)


class EntryWindowConfig(BaseModel):
    start: time
    end:   time

    @model_validator(mode="after")
    def start_before_end(self) -> EntryWindowConfig:
        if self.start >= self.end:
            raise ValueError("entry window start must be before end")
        return self


class EntryConfig(BaseModel):
    window:              EntryWindowConfig
    conditions:          list[str] = []
    min_credit:          NumericSweep | None = None  # enter only if open_mark >= this value
    max_debit:           NumericSweep | None = None  # enter only if open_mark <= this value
    max_entries_per_day: int | None = None           # None = unlimited re-entries per day


class LegConfig(BaseModel):
    name:            str
    right:           Literal["call", "put"]
    action:          Literal["buy_to_open", "sell_to_open"]
    dte:             IntSweep | None = None   # required for delta legs; None = inherit parent expiry for offset legs
    quantity:        int = 1
    # Selection mode A: delta-targeted
    delta:           NumericSweep | None = None
    delta_tolerance: float = 0.10
    dte_tolerance:   int = 5
    # Selection mode B: fixed offset from a reference leg
    strike_offset:   float | None = None   # positive = above ref strike, negative = below
    reference_leg:   str | None = None     # name of the delta-targeted leg to offset from


class StopLossConfig(BaseModel):
    price:     NumericSweep         # per-point distance above open_mark that triggers SL
    condition: str | None = None   # AND-gated: SL only fires when this expression is also true


class TakeProfitConfig(BaseModel):
    price:             NumericSweep | None = None  # fixed per-point offset from open_mark
    pct:               NumericSweep | None = None  # fraction of open_mark to retain (e.g. 0.70 = exit at 70% profit)
    condition:         str | None = None           # AND-gated: TP only fires when this expression is also true
    confirmation_bars: int = 1                     # consecutive 1-min bars at/below TP threshold before exit fires


class LiquidityConfig(BaseModel):
    min_exit_volume:         Optional[int] = None   # None → volume gate disabled
    lookback_minutes:        int = 3
    pre_expiry_lock_minutes: Optional[int] = None   # None → lock disabled
    slippage_model:          Literal["flat", "spread"] = "flat"


class ExitConfig(BaseModel):
    stop_loss:      NumericSweep | StopLossConfig | None = None
    take_profit:    NumericSweep | TakeProfitConfig | None = None
    take_profit_pct: NumericSweep | None = None   # legacy top-level form
    dte_exit:       IntSweep | None = None
    expiry_exit:    bool = True
    conditions:     list[str] = []   # exit if any condition is true (OR logic)
    liquidity:      LiquidityConfig = LiquidityConfig()


class FeesConfig(BaseModel):
    entry_fee_per_contract:      float = 0.0
    exit_fee_per_contract:       float = 0.0
    expiration_fee_per_contract: float = 0.0


class CostsConfig(BaseModel):
    slippage_pct:     float = 0.0
    fee_per_contract: float = 0.0       # legacy: split evenly as entry=fee/2, exit=fee/2
    fees: FeesConfig | None = None      # structured form; mutually exclusive with fee_per_contract


class MatrixConfig(BaseModel):
    max_combinations: int = 100


# ---------------------------------------------------------------------------
# Explicit combinations types
# ---------------------------------------------------------------------------

# Structured mode: leg/section names → parameter overrides
# e.g. {"short_put": {"delta": -0.20, "dte": 30}, "exit": {"stop_loss": 1.50}}
StructuredCombination = dict[str, dict[str, Any]]


class TableCombinations(BaseModel):
    mode:    Literal["table"]
    columns: list[str]              # dot-path refs e.g. "short_put.delta"
    rows:    list[list[float | int]]

    @model_validator(mode="after")
    def row_lengths_match_columns(self) -> TableCombinations:
        for i, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"row {i} has {len(row)} values but {len(self.columns)} columns defined"
                )
        return self


# ---------------------------------------------------------------------------
# Trade definition — one per independent position structure
# ---------------------------------------------------------------------------

class TradeDefinition(BaseModel):
    name:       str
    instrument: InstrumentConfig
    entry:      EntryConfig
    legs:       list[LegConfig]
    exit:       ExitConfig

    @model_validator(mode="after")
    def leg_names_unique(self) -> TradeDefinition:
        names = [leg.name for leg in self.legs]
        if len(names) != len(set(names)):
            raise ValueError("leg names must be unique within a trade")
        return self


# ---------------------------------------------------------------------------
# Top-level strategy definition
# ---------------------------------------------------------------------------

class StrategyDefinition(BaseModel):
    name:         str
    version:      str = "1.0"
    universe:     UniverseConfig
    costs:        CostsConfig = CostsConfig()
    matrix:       MatrixConfig = MatrixConfig()
    trades:       list[TradeDefinition]
    combinations: list[StructuredCombination] | TableCombinations | None = None

    @model_validator(mode="after")
    def trades_share_underlying(self) -> StrategyDefinition:
        """All trades must reference the same root_symbol."""
        symbols = {t.instrument.root_symbol for t in self.trades}
        if len(symbols) > 1:
            raise ValueError(
                f"all trades must share the same underlying; got {symbols}"
            )
        return self

    @model_validator(mode="after")
    def trade_names_unique(self) -> StrategyDefinition:
        names = [t.name for t in self.trades]
        if len(names) != len(set(names)):
            raise ValueError("trade names must be unique within a strategy")
        return self

    @model_validator(mode="after")
    def validate_combinations_vs_sweeps(self) -> StrategyDefinition:
        """
        Sweep parameters and explicit combinations are mutually exclusive.
        Raises if both are present in the same definition.
        """
        ...
```

---

## Entry Window

The `entry.window` block defines a time range within each session during which new
positions may be opened. The `EntryScanner` evaluates entry conditions at every
1-minute bar whose timestamp falls within the window.

Each trade enforces a **one-at-a-time constraint**: a new entry for a given trade is
only opened once the previous position in that trade has closed. This is enforced
exactly after Pass 2 (when real exit times are known), not approximated during Pass 1.
See [Engine Design](#one-at-a-time-constraint) below.

### Max entries per day (`max_entries_per_day`)

`entry.max_entries_per_day` caps the total number of positions opened for a trade on
any single calendar day. It is applied after one-at-a-time enforcement and counts all
positions taken that day — including the first entry, not just re-entries.

```yaml
entry:
  window:
    start: "09:45"
    end:   "14:30"
  max_entries_per_day: 1   # one position per day; never re-enter after an early TP exit
```

Setting `max_entries_per_day: 1` means the strategy takes its first valid entry of the
day and holds or expires, even if that position closes at 70% TP by 10:30 AM and
the entry window remains open for another four hours.

**Use case:** The vectorized backtest scans every 1-minute bar and re-enters
immediately after each exit, accumulating more positions per day than a live system
typically does. This inflates the total position count (and aggregate P&L) relative to
live execution. Capping at the observed live re-entry rate closes this gap without
changing per-trade economics.

`None` (default) means no cap — the engine re-enters as many times as the one-at-a-time
constraint and entry window allow.

**Design note:** A fixed `time` field was considered but rejected in favour of a
`window` to support strategies where signals can fire multiple times per session. The
one-at-a-time constraint is the guard against concurrent positions accumulating within
a single trade.

---

## Entry Condition Evaluation

All entry conditions — indicator values, underlying bar values, leg properties, and
if/then expressions — are evaluated in a **single unified phase** after leg selection
and open mark computation. The entry pipeline is:

```
1. Window + session filters     (time-based, cheap — eliminates most timestamps)
2. Leg selection                (single batched DuckDB query on remaining timestamps)
3. Open mark computation
4. All conditions evaluated     (single phase — full namespace available)
5. min_credit / max_debit       (mark-based filters)
```

The one-at-a-time constraint (at most one open position per trade at a time) is **not**
part of the Pass 1 pipeline. It is applied after Pass 2 once real exit times are known.
See the engine design notes in `classes.md`.

**Design note:** An earlier design split conditions into pre-leg and post-leg phases
to filter timestamps before leg selection as a performance optimisation. This was
dropped because leg selection in btkit 2.0 is a single batched DuckDB query — not a
Python loop — so the cost of querying more timestamps is modest. The simpler
single-phase design avoids the need to classify conditions at load time and gives
every condition access to the full column namespace.

### Condition Namespace

Every condition expression has access to the following columns:

| Source | Examples |
|---|---|
| Underlying bar values | `close`, `open`, `high`, `low`, `volume` |
| Indicator columns | `rsi_14`, `vwap`, `sma_20`, `vix_close` (any column in `indicator_bars`) |
| Leg properties | `short_put.strike`, `short_put.delta`, `short_put.iv`, `short_put.dte` |
| Position mark | `open_mark` |

Leg property references use dot notation (`<leg_name>.<field>`). Indicator names must
not contain dots — this is the delimiter that identifies leg property references.

### Condition Syntax

**Simple comparison:**
```yaml
conditions:
  - "rsi_14 < 40"
  - "vix_close < 30"
```
All conditions in the list must be true for entry (AND logic).

**If/then (logical implication):**
```yaml
conditions:
  - "if close < vwap then short_put.strike < vwap_lower_1std"
```
`if A then B` is equivalent to `(NOT A) OR B`. The entry is only rejected when the
predicate is true and the consequence is false. When the predicate is false, the
condition passes unconditionally (implied `else true`).

**Supported operators:** `>`, `<`, `>=`, `<=`, `==`, `!=`, `and`, `or`, `not`

```python
# strategy/loader.py
def parse_condition(expr: str) -> pl.Expr:
    """
    Parses a condition string (simple comparison or if/then) into a Polars
    expression. Resolves identifiers against the post-selection namespace:
    indicator columns, underlying bar columns, and leg property columns.
    If/then is rewritten as (~predicate) | consequence before compilation.
    """
    ...
```

Conditions referencing names not present in the namespace raise a validation error
at load time. Complex expressions beyond this syntax (e.g. arithmetic, rolling
lookbacks) should be implemented as indicator columns in the user's indicator script.

---

## Exit Priority Order

When multiple exit conditions apply, they are evaluated in this fixed order:

| Priority | Condition | Notes |
|---|---|---|
| 1 | Gap open past SL | Check position open mark before bar close processing |
| 2 | Gap open past TP | Same |
| 3 | Stop loss | `position_mark >= sl_price` |
| 4 | Take profit | `position_mark <= tp_price` |
| 5 | Indicator exit | any condition in `exit.conditions` is true |
| 6 | DTE exit | `dte <= dte_exit` |
| 7 | Expiry | Last bar before expiration |

SL and TP take priority over indicator conditions because they are hard risk management
rules. Indicator conditions represent "the trade thesis has changed" and are softer
signals that yield to active risk limits. See `fill_price_and_costs.md` for fill price
rules.

---

## Credit and Debit Filters

`entry.min_credit` and `entry.max_debit` are optional mark-based entry filters. Like
all other conditions, they are evaluated in the single unified condition phase after
leg selection and mark computation (see Entry Condition Evaluation below).

For credit strategies (net STO position): use `min_credit`. The entry is skipped if
`open_mark < min_credit`.

For debit strategies (net BTO position): use `max_debit`. The entry is skipped if
`open_mark > max_debit`.

Both fields are sweepable (`NumericSweep | None`) — testing different credit thresholds
is a common DOE parameter. Using both simultaneously on the same strategy is permitted
but unusual; validation does not prevent it.

---

## Exit Indicator Conditions

`exit.conditions` accepts the same expression syntax as `entry.conditions`, referencing
the same indicator column names. They are evaluated on each bar during the exit scan
using the indicator values at that bar's timestamp.

**Logic:** OR — the position is closed if *any single condition* is true. AND logic can
be composed within one condition string using the `and` keyword:

```yaml
exit:
  conditions:
    - "rsi_14 > 70"                    # exits if RSI spikes
    - "vix_close > 40"                 # OR if VIX spikes
    - "rsi_14 > 60 and sma_20 < close" # OR if both sub-conditions hold
```

When an indicator exit fires, `exit_reason` is recorded as `'condition'` and the fill
price follows the standard bar-close mark rule (not a TP/SL threshold price), since
the exit is not triggered by a specific price level. See `fill_price_and_costs.md`.

---

## Exit Liquidity Modelling

By default the exit engine assumes every bar is fillable at the bar's mark price —
the same naive assumption used in most backtesting frameworks. For short-dated,
far-OTM options this can overstate the number of price-triggered exits (TP/SL) that
would actually fill in live trading, leading to optimistic P&L expectations.

The optional `exit.liquidity` block exposes three independent controls:

```yaml
exit:
  stop_loss:   5.00
  take_profit_pct: 0.70
  expiry_exit: true

  liquidity:
    # 1. Volume gate — suppress price-triggered exits when the option is illiquid
    min_exit_volume:  100          # cumulative contracts over the lookback window
    lookback_minutes: 3            # rolling time window (default: 3 min)

    # 2. Pre-expiry lock — suppress price-triggered exits near option expiration
    pre_expiry_lock_minutes: 30    # lock TP/SL in the final N minutes before expiry close

    # 3. Spread slippage — add half the bar high-low range per leg to the fill price
    slippage_model: spread         # "flat" (default) | "spread"
```

All fields are optional. Omitting `liquidity:` entirely (or any individual field)
preserves the original naive-fill behaviour — the engine adds no extra columns to
its queries and incurs no overhead.

### Feature 1 — Volume Gate (`min_exit_volume`)

A price-triggered exit (TP, SL, or gap) is only allowed when the **rolling sum of
option volume across all legs** over the preceding `lookback_minutes` minutes is at
least `min_exit_volume` contracts.

- Volume is accumulated from the same 1-minute OHLCV bars used for mark prices.
- Bars missing from the database (common near expiration for illiquid strikes) count
  as **zero volume** — the engine does not assume they traded.
- The minimum volume is the *weakest leg*: if any single leg has zero recent volume
  the gate is closed regardless of volume in the other legs.
- Expiry and DTE exits fire unconditionally — the volume gate only applies to
  price-triggered exits.

**Practical guidance:** For 0-DTE short credit spreads on ES, values in the range
50–200 contracts per 3-minute window are a reasonable starting point. Run a study
sweep over this parameter against live-trading results to calibrate.

### Feature 2 — Pre-expiry Lock (`pre_expiry_lock_minutes`)

Suppresses all price-triggered exits (TP, SL, gap) during the final N minutes before
the instrument's expiry close time (`instrument.expiry_close_time`). This models the
illiquid, wide-spread conditions that typically prevail as an option approaches its
last print.

- `expiry_close_time` must be set on the instrument when this feature is used; if it
  is omitted the engine falls back to the session end time.
- Expiry exits (i.e. the position is held to expiration) still fire normally.
- The lock applies only on expiration day (`dte == 0`).

**Practical guidance:** For 0-DTE ES options, 15–30 minutes before the 16:00 ET
expiry close is a common lock window (i.e. `pre_expiry_lock_minutes: 15` suppresses
TP/SL after 15:45).

### Feature 3 — Spread Slippage (`slippage_model: spread`)

When set to `"spread"`, the engine adds **half the bar's high-low range** per leg to
the fill price on every exit. This estimates the cost of crossing the bid-ask spread
when closing a position.

- For credit spreads (sell-to-open), closing the position is a debit: the fill price
  is the mark plus slippage, increasing the cost.
- The bar high-low range is used as a proxy for the bid-ask spread width because it
  is available in the existing 1-minute OHLCV data with no additional queries.
- This is additive to `costs.slippage_pct` and `costs.fee_per_contract` — all three
  apply simultaneously.
- Expiry exits (position held to 0) are unaffected by spread slippage.

**Design note:** Half-spread slippage applied to every exit is a conservative
assumption. Realistically, limit orders often fill inside the spread; however,
modelling limit-order fill probability requires tick data not available at 1-minute
resolution. The conservative assumption produces a lower bound on strategy
profitability that is more consistent with live results for illiquid 0-DTE options.

### Interaction with `costs.slippage_pct`

`costs.slippage_pct` applies a percentage-of-mark haircut to both entry and exit fills
and represents general market impact / spread cost on both sides of the trade.
`liquidity.slippage_model: spread` adds a separate, bar-specific cost only to exits.
They model different phenomena and can be used simultaneously.

---

## TP Confirmation Bars

`take_profit.confirmation_bars` requires the TP condition to hold for N **consecutive**
1-minute bars before the exit fires. The default of 1 preserves the original behaviour
(exit immediately when the mark first crosses the TP threshold).

```yaml
exit:
  take_profit:
    pct: 0.70
    confirmation_bars: 2   # require 2 consecutive bars at/below TP level before exiting
  stop_loss: 5.00
  expiry_exit: true
```

`confirmation_bars` only applies to the regular close-based TP check
(`position_mark <= tp_price`). Gap-open TP (`spread_open_mark <= tp_price` at bar
open) is instantaneous and bypasses confirmation, since a gap past the TP level
is a resolved event, not a brief touch.

**Use case:** The backtest engine evaluates every 1-minute bar and captures
*fleeting TP touches* — moments where the spread mark dips to the TP threshold for
a single bar then rebounds — that a live trading system with scanning latency and
order-routing delays would miss. These single-bar touches are captured as TP exits
in the backtest but result in expiry-worthless outcomes in live execution. Since an
expiry-worthless exit and a near-zero TP exit produce nearly identical P&L, the
difference shows up in exit composition (high backtest TP rate, high live expiry rate)
but not in per-trade return bias.

Setting `confirmation_bars: 2` filters out almost all single-bar TP touches while
still allowing genuine, sustained TP conditions to fire on the second consecutive bar.
This closes the exit composition gap without disturbing per-trade P&L calibration.

**Interaction with other exit controls:** Confirmation is applied after all other
`_price_ok` guards (session window, volume gate, pre-expiry lock). A bar where the
TP condition is met but the position is locked or out-of-session does not count toward
the confirmation streak — the streak only accumulates on bars where the TP would
otherwise be valid.

---

## Tick Size

Options trade at discrete price increments. Setting `instrument.tick_size` causes all
fill prices and the MAE high-water mark to be rounded to the nearest tick before being
written to the output database. This prevents the backtest from recording fills at
prices the exchange cannot actually print.

```yaml
instrument:
  root_symbol: ES
  asset_class: future
  tick_size: 0.05      # $0.05 per point — standard for ES/MES options
```

**Common values:**

| Instrument | Typical option tick |
|---|---|
| ES / MES futures options | 0.05 |
| NQ / MNQ futures options | 0.05 |
| SPY / QQQ equity options | 0.01 |
| SPX / XSP index options | 0.05 (strikes > $3) or 0.10 |
| IWM / EEM equity options | 0.01 |

Check your broker's contract spec if unsure — tick sizes occasionally change or differ
between regular and mini contracts.

**What gets rounded:**

| Price | Stage |
|---|---|
| `open_mark` | Entry — after summing leg close prices |
| `tp_price`, `sl_price` | Entry — after deriving from `open_mark` |
| `exit_mark` | Exit — gap fills, condition/DTE/expiry bar-close prices |
| `worst_mark` | Exit — MAE high-water mark at output |

TP and SL fills are already on-tick (thresholds are rounded at entry time) so they are
not double-rounded. Only bar-level prices (gap opens, condition exits, expiry marks)
need rounding at exit time.

**Default:** `tick_size: 0.0` — all rounding disabled, prices are continuous. This is
the default and reproduces the original engine behaviour exactly. Omitting the field is
equivalent to `tick_size: 0.0`.

See [fill_price_and_costs.md](fill_price_and_costs.md) for the full rounding rules.

---

## Leg Selection

At each entry timestamp, the `EntryScanner` queries `option_greeks` to find the
best-matching option for each leg by minimising the distance to the target delta,
subject to a DTE window around the target DTE.

```
best_match = argmin |actual_delta - target_delta|
             where |actual_dte - target_dte| <= dte_tolerance
```

Default tolerances (configurable per leg):
- `delta_tolerance: 0.05`
- `dte_tolerance: 5`

If no option is found within tolerance at a given entry timestamp, that entry is
skipped entirely — all legs must be fillable for an entry to be recorded.
