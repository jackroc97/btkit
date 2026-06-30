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
        no_reentry_after_loss: true # block same-day re-entry after a stop-loss exit (optional)

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
        #   max_leg_stale_minutes:   5       # suppress TP/SL when any leg's last bar is older than N minutes
        #   slippage_model: spread
        # allow_after_hours_exits: false     # set true to allow TP/SL to trigger outside session hours
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
| `vega_exit` | `exit` |
| `roll.dte` | `roll` |
| `roll.vega` | `roll` |

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
    no_reentry_after_loss: bool = False              # block same-day re-entry after a stop-loss exit


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
    vega_exit:      NumericSweep | None = None    # exit when spread net vega < this; see Vega Exit
    expiry_exit:    bool = True
    conditions:     list[str] = []   # exit if any condition is true (OR logic)
    liquidity:      LiquidityConfig = LiquidityConfig()
    allow_after_hours_exits: bool = False   # see After-Hours Exits
    leg_out:        bool = False             # see Leg-Out Independent Exit
    on_sl_long_continuation: bool = False   # see Long-Leg Continuation After SL
    long_trailing_stop_pct:  float = 0.50   # trailing stop fraction for on_sl_long_continuation


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
# Roll block
# ---------------------------------------------------------------------------

class RollConfig(BaseModel):
    window: EntryWindowConfig | None = None  # roll only within this window; None = use trade entry window
    dte:    IntSweep | None = None           # roll when remaining DTE <= this value
    vega:   NumericSweep | None = None       # roll when spread net vega < this value
    # At least one of dte or vega must be set (validator enforced)


# ---------------------------------------------------------------------------
# Trade definition — one per independent position structure
# ---------------------------------------------------------------------------

class TradeDefinition(BaseModel):
    name:       str
    instrument: InstrumentConfig
    entry:      EntryConfig
    legs:       list[LegConfig]
    exit:       ExitConfig
    roll:       RollConfig | None = None   # see Roll Block

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

The definition of "closed" depends on which exit features are active:

- **Default / `leg_out: true`** — the gate opens at the spread's `exit_time`, i.e. the
  bar where TP or SL fired. A new spread may be entered immediately on the next valid
  bar. With `leg_out`, the long leg's fill price is adjusted to the next real bar after
  the exit, but that adjustment does not hold the gate open — the long leg is not
  modelled as an ongoing open position.
- **`on_sl_long_continuation: true`** — for SL exits with an active continuation, the
  gate is extended to `max(continuation_exit_time, next_trading_day_open_after_sl)`.
  No new spread is entered until the long leg continuation has closed **and** the next
  session has started. TP and expiry exits use the standard `exit_time` gate and are
  not affected.

See [Leg-Out Independent Exit](#leg-out-independent-exit-leg_out) and
[Long-Leg Continuation After SL](#long-leg-continuation-after-sl-on_sl_long_continuation)
for full details on each feature.

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

### No re-entry after loss (`no_reentry_after_loss`)

`entry.no_reentry_after_loss: true` prevents the strategy from entering a new position
on the same calendar day as a stop-loss exit. Once a position closes as a `stop_loss`
or `gap_sl`, the engine blocks all further entries for the remainder of that trading day.

```yaml
entry:
  window:
    start: "09:45"
    end:   "14:30"
  no_reentry_after_loss: true
```

**Loss definition:** A loss is any position whose `exit_reason` is `stop_loss` or
`gap_sl`. TP, expiry, DTE, and condition exits do not trigger the block, even if
`net_pnl` happens to be negative (e.g., a TP that fires below the open mark due to
gap-down fill prices).

**How it works:** Applied after one-at-a-time enforcement. The engine walks positions
chronologically; when it encounters a loss on calendar date D (in the session timezone),
it records D as blocked and drops all subsequent entries with `entry_date == D`. The
losing position itself is kept — only re-entries on the same day are dropped.

**Interaction with `max_entries_per_day`:** The two settings can be used together.
`max_entries_per_day` is applied first, capping total entries per day. The loss block
then removes any same-day entries that follow a stop-loss within the surviving set.

**Interaction with `on_sl_long_continuation`:** When the continuation feature is active,
a stop-loss exit already extends the one-at-a-time gate past midnight (into the next
trading day). `no_reentry_after_loss` adds no additional restriction in that case — the
gate already prevents same-day re-entry. Both settings may coexist safely.

**Use case:** Live traders often step away after a stop-loss — either as a discipline
rule ("two strikes and done") or because a stop-out signals adverse intraday conditions.
This setting models that behaviour in the backtest, preventing the engine from
accumulating multiple losing positions on the same day when the entry window remains
open after the stop.

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

**Arithmetic expressions:**
The `+`, `-`, `*`, and `/` operators are supported within condition strings, along with
`abs()` as the sole supported function call:

```yaml
exit:
  conditions:
    # Exit when MTM gain >= 2× the absolute open mark (doubles the initial credit/debit)
    - "position_mark - open_mark >= 2.0 * abs(open_mark)"
    # Exit when underlying move from entry exceeds 1% of open mark
    - "abs(close - open_mark) > 0.01 * abs(open_mark)"
```

Arithmetic can reference any column available in the evaluation namespace. `abs()` is
the only function call supported — other functions (e.g. `min`, `sqrt`) raise a
`ValueError` at parse time.

**Supported operators:** `>`, `<`, `>=`, `<=`, `==`, `!=`, `and`, `or`, `not`, `+`, `-`, `*`, `/`

**Supported function calls:** `abs(expr)`

```python
# strategy/loader.py
def parse_condition(expr: str) -> pl.Expr:
    """
    Parses a condition string (comparison, boolean, arithmetic, or if/then)
    into a Polars expression. Resolves identifiers against the post-selection
    namespace: indicator columns, underlying bar columns, leg property columns,
    and (in the exit context) position_mark and open_mark.
    """
    ...
```

Conditions referencing names not present in the namespace at evaluation time raise an
error that is captured per-trade and logged as a warning (the condition evaluates to
`False`). Rolling lookbacks or multi-row aggregations should be pre-computed as
indicator columns in the user's indicator script.

---

## Exit Priority Order

When multiple exit conditions apply on the same bar, they are resolved in this fixed order:

| Priority | Exit reason | Notes |
|---|---|---|
| 1 | `gap_sl` | Bar open already past SL threshold |
| 2 | `gap_tp` | Bar open already past TP threshold |
| 3 | `stop_loss` | `position_mark >= sl_price` at bar close |
| 4 | `take_profit` | `position_mark <= tp_price` at bar close |
| 5 | `condition` | Any condition in `exit.conditions` is true |
| 6 | `roll` | Roll threshold crossed within roll window (see [Roll Block](#roll-block-roll)) |
| 7 | `vega_exit` | Spread net vega below threshold (see [Vega Exit](#vega-exit-vega_exit)) |
| 8 | `dte_exit` | `dte <= dte_exit` |
| 9 | `expiry` | Last bar before expiration |

SL and TP take priority over management exits (condition, roll, vega, DTE) because they
are hard risk limits. Conditions (5) represent "the trade thesis has changed" and fire
before the structural management exits below them. Roll (6) closes the position to
re-enter fresh; vega (7) and DTE (8) are passive decay/management thresholds. Expiry
(9) is the backstop when nothing else fires first.

See `fill_price_and_costs.md` for fill price rules per exit type.

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

## Arithmetic in Exit Conditions

Exit conditions (and entry conditions) support arithmetic operators and the `abs()`
function, enabling expressions that relate the current position mark to the entry price
or other runtime values.

### Available columns in exit conditions

In addition to indicator columns and underlying bar values, the following columns are
always available when evaluating `exit.conditions`:

| Column | Description |
|---|---|
| `position_mark` | Current bar-close spread mark (Σ signed_qty × leg_close) |
| `open_mark` | Spread mark at entry time |
| `spread_open_mark` | Spread mark at current bar open (gap detection) |
| `_dte_now` | Remaining calendar days to expiration at this bar |
| `_spread_vega` | Current bar net spread vega (Σ signed_qty × leg_vega); `null` when `vega_exit` / `roll.vega` is not configured |
| `open_vega` | Spread vega at entry time; constant per position; `null` when vega is not computed (see above) |

### Arithmetic operators

```yaml
exit:
  conditions:
    # Exit when the MTM gain is at least 2× the initial credit/debit in absolute terms.
    # Works for both credit spreads (open_mark > 0) and debit spreads (open_mark < 0).
    - "position_mark - open_mark >= 2.0 * abs(open_mark)"

    # Exit when the spread has compressed to less than 10% of the open mark
    - "abs(position_mark) < 0.10 * abs(open_mark)"

    # Exit when spread has lost more than 1.5 points relative to a specific threshold
    - "open_mark - position_mark > 1.5"
```

**Supported operators:** `+`, `-`, `*`, `/`

**Supported function:** `abs(expr)` — takes a single argument, which can itself be an
arithmetic expression. No other function calls are supported; attempting to use `min`,
`max`, `sqrt`, etc. raises a `ValueError` at load time.

**Operator precedence** follows standard Python rules: `*` and `/` bind tighter than
`+` and `-`, and all arithmetic binds tighter than comparisons.

### MTM gain exit — worked example

A put-backspread opened for a -$2.00 debit (`open_mark = -2.00`) should exit when the
gain equals 2× the initial debit (i.e. when the position marks at +$2.00 profit):

```
position_mark - open_mark >= 2.0 * abs(open_mark)
⟹  position_mark - (-2.00) >= 2.0 * 2.00
⟹  position_mark >= 2.00
```

The same expression works equally for credit spreads (where `open_mark > 0`):
with `open_mark = 1.50`, it exits when `position_mark - 1.50 >= 3.00`,
i.e. `position_mark >= 4.50` — when the gain exceeds 3 full credits.

---

## Vega Exit (`vega_exit`)

`exit.vega_exit` closes the position when the spread's **net vega** drops below a
configured threshold. It is a decay-management exit: as time passes and the option
approaches expiration, vega collapses. A falling vega means the spread has little
remaining sensitivity to volatility — a signal that the risk/reward of holding has
deteriorated and the position should be closed.

```yaml
exit:
  stop_loss:   5.00
  take_profit: 0.50
  dte_exit:    7
  vega_exit:   0.15   # exit when net spread vega drops below 0.15
```

### What is "net vega"?

Net vega is computed per bar as the sum of each leg's vega weighted by signed quantity:

```
spread_vega = Σ(signed_qty × leg_vega)
```

where `signed_qty = +quantity` for STO legs and `-quantity` for BTO legs. For a short
put spread (-1P +1P), the net vega is negative (the spread is net short vega); vega_exit
fires when this value rises above the threshold (becomes less negative). For long-skew
structures like a put backspread (-1P +2P), the net vega is positive and falls as
expiration approaches.

Set `vega_exit` to match the sign convention of your spread:
- **Net short vega spreads (credit spreads):** net vega is negative. Use a negative
  threshold, e.g. `vega_exit: -0.05` (exit when vega is no longer meaningfully negative).
- **Net long vega spreads (backspreads, debit spreads):** net vega is positive. Use a
  positive threshold, e.g. `vega_exit: 0.15` (exit when vega falls below 0.15).

### Data source

Vega values are pulled from the `option_greeks` table in the input database, fetched
once per expiration cohort to avoid per-entry round-trips. The greeks data is joined to
each position's bar stream by `(instrument_id, ts_event)`.

If `option_greeks` data is absent for a bar, `_spread_vega` will be null for that bar
and `_vega_exit` will not fire — the position continues.

### Priority

Vega exit fires at priority 7 (after condition exits at 5 and roll at 6, before
dte_exit at 8). See [Exit Priority Order](#exit-priority-order).

### Sweepability

`vega_exit` is a `NumericSweep` field — it accepts a scalar, list, or range for use in
study sweeps:

```yaml
exit:
  vega_exit: [0.05, 0.10, 0.15, 0.20]   # 4-point sweep
```

### Relative vega thresholds via exit conditions

When the desired exit threshold is a **fraction of the entry vega** rather than a fixed
absolute value, use `open_vega` in `exit.conditions` instead of `vega_exit`:

```yaml
exit:
  conditions:
    # Exit when spread vega has decayed to 30% of entry vega
    - "_spread_vega < 0.3 * open_vega"
```

`open_vega` is the spread net vega captured at the first greeks bar at or after entry
time. It is a constant per position, so `0.3 * open_vega` evaluates to a fixed dollar
amount for each trade and the comparison with the current `_spread_vega` works correctly.

Both `_spread_vega` and `open_vega` are only populated when vega computation is active.
The engine automatically activates the greeks fetch when any of the following are true:
`exit.vega_exit` is set, `roll.vega` is set, **or** any condition string contains
`_spread_vega` or `open_vega`. So using them in a condition is self-contained — no
extra configuration required:

```yaml
exit:
  dte_exit: 3
  conditions:
    - "_spread_vega < 0.3 * open_vega"
```

---

## Roll Block (`roll`)

The `roll` block closes the current position when a threshold is crossed and
**immediately re-opens** a fresh position with new delta targeting. A roll is
conceptually a continuation of the same trade thesis under better terms — the position
is closed to capture decay/gain and re-opened at the current market delta.

```yaml
roll:
  dte:    10               # roll when ≤ 10 days remain
  vega:   0.15             # roll when spread net vega < 0.15
  window:                  # roll only within this time window (optional)
    start: "09:45"
    end:   "14:30"
```

At least one of `dte` or `vega` must be specified. Both may be set simultaneously —
the roll fires when *either* threshold is crossed (OR logic).

### How a roll works

1. The exit scanner detects the roll condition at some bar and records `exit_reason = "roll"`.
2. The entry scanner independently finds the next valid entry bar after the roll exit time.
3. The engine's enforcement logic:
   - `_enforce_one_at_a_time` gates the re-entry at the roll exit time (standard behaviour).
   - `_enforce_max_entries_per_day` **exempts roll re-entries** from the daily cap — the
     re-entry immediately following a roll exit on the same calendar day is not counted
     against `max_entries_per_day`.

The re-opened position gets fresh delta selection (new strike, same DTE target) and a new
`open_mark`. Its P&L is tracked independently from the closed position in the output database.

### Roll window

`roll.window` restricts when a roll can fire. If the roll threshold is crossed outside the
window, the position is held. If `window` is omitted, the trade's `entry.window` is used.

A common pattern is to restrict rolls to core hours to avoid re-entering near the open or
close:

```yaml
roll:
  dte:    7
  window:
    start: "09:45"
    end:   "14:30"
```

### Interaction with `max_entries_per_day`

Without the roll bypass, a `max_entries_per_day: 1` strategy that rolls once per day
would allow only the morning entry and drop the roll re-entry as a second daily entry.
The engine detects roll exits and exempts the immediately following same-day re-entry
from the daily cap, so the sequence `entry → roll → re-entry` is treated as a single
position lineage, not two independent trades.

```yaml
entry:
  max_entries_per_day: 1   # one entry per day; roll re-entries are exempt
roll:
  dte: 10
```

**Only the single re-entry immediately after each roll is exempt.** If the re-opened
position also rolls the same day (a second roll), its re-entry would be the third daily
entry and subject to the normal cap.

### Interaction with `no_reentry_after_loss`

A `roll` exit reason is not considered a loss — `no_reentry_after_loss` only blocks
re-entries after `stop_loss` or `gap_sl` exits. A position that rolls does not trigger
the loss block, and the roll re-entry proceeds normally even when `no_reentry_after_loss: true`.

### Interaction with `exit.vega_exit`

`roll.vega` and `exit.vega_exit` both use the same `_spread_vega` column computed per
cohort from `option_greeks`. When both are configured, vega is fetched once and both
thresholds are evaluated from the same data.

However, they have different priorities: `roll` fires at priority 6 and `vega_exit` at
priority 7. If both thresholds are crossed on the same bar, the roll takes precedence and
the position closes as `"roll"` (with a planned re-entry), not `"vega_exit"` (standalone close).

### Fill price

Roll exits use the standard bar-close mark + slippage as the fill price (same as
condition and DTE exits). The slippage model configured in `exit.liquidity` applies.

### Sweepability

`roll.dte` and `roll.vega` are both sweepable:

```yaml
roll:
  dte:  [7, 10, 14]   # 3-point sweep on roll DTE threshold
  vega: 0.15
```

### Full example: put-backspread with roll and vega management

```yaml
strategy:
  name: put_backspread_managed
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
    slippage_pct: 0.01
    fees:
      entry_fee_per_contract:  0.65
      exit_fee_per_contract:   0.65

  trades:
    - name: put_backspread

      instrument:
        root_symbol: ES
        asset_class: future
        tick_size: 0.05

      entry:
        window:
          start: "09:45"
          end:   "14:00"
        max_entries_per_day: 1
        conditions:
          - "ves1d_close > 20"    # enter only when 1-day VES is elevated

      legs:
        - name:     short_put
          right:    put
          action:   sell_to_open
          delta:    -0.30
          dte:      21
          quantity: 1

        - name:          long_put_1
          right:         put
          action:        buy_to_open
          reference_leg: short_put
          strike_offset: -25.0
          quantity:      2

      exit:
        stop_loss:   8.00
        dte_exit:    3
        vega_exit:   0.10     # exit when vega decays below 0.10
        conditions:
          # MTM gain exit: when spread has gained 2× the initial debit
          - "position_mark - open_mark >= 2.0 * abs(open_mark)"
          # Spike exit: large move on the VES1D indicator
          - "ves1d_close > 60"

      roll:
        dte:  10              # roll when ≤ 10 DTE
        window:
          start: "09:45"
          end:   "14:30"
```

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

## After-Hours Exits (`allow_after_hours_exits`)

By default, all price-triggered exits — stop-loss, take-profit, and their gap variants
— are restricted to bars that fall within the session window (`session.start_time` to
`session.end_time`). After-hours option prices are often stale or illiquid, and allowing
them to fire stops would produce unrealistic fills.

Setting `allow_after_hours_exits: true` removes the session-time constraint so that
TP and SL can trigger on any bar, including pre-market and post-market bars.

```yaml
exit:
  stop_loss:   2.00
  take_profit: 0.50
  allow_after_hours_exits: true
```

**Default:** `false` — all price-triggered exits are restricted to session hours.

### When to use it

Use this when your bar data includes liquid after-hours quotes and you want the
backtest to reflect the real behaviour of a GTC stop order that would be monitored
around the clock. It is most relevant for instruments with significant extended-hours
volume (e.g. ES futures options, which trade nearly 24 hours on weekdays).

### What is and is not affected

| Exit type | Affected by this flag |
|---|---|
| `stop_loss` (bar-close trigger) | Yes |
| `take_profit` (bar-close trigger) | Yes |
| `gap_sl` (bar-open trigger) | Yes |
| `gap_tp` (bar-open trigger) | Yes |
| `dte_exit` | No — never session-gated |
| `expiry_exit` | No — never session-gated |
| `conditions` | No — condition exits are not session-gated |

The `weekdays_only` constraint is always respected regardless of this flag — exits
will never trigger on Saturday or Sunday bars.

### Interaction with gap stop-loss

The gap SL (`gap_sl`) fires when a bar's **open** price is already through the stop
level — i.e. the market gapped past the threshold between the close of the previous
bar and the open of the next. This is the primary overnight risk vector for spread positions.

**With `allow_after_hours_exits: false` (default):** The gap SL cannot fire on
after-hours bars even if the market has already moved through the stop level overnight.
The exit will fire on the first in-session bar the following morning, using that bar's
open price as the fill. You are not at risk of missing the stop entirely — but the
recorded fill is the session open, not the true overnight gap open. This tends to be
more favourable than what a live GTC order would have achieved.

**With `allow_after_hours_exits: true`:** The gap SL fires at whichever bar first
prints an open through the stop, including overnight bars. The fill price is that
bar's open. This more closely models what a live GTC stop order would produce and
will generally result in worse (more realistic) SL fill prices for large overnight gaps.

**Summary:** If your bar data has reliable extended-hours option prices, enabling this
flag will tend to reduce reported P&L on SL events by capturing overnight gap moves at
their true gap-open price rather than deferring to the next session open.

---

## Leg-Out Independent Exit (`leg_out`)

By default, when a spread's TP or SL fires, all legs are closed simultaneously using
the position mark at the trigger bar. For credit spreads, the long leg's closing price
is typically the forward-filled value from the last bar it actually traded — which may
be hours stale by the time the short leg triggers.

Setting `leg_out: true` replaces that stale forward-filled long-leg component with the
**first real bar strictly after the spread's exit time** (market-order semantics). The
short leg still exits at the trigger bar; the long leg's price is patched to the next
available traded bar.

```yaml
exit:
  stop_loss:   2.00
  take_profit: 0.50
  leg_out:     true
```

**Default:** `false` — all legs exit at the same bar using forward-filled prices.

### When to use it

Use `leg_out` when your long leg is illiquid enough that its last-traded bar is
frequently stale at the time the short leg triggers. For ES put spreads at DTE ≤ 7,
the long leg (further OTM) often has gaps of 30–120 minutes between real bars; the
forward-filled price can materially misstate the true exit cost.

### How it works

For each TP or SL exit, `_adjust_leg_out_exits` looks up the first real bar for the
long leg **after** `exit_time` and adjusts `exit_mark` by:

```
exit_mark' = exit_mark
             − stale_close × signed_qty   [remove forward-fill component]
             + fill_close  × signed_qty   [add market-order fill]
```

where `signed_qty = −1 × quantity` for a BTO leg. If no real bar exists after
`exit_time` for a given leg (the option has no remaining liquidity), the fill is `0` —
modelling an option that cannot be sold.

Expiry exits are unaffected: both legs' terminal values are computed from intrinsic
value or the last bar close, which is already correct.

### Re-entry gating

`leg_out` does **not** extend the one-at-a-time gate. The gate opens at the spread's
`exit_time` — the bar where TP or SL fired — and the engine may enter a new spread
on the next valid entry bar. The long leg fill happens in the same pass as a price
adjustment; the engine does not treat the leg as an ongoing open position.

This means it is possible (and valid) for a new spread to be entered while the
previous long leg's fill bar has not yet printed. The two events are logically
independent: the old spread is closed, the long leg's exit price is simply unknown
until the next bar clears.

### Constraints

- Mutually exclusive with `on_sl_long_continuation` — raises a validation error if
  both are set. (`on_sl_long_continuation` replaces `leg_out` for the SL case and adds
  trailing-stop logic on top.)
- Volume filtering (via `exit.liquidity`) uses only STO (short) leg volume when
  `leg_out` is enabled, since long-leg volume gaps should not suppress the spread's
  TP/SL trigger.

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

---

## Long-Leg Continuation After SL (`on_sl_long_continuation`)

When a spread's short leg hits its stop-loss, the default behaviour closes the long
leg immediately at market. `on_sl_long_continuation` instead holds the long leg open
under a trailing stop, allowing it to capture profit if the market continues to trend
in the adverse direction.

```yaml
exit:
  stop_loss:   2.00
  take_profit: 0.50
  on_sl_long_continuation: true
  long_trailing_stop_pct:  0.40   # exit when price pulls back 40% from post-SL peak
```

**Default:** `false`. When disabled, all legs exit together at the spread's SL trigger
(or at the first real bar after SL, when `leg_out: true` is also set).

### What happens on an SL exit

1. The spread position closes normally at the SL trigger bar. The spread P&L is
   recorded in the `position` table as usual.
2. The long leg is picked up at the **first real bar strictly after the SL exit time**
   (market-order semantics — same as `leg_out`). This price is `continuation_entry_price`.
3. From that bar forward, a trailing stop is monitored using the long leg's bar closes:
   - `peak` = running maximum close price seen since continuation started
   - Trailing stop fires when `close <= peak × (1 − long_trailing_stop_pct)`
4. If the trailing stop never fires before expiry, the long leg settles at intrinsic
   value (`expiry_continuation` exit).
5. Results are written to the `position_continuation` table, linked to the spread
   position by `position_id`.

### Re-entry gating

A new spread is not opened until **both** conditions are met:

- The continuation has exited (trailing stop or expiry)
- The next trading session has started (calendar day after the SL exit, respecting
  `weekdays_only` and `skip_dates`)

The effective gate time for an SL exit with an active continuation is therefore:

```
gate = max(continuation_exit_time, next_trading_day_open_after_sl)
```

This is computed in `_build_gate_overrides` (engine.py) and replaces the raw
`exit_time` in the one-at-a-time enforcer for affected entry IDs.

**Contrast with `leg_out`:** `leg_out` does not extend the gate at all — the next
spread may enter as soon as the short leg exits. `on_sl_long_continuation` explicitly
holds the gate open because the long leg is being actively managed under a trailing
stop, not just price-adjusted at the next available bar.

**SL exits without a continuation** (e.g. the option had no bars in the post-SL
window, or the trade has no long leg) fall back to the raw SL `exit_time` as the gate
— same as any other exit type.

**TP and non-SL exits** are never subject to the extended gate, even when
`on_sl_long_continuation` is enabled. They use the normal `exit_time` gate.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `on_sl_long_continuation` | bool | `false` | Enable long-leg continuation after SL |
| `long_trailing_stop_pct` | float (0–1) | `0.50` | Exit when close pulls back this fraction from post-SL peak |

### Constraints

- Requires `stop_loss` to be configured (raises a validation error otherwise)
- Mutually exclusive with `leg_out` (raises a validation error if both are set)
- Only supported for trades with **one** long leg. Trades with multiple BTO legs
  emit a warning and skip continuation tracking
- `long_trailing_stop_pct` must be strictly between 0 and 1

### Output

Continuation results appear in the `position_continuation` table:

| Column | Description |
|---|---|
| `position_id` | FK to the parent spread position |
| `continuation_entry_price` | Long leg price at the start of continuation (first real bar after SL) |
| `continuation_exit_time` | Timestamp when continuation exited |
| `continuation_exit_price` | Long leg price at exit |
| `continuation_exit_reason` | `trailing_stop` or `expiry_continuation` |
| `continuation_pnl` | Dollar P&L of the continuation leg: `(exit_price − entry_price) × qty × multiplier` |

In the dashboard, the Trade detail page shows a "Long Leg Continuation" breakdown card
with entry/exit prices, exit reason, continuation P&L, and combined (spread + continuation)
P&L. The positions grid on the Backtest page adds a "Cont. P&L" column.

### Interaction with `allow_after_hours_exits`

`on_sl_long_continuation` and `allow_after_hours_exits` are independent. The after-hours
flag controls when the **spread's** SL can fire. Once the spread exits (at whatever bar
that is), continuation monitoring begins immediately on the next available bar of the
long leg, with no session-time constraint — the trailing stop is checked on every bar
including after-hours bars.

### Example

```yaml
exit:
  stop_loss:   2.00
  take_profit: 0.50
  dte_exit:    7
  on_sl_long_continuation: true
  long_trailing_stop_pct:  0.50   # trail 50% from peak
```

With `long_trailing_stop_pct: 0.50`: if the long put is at $0.80 when the spread SL
fires, then rises to $3.20 (peak), the trailing stop fires at $3.20 × 0.50 = $1.60.
If it never reaches a new peak before expiry, the position expires at intrinsic value.
