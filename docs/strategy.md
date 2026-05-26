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
    slippage_pct:     0.01
    fee_per_contract: 0.65

  # Matrix expansion settings (only relevant for parameterized strategies)
  matrix:
    max_combinations: 100               # error before running if expansion exceeds this

  # One or more independent trade definitions
  trades:
    - name: put_spread

      instrument:
        root_symbol: ES
        asset_class: future         # future | equity | etf

      entry:
        window:
          start: "09:30"            # no entries before this time
          end:   "14:00"            # no entries after this time
        conditions:
          - "rsi_14 < 40"
          - "vix_close < 30"
        min_credit:  0.50           # skip entry if open_mark < this (optional)
        max_debit:   2.00           # skip entry if open_mark > this (optional)

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

        - name:     long_put
          right:    put
          action:   buy_to_open
          delta:    -0.15                    # scalar — fixed across all combinations
          dte:      45
          quantity: 1

      exit:
        stop_loss:   [1.50, 2.00, 2.50]     # list sweep
        take_profit: 0.50                    # scalar
        dte_exit:    21
        expiry_exit: true
        conditions:                          # exit if any condition is true (OR logic)
          - "rsi_14 > 70"
          - "vix_close > 40"
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
        - {name: short_put, right: put, action: sell_to_open, delta: -0.20, dte: 45, quantity: 1}
        - {name: long_put,  right: put, action: buy_to_open,  delta: -0.10, dte: 45, quantity: 1}
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
        - {name: long_call,  right: call, action: buy_to_open,  delta: 0.10, dte: 45, quantity: 1}
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


class EntryWindowConfig(BaseModel):
    start: time
    end:   time

    @model_validator(mode="after")
    def start_before_end(self) -> EntryWindowConfig:
        if self.start >= self.end:
            raise ValueError("entry window start must be before end")
        return self


class EntryConfig(BaseModel):
    window:      EntryWindowConfig
    conditions:  list[str] = []
    min_credit:  NumericSweep | None = None  # enter only if open_mark >= this value
    max_debit:   NumericSweep | None = None  # enter only if open_mark <= this value


class LegConfig(BaseModel):
    name:     str
    right:    Literal["call", "put"]
    action:   Literal["buy_to_open", "sell_to_open"]
    delta:    NumericSweep
    dte:      IntSweep
    quantity: int = 1


class ExitConfig(BaseModel):
    stop_loss:   NumericSweep
    take_profit: NumericSweep
    dte_exit:    IntSweep | None = None
    expiry_exit: bool = True
    conditions:  list[str] = []  # exit if any condition is true (OR logic)


class CostsConfig(BaseModel):
    slippage_pct:     float = 0.0
    fee_per_contract: float = 0.0


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
