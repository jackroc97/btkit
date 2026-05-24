# btkit

Vectorized options backtesting framework for futures options (ES, NQ, etc.).

btkit runs Pass 1 (entry selection) and Pass 2 (exit monitoring) as batch Polars
operations over DuckDB, keeping the full backtest in memory with no per-bar Python
loops. A strategy with 2 years of daily entries typically finishes in seconds.

---

## Requirements

- Python 3.11+
- Databento `.dbn` data files (definitions + OHLCV-1m schemas)

## Installation

```bash
git clone https://github.com/yourorg/btkit.git
cd btkit
pip install -e ".[viz,dev]"    # viz = dashboard; dev = pytest + ruff
```

---

## Quick Start

**1. Build the input database from raw data files:**

```bash
btkit build \
  --data-path /path/to/dbn/files/ \
  --db-path   input.db \
  --indicators indicators.py   # optional: user-supplied indicator script
```

**2. Run a backtest:**

```bash
btkit run \
  --db-path     input.db \
  --output-db   results.db \
  --strategy    strategies/short_put.yaml
```

**3. View results:**

```bash
btkit analyze --output-db results.db   # print metrics to terminal

btkit serve \
  --output-db results.db \
  --input-db  input.db    # launch interactive dashboard at http://localhost:8765
```

**Or run the full pipeline in one command:**

```bash
btkit pipeline \
  --data-path  /path/to/dbn/files/ \
  --db-path    input.db \
  --output-db  results.db \
  --strategy   strategies/short_put.yaml \
  --indicators indicators.py
```

---

## Defining a Strategy

Strategies are YAML files. Every strategy has a `universe` (date range + session),
one or more `trades` (each with `instrument`, `entry`, `legs`, and `exit`), and
optional `costs`.

### Minimal example — short put spread

```yaml
strategy:
  name: short_put_spread
  version: "1.0"

  universe:
    start_date: "2025-01-01"
    end_date:   "2025-12-31"
    session:
      timezone:      "America/New_York"
      start_time:    "09:30"
      end_time:      "16:00"
      weekdays_only: true

  costs:
    slippage_pct:     0.01    # 1% of exit mark dollar value
    fee_per_contract: 0.65    # flat per round-trip

  trades:
    - name: put_spread
      instrument:
        root_symbol: ES
        asset_class: future

      entry:
        window:
          start: "10:00"
          end:   "11:00"

      legs:
        - name: short_put
          right:           put
          action:          sell_to_open
          delta:           -0.20         # target delta
          delta_tolerance: 0.05          # ± search band
          dte:             21            # target days to expiry
          dte_tolerance:   3             # ± search band
          quantity:        1

        - name: long_put
          right:           put
          action:          buy_to_open
          strike_offset:   -50.0         # 50 points below short_put strike
          reference_leg:   short_put
          dte:             21            # ignored — inherits short_put expiry
          quantity:        1

      exit:
        stop_loss:       2.00   # exit when spread mark rises 2.00 above open
        take_profit_pct: 0.50   # exit when spread mark falls to 50% of open
        dte_exit:        5      # exit 5 DTE before expiry
        expiry_exit:     true
```

### Leg selection modes

Each leg uses **one** of two selection modes:

| Mode | Fields required | How it works |
|---|---|---|
| Delta-targeted | `delta`, `delta_tolerance`, `dte`, `dte_tolerance` | Finds the option closest to target delta within both tolerance bands |
| Strike offset | `strike_offset`, `reference_leg` | Computes `reference_strike + offset`, selects the nearest available strike at the reference leg's expiration |

### Entry conditions

```yaml
entry:
  window:
    start: "10:00"
    end:   "12:00"
  conditions:
    - "sma_5 > sma_20"      # indicator column reference
    - "close > 5000"         # underlying bar column
  min_credit: 1.50           # skip entry if open mark < this
```

Conditions use AND logic — all must be true to enter.

### Exit conditions

```yaml
exit:
  stop_loss:   2.00
  take_profit: 1.00          # or take_profit_pct: 0.50
  dte_exit:    5             # close when DTE reaches this value
  expiry_exit: true          # close at expiration
  conditions:
    - "sma_5 < 0"            # close when any exit condition is true (OR logic)
```

Exit priority when multiple conditions trigger on the same bar:
`gap-open SL → gap-open TP → SL → TP → indicator condition → DTE → expiry`

### Multi-trade strategy (iron condor)

```yaml
trades:
  - name: put_spread
    instrument: { root_symbol: ES, asset_class: future }
    entry: { window: { start: "10:00", end: "11:00" } }
    legs:
      - { name: short_put, right: put, action: sell_to_open, delta: -0.20, dte: 21 }
      - { name: long_put,  right: put, action: buy_to_open,  strike_offset: -50, reference_leg: short_put, dte: 21 }
    exit: { stop_loss: 2.0, take_profit_pct: 0.5, expiry_exit: true }

  - name: call_spread
    instrument: { root_symbol: ES, asset_class: future }
    entry: { window: { start: "10:00", end: "11:00" } }
    legs:
      - { name: short_call, right: call, action: sell_to_open, delta: 0.20, dte: 21 }
      - { name: long_call,  right: call, action: buy_to_open,  strike_offset: 50, reference_leg: short_call, dte: 21 }
    exit: { stop_loss: 2.0, take_profit_pct: 0.5, expiry_exit: true }
```

Each trade maintains its own one-at-a-time constraint independently — the put spread
and call spread run concurrently as separate position streams.

---

## Writing Indicator Scripts

Pass one or more `--indicators` scripts to `btkit build`. Each script must export
a `compute(df: pl.DataFrame) -> pl.DataFrame` function that receives a wide DataFrame
of underlying OHLCV bars and returns a wide DataFrame with a `ts_event` column plus
one column per indicator.

```python
# indicators.py
import polars as pl

def compute(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([
        "ts_event",
        pl.col("close").rolling_mean(5).alias("sma_5"),
        pl.col("close").rolling_mean(20).alias("sma_20"),
        pl.col("close").rolling_std(14).alias("volatility_14"),
    ])
```

Indicator column names become available in entry and exit `conditions` expressions.

---

## CLI Reference

| Command | Description |
|---|---|
| `btkit build` | Ingest Databento files → DuckDB input database |
| `btkit run` | Run a single backtest from a strategy YAML |
| `btkit analyze` | Print metrics for a completed backtest |
| `btkit pipeline` | build → run → analyze in sequence |
| `btkit serve` | Launch the Dash dashboard |

```
btkit build   --data-path DIR --db-path FILE [--indicators FILE ...]
btkit run     --db-path FILE --output-db FILE --strategy FILE
btkit analyze --output-db FILE [--backtest-id INT]
btkit pipeline --data-path DIR --db-path FILE --output-db FILE --strategy FILE \
               [--indicators FILE ...] [--initial-equity FLOAT] [--rebuild]
btkit serve   --output-db FILE --input-db FILE [--port INT] [--debug]
```

---

## Project Structure

```
btkit/
├── pipeline/        # DatabaseBuilder, GreeksCalculator, IndicatorRunner
├── db/              # InputDatabase, OutputDatabase
├── strategy/        # StrategyDefinition Pydantic models, YAML loader, condition parser
├── backtest/        # BacktestEngine, EntryScanner, ExitScanner, PnLCalculator
├── analysis/        # PostProcessor (metrics), Dashboard (Dash)
└── cli.py           # Typer CLI wiring

tests/
├── fixtures/
│   ├── data/        # Databento .dbn fixture files
│   ├── strategies/  # Test strategy YAMLs
│   └── indicators.py
├── unit/            # Pure-function tests (black76, validators, parser, metrics, pnl)
├── integration/     # DB-backed tests (EntryScanner, BacktestEngine)
├── output/          # Pre-built test input.db (git-ignored)
└── conftest.py

docs/
├── strategy.md      # Strategy YAML reference (full field docs)
├── database.md      # Input/output DB schema reference
└── fill_price_and_costs.md  # Cost model details
```

---

## Running Tests

```bash
# Unit tests only (fast, no DB required)
pytest tests/unit/

# Full suite (requires tests/output/input.db — build it first)
btkit build \
  --data-path tests/fixtures/data/ \
  --db-path   tests/output/input.db \
  --indicators tests/fixtures/indicators.py

pytest tests/
```

---

## Contributing

1. **Fork** the repo and create a feature branch.
2. **Install dev dependencies:** `pip install -e ".[dev]"`
3. **Write tests** — unit tests for pure functions in `tests/unit/`, integration tests
   that use the fixture DB in `tests/integration/`.
4. **Run the linter:** `ruff check . && ruff format .` — the CI gate requires zero errors.
5. **Run the test suite:** `pytest tests/` — all 143 tests must pass.
6. **Open a PR** — describe what changed and why.

### Code style

- Python 3.11+, type hints throughout.
- `ruff` for formatting and linting (config in `pyproject.toml`).
- No per-bar Python loops in backtest code — use Polars expressions.
- No mocking of the database in tests — use DuckDB in-memory or the fixture DB.
- Default to no comments; add one only when the WHY is non-obvious.

---

## Versioning

btkit follows [Semantic Versioning](https://semver.org):

| Change | Version bump |
|---|---|
| Bug fixes, no API or schema changes | PATCH (2.0.x) |
| New features, backward-compatible | MINOR (2.x.0) |
| Strategy YAML schema changes, output DB schema changes, API breaks | MAJOR (x.0.0) |

Any MINOR release may add optional YAML fields with defaults. Removing a field or
making an optional field required is a MAJOR change.
