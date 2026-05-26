# btkit 2.0 — Studies

A **study** is a named collection of one or more strategies that expands into a set of
parameter combinations, runs each combination as an independent backtest, and writes all
results to a single output database. Studies are the primary mechanism for
design-of-experiments work in btkit 2.0.

---

## Concepts

| Term | Meaning |
|---|---|
| **Study** | A named experiment defined in a YAML file. Contains one or more strategy references. |
| **Strategy** | A single strategy YAML file (see [strategy.md](strategy.md)). May contain scalar or sweep parameters. |
| **Combination** | One fully-scalar instance of a strategy after sweep expansion. Each combination runs as an independent backtest. |
| **Sweep** | A parameter expressed as a list or range rather than a scalar. The cartesian product of all sweeps in a strategy forms that strategy's combinations. |

A scalar strategy (no sweep parameters) contributes exactly one combination to the study.
A sweep strategy with 3 delta values × 2 stop-loss values contributes 6 combinations.
Multiple strategies in a single study have their combinations concatenated: the study
assigns each combination a globally unique `combination_id` starting from 1.

---

## Study YAML Format

A study file has a single top-level `study` key. Strategy paths are relative to the study
YAML file's directory.

```yaml
study:
  name: es_put_sweep_q1_2026
  workers: 4                     # parallel worker processes (default: CPU count)
  max_combinations: 200          # error before running if expansion exceeds this

  strategies:
    - path: strategies/short_put.yaml
    - path: strategies/short_put_spread.yaml
```

### Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Human-readable study name; written to the `study` table. |
| `strategies` | list | yes | One or more strategy file references (at least one required). |
| `strategies[n].path` | string | yes | Path to a strategy YAML, relative to the study file. |
| `workers` | integer | no | Number of parallel worker processes. Defaults to `os.cpu_count()`. |
| `max_combinations` | integer | no | Safety cap on total expansion. Raises before any run begins if exceeded. |

---

## Sweep Expansion

Each strategy in the study expands independently using its own sweep parameters. See
[strategy.md — Sweep Parameters](strategy.md#sweep-parameters) for the full list of
sweepable fields and syntax.

### List sweep

```yaml
# strategy YAML
legs:
  - name: short_put
    delta: [-0.20, -0.25, -0.30]   # 3 values
exit:
  stop_loss: [1.5, 2.0]             # 2 values
# → 3 × 2 = 6 combinations
```

### Range sweep

```yaml
legs:
  - name: short_put
    dte:
      start: 14
      stop:  28
      step:  7                       # → [14, 21, 28] → 3 combinations
```

### Explicit table

When full-factorial expansion is not desired, specify only the design points you want.
Column headers use dot-path notation (`<trade_name>.<leg_name>.<field>` or
`<trade_name>.exit.<field>`):

```yaml
combinations:
  mode: table
  columns: ["put_spread.short_put.delta", "put_spread.exit.stop_loss"]
  rows:
    - [-0.20, 1.50]
    - [-0.25, 2.00]
    - [-0.30, 2.50]
```

### Multi-strategy example

A study referencing two strategies — one scalar, one sweep — produces three total
combinations and assigns sequential IDs across both:

```yaml
study:
  name: comparison
  strategies:
    - path: iron_condor.yaml       # scalar → combination_id 1
    - path: short_put.yaml         # 2-delta sweep → combination_ids 2, 3
```

---

## Running a Study

```bash
btkit study run \
  --study   studies/es_put_sweep.yaml \
  --input-db  data/input.db \
  --output-db results/study_output.db
```

### Full CLI reference

```
btkit study run [OPTIONS]

  --study PATH            Study YAML file  [required]
  --input-db PATH         Input database path  [required]
  --output-db PATH        Output database path (created if absent)  [required]
  --workers INTEGER       Override workers setting from the study YAML
  --max-combinations INT  Override max_combinations from the study YAML
  --initial-equity FLOAT  Starting account equity  [default: 100000.0]
```

The `--workers` and `--max-combinations` flags take precedence over values in the study
YAML, allowing ad-hoc overrides without editing the file.

---

## Execution Model

The runner uses `ProcessPoolExecutor` to dispatch combinations across worker processes.
Each worker:

1. Receives a single combination (a fully-scalar `StrategyDefinition`) and its
   `combination_id`.
2. Opens the shared input database (read-only).
3. Creates an isolated temporary output database in a `btkit_study_*` temp directory.
4. Runs `BacktestEngine` for that combination and writes results to the temp DB.
5. Exits — the temp DB is a self-contained file ready for merging.

After all workers finish, `OutputMerger` attaches each temp database to the output
database via DuckDB `ATTACH` and copies rows into the main tables, resequencing primary
keys to avoid collisions. Foreign key relationships (`position.backtest_id`,
`position_leg.position_id`) are preserved through the resequencing. The temp directory
is then deleted.

The `study` row in the output database is written before workers start and its
`finished_at` timestamp is set after the merge completes.

Progress is displayed per combination:

```
[study] es_put_sweep — 6 combinations, 4 workers
  Combination 3/6 [combination_id=3] ███████████░░░░░░░░░ 50%  0:00:12
```

### Worker isolation

Each combination runs in its own OS process with its own DuckDB connection, eliminating
GIL contention and connection sharing issues. The shared input database is opened
read-only by each worker; the temporary output databases are never shared. This design
means worker failures are isolated — a failed combination records its error and the
merger skips it rather than corrupting the shared output.

---

## Analyzing Study Results

### `study_summary()`

Returns a Polars DataFrame with one row per combination, containing standard performance
metrics alongside the parameter values that defined that combination.

```python
from btkit.analysis.metrics import PostProcessor

pp = PostProcessor(output_db="results/study_output.db", study_id=1)
summary = pp.study_summary()
print(summary)
```

Columns include:

| Column | Description |
|---|---|
| `combination_id` | Combination index within the study (1-indexed) |
| `strategy_name` | Strategy name for this combination |
| `total_trades` | Number of completed positions |
| `win_rate` | Fraction of positions with positive net P&L |
| `total_pnl` | Sum of net P&L across all positions |
| `avg_pnl` | Mean net P&L per position |
| `sharpe` | Annualized Sharpe ratio |
| `max_drawdown` | Maximum peak-to-trough equity drawdown |
| `status` | `completed` or `failed` |

### Loading a specific combination

To load positions from a single combination for deeper analysis:

```python
pp = PostProcessor(output_db="results/study_output.db", study_id=1)
positions = pp.positions(combination_id=3)
```

### Using `--study-id` with `btkit analyze`

```bash
btkit analyze \
  --output-db results/study_output.db \
  --study-id 1
```

This opens the dashboard with the full study context: a combination selector, per-
combination equity curves and metrics, and a study-level summary table for ranking
parameter configurations.

---

## Pydantic Models

```python
from pydantic import BaseModel, model_validator


class StrategyRef(BaseModel):
    path: str                   # relative to study YAML directory


class StudyDefinition(BaseModel):
    name: str
    strategies: list[StrategyRef]
    max_combinations: int | None = None
    workers: int | None = None

    @model_validator(mode="after")
    def at_least_one_strategy(self) -> StudyDefinition:
        if not self.strategies:
            raise ValueError("study must list at least one strategy")
        return self
```

---

## Loading a Study

```python
from btkit.study.loader import load_study

definition, study_dir = load_study("studies/es_put_sweep.yaml")
# definition: StudyDefinition
# study_dir:  Path to the directory containing the study YAML
#             (used to resolve relative strategy paths)
```

`load_study()` raises `ValueError` if the file does not contain a top-level `study` key
or if the model fails validation.

---

## `StudyExpander`

`StudyExpander` takes a `StudyDefinition` and its directory, resolves each strategy
path, loads the corresponding `StrategyDefinition`, and expands sweeps into a flat list
of `(combination_id, StrategyDefinition)` tuples where every `StrategyDefinition` is
fully scalar.

```python
from btkit.study.expander import StudyExpander

expander = StudyExpander(definition, study_dir)
combinations = expander.combinations   # list[tuple[int, StrategyDefinition]]

# Inspect the parameter table
params_df = expander.params_df
# Columns: combination_id, strategy_name, <swept_param_paths...>
# e.g.:    combination_id | strategy_name | put_spread.short_put.delta | put_spread.exit.stop_loss
```

`expander.combinations` raises `ValueError` before returning if the total expansion
exceeds `max_combinations`.

---

## Relationship to Single Runs

Single scalar backtests (`btkit run`) are unaffected by the study feature. The `study_id`
and `combination_id` columns in the `backtest` table are `NULL` for single runs. The
`run` and `pipeline` commands check `strategy.is_parameterized()` at startup and redirect
the user to `btkit study run` if sweep parameters are detected, rather than silently
running only the first combination.

---

## Workflow Example

```bash
# 1. Write a strategy with sweep parameters
cat > strategies/short_put.yaml << 'EOF'
strategy:
  name: short_put
  universe:
    start_date: "2025-01-01"
    end_date:   "2025-12-31"
  trades:
    - name: put
      instrument: {root_symbol: ES, asset_class: future}
      entry:
        window: {start: "09:30", end: "14:00"}
      legs:
        - name: short_put
          right: put
          action: sell_to_open
          delta: [-0.15, -0.20, -0.25, -0.30]
          dte: 21
      exit:
        stop_loss:   [1.5, 2.0, 2.5]
        take_profit: 0.5
        dte_exit:    7
EOF

# 2. Write a study YAML that references it
cat > studies/put_sweep_2025.yaml << 'EOF'
study:
  name: put_sweep_2025
  workers: 8
  max_combinations: 50
  strategies:
    - path: ../strategies/short_put.yaml
EOF

# 3. Run — 4 deltas × 3 stop-losses = 12 combinations
btkit study run \
  --study   studies/put_sweep_2025.yaml \
  --input-db  data/input.db \
  --output-db results/put_sweep_2025.db

# 4. Analyze
btkit analyze \
  --output-db results/put_sweep_2025.db \
  --study-id 1
```
