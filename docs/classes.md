# btkit 2.0 — Class Design

## Package Structure

```
btkit/
├── cli.py                      # Typer CLI — entry point for all commands
├── db/
│   ├── input_db.py             # InputDatabase  — read access to input DB
│   ├── output_db.py            # OutputDatabase — write access to output DB
│   └── merger.py               # OutputMerger   — consolidates parallel worker DBs
├── pipeline/
│   ├── builder.py              # DatabaseBuilder — orchestrates DB prep
│   ├── greeks.py               # GreeksCalculator — numba Black-76 batch
│   └── indicators.py           # IndicatorRunner — loads + executes user scripts
├── backtest/
│   ├── engine.py               # BacktestEngine — single-run orchestrator
│   ├── entry.py                # EntryScanner — Pass 1: find entry signals
│   ├── exit.py                 # ExitScanner — Pass 2: find first exit per entry
│   ├── pnl.py                  # PnLCalculator — Pass 3: compute net PnL
│   └── matrix_runner.py        # MatrixRunner — parallel multi-combination orchestrator
├── strategy/
│   ├── definition.py           # Pydantic models (includes SweepRange, SweepParam types)
│   ├── loader.py               # YAML loader + validator
│   └── matrix.py               # StrategyMatrix — expands parameterized definitions
└── analysis/
    ├── metrics.py              # PostProcessor — standard backtest metrics + MAE + heatmaps
    └── charts.py               # Chart functions (Plotly + lightweight-charts)
```

---

## Database Layer

### `InputDatabase`

Read-only access to the input database. All methods return Polars DataFrames.
Keeps a single DuckDB connection open for the lifetime of the object.

```python
class InputDatabase:
    def __init__(self, db_path: str): ...

    def underlying_bars(
        self,
        instrument_id: int,
        start: datetime,
        end: datetime
    ) -> pl.DataFrame: ...

    def option_bars(
        self,
        instrument_id: int,
        start: datetime,
        end: datetime
    ) -> pl.DataFrame: ...

    def greeks_at_entry(
        self,
        underlying_id: int,
        ts_event: datetime,
        right: str,
        target_delta: float,
        target_dte: int,
        delta_tolerance: float = 0.05,
        dte_tolerance: int = 5
    ) -> pl.DataFrame:
        """
        Returns candidate options near the desired delta and DTE.
        Caller selects the best match.
        """
        ...

    def option_bars_for_legs(
        self,
        instrument_ids: list[int],
        start: datetime,
        end: datetime
    ) -> pl.DataFrame:
        """
        Batch-loads bars for a set of instrument IDs. Used by ExitScanner
        to load all legs of all open positions in one query.
        """
        ...

    def indicators(
        self,
        underlying_id: int,
        start: datetime,
        end: datetime
    ) -> pl.DataFrame:
        """
        Returns a wide DataFrame with one column per indicator name, indexed
        by ts_event. Internally executes a DuckDB PIVOT over indicator_bars
        joined with indicator_definition. The tall-format storage is
        transparent to callers — they always receive a wide DataFrame ready
        for Polars filter expressions.
        """
        ...

    def close(self) -> None: ...
```

---

### `OutputDatabase`

Write access to the output database.

```python
class OutputDatabase:
    def __init__(self, db_path: str): ...

    def create_schema(self) -> None:
        """Creates tables if they do not exist."""
        ...

    def write_backtest(self, metadata: dict) -> int:
        """Inserts a backtest record and returns the generated id."""
        ...

    def write_results(
        self,
        backtest_id: int,
        positions: pl.DataFrame,
        legs: pl.DataFrame
    ) -> None: ...

    def close(self) -> None: ...
```

---

### `OutputMerger`

Consolidates multiple worker-specific output databases produced by `MatrixRunner`
into a single output database file.

```python
class OutputMerger:
    def merge(
        self,
        worker_db_paths: list[str],
        output_db_path: str,
        cleanup: bool = True
    ) -> None:
        """
        Reads each worker DB in sequence. Re-sequences backtest.id,
        position.id, and position_leg.id to be globally unique across
        all workers. Preserves combination_id and matrix_id.
        Writes all rows to output_db_path.
        Deletes worker DB files after merging if cleanup=True.
        """
        ...
```

**Design note:** Each parallel worker writes to its own isolated DB file to avoid
concurrent write contention — DuckDB does not support simultaneous multi-process
writes cleanly. The merge step is fast relative to backtest runtime and requires
only sequential reads from each worker file and a single write pass.

---

## Pipeline Layer

### `DatabaseBuilder`

Orchestrates the full input database build from raw databento files.

```python
class DatabaseBuilder:
    def __init__(
        self,
        raw_data_path: str,
        db_path: str,
        indicator_scripts: list[str] = None
    ): ...

    def build(self) -> None:
        """Full build: ingest → greeks → indicators."""
        ...

    def _ingest_definitions(self) -> None:
        """
        Reads .dbn definition files. Builds an internal instrument map
        used when joining metadata into option_bars at ingest time.
        """
        ...

    def _ingest_ohlcv(self) -> None:
        """
        Reads .dbn OHLCV files. Splits into underlying_bars and option_bars,
        pre-joining definition metadata into option_bars at write time so
        no joins are required at backtest runtime.
        """
        ...

    def _compute_greeks(self) -> None:
        """Instantiates GreeksCalculator and processes option_bars in batches."""
        ...

    def _run_indicators(self) -> None:
        """
        Instantiates one IndicatorRunner per script path and runs each
        against the underlying_bars for the configured underlying.
        """
        ...
```

---

### `GreeksCalculator`

Wraps the numba Black-76 implementation. Reads from `option_bars` (joined with
`underlying_bars` for the underlying close price) and writes to `option_greeks`
in configurable batch sizes.

```python
class GreeksCalculator:
    def __init__(
        self,
        db: InputDatabase,
        risk_free_rate: float = 0.01,
        batch_size: int = 50_000
    ): ...

    def run(self) -> None: ...

    def _compute_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Receives a batch DataFrame joining option_bars with underlying_bars
        (for the underlying close). Extracts numpy arrays, calls numba
        implied_vol and greek functions, and returns a DataFrame matching
        the option_greeks schema:
            ts_event, instrument_id, underlying_id, dte, T,
            iv, delta, gamma, theta, vega
        Note: expiration, right, strike_price, underlying_close, and
        option_close are used as inputs to the computation but are not
        written to option_greeks — they already exist in option_bars.
        """
        ...
```

---

### `IndicatorRunner`

Loads a user-supplied indicator script, calls its `compute()` function with the
underlying bars DataFrame, and writes the results to `indicator_definition` and
`indicator_bars`.

```python
class IndicatorRunner:
    def __init__(self, db: InputDatabase, script_path: str):
        """
        Reads and stores the script source at init time. The source is later
        written to indicator_definition for reproducibility. The script is
        executed in the current Python environment — any imported external
        modules must be installed by the user.
        """
        ...

    def run(self, underlying_id: int) -> None:
        """
        1. Load underlying_bars for underlying_id into a Polars DataFrame.
        2. Call the user's compute(df) → wide DataFrame with indicator columns.
        3. For each indicator column in the result:
               a. Insert or update a row in indicator_definition
                  (name, underlying_id, underlying_symbol, script_source).
               b. Melt that column to tall format (ts_event, indicator_id, value).
               c. Write tall rows to indicator_bars.
        Each output column from compute() becomes its own independent series,
        allowing a single script to produce multiple indicators.
        """
        ...
```

**User indicator script interface:**

```python
# user_indicators/my_indicators.py
import polars as pl

def compute(df: pl.DataFrame) -> pl.DataFrame:
    """
    Receives a DataFrame of underlying_bars columns:
        ts_event, instrument_id, symbol, open, high, low, close, volume

    Must return the same DataFrame with one or more additional indicator
    columns appended. Each appended column becomes its own independent
    series in indicator_bars, keyed by its column name. Column names must
    be unique across all indicator scripts for the same underlying.

    External imports are permitted. The user is responsible for ensuring
    any imported modules are installed in the active environment.
    """
    return df.with_columns([
        pl.col("close").rolling_mean(20).alias("sma_20"),
        pl.col("close").rolling_mean(50).alias("sma_50"),
        # A script may return as many indicator columns as needed.
    ])
```

---

## Strategy Layer

### `StrategyMatrix`

Expands a parameterized `StrategyDefinition` into a list of fully-scalar instances,
one per combination. Handles both sweep-parameter (full-factorial) and explicit
combination modes.

```python
class StrategyMatrix:
    def __init__(self, definition: StrategyDefinition):
        """
        Validates the definition and pre-computes the expansion.
        Raises if the number of combinations exceeds
        definition.matrix.max_combinations.
        """
        ...

    @property
    def combinations(self) -> list[StrategyDefinition]:
        """
        Returns a list of fully-scalar StrategyDefinition instances.
        Each instance has combination_id set (1-indexed).
        Sweep params and SweepRange objects have been resolved to plain
        floats/ints. Explicit combinations have been merged with base
        definition defaults.
        """
        ...

    @property
    def params_df(self) -> pl.DataFrame:
        """
        Summary DataFrame of the parameter space. One row per combination,
        with columns for each swept or explicitly varied parameter plus
        combination_id. Used by PostProcessor for heatmap generation and
        matrix-level analysis.
        """
        ...

    def _expand_sweeps(self) -> list[dict]:
        """
        Resolves all SweepParam fields to lists of values, then computes
        the cartesian product across all list-valued fields. Each element
        of the result is a flat dict of scalar parameter overrides.
        """
        ...

    def _expand_combinations(self) -> list[dict]:
        """
        Resolves explicit combinations (structured or table mode) into
        the same flat dict format as _expand_sweeps, applying overrides
        against the base definition's scalar defaults.
        """
        ...
```

---

## Backtest Layer

### `BacktestEngine`

Top-level orchestrator for a single backtest run. Wires together the three passes
and writes results to the output database.

```python
class BacktestEngine:
    def __init__(
        self,
        input_db: InputDatabase,
        output_db: OutputDatabase,
        strategy: StrategyDefinition,
        initial_equity: float = 100_000.0
    ): ...

    def run(self) -> int:
        """
        Executes the three-pass vectorized backtest.
        Returns the backtest_id written to the output database.
        strategy must be a fully-scalar StrategyDefinition (all SweepParam
        fields resolved to plain values). MatrixRunner handles expansion
        before dispatching to BacktestEngine.
        """
        backtest_id = self._write_backtest_record()
        entries     = EntryScanner(self.input_db, self.strategy).scan()
        exits       = ExitScanner(self.input_db, self.strategy).scan(entries)
        positions   = PnLCalculator(self.strategy).compute(entries, exits)
        self.output_db.write_results(backtest_id, positions.positions, positions.legs)
        return backtest_id
```

---

### `MatrixRunner`

Orchestrates parallel execution of a `StrategyMatrix`. Each combination runs in an
isolated worker process with its own output database file. Results are merged into
a single output database on completion.

```python
class MatrixRunner:
    def __init__(
        self,
        matrix: StrategyMatrix,
        input_db_path: str,
        output_db_path: str,
        max_workers: int = None     # defaults to os.cpu_count()
    ): ...

    def run(self) -> None:
        """
        1. Assigns each combination a worker-specific output DB path
           (e.g. {tmp_dir}/worker_{combination_id}.db).
        2. Submits (strategy, input_db_path, worker_output_path) tuples
           to ProcessPoolExecutor with max_workers.
        3. Tracks progress with tqdm across futures as they complete.
        4. On full completion, calls OutputMerger to consolidate all
           worker DBs into output_db_path.
        """
        ...
```

**Design note:** `ProcessPoolExecutor` is used rather than `ThreadPoolExecutor`.
The legacy project was limited to threads because its Python tick loop held the GIL.
Since btkit 2.0's hot path is Polars + numba + DuckDB — all of which release the GIL
— true process parallelism is achievable. Each worker process receives only a scalar
`StrategyDefinition` (a Pydantic model with primitive values, fully picklable) and
two path strings. DuckDB connections are created inside each worker process and are
never shared across process boundaries.

---

### `EntryScanner`

Pass 1. Scans underlying bars and indicators to identify all valid entry timestamps
within the configured entry window, then selects the specific option legs for each
entry via a greeks lookup.

```python
class EntryScanner:
    def __init__(self, db: InputDatabase, strategy: StrategyDefinition): ...

    def scan(self) -> pl.DataFrame:
        """
        Returns one row per valid entry with columns:
            entry_time, tp_price, sl_price, dte_exit_time,
            + one group of columns per leg:
                leg_{n}_instrument_id, leg_{n}_open_price, leg_{n}_multiplier,
                leg_{n}_strike, leg_{n}_expiration, leg_{n}_right, leg_{n}_action
        """
        ...

    def _apply_window_filters(self, bars: pl.DataFrame) -> pl.DataFrame:
        """
        Fast first-pass filter on time alone — no DB access, no condition
        evaluation. Keeps only bars that fall within:
          - entry.window (start/end time of day)
          - universe.session (weekdays_only, skip_dates)
        Eliminates the majority of timestamps cheaply before leg selection.
        """
        ...

    def _select_legs(self, entry_times: pl.DataFrame) -> pl.DataFrame:
        """
        For each candidate timestamp, issues a single batched DuckDB query
        against option_greeks to find the best-matching option for each leg
        definition (by delta + DTE). Timestamps where any leg cannot be
        matched within tolerance are dropped.
        """
        ...

    def _compute_open_mark(self, entries: pl.DataFrame) -> pl.DataFrame:
        """
        Computes open_mark (sum of leg open prices weighted by signed quantity)
        and derives tp_price and sl_price from the strategy's take_profit and
        stop_loss parameters applied to open_mark.
        """
        ...

    def _evaluate_conditions(self, entries: pl.DataFrame) -> pl.DataFrame:
        """
        Condition evaluation phase — runs after leg selection and open mark
        computation so that all column namespaces are available:
            - Underlying bar values: close, open, high, low, volume
            - Indicator columns: any column in indicator_bars (via PIVOT)
            - Leg properties: short_put.strike, short_put.delta, etc.
            - Position mark: open_mark

        Applies in sequence:
          1. All entry.conditions (AND logic; if/then parsed to implication) — vectorized
          2. min_credit / max_debit filters — vectorized
          3. max_open_positions (stateful count) — vectorized guard
          4. minimum_equity filter (sequential): sweeps candidates in chronological
             order, tracking current_equity = initial_equity + cumulative net_pnl of
             all positions closed before each candidate entry time. Drops candidates
             where current_equity < minimum_equity. This step runs after steps 1–3
             because it is inherently sequential and cannot be expressed as a Polars
             filter over the candidate DataFrame alone.

        Steps 1–3 are compiled Polars expressions applied in a single vectorized pass.
        Step 4 runs only when entry.minimum_equity is set.
        """
        ...
```

---

### `ExitScanner`

Pass 2. For each entry, scans forward through 1-minute bars to find the first bar
where a TP, SL, DTE, or expiry exit condition is met. Also tracks `worst_mark`
across the scan. Implements the fill price rules from `fill_price_and_costs.md`.

```python
class ExitScanner:
    def __init__(self, db: InputDatabase, strategy: StrategyDefinition): ...

    def scan(self, entries: pl.DataFrame) -> pl.DataFrame:
        """
        Returns one row per entry with columns:
            entry_id, exit_time, exit_mark, worst_mark, exit_reason

        exit_reason: 'take_profit' | 'stop_loss' | 'condition' | 'dte_exit' | 'expiry'
        """
        ...

    def _load_exit_data(self, entries: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Batch-loads all data needed to monitor open positions. Returns two DataFrames:
          - option_bars: all leg bars from entry_time to latest possible exit,
            covering all leg instrument IDs across all entries in one DB query.
          - indicators: wide indicator DataFrame for the underlying over the same
            time window, used to evaluate exit.conditions on each bar.
        This is the only DB read in Pass 2.
        """
        ...

    def _compute_position_marks(self, bars: pl.DataFrame) -> pl.DataFrame:
        """
        Computes position_mark per bar for each entry as:
            sum(leg_close * signed_quantity for each leg)
        where signed_quantity is positive for BTO legs and negative for STO legs.
        Returns a long DataFrame of (entry_id, ts_event, position_mark).
        See fill_price_and_costs.md for the rationale for using bar closes
        rather than individual leg highs/lows for multi-leg positions.
        """
        ...

    def _find_first_hit(
        self,
        position_marks: pl.DataFrame,
        indicators: pl.DataFrame,
        entries: pl.DataFrame
    ) -> pl.DataFrame:
        """
        For each entry, scans position_marks forward from entry_time to find
        the first bar satisfying any exit condition (in priority order):
            1. Gap open past SL:   position_mark at bar open >= sl_price
            2. Gap open past TP:   position_mark at bar open <= tp_price
            3. Stop loss:          position_mark >= sl_price
            4. Take profit:        position_mark <= tp_price
            5. Indicator condition: any expression in exit.conditions is true
                                    at this bar (OR logic; uses indicator values
                                    joined on ts_event)
            6. DTE exit:           dte <= strategy.exit.dte_exit
            7. Expiry:             ts_event >= expiration

        Also computes worst_mark as the most adverse position_mark observed
        across all bars from entry_time to exit_time. This is a free
        aggregation — the scan is already iterating over all bars per entry.

        Indicator exit fills at the bar close mark (not a price threshold),
        since no specific price level triggered the exit.

        Handles gap opens and the SL-priority tie-breaking rule per
        fill_price_and_costs.md.
        """
        ...
```

---

### `PnLCalculator`

Pass 3. Pure arithmetic — joins entries and exits, applies slippage and fees to
produce final position and leg records ready for the output database.

```python
@dataclass
class BacktestPositions:
    positions: pl.DataFrame   # matches position table schema
    legs: pl.DataFrame        # matches position_leg table schema


class PnLCalculator:
    def __init__(self, strategy: StrategyDefinition): ...

    def compute(
        self,
        entries: pl.DataFrame,
        exits: pl.DataFrame
    ) -> BacktestPositions:
        """
        1. Join entries + exits on entry_id.
        2. gross_pnl = open_mark - exit_mark
           (sign convention follows leg actions; no spread-specific assumption)
        3. slippage_cost = exit_mark * slippage_pct
        4. fee_cost = fee_per_contract * total_contracts * 2  (open + close)
        5. net_pnl = gross_pnl - slippage_cost - fee_cost
        worst_mark is passed through from exits unchanged.
        """
        ...
```

---

## Analysis Layer

### `PostProcessor`

Loads results from the output database and computes standard backtest metrics,
including MAE derived from `worst_mark`. Supports both single-run and matrix-run
analysis.

```python
class PostProcessor:
    def __init__(
        self,
        output_db: OutputDatabase,
        backtest_id: int | None = None,
        matrix_id: int | None = None
    ):
        """
        Initialise with either backtest_id (single run) or matrix_id (all runs
        from a matrix expansion). Exactly one must be provided.
        """
        ...

    def metrics(self) -> dict:
        """
        Returns a dict of standard metrics:
            net_profit, total_trades, percent_profitable,
            profit_factor, avg_win, avg_loss, median_pnl,
            max_drawdown, max_drawdown_pct, cagr, mar,
            sharpe_ratio, sortino_ratio, calmar_ratio,
            premium_capture_rate,
            avg_mae, median_mae, worst_mae
        MAE is derived from worst_mark and open_mark in the position table.
        Only valid when initialised with backtest_id.
        """
        ...

    def heatmap(
        self,
        metric: str,
        x_param: str,
        y_param: str,
        fixed_params: dict[str, Any] = None
    ) -> go.Figure:
        """
        Matrix-only (requires matrix_id). Plots a heatmap of the given metric
        across two swept parameters. fixed_params filters the matrix to a
        specific slice when more than two parameters were swept.
        x_param and y_param must be column names in StrategyMatrix.params_df.
        """
        ...

    def equity_curve(self) -> pl.DataFrame: ...
    def trade_pnl_series(self) -> pl.DataFrame: ...
    def summarize(self, formatted: bool = False) -> pl.DataFrame: ...
```

---

## CLI

Three subcommands, one for each pipeline stage. The `run` command auto-detects
single vs. matrix mode — no separate subcommand is needed.

```python
# cli.py
import typer
app = typer.Typer()

@app.command()
def build(
    data_path: str,
    db_path: str,
    indicators: list[str] = typer.Option(default=[])
) -> None:
    """Build the input database from raw databento files."""
    ...

@app.command()
def run(
    strategy: str,
    input_db: str,
    output_db: str,
    initial_equity: float = typer.Option(
        default=100_000.0,
        help="Starting account equity. Used for equity curve and minimum_equity filtering."
    ),
    workers: int = typer.Option(
        default=None,
        help="Number of parallel workers for matrix runs. Defaults to cpu_count()."
    ),
    max_combinations: int = typer.Option(
        default=None,
        help="Override matrix.max_combinations from the strategy YAML."
    )
) -> None:
    """
    Run a backtest from a strategy YAML. Auto-detects single vs. matrix run.

    If the strategy contains sweep parameters or an explicit combinations
    block, StrategyMatrix is used to expand it and MatrixRunner executes
    the combinations in parallel. Otherwise BacktestEngine runs directly.
    The --workers flag is ignored for single-combination runs.
    """
    ...

@app.command()
def analyze(
    output_db: str,
    backtest_id: int = typer.Option(default=None),
    matrix_id: int = typer.Option(default=None),
    open_browser: bool = True
) -> None:
    """
    Compute metrics and open the results dashboard.
    Provide either --backtest-id (single run) or --matrix-id (matrix run).
    """
    ...

@app.command()
def pipeline(
    data_path: str,
    strategy: str,
    db_path: str,
    output_db: str,
    indicators: list[str] = typer.Option(default=[]),
    initial_equity: float = typer.Option(default=100_000.0),
    workers: int = typer.Option(default=None),
    max_combinations: int = typer.Option(default=None),
    rebuild: bool = typer.Option(
        default=False,
        help="Force rebuild of the input DB even if it already exists at db_path."
    ),
    open_browser: bool = True
) -> None:
    """
    Full pipeline in one command: build → run → analyze.

    The build step is skipped if the input DB file already exists at db_path,
    unless --rebuild is specified. This allows iterating on strategies without
    re-ingesting and re-computing greeks each time. All other flags are passed
    through to the corresponding sub-commands.
    """
    ...
```
