# btkit 2.0 — Database Design

btkit 2.0 uses two separate DuckDB databases: an **input database** built once from raw data,
and an **output database** written to by each backtest run. They are kept separate so the input
database can be treated as read-only and shared across many backtests.

---

## Input Database

Built during `btkit build`. The guiding principle is **no runtime joins** — all definition
metadata is pre-joined into the price tables at build time so the backtest engine only
performs range scans at backtest time.

---

### `underlying_bars`

1-minute OHLCV for root instruments (futures, ETFs, equities).

```sql
CREATE TABLE underlying_bars (
    ts_event        TIMESTAMPTZ     NOT NULL,
    instrument_id   INTEGER         NOT NULL,
    symbol          VARCHAR         NOT NULL,
    open            DOUBLE          NOT NULL,
    high            DOUBLE          NOT NULL,
    low             DOUBLE          NOT NULL,
    close           DOUBLE          NOT NULL,
    volume          BIGINT
);

CREATE INDEX idx_underlying_bars ON underlying_bars (instrument_id, ts_event);
```

---

### `option_bars`

1-minute OHLCV for options, with definition metadata pre-joined. No reference to a
separate definition table is needed at backtest time.

```sql
CREATE TABLE option_bars (
    ts_event        TIMESTAMPTZ     NOT NULL,
    instrument_id   INTEGER         NOT NULL,
    underlying_id   INTEGER         NOT NULL,
    symbol          VARCHAR         NOT NULL,
    expiration      DATE            NOT NULL,
    strike_price    DOUBLE          NOT NULL,
    right           VARCHAR(1)      NOT NULL,   -- 'C' or 'P'
    multiplier      INTEGER         NOT NULL,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE,
    volume          BIGINT
);

CREATE INDEX idx_option_bars_lookup
    ON option_bars (underlying_id, right, expiration, strike_price, ts_event);
CREATE INDEX idx_option_bars_instrument
    ON option_bars (instrument_id, ts_event);
```

---

### `option_greeks`

Pre-computed per 1-minute bar via numba Black-76. Stored separately from `option_bars`
to allow re-computation without rebuilding price data.

```sql
CREATE TABLE option_greeks (
    ts_event        TIMESTAMPTZ     NOT NULL,
    instrument_id   INTEGER         NOT NULL,
    underlying_id   INTEGER         NOT NULL,
    dte             INTEGER         NOT NULL,
    T               DOUBLE          NOT NULL,   -- time to expiry in years
    iv              DOUBLE,
    delta           DOUBLE,
    gamma           DOUBLE,
    theta           DOUBLE,
    vega            DOUBLE
);

CREATE INDEX idx_option_greeks_lookup
    ON option_greeks (underlying_id, dte, ts_event);
```

**Design notes:**

- `expiration`, `right`, and `strike_price` are intentionally omitted — they already exist
  in `option_bars` and storing them here would be pure duplication.
- `underlying_close` and `option_close` are also omitted — they are the `close` columns of
  `underlying_bars` and `option_bars` respectively.
- `dte` and `T` are kept despite being derivable from `ts_event` and `expiration`. The leg
  selection query at entry time filters on `dte` as its primary range constraint
  (`WHERE dte BETWEEN 40 AND 50`). Keeping `dte` in `option_greeks` makes this a
  single-table range scan. Without it, every leg selection query would require a join
  back to `option_bars` to compute `(expiration::date - ts_event::date)` at query time.
  The storage cost is trivial; the query benefit is real.
- `T` is the direct floating-point input to the Black-76 formula. It is computed once at
  greek computation time and stored to avoid recomputing it during downstream queries.

---

### `indicator_definition`

One row per named indicator series per underlying. Stores metadata about how each
indicator was produced, including the full source code of the script that generated it.

```sql
CREATE TABLE indicator_definition (
    id                  INTEGER PRIMARY KEY,
    name                VARCHAR     NOT NULL,   -- e.g. 'rsi_14', 'sma_20'
    underlying_id       INTEGER     NOT NULL,
    underlying_symbol   VARCHAR     NOT NULL,
    params              JSON,                   -- e.g. {"period": 14}
    script_source       TEXT        NOT NULL    -- full source of the generating script
);

CREATE UNIQUE INDEX idx_indicator_def_unique
    ON indicator_definition (name, underlying_id);
```

**Design notes:**

- The `indicators` table is split into `indicator_definition` + `indicator_bars` (tall
  format) rather than a single wide table with one column per indicator. This avoids the
  need to create dynamic schemas at build time — the two tables have a fixed, known
  structure regardless of which indicators the user computes.
- The tall format also supports incremental builds: a new indicator can be added by
  inserting rows into both tables without modifying existing data or schema.
- `script_source TEXT` stores the full source code of the generating script rather than
  a file path. This ensures reproducibility — if the script is moved, edited, or deleted,
  the DB retains an exact record of what generated the stored values. If the user changes
  the script and wants updated values, they rerun `btkit build`, which regenerates the
  indicator and updates the stored source.
- External module imports within indicator scripts are permitted. Ensuring those modules
  are installed is the user's responsibility. Only the top-level script source is stored;
  imported module source is not captured.
- The `UNIQUE INDEX` on `(name, underlying_id)` prevents two scripts from producing an
  indicator with the same name for the same underlying. This is caught at build time.

---

### `indicator_bars`

Time-series values for each indicator defined in `indicator_definition`.

```sql
CREATE TABLE indicator_bars (
    ts_event        TIMESTAMPTZ     NOT NULL,
    indicator_id    INTEGER         NOT NULL REFERENCES indicator_definition(id),
    value           DOUBLE
);

CREATE INDEX idx_indicator_bars ON indicator_bars (indicator_id, ts_event);
```

**Design notes:**

- Indicators are stored as single scalar `value` per timestamp, not as OHLC tuples.
  OHLC structure is meaningful for price bars because a price genuinely has an open,
  high, low, and close within each period — four physically distinct events. Indicators
  are derived scalar values computed from historical bars; there is no intrabar "high RSI"
  or "low SMA." This is universal practice across quant platforms.
- When an indicator script produces multiple output series (e.g. MACD producing `macd`,
  `macd_signal`, and `macd_histogram`), each series becomes its own row in
  `indicator_definition` and its own series in `indicator_bars`. The `IndicatorRunner`
  handles this automatically by splitting the columns returned from `compute()`.
- At backtest time, indicators are loaded into a wide Polars DataFrame via a DuckDB
  PIVOT before entry scanning begins. This converts the tall storage format into wide
  columns like `rsi_14`, `vix_close`, etc. — the format required for evaluating entry
  conditions.

---

## Incremental Builds (`--append`)

By default, `btkit build` creates a new input database from scratch. The `--append` flag
switches to an incremental mode that adds new data to an **existing** database without
touching rows that are already there. This is the correct workflow whenever you want to
extend the date range covered by a database, or add data for new symbols alongside
existing ones.

---

### When to use it

| Scenario | Command |
|---|---|
| Initial build from scratch | `btkit build --data-path data/ --db-path input.db` |
| Extend date range (new zips, same symbols) | `btkit build --data-path new_data/ --db-path input.db --append` |
| Add new symbols (e.g. add NQ alongside existing ES) | `btkit build --data-path nq_data/ --db-path input.db --append` |
| Mix: new time range + new symbols | `btkit build --data-path mixed_data/ --db-path input.db --append` |

---

### Basic workflow

```bash
# 1. Initial build — May data
btkit build \
  --data-path data/may/ \
  --db-path output/input.db

# 2. Download June data from Databento and drop the zips into data/june/

# 3. Append June to the existing database
btkit build \
  --data-path data/june/ \
  --db-path output/input.db \
  --append

# 4. Run a backtest spanning May–June — no other changes needed
btkit run \
  --strategy strategies/iron_condor_0dte.yaml \
  --input-db output/input.db \
  --output-db output/output.db
```

The strategy YAML `start_date` / `end_date` controls the backtest window. After
appending June data you only need to widen those dates and re-run — the database
already contains everything.

---

### What happens at each layer

#### Definitions

Databento includes a full instrument definition file in every download. When appending,
btkit reads the definition files from the **new** zip batch and builds an in-memory
instrument map for that batch. This map is used solely to enrich the new OHLCV rows at
ingest time — it does not touch anything already in the database.

The key design point is that **definition metadata is pre-joined into every `option_bars`
row at ingest time** (`underlying_id`, `expiration`, `strike_price`, `right`,
`multiplier`). There is no separate definitions table to keep in sync. When you append
new option bar data, those rows carry their `underlying_id` directly from the Databento
definition record, so the link between options and their underlying futures is always
exact and current.

#### OHLCV (`underlying_bars`, `option_bars`)

Each row is deduplicated against the existing database on `(ts_event, instrument_id)`:

```sql
INSERT INTO underlying_bars
SELECT t.* FROM new_batch t
WHERE NOT EXISTS (
    SELECT 1 FROM underlying_bars e
    WHERE e.ts_event = t.ts_event AND e.instrument_id = t.instrument_id
)
```

Rows that already exist are silently skipped. Rows for new timestamps or new
instrument IDs are inserted. This is safe for overlapping, non-overlapping, and
gap-free date ranges.

> **Note:** Even within a single `--append` invocation, the input batch is first
> deduplicated internally before the `NOT EXISTS` check runs, so duplicate rows
> within the same zip do not accumulate.

#### Greeks (`option_greeks`)

Greeks are computed only for `option_bars` rows that do not yet have a corresponding
entry in `option_greeks`:

```sql
-- Only process option bars not already in option_greeks
WHERE NOT EXISTS (
    SELECT 1 FROM option_greeks og
    WHERE og.ts_event = ob.ts_event AND og.instrument_id = ob.instrument_id
)
```

Previously computed greeks are never touched. The batching and progress counter reflect
only the new rows being processed.

#### Indicators

Indicators are checked for staleness automatically whenever you run `btkit run` (or
`btkit pipeline`). The check compares the maximum timestamp in `indicator_bars` against
the maximum timestamp in `underlying_bars`:

- **Up to date** — indicator coverage reaches the latest underlying bar. No action.
- **Stale** — new underlying bars extend beyond the current indicator coverage. The
  indicator is recomputed on the full underlying bar history and only the new timestamps
  are inserted (`NOT EXISTS` on `(ts_event, indicator_id)`). The console prints:

  ```
  Checking strategy indicators...
    vwap.py: underlying data extended (2026-05-21 → 2026-06-18), recalculating...
    vwap.py: done
  ```

- **Not yet built** — no entry found in `indicator_definition` for this script. The
  indicator is built from scratch and the console prints:

  ```
  Checking strategy indicators...
    vwap.py: new indicator detected, building...
    vwap.py: done
  ```

> **Why recompute on full history?** Indicators like cumulative VWAP, moving averages,
> and any signal that uses a lookback window need to see all prior bars to produce
> correct values for the new bars. Only the result rows for new timestamps are written
> to the database — the full recompute is a correctness requirement, not a bug.

---

### Adding new symbols

When `--append` is used with data for a symbol not yet in the database (e.g. adding NQ
alongside existing ES data), the full pipeline runs for the new symbol:

1. The new zip's definition files supply the `underlying_id` mapping for NQ options.
2. NQ `underlying_bars` and `option_bars` rows are inserted (they don't exist yet, so
   no `NOT EXISTS` filtering is needed in practice — all rows are new).
3. Greeks are computed for all new NQ option bars.
4. If any strategy indicators are specified, they are recalculated because the new NQ
   underlying bars extend beyond the last indicator timestamp for the NQ
   `underlying_id`.

No manual steps are required. The strategy YAML controls which `root_symbol` is active,
so NQ and ES strategies each only see their own underlying's data.

---

### Overlapping date ranges

If the new zip files contain dates that already exist in the database (e.g. you
re-download a date range you already have), the `NOT EXISTS` checks ensure no
duplicates are created. The append is effectively a no-op for rows that are already
present. This makes `--append` safe to run even when you're not certain whether new
data overlaps with existing data.

---

### Full command reference

```
btkit build [OPTIONS]

  --data-path PATH        Directory containing Databento .zip files  [required]
  --db-path PATH          Input database path (created if absent)     [required]
  --indicators PATH       Indicator script(s). May be repeated.
  --append / --no-append  Append to existing DB instead of rebuilding.
                          Skips OHLCV rows, greeks, and indicator bars
                          that are already present.  [default: no-append]
```

---

## Output Database

Written by `btkit run`. One output database file per run (or per matrix run, containing
results for all parameter combinations within that matrix).

---

### `backtest`

One row per backtest run. Strategy parameters are stored as JSON so the output database
is self-describing — results can be interpreted without the original YAML file.

```sql
CREATE TABLE backtest (
    id                  INTEGER PRIMARY KEY,
    matrix_id           INTEGER,                -- NULL for single runs;
                                                -- groups all runs from one matrix expansion
    combination_id      INTEGER,                -- NULL for single runs;
                                                -- 1-indexed position within the matrix
    strategy_name       VARCHAR         NOT NULL,
    strategy_version    VARCHAR,
    strategy_params     JSON            NOT NULL,
    initial_equity      DOUBLE          NOT NULL,   -- account equity at backtest start
    slippage_pct        DOUBLE          NOT NULL,
    fee_per_contract    DOUBLE          NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL
);
```

**Design notes:**

- `matrix_id` is assigned by `MatrixRunner` and shared across all backtests from the same
  matrix expansion. It allows `PostProcessor` to load and compare the full set of
  combinations from a single matrix run.
- `combination_id` is the 1-indexed position of this run within the expanded matrix,
  corresponding to the row index in `StrategyMatrix.params_df`. Together with
  `strategy_params` (which stores the scalar parameter values for this specific run),
  it allows any individual combination to be identified and re-run in isolation.
- Both columns are NULL for single-combination backtests, which never pass through
  `MatrixRunner`.

---

### `position`

One row per complete trade lifecycle (open through close). All monetary values are
net of costs unless otherwise noted.

```sql
CREATE TABLE position (
    id              INTEGER PRIMARY KEY,
    backtest_id     INTEGER         NOT NULL REFERENCES backtest(id),
    trade_name      VARCHAR         NOT NULL,   -- matches trades[n].name from strategy YAML
    open_time       TIMESTAMPTZ     NOT NULL,
    exit_time       TIMESTAMPTZ,                -- NULL if still open at end of backtest
    exit_reason     VARCHAR,                    -- 'take_profit' | 'stop_loss' |
                                                -- 'condition' | 'dte_exit' |
                                                -- 'expiry' | NULL
    open_mark       DOUBLE          NOT NULL,   -- position mark value at open
    exit_mark       DOUBLE,                     -- position mark value at exit (pre-slippage)
    worst_mark      DOUBLE,                     -- most adverse mark seen during trade lifetime
    slippage_cost   DOUBLE,
    fee_cost        DOUBLE,
    net_pnl         DOUBLE
);
```

**Design notes:**

- `trade_name` links each position back to the trade definition in the strategy YAML
  that produced it. For single-trade strategies this is a constant; for multi-trade
  strategies (e.g. independent put and call spreads) it allows `PostProcessor` to
  report per-trade metrics and lets users filter results by wing.
- `open_mark` and `exit_mark` use generic "mark" naming rather than spread-specific
  terminology. "Mark" is the standard financial term for the current market value of a
  position and applies equally to single-leg options, multi-leg spreads, and future
  instrument types. Direction and leg structure are captured in `position_leg`.
- `worst_mark` records the most adverse mark value observed across all bars during the
  trade's lifetime, before exit. This enables Maximum Adverse Excursion (MAE) to be
  computed in post-processing. Tracking it during Pass 2 is essentially free — the exit
  scanner is already iterating over all bars per position.

---

### `position_leg`

One row per leg per position. Supports single-leg and multi-leg strategies.

```sql
CREATE TABLE position_leg (
    id              INTEGER PRIMARY KEY,
    position_id     INTEGER         NOT NULL REFERENCES position(id),
    instrument_id   INTEGER         NOT NULL,
    symbol          VARCHAR         NOT NULL,
    expiration      DATE,
    strike_price    DOUBLE,
    right           VARCHAR(1),                 -- 'C' or 'P'; NULL for non-option legs
    action          VARCHAR(3)      NOT NULL,   -- 'STO' or 'BTO'
    quantity        INTEGER         NOT NULL,   -- always positive; action encodes direction
    multiplier      INTEGER         NOT NULL,
    open_price      DOUBLE          NOT NULL,
    exit_price      DOUBLE
);
```
