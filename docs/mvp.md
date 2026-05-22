# btkit 2.0 — MVP Definition

This document defines the scope of the Minimum Viable Product: what constitutes a
successful MVP, what will be implemented, and what is explicitly deferred to a later
milestone.

---

## Success Criteria

The MVP is complete when all of the following hold:

1. **End-to-end pipeline.** A user can go from raw Databento `.dbn` files to backtest
   results using three CLI commands (`btkit build`, `btkit run`, `btkit analyze`) or
   the combined `btkit pipeline` wrapper.

2. **Correct database build.** `btkit build` ingests raw OHLCV and definition data,
   pre-computes option Greeks, and runs user-supplied indicator scripts. The resulting
   input database is self-contained and requires no further modification before a backtest.

3. **Multi-leg strategy support.** The backtest engine correctly handles strategies with
   any number of legs, including four-leg structures such as iron condors. Entry selection,
   spread mark computation, exit detection, and P&L calculation all work for multi-leg
   positions.

4. **Correct exit logic.** All seven exit conditions are implemented in priority order:
   gap-open SL, gap-open TP, stop loss, take profit, indicator condition, DTE exit, expiry.
   The fill price rules from `fill_price_and_costs.md` are applied correctly.

5. **Indicator-based conditions.** Simple comparison conditions referencing indicator
   columns (e.g. `rsi_14 < 40`, `vix_close < 30`) work as entry and exit filters.

6. **Correct metrics.** `btkit analyze` produces the full set of standard backtest
   metrics — net profit, win rate, profit factor, drawdown, CAGR, Sharpe, Sortino,
   Calmar, MAR, MAE — computed from the output database and printed to the terminal.

7. **Output database integrity.** The output database accurately records every position
   and leg, with correct open/exit marks, fill prices, costs, and net P&L.

---

## What Will Be Implemented

### Pipeline — `btkit build`

| Component | Scope |
|---|---|
| `DatabaseBuilder` | Full: ingest definitions, ingest OHLCV, orchestrate greeks and indicators |
| Databento ingest | `.dbn` definition and OHLCV files → `underlying_bars`, `option_bars` (pre-joined) |
| `GreeksCalculator` | Full: numba Black-76, batch computation → `option_greeks` |
| `IndicatorRunner` | Full: user-supplied `compute(df)` scripts → `indicator_definition` + `indicator_bars`; multi-series output supported |
| `InputDatabase` | Full read interface; all query methods |
| `OutputDatabase` | Full write interface; schema creation |

### Strategy Layer

| Component | Scope |
|---|---|
| `StrategyDefinition` Pydantic models | Full model definitions; all fields validated at load time |
| Sweep parameter types (`NumericSweep`, `IntSweep`, `SweepRange`) | Types defined and validated; expansion deferred — all sweep fields must be scalar for MVP runs |
| YAML loader | Full validation, clear error messages for malformed strategies |
| Condition parser | **Simple comparisons only:** `<`, `>`, `<=`, `>=`, `==`, `!=`, `and`, `or`, `not`; leg property references (`short_put.delta`); indicator and bar column references |

### Backtest Engine — Single Run

| Component | Scope |
|---|---|
| `BacktestEngine` | Full single-run orchestration |
| `EntryScanner._apply_window_filters()` | Full: entry window, session config, weekday/skip-date filters |
| `EntryScanner._select_legs()` | Full: batched DuckDB query on `option_greeks`; best delta+DTE match per leg; drops entries where any leg is unmatched |
| `EntryScanner._compute_open_mark()` | Full: multi-leg signed sum; TP/SL price derivation |
| `EntryScanner._evaluate_conditions()` | Steps 1–3: entry conditions (vectorized), min_credit/max_debit, max_open_positions; `minimum_equity` filter (step 4) deferred |
| `ExitScanner._load_exit_data()` | Full: single batch query for all leg bars and indicators |
| `ExitScanner._compute_position_marks()` | Full: close-based spread marks for multi-leg positions |
| `ExitScanner._find_first_hit()` | Full: all 7 exit conditions in priority order; gap-open detection; worst_mark tracking; indicator exit conditions |
| `PnLCalculator` | Full: gross P&L, slippage, fees, net P&L |
| Multi-leg support | Full throughout — required for iron condors |

### Output Database

| Component | Scope |
|---|---|
| `backtest` table | Full schema; `matrix_id` and `combination_id` always NULL for MVP |
| `position` table | Full schema including `worst_mark` |
| `position_leg` table | Full schema |

### Analysis — `btkit analyze`

| Component | Scope |
|---|---|
| `PostProcessor.metrics()` | Full metric set: net profit, total trades, win rate, profit factor, avg win/loss, median P&L, max drawdown, CAGR, MAR, Sharpe, Sortino, Calmar, premium capture rate, avg/median/worst MAE |
| `PostProcessor.equity_curve()` | Returns Polars DataFrame; no chart rendering |
| `PostProcessor.trade_pnl_series()` | Returns Polars DataFrame; no chart rendering |
| `PostProcessor.summarize()` | Formatted table to terminal |

### CLI

| Command | Scope |
|---|---|
| `btkit build` | Full: data path, db path, optional indicator script paths |
| `btkit run` | Single-run only; `--initial-equity`; `--workers` and `--max-combinations` absent for MVP |
| `btkit analyze` | Terminal output only; `--open-browser` absent for MVP |
| `btkit pipeline` | Full: chains build (with skip-if-exists logic) → run → analyze |

---

## What Is Deferred

### Matrix Runs and Parallelism

Deferred in their entirety. The strategy YAML schema and Pydantic models support sweep
parameter syntax (for forward compatibility), but the engine will reject any strategy
definition that contains non-scalar parameter values.

| Deferred Component | Notes |
|---|---|
| `StrategyMatrix` | Sweep expansion, cartesian product, explicit combinations |
| `MatrixRunner` | `ProcessPoolExecutor`-based parallel dispatch |
| `OutputMerger` | Worker DB consolidation |
| `PostProcessor.heatmap()` | Requires matrix results |
| `--workers` CLI flag | No-op until `MatrixRunner` exists |
| `--max-combinations` CLI flag | Same |

### Condition Parser — if/then Syntax

The `if A then B` logical implication form is deferred. Basic comparisons and boolean
operators (`and`, `or`, `not`) are sufficient for the MVP. The parser is designed so
that if/then can be added as an extension without touching the rest of the evaluation
pipeline.

### Minimum Equity Entry Filter

`entry.minimum_equity` is validated and stored in the `StrategyDefinition` model but not
evaluated during Pass 1. Entries are not filtered by running equity in the MVP. The field
is a no-op until the sequential equity filter step is implemented.

### Visualization

`charts.py` and all browser-based output are deferred. `btkit analyze` writes results to
the terminal. The dashboard format has not yet been specified and will be designed as a
separate milestone.

---

## Testing and Verification

The MVP is verified through end-to-end tests that drive the CLI commands and inspect the
resulting databases and terminal output. No unit test framework is required at this stage.
Each test can be executed manually and verified against the expected outputs described
below.

---

### Test Fixtures

All tests share the same test dataset and a set of purpose-built strategy YAML files.

**Test dataset**

A small slice of Databento data for a single underlying (e.g. ES front-month futures)
covering approximately 3 months. Requirements:

- At minimum 20 trading days of 1-minute OHLCV bars for the underlying
- Options data covering at least two expiration cycles and a range of strikes/deltas
- At least one expiration where positions can be held through to expiry within the date range
- A known subset of timestamps and prices that can be used for manual verification of
  marks and P&L (even just 5–10 rows)

**Test indicator script** (`tests/fixtures/indicators.py`)

```python
import polars as pl

def compute(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("close").rolling_mean(20).alias("sma_20"),
        pl.col("close").rolling_mean(5).alias("sma_5"),
    ])
```

`sma_20` and `sma_5` are simple enough that expected values can be computed by hand for
spot-checking.

**Test strategy files** (`tests/fixtures/`)

Six YAML strategy files are needed (described under each test case below). All use the
same underlying and date range. All use scalar parameters only.

---

### Test 1 — End-to-end pipeline (SC-1)

**Goal:** Verify the full pipeline runs without errors from raw data to metrics output.

**Execution:**
```
btkit pipeline \
  --data-path tests/fixtures/data/ \
  --strategy  tests/fixtures/strategy_base.yaml \
  --db-path   tests/output/test_input.db \
  --output-db tests/output/test_output.db \
  --indicators tests/fixtures/indicators.py
```

**Verification:**
- Command exits with code 0
- `tests/output/test_input.db` and `tests/output/test_output.db` both exist
- No error or traceback in stdout/stderr
- Terminal prints a metrics summary (even if results are sparse)

---

### Test 2 — Database build correctness (SC-2)

**Goal:** Verify that `btkit build` produces the correct tables with correct content.

**Execution:** Run `btkit build` against the test dataset. Then query the resulting DB.

**Verification — schema:**
```sql
-- All five tables exist
SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';
```
Expected: `underlying_bars`, `option_bars`, `option_greeks`, `indicator_definition`,
`indicator_bars`.

**Verification — option_bars pre-join:**
```sql
SELECT COUNT(*) FROM option_bars WHERE strike_price IS NULL OR expiration IS NULL OR right IS NULL;
```
Expected: 0 rows. Definition metadata must be fully pre-joined — no NULLs.

**Verification — Greeks completeness:**
```sql
SELECT COUNT(*) FROM option_greeks WHERE delta IS NULL AND iv IS NULL;
```
Expected: 0 (or a small number attributable to edge cases where IV does not converge).
Spot-check a handful of rows against a manual Black-76 calculation using known inputs.

**Verification — indicators:**
```sql
SELECT name FROM indicator_definition;
```
Expected: `sma_20`, `sma_5`.
```sql
SELECT COUNT(*) FROM indicator_bars WHERE value IS NULL;
```
Expected: only the leading rows where the rolling window has not yet filled (20 for
`sma_20`, 5 for `sma_5`).

---

### Test 3 — Multi-leg strategy / iron condor (SC-3)

**Goal:** Verify that a four-leg iron condor strategy produces correctly structured
positions and legs.

**Strategy file:** `strategy_iron_condor.yaml` — an iron condor with four legs:
`long_put`, `short_put`, `short_call`, `long_call`.

**Execution:** `btkit run` with the iron condor strategy.

**Verification — leg count:**
```sql
SELECT position_id, COUNT(*) AS leg_count FROM position_leg GROUP BY position_id;
```
Expected: every position has exactly 4 legs.

**Verification — leg actions:**
```sql
SELECT action, COUNT(*) FROM position_leg GROUP BY action;
```
Expected: equal counts of `STO` and `BTO` (2 of each per position).

**Verification — leg structure:**
```sql
SELECT right, action FROM position_leg WHERE position_id = 1 ORDER BY strike_price;
```
Expected (for a standard iron condor): `P/BTO` (lowest strike), `P/STO`, `C/STO`, `C/BTO`
(highest strike).

**Verification — spread mark:**
For one known entry timestamp, manually compute:
```
open_mark = (long_put_price * 1) + (short_put_price * -1) + (short_call_price * -1) + (long_call_price * 1)
```
Compare against `position.open_mark` for that row.

---

### Test 4 — Exit condition coverage (SC-4)

**Goal:** Verify that each of the seven exit conditions fires correctly.

Run five separate strategies, each engineered to produce a specific `exit_reason`. After
each run, query:

```sql
SELECT DISTINCT exit_reason FROM position;
```

| Strategy file | Engineering | Expected `exit_reason` |
|---|---|---|
| `strategy_exit_tp.yaml` | Very tight `take_profit` (e.g. 0.05), wide `stop_loss` | `take_profit` (all or vast majority) |
| `strategy_exit_sl.yaml` | Tight `stop_loss` (e.g. 0.10), no `take_profit` | `stop_loss` (all or vast majority) |
| `strategy_exit_dte.yaml` | `dte_exit` set to a value well above typical holding period, no TP/SL | `dte_exit` (all positions) |
| `strategy_exit_expiry.yaml` | `dte_exit` disabled, `expiry_exit: true`, TP/SL set wide | `expiry` (all positions) |
| `strategy_exit_condition.yaml` | `exit.conditions: ["sma_5 > 0"]` (always true after warmup), tight TP/SL disabled | `condition` (all positions after indicator warmup) |

**Gap-open verification (priority 1 and 2):** After any run, check whether any positions
have an `exit_time` that coincides with the first bar of a session (overnight gap). If
found, verify that `exit_mark` equals the spread open mark at that bar, not the TP/SL
threshold price. This can be confirmed by checking `exit_mark` against a manually computed
spread open mark from `option_bars`.

---

### Test 5 — Indicator-based entry conditions (SC-5)

**Goal:** Verify that indicator column values correctly gate entry signals.

**Step A — Baseline:** Run `strategy_base.yaml` (no indicator conditions). Record total
entry count `N_base`.

**Step B — Always-true condition:** Add `conditions: ["sma_20 > 0"]` to the same strategy
(always true after warmup; `sma_20` of a price series is always positive). Record entry
count `N_true`.

Expected: `N_true` ≈ `N_base` (minor difference only from the leading warmup window where
`sma_20` is NULL and the condition cannot be evaluated).

**Step C — Always-false condition:** Add `conditions: ["sma_20 < 0"]`. Record entry count
`N_false`.

Expected: `N_false` = 0.

**Step D — Selective condition:** Add `conditions: ["sma_5 > sma_20"]` (true roughly half
the time). Record entry count `N_selective`.

Expected: `0 < N_selective < N_base`.

Spot-check: for a handful of entries in Step D, look up `sma_5` and `sma_20` values at
the entry timestamp in `indicator_bars` and confirm `sma_5 > sma_20` holds.

---

### Test 6 — Metrics correctness (SC-6)

**Goal:** Verify that computed metrics match manually calculated values.

**Setup:** Use a minimal strategy over a short window such that the full set of resulting
trades (ideally 5–15) can be enumerated by hand. Record each trade's `open_mark`,
`exit_mark`, `slippage_cost`, `fee_cost`, and `net_pnl` from the `position` table.

**Manual calculations to verify:**

| Metric | Manual formula |
|---|---|
| `net_profit` | `SUM(net_pnl)` across all positions |
| `total_trades` | `COUNT(*)` from `position` |
| `percent_profitable` | `COUNT(*) WHERE net_pnl > 0` / `total_trades` |
| `profit_factor` | `SUM(net_pnl WHERE net_pnl > 0)` / `ABS(SUM(net_pnl WHERE net_pnl < 0))` |
| `avg_mae` | `MEAN(ABS(worst_mark - open_mark))` across all positions |
| `net_pnl` per row | `(open_mark - exit_mark) - slippage_cost - fee_cost` |

Run `btkit analyze --backtest-id <id>` and compare each metric in the terminal output
against the manually computed values. Allow floating-point tolerance of ±0.01.

---

### Test 7 — Output database integrity (SC-7)

**Goal:** Verify structural and referential correctness of the output database after any
backtest run.

Run the following queries after any test run. All should return 0 rows (or the values
noted).

**Referential integrity:**
```sql
-- Every position_leg references a valid position
SELECT COUNT(*) FROM position_leg pl
LEFT JOIN position p ON pl.position_id = p.id
WHERE p.id IS NULL;
-- Expected: 0
```

**Temporal validity:**
```sql
-- No position closes before it opens
SELECT COUNT(*) FROM position WHERE exit_time IS NOT NULL AND exit_time <= open_time;
-- Expected: 0
```

**Valid exit reasons:**
```sql
-- exit_reason is always one of the five valid values or NULL
SELECT COUNT(*) FROM position
WHERE exit_reason NOT IN ('take_profit','stop_loss','condition','dte_exit','expiry')
AND exit_reason IS NOT NULL;
-- Expected: 0
```

**P&L consistency:**
```sql
-- net_pnl = (open_mark - exit_mark) - slippage_cost - fee_cost
-- Allow for floating-point tolerance
SELECT COUNT(*) FROM position
WHERE ABS(net_pnl - (open_mark - exit_mark - slippage_cost - fee_cost)) > 0.01
AND exit_mark IS NOT NULL;
-- Expected: 0
```

**Leg count consistency (iron condor test only):**
```sql
SELECT position_id, COUNT(*) AS leg_count FROM position_leg
GROUP BY position_id HAVING leg_count != 4;
-- Expected: 0 rows
```

**Aggregate consistency:**
```sql
-- Sum of net_pnl in DB matches net_profit from PostProcessor
SELECT SUM(net_pnl) FROM position WHERE backtest_id = <id>;
-- Compare against PostProcessor.metrics()["net_profit"]
```

---

## Out-of-Scope for btkit 2.0 (not MVP-specific)

The following items are not planned for any near-term milestone and are noted in
`constraints.md`:

- Tick data resolution
- Margin and buying power modeling
- Multi-underlying strategies
- Dollar-amount-based position sizing
