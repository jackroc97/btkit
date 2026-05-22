# btkit 2.0

## Overview

btkit 2.0 is a ground-up re-engineering of [btkit](~/dev/btkit), an options backtesting
framework. The aim is to allow users to construct and evaluate options strategies against
historical data. btkit 2.0 builds on lessons learned from the legacy project with an
emphasis on a vectorized backtest core, indicator support, declarative strategy definitions,
and a streamlined one-command pipeline.

---

## Legacy Project

### Tech Stack

- **Raw data:** OHLCV and definition schema data from Databento
- **Database:** DuckDB
- **Library logic and strategy implementation:** Python, numba JIT for compute-heavy operations
- **Post-processing and visualization:** Jupyter notebooks / Python

### Shortfalls

The legacy btkit project was successful at its primary goal of allowing users to construct
simple options-focused backtests, but had four significant shortfalls:

1. **Runtime.** Single backtests spanning 2–3 years on 5-minute intervals took 2–10 minutes
   to run. Data was stored in columnar format, which helped reduce runtimes, but the
   time-stepped architecture was the primary bottleneck.

2. **Resolution.** The legacy project was built primarily for 5-minute OHLCV data. This is
   not enough resolution for intraday options strategies, where higher-frequency data is
   preferred — especially for strategies that rely on take-profit and stop-loss orders for
   risk management.

3. **No indicators.** Indicators were not natively supported by the framework. Custom
   implementation was required, greatly complicating otherwise simple backtests.

4. **Clunky pipeline.** Building the database from source data (Databento `.dbn` files)
   required one script. Pre-computing option Greeks required a separate script. Running a
   backtest required yet another script. Running a matrix of backtests required creating a
   DOE in YAML format and yet another script. Post-processing results required more scripts
   still.

---

## Goals for btkit 2.0

The overall goal is to fix the legacy shortfalls from the bottom up. This is a complete
re-engineering of the project where necessary.

1. **Runtime.** Significantly reduce backtest runtime via:
   - A vectorized backtest core rather than a time-stepped loop. Given a declarative
     strategy definition, all or most operations can be completed in the columnar store,
     avoiding per-bar Python overhead.
   - A flat database schema that eliminates runtime joins. Pre-joining definition metadata
     into price tables at build time produces a less normalized but substantially faster
     schema.
   - Process-level parallelism for concurrent backtests (matrix runs).

2. **Resolution.** Move from 5-minute to 1-minute OHLCV bars. Tick data is the long-term
   goal but is deferred due to storage, compute, and data availability constraints. See
   [constraints.md](constraints.md) for details.

3. **Indicators.** Pre-computed indicators are stored in the database and available as
   entry and exit conditions. Rather than shipping built-in indicators, users provide
   scripts that generate indicator columns from underlying bar data.

4. **Simple strategy definition.** Strategies are defined declaratively in YAML. Numeric
   parameters support scalar values, list sweeps, and range sweeps for
   design-of-experiments runs. Explicit combination lists are also supported for targeted
   parameter spaces.

5. **Simple pipeline.** One command (`btkit pipeline`) to go from raw data to visualized
   output. Individual sub-commands are also available for incremental use:
   - **`btkit build`** — Ingest raw data, compute Greeks, run indicator scripts, write
     input database.
   - **`btkit run`** — Load strategy YAML, run vectorized backtest (single or matrix),
     write output database.
   - **`btkit analyze`** — Compute metrics, open results dashboard.

---

## Design Documents

| Document | Contents |
|---|---|
| [database.md](database.md) | Input and output database schemas (DuckDB) |
| [classes.md](classes.md) | Package structure and class signatures for the logic layer |
| [strategy.md](strategy.md) | Strategy YAML schema, Pydantic models, condition expressions, exit priority |
| [fill_price_and_costs.md](fill_price_and_costs.md) | Fill price rules, slippage model, fee model |
| [constraints.md](constraints.md) | Known constraints, limitations, and reach goals |
| [mvp.md](mvp.md) | MVP success criteria, in-scope components, and deferred features |
