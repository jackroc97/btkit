# btkit 2.0 — Constraints and Limitations

This document records the known boundaries of the current design: deliberate scope
decisions, architectural trade-offs, and features explicitly deferred to future work.
It exists so that these constraints are visible without having to read the full design
documents, and so that future contributors understand what was intentional vs. what
is left undone.

---

## Data Resolution

**Constraint:** btkit 2.0 is built on **1-minute OHLCV bars**, not tick data.

The original goal stated in `instructions.md` was to "accept, and even prefer,
tick-by-tick data." This was set aside for pragmatic reasons: tick data requires
substantially more storage, a different ingest pipeline, and a different bar/event
model throughout the backtest engine.

**Impact:**
- TP/SL events that occur and reverse within a single 1-minute bar cannot be detected.
  With 1-min resolution this error is small but non-zero.
- For multi-leg spreads, the position mark is computed from bar closes, not intrabar
  extremes. A TP/SL triggered mid-bar but reversed by close will be missed until the
  next bar.
- The entire database schema (OHLCV tables, spread mark computation, fill price rules)
  is built around 1-minute bars. Adding tick data in the future would require significant
  schema and engine changes.

**Reach goal:** Refactor the data layer to support tick events; reframe 1-min bars as
a derived aggregation rather than the primary data source.

---

## Vectorized Design and Sequential State

**Constraint:** The three-pass vectorized backtest (Entry → Exit → PnL) cannot express
conditions that depend on the sequential history of the run in a pure Polars pass.

The design resolves this with targeted sequential steps at well-defined points:
- `max_open_positions` is a vectorized approximation (count of open positions at each
  timestamp), not a true sequential check.
- `minimum_equity` is evaluated as a sequential final step in Pass 1, after all
  vectorized filters, by sweeping entry candidates in chronological order and tracking
  running equity. This is the only non-vectorized filter in the critical path.

**Impact:** Complex position-sizing rules that depend on current portfolio state (e.g.,
Kelly-fraction sizing, risk-parity weighting) cannot be expressed in the current model.
`fixed_contracts` sizing is the only supported method.

**Reach goal:** A portfolio-level post-filter pass that adjusts or rejects entries based
on full portfolio state, run after the vectorized entry scan.

---

## Margin and Capital Requirements

**Constraint:** btkit 2.0 does not model margin requirements or buying power reduction.

`initial_equity` is a run-level parameter used as the starting point for equity curve
and drawdown metrics. `entry.minimum_equity` provides a simple equity floor that can
approximate a minimum account requirement, but it does not compute margin consumed by
individual positions.

**Impact:** Strategies that would be margin-constrained in live trading may appear
over-deployed in backtests.

**Mitigation:** Use `max_open_positions` and `minimum_equity` together as conservative
proxies for capital constraints. For example, if each short put spread consumes roughly
$5,000 in margin on a $50,000 account, set `max_open_positions: 10` and
`minimum_equity: 5000`.

**Reach goal:** A per-position margin model that deducts estimated margin from available
equity as positions are opened and credits it back on close.

---

## Single Underlying per Strategy

**Constraint:** Each strategy definition targets a single underlying instrument
(`instrument.root_symbol`). Multi-leg strategies across different underlyings (e.g.,
a pairs trade between SPY and QQQ) are not supported.

**Impact:** Strategies referencing indicator values or leg properties from more than one
underlying cannot be expressed in the current YAML schema.

**Reach goal:** A multi-instrument strategy type that allows leg definitions to reference
different underlyings.

---

## Indicator Reproducibility and External Dependencies

**Constraint:** `indicator_definition.script_source` stores the source of the top-level
indicator script, not its transitive dependencies.

If an indicator script imports external modules, those modules' source is not captured.
The stored script is sufficient to re-identify what was run, but is not guaranteed to
re-execute identically if imported module versions change.

**Impact:** Reproducibility depends on the user maintaining a stable environment.
Pinning dependencies in a `requirements.txt` or `pyproject.toml` alongside the
indicator script is strongly recommended.

---

## Visualization

**Constraint:** The visualization/dashboard layer (`charts.py`, `btkit analyze`) is not
yet fully specified. The class design includes a stub for `charts.py` with Plotly and
lightweight-charts as the intended libraries. The output format (static HTML report,
interactive web app, etc.) is to be decided before implementation of the analysis layer.

**Impact:** The `btkit analyze` command and `btkit pipeline` commands are designed but
cannot be implemented until the dashboard format is specified.

**Action required:** Design the dashboard before beginning implementation of
`analysis/charts.py`.
