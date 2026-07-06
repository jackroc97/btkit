# btkit audit — Input Database Data Quality Audit

The `btkit audit` command runs a structured data quality check against an input database and writes its results to an `option_audit` table that the backtest engine reads at entry time to filter out unreliable instruments.

---

## Quick start

```bash
# Run the full audit and write results
btkit audit --input-db /path/to/input.db

# Skip Phase 2 (expensive 3-way join) for a fast pass
btkit audit --input-db /path/to/input.db --skip-phase2

# Inspect flags without modifying the database
btkit audit --input-db /path/to/input.db --dry-run

# Machine-readable output
btkit audit --input-db /path/to/input.db --output-format json
```

Once the audit has been run, the backtest engine automatically applies the configured filter. No changes to strategy YAML are required if you are happy with the default (`hard_errors_only`).

---

## Why audit matters

The most operationally dangerous issue is `BARS_TRUNCATED`: options whose bar data ends weeks before the stated expiration date. When this happens, the btkit engine treats the last available bar date as expiry and exits the position immediately on that bar — regardless of the actual expiry — at whatever the last observed price was. This can produce large, unexpected losses. A confirmed example: `instrument_id=325994` (ES put, expiry 2023-01-26) had bars only through 2022-12-13 (44 days short). The engine entered and immediately closed the position at a -$1,451 loss.

---

## Audit phases and flags

The audit runs in four phases. Phases 1, 3, and 4 are cheap SQL scans. Phase 2 requires a 3-way join across `option_greeks`, `option_bars`, and `underlying_bars` and is the most time-consuming step on large databases.

### Phase 1 — Implied volatility

| Flag | Severity | Condition |
|---|---|---|
| `IV_NAN` | soft | `isnan(iv)` — Black-76 IV computation failed (typically T=0 at expiry) |
| `IV_SENTINEL` | soft | `iv = 10.0` — greeks engine bisection hit its upper cap; option is deep ITM with no extrinsic value |
| `IV_HIGH` | soft | `iv > 2.0`, finite, not sentinel — IV above 200%; may be legitimate during stress events |

**Note on DuckDB NaN:** The `iv` column stores IEEE 754 NaN as a valid float, distinct from SQL NULL. The audit uses `isnan(iv)` — not `IS NULL` — to detect NaN values correctly.

### Phase 2 — Black-76 delta consistency

| Flag | Severity | Condition |
|---|---|---|
| `DELTA_INCONSISTENT` | soft | `|reported_delta − theoretical_delta| > 0.10` |

Theoretical delta is computed using the **Black-76 model** (correct for futures options) with the reported IV as the volatility input and `r = 0.01` (matching `GreeksCalculator`). The computation reuses the numba-JIT `_greeks()` kernel from `btkit.pipeline.greeks` — no external dependencies (no scipy; uses `math.erfc` for the CDF).

Rows with NaN/infinite delta or IV, `T ≤ 0`, or non-positive underlying price are skipped.

High IV (e.g. 200%) is not automatically treated as corrupt: a 30% OTM put during a genuine stress event can have a theoretically consistent delta > 0.02 once the higher IV is accounted for. `DELTA_INCONSISTENT` compares reported vs. Black-76 at the *reported* IV — so the flag catches cases where the delta and IV are mutually inconsistent, not merely unusual.

### Phase 3 — Bar coverage

| Flag | Severity | Condition |
|---|---|---|
| `BARS_TRUNCATED` | **hard** | `(expiration − last_bar_date) / (expiration − first_bar_date) > 0.15` |
| `BARS_SPARSE` | soft | bars per active trading day < 10 |
| `NO_EXPIRY_BARS` | soft | instrument has zero bars on its expiration date |

**BARS_TRUNCATED** uses a proportional threshold to avoid false positives from the original absolute-day approach. The 15% threshold means: if more than 15% of the observable life is missing from the right (expiry) end, the instrument is flagged. Example: a 45DTE option with its last bar 7+ days before expiry (7/45 = 15.6%) trips the flag.

**NO_EXPIRY_BARS** is complementary to `BARS_TRUNCATED`. A 45DTE option whose data runs through day 44 but has nothing on day 45 has a truncation ratio of 1/45 = 2.2% — well below the 15% threshold — but is still flagged by `NO_EXPIRY_BARS`. This matters when positions survive to expiry and need the final day's price for settlement.

**Coverage flags are instrument-level.** Each flagged instrument produces a single row in `option_audit` (using its earliest `ts_event` as the representative timestamp). The entry-time filter excludes the entire instrument, not just individual bars.

### Phase 4 — Basic integrity

| Flag | Severity | Condition |
|---|---|---|
| `NEGATIVE_CLOSE` | **hard** | `close < 0` — impossible for options |
| `NEGATIVE_DTE` | **hard** | `dte < 0` — greeks computed after expiry |
| `ZOMBIE_BAR` | **hard** | `expiration < ts_event::date` — bar dated after the option's own expiry |
| `DELTA_SIGN_ERROR` | **hard** | put delta > 0 or call delta < 0 (finite, non-NaN) |
| `DELTA_MAGNITUDE_ERROR` | **hard** | `abs(delta) > 1.0` (finite, non-NaN) |

---

## The option_audit table

Results are written to an `option_audit` table in the input database itself (alongside `option_bars` and `option_greeks`). Schema:

```sql
CREATE TABLE option_audit (
    instrument_id  INTEGER     NOT NULL,
    ts_event       TIMESTAMPTZ NOT NULL,
    flag_code      VARCHAR     NOT NULL,
    flag_severity  VARCHAR     NOT NULL,  -- 'hard' | 'soft'
    flag_value     DOUBLE,                -- the observed value that triggered the flag
    threshold      DOUBLE,                -- the threshold it exceeded
    PRIMARY KEY (instrument_id, ts_event, flag_code)
);
```

The table is **truncated and rebuilt** on every run so results always reflect the current database state. Running the audit twice is idempotent.

---

## Entry-time filtering

The backtest engine reads `option_audit` when selecting legs. Any instrument whose `instrument_id` appears in `option_audit` with a matching `flag_code` is excluded from all entry candidates. If the `option_audit` table is absent (database was built before the audit feature existed), filtering is silently skipped — full backward compatibility.

### Configuring the filter in a strategy

Add `audit_filter` under `universe` in the strategy YAML:

```yaml
strategy:
  name: my_strategy
  universe:
    start_date: "2022-01-01"
    end_date:   "2026-01-01"
    audit_filter: hard_errors_only   # default

  trades:
    ...
```

#### Preset values

| Preset | Effect |
|---|---|
| `none` | No filter; all instruments are eligible for entry |
| `hard_errors_only` | Exclude instruments with any hard flag (**default**) |
| `strict` | Exclude instruments with any flag (hard or soft) |

#### Explicit flag list

Pass a list of specific flag codes to filter on a custom subset:

```yaml
audit_filter: ["BARS_TRUNCATED", "NEGATIVE_CLOSE", "ZOMBIE_BAR"]
```

This is useful when you want to exclude only the most operationally dangerous flags without discarding instruments with soft issues like `IV_NAN` at a single bar.

---

## CLI reference

```
btkit audit --input-db PATH [OPTIONS]

Options:
  --output-format TEXT   Output format: text (default), json, csv
  --skip-phase2          Skip Phase 2 (expensive 3-way join)
  --dry-run              Compute flags but do NOT write to option_audit
  --no-quintile          Suppress the quintile breakdown table
```

### Sample text output

```
Auditing: /path/to/input_es_full.db
  Phase 1  IV flags:     4,583,097 rows  (0.8s)
  Phase 2  Delta consistency: materialising 3-way join ...
     1008 / 1008 dates  (100%)
  Phase 2  Delta consistency:   183,429 rows  (47.3s)
  Phase 3  Bar coverage:         84,526 rows  (2.1s)
  Phase 4  Integrity:               337 rows  (0.6s)

==============================================================
  btkit audit — input_es_full.db
==============================================================
  Duration    : 51.0s
  Mode        : written to option_audit table

  Flag                       Sev              Rows  Instruments
  ----------------------------------------------------------
  BARS_TRUNCATED             hard            1,247          312
  BARS_SPARSE                soft            8,103        8,103
  DELTA_INCONSISTENT         soft          183,429       41,284
  DELTA_MAGNITUDE_ERROR      hard               14            7
  DELTA_SIGN_ERROR           hard               88           44
  IV_HIGH                    soft        1,204,771       98,733
  IV_NAN                     soft        2,891,204      183,492
  IV_SENTINEL                soft          487,122       50,234
  NEGATIVE_DTE               hard               31           12
  NO_EXPIRY_BARS             soft           76,423       76,423
  ZOMBIE_BAR                 hard              204           31
  ----------------------------------------------------------
  TOTAL                                              195,447
==============================================================

Quintile breakdown (by first-bar delta):

Puts
  Q     Delta range                Total   Flagged       %
  ----------------------------------------------------------
  1     -1.000 → -0.528           53,182    15,658    29.4%
  2     -0.528 → -0.203           53,181     9,845    18.5%
  3     -0.203 → -0.054           53,182     6,288    11.8%
  4     -0.054 → -0.014           53,181     4,666     8.8%
  5     -0.014 → +0.000           53,182     7,370    13.9%

Calls
  Q     Delta range                Total   Flagged       %
  ----------------------------------------------------------
  1     +0.000 → +0.063           36,064     6,360    17.6%
  2     +0.063 → +0.298           36,063     7,942    22.0%
  3     +0.298 → +0.568           36,064     9,219    25.6%
  4     +0.568 → +0.959           36,063     9,275    25.7%
  5     +0.959 → +1.000           36,064    15,665    43.4%
```

---

## Recommended workflow

1. **Build the input database** as normal with `btkit build`.
2. **Run the audit once** after building:
   ```bash
   btkit audit --input-db /path/to/input.db
   ```
   Re-run whenever the database is updated with new data.
3. **Review the report.** Particularly note:
   - `BARS_TRUNCATED` count — each flagged instrument is a potential premature-exit risk.
   - `IV_NAN` at high counts near expiry is expected (T=0 → NaN IV); near-expiry soft flags are less operationally dangerous.
4. **Configure the filter** in strategy YAML if the default `hard_errors_only` is not appropriate.
5. **Run backtests** as normal. The engine silently applies the filter; no further action required.

---

## Implementation notes

### Where results are stored

The `option_audit` table is written directly into the input database file — the same DuckDB file that contains `option_bars` and `option_greeks`. This keeps audited data co-located with the input data, so any copy or backup of the input database automatically includes audit results.

### Concurrency

The audit runner opens a **writable** DuckDB connection (the input database is not opened read-only during audit). The backtest engine opens it read-only. DuckDB does not support concurrent writers to the same file, so do not run `btkit audit` while a backtest or study is in progress against the same input database.

### Black-76 vs Black-Scholes

Phase 2 uses **Black-76** (options on futures), not BSM. This is correct for CME futures options (ES, SPX). The model discount factor is `exp(-r·T)`, and delta for a put is `df·(N(d1)−1)`. The same kernel (`_greeks` in `btkit.pipeline.greeks`) is used by `GreeksCalculator` when building the database, so the audit's theoretical delta is computed identically to how the stored delta was originally computed.

### Coverage flags are instrument-level

`BARS_TRUNCATED`, `BARS_SPARSE`, and `NO_EXPIRY_BARS` store one row per instrument (using `min(ts_event)` as the representative timestamp) rather than one row per bar. The entry-time filter queries `SELECT DISTINCT instrument_id FROM option_audit WHERE flag_code IN (...)` and excludes the entire instrument — so a single sentinel row is sufficient for filtering while keeping the audit table compact.

### Backward compatibility

`greeks_for_all_legs` and `greeks_for_strike_legs` accept `audit_filter_codes=None` as a default. When no codes are provided (or `audit_filter: "none"` is set), no query is made against `option_audit`. Databases that pre-date the audit feature work unchanged.
