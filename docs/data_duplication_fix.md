# Input Database Duplication — Findings and Remediation Plan

Status: diagnosis complete, remediation not started. Target database:
`~/dev/wealthy-option-backtest/db/input/es.db` (67 GB). All audit queries so far were
read-only; nothing has been modified.

---

## 1. Two separate defects

| # | Defect | Scale | Window | Cause |
|---|---|---|---|---|
| 1 | Exact duplicate rows | 4.84M option bars, 4.82M greeks, 192k underlying bars, plus indicator bars | 2026-01-01 → 2026-05-28 | Ingest dedup bypassed (§2) |
| 2 | One `instrument_id` carrying two contracts **on the same day** | 3,875 id-days over 1,048 ids | 2016 → 2026 | Definition routing (§5) |

### The correct Databento invariant

Databento assigns `instrument_id` from the publisher's own numeric id — for GLBX.MDP3,
tag 48 `SecurityID` from the Security Definition messages — and **only guarantees that it
is unique within a given day**. Some publishers remap ids daily. Databento also never
retroactively reassigns symbols in historical data.

So the invariant to enforce is:

```
(instrument_id, day)  ->  exactly one contract definition
```

**not** "one `instrument_id` maps to one contract over a contiguous window". An id may
legitimately serve contract A one week and contract B the next, and the two contracts'
overall bar windows may even interleave across different days. 105,899 ids carry 232,241
definitions, which is normal.

An earlier version of this analysis tested window overlap, which is the wrong test. Of the
1,051 window-overlapping pairs it found, only **40** are legitimate interleaving on
distinct days; **1,011 share at least one day** and are real violations.

---

## 2. Defect 1: root cause (confirmed)

The duplicate window matches the source-zip overlap exactly.

`raw_data/ES/` coverage:

| Zip | OHLCV coverage |
|---|---|
| `GLBX-20260525-CSJV83YFJA` | 2026-01-01 → 2026-04-21 |
| `GLBX-20260522-55XRGWV8NM` | 2026-04-22 → 2026-05-21 |
| `GLBX-20260602-9FVF6BRTUL` | 2026-05-22 → 2026-05-28 |
| `GLBX-20260603-39RSDHYR76` | 2010-06-06 → **2026-06-02** (full history, 193 monthly files) |
| `GLBX-20261002-CCFTPMWLYW` | 2026-06-01 → 2026-06-14 |
| `GLBX-20261002-QXNC7PMEDJ` | 2026-06-15 → 2026-09-30 |

The full-history re-download overlaps the three incremental zips over
**2026-01-01 → 2026-05-28** — precisely the observed duplicate window, and precisely
2 copies of every row.

Two code defects let it through:

**(a) The static cutoff fast path.** `ub_cutoff` / `ob_cutoff` are computed **once before
the ingest loop** ([builder.py:351-353](../btkit/pipeline/builder.py#L351)) and never
advanced. In `_write_df`, any row with `ts_event > cutoff` bypasses the `NOT EXISTS`
check entirely ([builder.py:810-816](../btkit/pipeline/builder.py#L810)). The in-run
`.unique()` at [builder.py:386](../btkit/pipeline/builder.py#L386) only dedups **within one
batch** of `_OHLCV_BATCH_SIZE` files. The same `(ts_event, instrument_id)` arriving from
two different zips lands in two different batches, both compare as "newer than cutoff",
and both are inserted.

**(b) `NOT EXISTS` does not dedup within the inserted batch.** Verified directly in
DuckDB: a batch holding two copies of a key inserts both, because the subquery is
evaluated against the pre-insert table state. So even the non-fast path leaks duplicates
whenever the incoming frame has internal duplicates.

### Downstream cascade

- **`option_greeks`** — computed per `option_bars` row missing from greeks, so doubled
  bars yield doubled greeks rows (4.82M vs 4.84M). Consistent.
- **`indicator_bars`** — computed from `underlying_bars`, so doubled underlying rows
  yield doubled output, and mechanism (b) inserts both. **The values are also wrong**,
  not merely duplicated: cumulative and rolling indicators (the whole VWAP family)
  consumed double-counted input. Up to 32 copies per key with disagreeing values; the
  32× amplification factor is not yet fully explained, but it does not change the fix,
  since these must be recomputed from clean input regardless.

### Why the backup is not a rollback point

`es.db.bak-20261002` carries the **same** duplicate counts (192,407 and 4,844,020) with
max `ts_event` 2026-06-02. The duplication therefore predates the 2026-10-02 append —
that run added Jun–Sep data cleanly. Rolling back is not an option; the data must be
deduplicated or rebuilt.

---

## 3. Impact on backtests

| Area | Effect |
|---|---|
| TP/SL trigger levels | **None.** Duplicate rows carry identical prices (0 price conflicts), so `position_mark` per row is unchanged. |
| Volume liquidity gate | **Wrong.** The rolling sum at [exit.py:302-310](../btkit/backtest/exit.py#L302) sums duplicated rows, so `min_exit_volume` sees ~2× real volume and the gate passes too easily. |
| Indicator conditions | **Wrong** over Jan–May 2026: values are computed from doubled input, and the `ts_event` join at [exit.py:174](../btkit/backtest/exit.py#L174) multiplies rows and picks an arbitrary copy. |
| Entry counts | **Unverified** — needs checking whether duplicate bars produce duplicate entry candidates. |
| Exit-scan performance | Degraded: leg joins multiply rows. |
| Live-vs-backtest comparison | Apr–May 2026 overlaps the live trading period, so the window used for validation is affected. |

---

## 4. Audit coverage gap

The existing audit has 12 flags across 4 phases
([schema.py](../btkit/audit/schema.py)) and **none of them check row multiplicity**:

| Phase | Flags |
|---|---|
| 1 — IV | `IV_NAN`, `IV_SENTINEL`, `IV_HIGH` |
| 2 — delta | `DELTA_INCONSISTENT` |
| 3 — coverage | `BARS_TRUNCATED`, `BARS_SPARSE`, `NO_EXPIRY_BARS` |
| 4 — integrity | `NEGATIVE_CLOSE`, `NEGATIVE_DTE`, `ZOMBIE_BAR`, `DELTA_SIGN_ERROR`, `DELTA_MAGNITUDE_ERROR` |

`ZOMBIE_BAR` (bar dated after its contract's expiration) is a partial proxy for defect 2,
but it caught **none** of the 1,050 overlapping ids: 954 carry some other flag, 0 carry
`ZOMBIE_BAR` or `NEGATIVE_DTE`. Both defect classes are invisible to `btkit audit` today.

---

## 5. Defect 2: same-day definition collisions

Violations of `(instrument_id, day) -> one definition`:

| Day definition | Violating id-days | Distinct ids | Range |
|---|---|---|---|
| UTC date | 3,875 | 1,048 | 2016-11-04 → 2026-09-15 |
| CME session date (18:00 ET boundary) | 3,986 | 1,253 | 2015-12-24 → 2026-09-15 |

By year (UTC date):

| Year | 2016 | 2017 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| id-days | 9 | 8 | 74 | 39 | 269 | 413 | 1,397 | 1,263 | 403 |
| ids | 2 | 2 | 27 | 20 | 60 | 105 | 379 | 349 | 119 |

Example — `instrument_id` 116864 on 2022-09-01 (one UTC day):

| Time | Symbol | Expiration | Strike | Right |
|---|---|---|---|---|
| 11:12 EDT | `E1AV2 P2880` | 2022-10-03 | 2880 | P |
| 15:11 EDT | `E1CU2 P4060` | 2022-09-07 | 4060 | P |

Two different contracts, four hours apart, same id, same day. `instrument_id` 120342
shows the same pattern repeatedly across March 2023, alternating between a put
(`E3DJ3 P3765`, exp 2023-04-20) and a call (`E5DH3 C3990`, exp 2023-03-30).

### Two distinct sub-causes

**(a) Inside the Jan–May 2026 duplicate window: a consequence of defect 1.** All 2,474
same-`ts_event` definition conflicts fall in that window. Each duplicate copy of a bar was
routed to a segment independently, and the two routing passes disagreed.

**(b) Outside that window: a confirmed routing defect — systematic tie-break ambiguity.**

`_route_bars_to_segments` ([builder.py:710-739](../btkit/pipeline/builder.py#L710)) scores
every candidate segment by `_dist` (date distance to the segment's
`[activation, expiration]` window) then `_win` (window length), sorts, and keeps the first
row per `(ts_event, instrument_id)`. Since `pl.DataFrame.sort()` is **not stable by
default**, candidates tied on both keys resolve in arbitrary order.

**The tie is systematic, not occasional.** CME lists options with fixed tenors, so when an
`instrument_id` is recycled to another contract of the same tenor, `expiration − activation`
is identical **by construction**. Verified by scanning all 316 definition files for a random
sample of 24 violating id-days:

```
SYSTEMATIC TIE CHECK: tie=24  unique=0  no_data=0
```

Every single one was an exact tie, with `_win` clustering on CME's tenors (28, 40, 41, 48
days). Examples:

| id | Date | Tied candidates | `_win` |
|---|---|---|---|
| 116864 | 2022-09-01 | `E1CU2 P4060` (act 08-03, exp 09-07) vs `E1AV2 P2880` (act 08-29, exp 10-03) | 35 both |
| 120342 | 2023-03-10 | `E5DH3 C3990` (act 02-17, exp 03-30) vs `E3DJ3 P3765` (act 03-10, exp 04-20) | 41 both |
| 42502098 | 2024-10-23 | `E4DV4 P5275` vs `E1AZ4 C6220` | 41 both |
| 42888436 | 2026-03-17 | `E4CJ6 C7385` vs `E1CJ6 P6250` | 40 both |

Note the recycled pairs are frequently a put and a call at unrelated strikes, so a
mis-route does not merely shift a strike — it can price a leg off a completely different
instrument.

> An earlier check appeared to disprove this, but only because it scanned the definition
> files for the violating month. The competing segment's definition record is published in
> a **later** snapshot file (`E1AV2 P2880` appears only in the October 2022 file), so the
> full 316-file scan is required to see the tie.

### The authoritative resolution signal

`activation`/`expiration` describe the **contract's** lifetime, not the period during which
the id referred to it, which is why they overlap and tie. The reliable signal is each
definition record's own **`ts_event`** — when the exchange published that id → contract
mapping. The mapping in force for a bar is the latest one published at or before it.

This is better than parsing the snapshot filename: it needs no filename convention and
handles a reassignment that happens part-way through a range file.

Publication windows turn out to be **disjoint per id**, so there is no tie left to break:

```
instrument_id 116864
  rank 0  E1CU2 P4060   pub [begin        -> 2022-10-02 16:01:37)   act=2022-08-03 exp=2022-09-07
  rank 1  E1AV2 P2880   pub [2022-10-02   -> 2023-02-09 23:55:54)   act=2022-08-29 exp=2022-10-03
  rank 2  E4AH3 C4375   pub [2023-02-09   -> end               )   act=2023-02-07 exp=2023-03-27
```

The activation windows overlap (2022-08-29 → 09-07) and are both 35 days, which is what
made the old tie-break a coin flip. The publication windows do not overlap at all.
`E1AV2 P2880` was first published 2022-10-02 — a month *after* the 2022-09-01 bar — even
though its activation precedes it.

Implemented and verified against real vendor definition files: all six probe cases route
correctly, including bars inside the later contract's window, with `n_overlap = 0`.

### Impact

The exit scanner filters leg bars to `ts_event::date <= leg_expiration`
([exit.py:1105-1110](../btkit/backtest/exit.py#L1105)), which blocks a *later*-expiring
contract's bars from leaking into an earlier position, but not the reverse: a bar
belonging to a shorter-dated contract can still be priced into a longer-dated leg. With
403 violating id-days in 2026, this reaches the live-comparison window.

---

## 6. Remediation plan

Phases 1 and 2 are independent of the exit redesign and should land first.

### Phase 1 — Add audit coverage (do first, to verify the fix)

New flags in a new `phase5_duplication` rule module:

| Flag | Severity | Check |
|---|---|---|
| `DUPLICATE_BAR` | hard | >1 `option_bars` row per `(instrument_id, ts_event)` |
| `ID_DAY_COLLISION` | hard | >1 contract definition for one `(instrument_id, day)` — Databento's actual guarantee |

Do **not** flag an id merely for carrying several definitions over time: that is normal
recycling (105,899 ids do it). The flag must be scoped to a single day.

`option_audit` is keyed `(instrument_id, ts_event, flag_code)`, so both fit the existing
table. `underlying_bars` / `option_greeks` / `indicator_bars` multiplicity is not
per-option-instrument, so it belongs in the audit **report** rather than as flag rows.

### Phase 2 — Deduplicate the base tables

Recommended approach: **build a clean database file and swap**, rather than deduping in
place. DuckDB does not shrink a file after `DROP`, and a new file gives a clean rollback.
Disk is sufficient (193 GB free vs a 67 GB source).

Order matters:

1. `underlying_bars` — keep one row per `(instrument_id, ts_event)`.
2. `option_bars` — same, but handle the 2,474 ambiguous rows separately (below).
3. `option_greeks` — same key; or drop and recompute from the clean bars.
4. `indicator_bars` — **do not dedup; rebuild** (Phase 3).

```sql
CREATE TABLE option_bars_clean AS
SELECT * FROM option_bars
QUALIFY row_number() OVER (PARTITION BY instrument_id, ts_event) = 1;
```

**The 2,474 definition-conflict rows cannot be deduped by picking arbitrarily** — the two
rows describe different contracts. Options: (a) delete both copies, since the attribution
is ambiguous and 2,474 of 74.5M rows is negligible; (b) re-derive the correct contract
from the Databento definition files via the existing `_route_bars_to_segments`. (a) is
safest and is the recommendation.

### Phase 3 — Rebuild indicators

Mandatory, because the **values** are wrong rather than merely duplicated. After
`underlying_bars` is clean, delete and recompute `indicator_bars` for the affected
underlyings. This is the expensive step: 713M rows across 4,352 indicator definitions.
Scoping the rebuild to the indicators the live strategies actually use would cut this
sharply.

### Phase 4 — Fix the ingest defect

Two changes, both needed:

- **Structural guarantee (recommended):** add `PRIMARY KEY` or a `UNIQUE` index on
  `(instrument_id, ts_event)` for `underlying_bars`, `option_bars` and `option_greeks`,
  and `(indicator_id, ts_event)` for `indicator_bars`, then switch the writers to
  `INSERT ... ON CONFLICT DO NOTHING`. This makes the defect impossible rather than
  merely checked. Cost: index build memory and some insert throughput; worth testing on
  a copy first.
- **Remove or repair the static cutoff fast path** in `_write_df`. Even with the
  constraint in place, the fast path's premise ("newer than the pre-run max is
  unambiguously new") is false whenever one run ingests overlapping sources.

### Phase 5 — Fix definition routing ✅ DONE

Cause confirmed (§5): systematic `(_dist, _win)` ties resolved by an unstable sort.

Implemented:

1. `_scan_definition_file` records each definition's `pub_from` (earliest `ts_event` at
   which it was published) and keys segments on full contract identity
   `(activation, expiration, raw_symbol)` rather than the validity window alone, which
   could collapse two contracts sharing a window.
2. `_order_segments` sorts an id's definitions by publication time; `_build_segments_frame`
   derives back-to-back windows `[_pub_from, _pub_to)` where each contract owns the id
   until the next mapping is published. The first segment opens at the beginning of time so
   bars predating our definition coverage still classify; the last runs to the end of time.
3. `_route_bars_to_segments` prefers the segment whose publication window contains the bar,
   falling back to the old `[activation, expiration]` distance rule only for ids with no
   publication timestamp, and using `_seg_rank` as a final tie-break so the outcome can
   never depend on row order.

Still outstanding:

- **Re-ingest the affected bars.** The fix corrects routing for *new* ingests; the 3,875
  existing violating id-days are repaired only by the rebuild (Phase 2/3).
- **Greeks must be recomputed** for any bar whose attribution changes: strike, right and
  expiration feed the Black-76 inputs, and delta drives leg selection, so entries can move.
- Consider an ingest-time assertion that no `(instrument_id, bar_date)` resolves to two
  definitions. The `ID_DAY_COLLISION` audit flag already detects it after the fact.

---

## 7. Synthetic contract id

A stable, synthetic contract key — derived from `(symbol, expiration)` — is worth adding,
and the rebuild is the right moment to do it. The justification is **correctness**; the
performance gain is real but modest and narrow.

### The proposed key validates cleanly

| Test | Result |
|---|---|
| `(symbol, expiration)` → >1 `(strike, right, multiplier)` | **0 violations** |
| `(symbol, expiration)` → >1 `underlying_id` | **0 violations** |
| Distinct contracts under `(symbol, expiration)` | 665,028 |
| Distinct `instrument_id` values | 538,686 |

So 126,342 contracts (≈23%) share an `instrument_id` with at least one other contract.

`symbol` **alone is not sufficient**: 303 symbols span more than one expiration, 240 of
them with a gap of 3,647–3,655 days — the CME 10-year symbol cycle (`EW2H5 P2100` expires
both 2015-03-13 and 2025-03-14). Adding `expiration` resolves these.

**Futures symbols recycle the same way** — `ESH4` → 2014-03-21 and 2024-03-15, two
`instrument_id`s. So `underlying_id` is *also* an unstable key, and the futures side needs
`(symbol, expiration)` too.

> **Caveat:** 63 symbols, all in the `EW4Z5` (week 4, Dec 2015) series, carry expirations
> exactly one day apart — 2015-12-23 and 2015-12-24 — with bars split between them. This
> looks like a Christmas holiday adjustment recorded inconsistently across snapshots. A
> `(symbol, expiration)` key would split each into two contracts. Isolated, pre-2016, and
> outside any current backtest window, but worth normalising or flagging.

### `(instrument_id, ts_event)` *is* the right uniqueness key

An earlier draft of this section argued that the 2,474 same-id/same-timestamp rows were
legitimately distinct contracts, so a unique constraint on `(instrument_id, ts_event)`
would wrongly drop one. **That was wrong.** All 2,474 carry **identical OHLCV**:

```
def_conflicts = 2474   def_conflict_same_ohlcv = 2474   def_conflict_diff_ohlcv = 0
```

Databento's OHLCV feed emits exactly one bar per `(instrument_id, minute)`, so two
contracts cannot share an id within one bar. These rows are **one source bar ingested
twice and routed to a different definition each time** — a compound of defect 1 and
defect 2, not two real bars.

Consequences:

- `UNIQUE (instrument_id, ts_event)` is a correct and sufficient constraint for
  `option_bars`, so the §6 Phase 4 recommendation stands.
- But **deduping these 2,474 keys by picking an arbitrary copy would preserve an arbitrary
  attribution.** Routing (Phase 5) must be fixed *before or during* the rebuild, so each
  source bar has one correct definition. Dedup alone is not enough for them.
- `contract_id` is therefore justified by the reasons below — downstream simplification,
  table width, cross-database stability — and **not** as the uniqueness key.

### What it buys

**Correctness (the main reason).** The recycling class of bug becomes structurally
impossible downstream. Two defensive filters exist *solely* because `instrument_id` is
unstable, and both could be removed:

- the post-expiration filter in `_find_first_hit` ([exit.py:164-166](../btkit/backtest/exit.py#L164))
- the equivalent in `_adjust_leg_out_exits` ([exit.py:1105-1110](../btkit/backtest/exit.py#L1105))

**Performance — modest, and not where you might expect.**

| | Effect |
|---|---|
| Bar/greeks joins | **No gain.** An integer join on `contract_id` costs the same as on `instrument_id`. |
| Removing the defensive filters | **Real gain.** Both do `ts_event.convert_time_zone(tz).dt.date()` over every bar of every cohort; tz conversion at that volume is expensive. |
| Fact-table width | **Real gain.** At 112 bars per contract, `symbol`/`expiration`/`strike_price`/`right`/`multiplier`/`underlying_id` are duplicated ~112×. Moving them to a dimension table cuts scan I/O on a 67 GB database. |
| Strike/greeks search | **No gain.** `_greeks_for_strike_legs_tolerant` and per-cohort bar loading dominate runtime, and neither changes. |

### Design — implemented ✅

- **`btkit/db/contract_key.py`** — `contract_id(symbol, expiration)` returns a 63-bit
  blake2b digest. Deterministic rather than a sequence, because `btkit study merge`
  ([study/merger.py](../btkit/study/merger.py)) combines separately-built databases and a
  sequence would give the same contract different ids in each. blake2b rather than the
  built-in `hash()`, whose string hashing is randomised per process. A pinned regression
  value in the tests makes an accidental hash change impossible to miss.
- **`contract` dimension table** with `contract_id` as `PRIMARY KEY` and
  `UNIQUE (symbol, expiration)`. The primary key means a hash collision fails the insert
  rather than silently merging two contracts. Populated for futures as well as options,
  since futures symbols recycle on the same 10-year cycle.
- **`contract_id` added to `option_bars` and `option_greeks`**, each with a
  `(contract_id, ts_event)` index. Computed once per segment in `_build_segments_frame`
  (~665k contracts) rather than per bar (74.5M), and inherited by bars through the routing
  join. `GreeksCalculator` carries it from `option_bars` into `option_greeks`.
- **`instrument_id` is retained** as provenance: ingest needs it to join OHLCV to
  definitions, and `position_leg` / the analysis UI reference it
  ([chart.py:247](../btkit/analysis/api/routes/chart.py#L247)).
- **Attribution is resolved once, at ingest**, by publication-time routing (§6 Phase 5).

### Deliberately deferred

- **The exit scanner still joins on `instrument_id`** and keeps its defensive filters.
  Switching it is a behaviour change that cannot be validated against the current database
  — the data is corrupt until the rebuild — so it should follow the rebuild and be A/B'd
  against the pre-change results.
- **The two defensive filters stay for now.** Recycling is no longer their justification,
  but they still screen out genuine zombie bars (358,540 `ZOMBIE_BAR` flags: bars dated
  after their own contract's expiration). Remove them only once ingest or the audit
  excludes those at source. This means the scan-time saving from dropping the two
  timezone-date conversions is not yet realised.
- **Normalisation is not done.** `option_bars` still carries `symbol`, `expiration`,
  `strike_price`, `right`, `multiplier` and `underlying_id` per row, duplicated ~112× per
  contract. Moving them into the dimension would cut scan I/O but touches most queries in
  `input_db.py`, so it is a separate change.
- **`underlying_bars` has no `contract_id`.** The futures contracts are in the dimension,
  but wiring the column through would also mean revisiting `indicator_definition`, which is
  keyed on the underlying's `instrument_id`.

---

## 8. Open decisions

1. **Dedup strategy** — clean-copy-and-swap (recommended) or in-place?
2. **The 2,474 ambiguous rows** — delete both copies (recommended) or re-derive from the
   definition files?
3. **Indicator rebuild scope** — all 4,352 definitions, or only those the live strategies
   reference?
4. **Constraints** — add `UNIQUE (contract_id, ts_event)` (recommended, see §7) or only
   repair the cutoff logic?
6. **Synthetic contract id** — adopt it as part of this rebuild (recommended) or keep
   `instrument_id` as the only key and retain the defensive filters?
5. **Verification** — re-run a known backtest before and after to quantify how much the
   volume gate and indicator conditions shifted results?
