"""
Replay stored backtests against two input databases and diff the results.

Every `backtest` row keeps a fully-resolved scalar `strategy_params`, so an
individual backtest can be re-run verbatim without expanding the sweep it came
from. That is what makes an A/B affordable: 1,301 backtests at ~36s each would be
~26 hours per side, but a stratified sample of ~10% is a couple of hours.

    # list what would be sampled, run nothing
    python scripts/ab_replay.py plan --source <study.db> [--tier all]

    # replay a sample against both databases
    python scripts/ab_replay.py run --source <study.db> \
        --old-db <es.db> --new-db <es_rebuild.db> --out-dir <dir> [--limit N]

    # diff the two result sets
    python scripts/ab_replay.py compare --out-dir <dir>

Both sides run the same btkit code; the only variable is the input database. See
docs/data_duplication_fix.md for what changed between them.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import yaml

# Fields that make a definition a *study* rather than a single backtest. The stored
# params carry them, and leaving them in would re-expand the sweep into hundreds of
# combinations instead of replaying the one.
SWEEP_KEYS = ("matrix", "combinations")

# Indicators whose values changed between the two databases, from the step-1 data
# diff. Backtests gating on ves1d are a different risk class (22.4% of its values
# moved) from those gating on the others (<=0.07%).
HIGH_DRIFT_INDICATORS = ("ves1d",)

# How many of the low-drift gated backtests to spot-check. Their gating indicators
# moved <=0.07%, so exhaustive coverage buys little, and at a measured ~600s each
# for both sides they are by far the most expensive backtests in the population.
LOW_DRIFT_SAMPLE = 4


def _is_high_drift(bt: dict) -> bool:
    return any(ind in cond for cond in bt["indicators"] for ind in HIGH_DRIFT_INDICATORS)


# ---------------------------------------------------------------------------
# Loading and normalising stored definitions
# ---------------------------------------------------------------------------


def _load_backtests(source: str) -> list[dict]:
    con = duckdb.connect(source, read_only=True)
    rows = con.execute(
        "SELECT id, study_id, strategy_name, strategy_params, duration_s "
        "FROM backtest WHERE status = 'completed' ORDER BY id"
    ).fetchall()
    con.close()
    out = []
    for bid, sid, name, params, dur in rows:
        p = json.loads(params) if isinstance(params, str) else params
        out.append(
            {
                "id": bid,
                "study_id": sid,
                "name": name,
                "params": p,
                "duration_s": dur or 0.0,
                "indicators": _conditions(p),
            }
        )
    return out


def _conditions(params: dict) -> list[str]:
    """Every entry/exit condition expression in a definition."""
    found: list[str] = []
    for tr in params.get("trades", []) or []:
        for section in ("entry", "exit"):
            found.extend((tr.get(section) or {}).get("conditions", []) or [])
    return found


def _to_strategy_yaml(params: dict) -> str:
    """Render a stored definition as a single-backtest strategy YAML.

    Strips the sweep fields and drops nulls: the resolved JSON spells out every
    optional field as null, and feeding those back makes the loader reject values
    that it would otherwise default.
    """
    clean = {k: v for k, v in params.items() if k not in SWEEP_KEYS}
    # The loader requires the definition under a top-level 'strategy' key; the stored
    # params are the inner mapping only.
    return yaml.safe_dump(
        {"strategy": _drop_nulls(clean)}, sort_keys=False, default_flow_style=False
    )


def _drop_nulls(obj):
    if isinstance(obj, dict):
        return {k: _drop_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_nulls(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def build_sample(backtests: list[dict], seed: int = 20261006) -> list[dict]:
    """Stratified sample: every indicator-gated backtest, plus a spread of the rest.

    Indicator-gated ones are taken whole because they are few and carry the most
    risk. The remainder is sampled per study so the parameter space (delta, DTE,
    stop, take-profit, width) stays covered rather than clustering in one sweep.
    """
    rng = random.Random(seed)
    gated = [b for b in backtests if b["indicators"]]
    plain = [b for b in backtests if not b["indicators"]]

    by_study: dict[int, list[dict]] = defaultdict(list)
    for b in plain:
        by_study[b["study_id"]].append(b)

    # Step 1's data diff measured how much each gating indicator actually moved:
    # ves1d changed 22.4% of its values, while elevated_risk, pvwap20 and slope20
    # changed <=0.07%. So the ves1d-gated backtests are taken whole and the rest are
    # spot-checked — they are also the slowest in the population (~156s each), so
    # sampling them is what keeps the run affordable.
    high_drift = [b for b in gated if _is_high_drift(b)]
    low_drift = [b for b in gated if not _is_high_drift(b)]
    sample = list(high_drift)
    sample.extend(rng.sample(low_drift, min(LOW_DRIFT_SAMPLE, len(low_drift))))
    for sid in sorted(by_study):
        group = sorted(by_study[sid], key=lambda b: b["id"])
        # Roughly 2% of a large sweep, floor 2 and ceiling 3. Combinations within one
        # sweep are near-duplicates, and measured cost is ~140s per backtest for both
        # sides, so extra draws cost real time and add little coverage.
        take = max(2, min(3, round(len(group) * 0.02)))
        sample.extend(rng.sample(group, min(take, len(group))))

    for b in sample:
        b["tier"] = (
            "A-gated-high-drift"
            if _is_high_drift(b)
            else "A-gated-low-drift"
            if b["indicators"]
            else "B-random"
        )
    return sorted(sample, key=lambda b: (b["tier"], b["id"]))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_plan(args) -> int:
    backtests = _load_backtests(args.source)
    sample = build_sample(backtests)
    tiers: dict[str, list[dict]] = defaultdict(list)
    for b in sample:
        tiers[b["tier"]].append(b)

    print(f"population: {len(backtests)} completed backtests")
    print(f"sample:     {len(sample)} ({100 * len(sample) / len(backtests):.1f}%)\n")
    print(f"  {'tier':22} {'count':>6} {'est. seconds (both sides)':>26}")
    total = 0.0
    for tier in sorted(tiers):
        secs = sum(b["duration_s"] for b in tiers[tier]) * 2
        total += secs
        print(f"  {tier:22} {len(tiers[tier]):>6} {secs:>26,.0f}")
    print(f"\n  estimated wall time: {total / 3600:.1f} h (sequential)")
    print("\nconditions represented:")
    for cond in sorted({c for b in sample for c in b["indicators"]}):
        n = sum(1 for b in sample if cond in b["indicators"])
        print(f"  [{n:>3}]  {cond}")
    return 0


def cmd_run(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backtests = _load_backtests(args.source)
    sample = build_sample(backtests)
    if args.limit:
        sample = sample[: args.limit]

    manifest_path = out_dir / "manifest.json"
    manifest: list[dict] = []
    yaml_dir = out_dir / "strategies"
    yaml_dir.mkdir(exist_ok=True)

    for i, b in enumerate(sample, 1):
        strategy_file = yaml_dir / f"bt{b['id']}.yaml"
        strategy_file.write_text(_to_strategy_yaml(b["params"]))
        entry = {"id": b["id"], "tier": b["tier"], "name": b["name"], "sides": {}}
        for side, db in (("old", args.old_db), ("new", args.new_db)):
            out_db = out_dir / f"ab_{side}.db"
            t0 = time.perf_counter()
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "btkit.cli",
                    "run",
                    "--strategy",
                    str(strategy_file),
                    "--input-db",
                    db,
                    "--output-db",
                    str(out_db),
                    "--note",
                    f"ab:{side}:bt{b['id']}",
                ],
                capture_output=True,
                text=True,
            )
            entry["sides"][side] = {
                "rc": proc.returncode,
                "seconds": round(time.perf_counter() - t0, 1),
                "stderr": proc.stderr[-400:] if proc.returncode else "",
            }
        status = "ok" if all(s["rc"] == 0 for s in entry["sides"].values()) else "FAILED"
        manifest.append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(
            f"[{i}/{len(sample)}] bt{b['id']} {b['tier']:22} {status} "
            f"old={entry['sides']['old']['seconds']}s new={entry['sides']['new']['seconds']}s",
            flush=True,
        )
    return 0


def cmd_compare(args) -> int:
    out_dir = Path(args.out_dir)
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{out_dir / 'ab_old.db'}' AS o (READ_ONLY)")
    con.execute(f"ATTACH '{out_dir / 'ab_new.db'}' AS n (READ_ONLY)")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    tier_of = {f"ab:old:bt{m['id']}": m["tier"] for m in manifest}

    # Each replay is one backtest row per side, paired by the note we wrote.
    pairs = con.execute(
        """
        SELECT o.id AS old_id, n.id AS new_id, o.note AS note
        FROM o.backtest o
        JOIN n.backtest n
          ON replace(o.note, 'ab:old:', '') = replace(n.note, 'ab:new:', '')
        WHERE o.status = 'completed' AND n.status = 'completed'
        ORDER BY o.id
        """
    ).fetchall()

    print(
        f"{'backtest':>10} {'tier':22} {'pos old':>8} {'pos new':>8} "
        f"{'same':>7} {'chg':>6} {'only_o':>7} {'only_n':>7} "
        f"{'pnl old':>12} {'pnl new':>12} {'d%':>8}"
    )
    rows = []
    for old_id, new_id, note in pairs:
        stats = con.execute(
            """
            WITH op AS (SELECT trade_name, open_time, exit_time, exit_reason,
                               open_mark, exit_mark, net_pnl
                        FROM o.position WHERE backtest_id = ?),
                 np AS (SELECT trade_name, open_time, exit_time, exit_reason,
                               open_mark, exit_mark, net_pnl
                        FROM n.position WHERE backtest_id = ?)
            SELECT (SELECT count(*) FROM op) AS n_old,
                   (SELECT count(*) FROM np) AS n_new,
                   (SELECT count(*) FROM op JOIN np USING (trade_name, open_time)
                     WHERE op.exit_time = np.exit_time
                       AND op.exit_reason = np.exit_reason
                       AND abs(op.net_pnl - np.net_pnl) < 0.005) AS same,
                   (SELECT count(*) FROM op JOIN np USING (trade_name, open_time)
                     WHERE op.exit_time IS DISTINCT FROM np.exit_time
                        OR op.exit_reason IS DISTINCT FROM np.exit_reason
                        OR abs(op.net_pnl - np.net_pnl) >= 0.005) AS changed,
                   (SELECT count(*) FROM op WHERE NOT EXISTS
                     (SELECT 1 FROM np WHERE np.trade_name = op.trade_name
                        AND np.open_time = op.open_time)) AS only_old,
                   (SELECT count(*) FROM np WHERE NOT EXISTS
                     (SELECT 1 FROM op WHERE op.trade_name = np.trade_name
                        AND op.open_time = np.open_time)) AS only_new,
                   (SELECT coalesce(sum(net_pnl), 0) FROM op) AS pnl_old,
                   (SELECT coalesce(sum(net_pnl), 0) FROM np) AS pnl_new
            """,
            [old_id, new_id],
        ).fetchone()
        n_old, n_new, same, changed, only_old, only_new, pnl_old, pnl_new = stats
        dpct = (100.0 * (pnl_new - pnl_old) / abs(pnl_old)) if pnl_old else float("nan")
        tier = tier_of.get(note, "?")
        rows.append(
            (note, tier, n_old, n_new, same, changed, only_old, only_new, pnl_old, pnl_new, dpct)
        )
        print(
            f"{note.replace('ab:old:', ''):>10} {tier:22} {n_old:>8} {n_new:>8} "
            f"{same:>7} {changed:>6} {only_old:>7} {only_new:>7} "
            f"{pnl_old:>12,.0f} {pnl_new:>12,.0f} {dpct:>7.2f}%"
        )

    if rows:
        print("\n=== summary by tier ===")
        by_tier: dict[str, list] = defaultdict(list)
        for r in rows:
            by_tier[r[1]].append(r)
        for tier in sorted(by_tier):
            g = by_tier[tier]
            tot_pos = sum(r[2] for r in g)
            tot_same = sum(r[4] for r in g)
            print(
                f"  {tier:22} backtests={len(g):>4}  positions={tot_pos:>7}  "
                f"identical={100.0 * tot_same / tot_pos if tot_pos else 0:5.1f}%  "
                f"pnl_old={sum(r[8] for r in g):>14,.0f}  pnl_new={sum(r[9] for r in g):>14,.0f}"
            )
    con.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="Show the sample without running anything.")
    p.add_argument("--source", required=True)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="Replay the sample against both databases.")
    p.add_argument("--source", required=True)
    p.add_argument("--old-db", required=True)
    p.add_argument("--new-db", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("compare", help="Diff the two result sets.")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
