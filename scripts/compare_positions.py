"""
Compare position results between two dump directories (baseline vs perf branch).

Usage:
    python scripts/compare_positions.py /tmp/btkit_baseline /tmp/btkit_perf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl


def compare(dir_a: Path, dir_b: Path) -> None:
    files_a = {f.stem: f for f in dir_a.glob("*.parquet")}
    files_b = {f.stem: f for f in dir_b.glob("*.parquet")}

    all_match = True
    for name in sorted(files_a):
        if name not in files_b:
            print(f"[MISSING] {name} not in {dir_b}")
            all_match = False
            continue

        a = pl.read_parquet(files_a[name]).sort(["trade_name", "open_time"])
        b = pl.read_parquet(files_b[name]).sort(["trade_name", "open_time"])

        if a.shape == b.shape:
            # Row-wise comparison on the key columns
            key_cols = ["trade_name", "open_time", "exit_time", "exit_reason",
                        "open_mark", "exit_mark", "net_pnl"]
            key_cols = [c for c in key_cols if c in a.columns and c in b.columns]
            a_key = a.select(key_cols)
            b_key = b.select(key_cols)
            if a_key.equals(b_key):
                print(f"[OK]   {name}: {len(a)} positions — identical")
            else:
                all_match = False
                diffs = (
                    a_key
                    .with_row_index("row")
                    .join(b_key.with_row_index("row"), on="row", suffix="_b")
                    .filter(
                        pl.any_horizontal(
                            pl.col(c) != pl.col(f"{c}_b")
                            for c in key_cols
                            if c != "trade_name"
                        )
                    )
                )
                print(f"[DIFF] {name}: same count ({len(a)}) but {len(diffs)} rows differ")
                print(diffs)
        else:
            all_match = False
            print(f"[DIFF] {name}: {len(a)} vs {len(b)} positions (Δ={len(b)-len(a):+d})")
            # Show which open_times exist in one but not the other
            a_times = set(a["open_time"].to_list())
            b_times = set(b["open_time"].to_list())
            only_a = sorted(a_times - b_times)
            only_b = sorted(b_times - a_times)
            if only_a:
                print(f"  Only in baseline ({len(only_a)}):")
                for t in only_a:
                    row = a.filter(pl.col("open_time") == t).to_dicts()[0]
                    print(f"    {t}  exit_reason={row.get('exit_reason')}  net_pnl={row.get('net_pnl')}")
            if only_b:
                print(f"  Only in perf ({len(only_b)}):")
                for t in only_b:
                    row = b.filter(pl.col("open_time") == t).to_dicts()[0]
                    print(f"    {t}  exit_reason={row.get('exit_reason')}  net_pnl={row.get('net_pnl')}")

    print()
    if all_match:
        print("All strategies: IDENTICAL results.")
    else:
        print("Differences found — investigation needed.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline_dir> <perf_dir>")
        sys.exit(1)
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
