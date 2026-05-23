"""
Dump detailed position-level results to Parquet for cross-branch comparison.

Usage:
    python scripts/dump_positions.py [output_dir]

Output: one parquet file per strategy in output_dir (default: /tmp/btkit_baseline/).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from btkit.backtest.engine import BacktestEngine
from btkit.db.input_db import InputDatabase
from btkit.db.output_db import OutputDatabase
from btkit.strategy.loader import load_strategy

INPUT_DB   = "/tmp/btkit_test_ingest.db"
STRATEGIES = sorted(Path("tests/fixtures/strategies").glob("*.yaml"))


def dump(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with InputDatabase(INPUT_DB) as idb:
        for strat_path in STRATEGIES:
            strategy = load_strategy(str(strat_path))
            with OutputDatabase(":memory:") as odb:
                odb.create_schema()
                engine = BacktestEngine(idb, odb, strategy, initial_equity=100_000.0)
                engine.run()
                positions = odb._con.execute(
                    "SELECT * FROM position ORDER BY trade_name, open_time"
                ).pl()

            out = output_dir / f"{strat_path.stem}.parquet"
            positions.write_parquet(out)
            print(f"  {strat_path.stem}: {len(positions)} positions → {out}")


if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/btkit_baseline")
    print(f"Dumping to {output_dir}/")
    dump(output_dir)
    print("Done.")
