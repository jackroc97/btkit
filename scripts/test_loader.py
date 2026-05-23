"""Verify YAML loader and condition parser work with updated fixtures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from btkit.strategy.loader import load_strategy, parse_condition
import polars as pl

fixtures = Path("tests/fixtures/strategies")
for yaml_file in sorted(fixtures.glob("*.yaml")):
    try:
        strat = load_strategy(yaml_file)
        print(f"OK  {yaml_file.name}: {strat.name!r}, {len(strat.trades)} trade(s)")
        for trade in strat.trades:
            print(f"      trade={trade.name!r}, legs={[l.name for l in trade.legs]}")
    except Exception as e:
        print(f"ERR {yaml_file.name}: {e}")

print()

# Test condition parser
test_conditions = [
    "sma_5 > sma_20",
    "sma_5 > 0",
    "rsi_14 < 40 and vix_close < 30",
    "not rsi_14 > 60",
    "short_put.delta > -0.30",
    "close > 7000 or volume < 100",
]
print("Condition parser:")
for cond in test_conditions:
    try:
        expr = parse_condition(cond)
        print(f"  OK  {cond!r}")
    except Exception as e:
        print(f"  ERR {cond!r}: {e}")

print("\nAll done.")
