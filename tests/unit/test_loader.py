"""
Unit tests for the strategy YAML loader and condition parser.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from btkit.strategy.loader import load_strategy, parse_condition

STRATEGIES_DIR = Path(__file__).parent.parent / "fixtures" / "strategies"


# ---------------------------------------------------------------------------
# load_strategy
# ---------------------------------------------------------------------------


class TestLoadStrategy:
    def test_loads_valid_yaml(self):
        strat = load_strategy(STRATEGIES_DIR / "short_put_spread.yaml")
        assert strat.name is not None
        assert len(strat.trades) > 0

    def test_rejects_parameterized_strategy(self, tmp_path):
        yaml_content = """
strategy:
  name: sweep_test
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: trade1
      instrument:
        root_symbol: ES
        asset_class: future
      entry:
        window:
          start: "10:00"
          end:   "12:00"
      legs:
        - name: short_put
          right:  put
          action: sell_to_open
          dte:    21
          delta:  [-0.20, -0.25]
      exit:
        stop_loss:   2.0
        take_profit: 1.0
"""
        f = tmp_path / "sweep.yaml"
        f.write_text(yaml_content)
        with pytest.raises(ValueError, match="Matrix runs are not supported"):
            load_strategy(f)

    def test_missing_strategy_key_raises(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("name: oops\n")
        with pytest.raises(ValueError, match="top-level 'strategy' key"):
            load_strategy(f)

    def test_invalid_asset_class_raises(self, tmp_path):
        yaml_content = """
strategy:
  name: bad_asset
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: trade1
      instrument:
        root_symbol: ES
        asset_class: crypto
      entry:
        window:
          start: "10:00"
          end:   "12:00"
      legs:
        - name: short_put
          right:  put
          action: sell_to_open
          dte:    21
          delta:  -0.25
      exit:
        stop_loss:   2.0
        take_profit: 1.0
"""
        f = tmp_path / "bad_asset.yaml"
        f.write_text(yaml_content)
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            load_strategy(f)


# ---------------------------------------------------------------------------
# parse_condition — comparisons
# ---------------------------------------------------------------------------


class TestParseConditionComparisons:
    """Test that parse_condition produces correct Polars filter results."""

    def _df(self, **cols) -> pl.DataFrame:
        return pl.DataFrame(cols)

    def test_greater_than(self):
        expr = parse_condition("sma_5 > 100.0")
        df = self._df(sma_5=[90.0, 100.0, 110.0])
        result = df.filter(expr)["sma_5"].to_list()
        assert result == [110.0]

    def test_less_than(self):
        expr = parse_condition("vix < 20")
        df = self._df(vix=[15.0, 20.0, 25.0])
        assert df.filter(expr)["vix"].to_list() == [15.0]

    def test_greater_equal(self):
        expr = parse_condition("vix >= 20")
        df = self._df(vix=[15.0, 20.0, 25.0])
        assert df.filter(expr)["vix"].to_list() == [20.0, 25.0]

    def test_less_equal(self):
        expr = parse_condition("vix <= 20")
        df = self._df(vix=[15.0, 20.0, 25.0])
        assert df.filter(expr)["vix"].to_list() == [15.0, 20.0]

    def test_equal(self):
        expr = parse_condition("status == 1")
        df = self._df(status=[0, 1, 2])
        assert df.filter(expr)["status"].to_list() == [1]

    def test_not_equal(self):
        expr = parse_condition("status != 0")
        df = self._df(status=[0, 1, 2])
        assert df.filter(expr)["status"].to_list() == [1, 2]

    def test_negative_literal(self):
        expr = parse_condition("delta > -0.30")
        df = self._df(delta=[-0.40, -0.25, -0.10])
        assert df.filter(expr)["delta"].to_list() == [-0.25, -0.10]

    def test_integer_literal(self):
        expr = parse_condition("dte > 5")
        df = self._df(dte=[3, 5, 7])
        assert df.filter(expr)["dte"].to_list() == [7]


# ---------------------------------------------------------------------------
# parse_condition — boolean operators
# ---------------------------------------------------------------------------


class TestParseConditionBooleans:
    def _df(self, **cols) -> pl.DataFrame:
        return pl.DataFrame(cols)

    def test_and(self):
        expr = parse_condition("sma_5 > 100 and vix < 20")
        df = self._df(sma_5=[110.0, 110.0, 90.0], vix=[15.0, 25.0, 15.0])
        result = df.filter(expr)
        assert len(result) == 1
        assert result["sma_5"][0] == 110.0
        assert result["vix"][0] == 15.0

    def test_or(self):
        expr = parse_condition("sma_5 > 110 or vix < 15")
        df = self._df(sma_5=[120.0, 100.0, 90.0], vix=[20.0, 10.0, 20.0])
        result = df.filter(expr)
        assert len(result) == 2

    def test_not(self):
        expr = parse_condition("not vix > 20")
        df = self._df(vix=[15.0, 20.0, 25.0])
        result = df.filter(expr)["vix"].to_list()
        assert result == [15.0, 20.0]

    def test_chained_and(self):
        expr = parse_condition("a > 0 and b > 0 and c > 0")
        df = self._df(a=[1, 1, -1], b=[1, -1, 1], c=[1, 1, 1])
        result = df.filter(expr)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# parse_condition — leg property dot-notation
# ---------------------------------------------------------------------------


class TestParseConditionDotNotation:
    def test_leg_property_reference(self):
        # "short_put.delta > -0.30" → col("short_put_delta") > -0.30
        expr = parse_condition("short_put.delta > -0.30")
        df = pl.DataFrame({"short_put_delta": [-0.40, -0.25, -0.10]})
        result = df.filter(expr)["short_put_delta"].to_list()
        assert result == [-0.25, -0.10]

    def test_multiple_dot_references(self):
        expr = parse_condition("short_put.delta > -0.30 and long_put.delta > -0.20")
        df = pl.DataFrame(
            {
                "short_put_delta": [-0.25, -0.35, -0.25],
                "long_put_delta": [-0.15, -0.15, -0.25],
            }
        )
        result = df.filter(expr)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# parse_condition — error handling
# ---------------------------------------------------------------------------


class TestParseConditionErrors:
    def test_invalid_syntax_raises(self):
        with pytest.raises(ValueError, match="Invalid condition syntax"):
            parse_condition("sma_5 >")

    def test_chained_comparison_raises(self):
        with pytest.raises(ValueError, match="Chained comparisons"):
            parse_condition("0 < sma_5 < 100")

    def test_unsupported_node_raises(self):
        with pytest.raises(ValueError):
            parse_condition("sma_5 + 10 > 100")
