"""
Unit tests for parse_condition arithmetic and abs() extensions.
"""
from __future__ import annotations

import polars as pl
import pytest

from btkit.strategy.loader import parse_condition


def _eval(expr_str: str, **col_values: float) -> bool:
    """Evaluate a condition string against a single-row DataFrame."""
    df = pl.DataFrame({k: [v] for k, v in col_values.items()})
    result = df.select(parse_condition(expr_str).alias("result"))["result"][0]
    return bool(result)


class TestArithmetic:

    def test_subtraction_in_comparison(self):
        assert _eval("a - b >= 1.0", a=3.0, b=1.5) is True
        assert _eval("a - b >= 1.0", a=2.0, b=1.5) is False

    def test_addition_in_comparison(self):
        assert _eval("a + b < 5.0", a=2.0, b=2.5) is True
        assert _eval("a + b < 5.0", a=2.5, b=3.0) is False

    def test_multiplication(self):
        assert _eval("a * 2.0 > 5.0", a=3.0) is True
        assert _eval("a * 2.0 > 5.0", a=2.0) is False

    def test_division(self):
        assert _eval("a / b < 1.0", a=1.0, b=2.0) is True
        assert _eval("a / b < 1.0", a=2.0, b=1.0) is False

    def test_compound_arithmetic(self):
        # (3.0 - 1.0) >= 2.0 * 0.5 → 2.0 >= 1.0 → True
        assert _eval("a - b >= 2.0 * c", a=3.0, b=1.0, c=0.5) is True

    def test_mtm_gain_condition(self):
        # position_mark - open_mark >= 2.0 * abs(open_mark)
        # 5.0 - (-2.0) = 7.0 >= 2.0 * 2.0 = 4.0 → True
        assert _eval(
            "position_mark - open_mark >= 2.0 * abs(open_mark)",
            position_mark=5.0,
            open_mark=-2.0,
        ) is True
        # 1.0 - (-2.0) = 3.0 < 2.0 * 2.0 = 4.0 → False
        assert _eval(
            "position_mark - open_mark >= 2.0 * abs(open_mark)",
            position_mark=1.0,
            open_mark=-2.0,
        ) is False


class TestAbsFunction:

    def test_abs_positive(self):
        assert _eval("abs(a) < 0.30", a=0.20) is True
        assert _eval("abs(a) < 0.30", a=0.35) is False

    def test_abs_negative_value(self):
        assert _eval("abs(a) < 0.30", a=-0.20) is True
        assert _eval("abs(a) < 0.30", a=-0.35) is False

    def test_abs_in_arithmetic(self):
        # 2.0 * abs(-3.0) = 6.0 > 5.0 → True
        assert _eval("2.0 * abs(a) > 5.0", a=-3.0) is True

    def test_abs_of_expression(self):
        # abs(a - b) = abs(1.0 - 3.0) = 2.0 >= 1.5 → True
        assert _eval("abs(a - b) >= 1.5", a=1.0, b=3.0) is True


class TestArithmeticWithBooleans:

    def test_arithmetic_combined_with_and(self):
        assert _eval("a - b > 1.0 and c < 10.0", a=5.0, b=2.0, c=8.0) is True
        assert _eval("a - b > 1.0 and c < 10.0", a=5.0, b=2.0, c=12.0) is False

    def test_arithmetic_combined_with_or(self):
        assert _eval("a * 2.0 > 10.0 or b < 0.0", a=6.0, b=1.0) is True
        assert _eval("a * 2.0 > 10.0 or b < 0.0", a=4.0, b=-1.0) is True
        assert _eval("a * 2.0 > 10.0 or b < 0.0", a=4.0, b=1.0) is False


class TestUnsupportedSyntax:

    def test_unsupported_operator_raises(self):
        with pytest.raises(ValueError, match="Unsupported arithmetic operator"):
            parse_condition("a ** 2 > 4")

    def test_unsupported_function_raises(self):
        with pytest.raises(ValueError, match="Only abs\\(\\) is supported"):
            parse_condition("min(a, b) > 0")

    def test_unknown_function_raises(self):
        with pytest.raises(ValueError, match="Only abs\\(\\) is supported"):
            parse_condition("sqrt(a) > 1.0")
