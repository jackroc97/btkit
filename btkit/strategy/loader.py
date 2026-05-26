"""
YAML strategy loader and condition parser.

load_strategy() is the main entry point — it reads a YAML file, validates it
against StrategyDefinition, and returns a ready-to-use model instance.

parse_condition() compiles a condition string (simple comparison or boolean
expression) into a Polars expression for vectorized evaluation. See docs/strategy.md
for supported syntax.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml

from btkit.strategy.definition import StrategyDefinition


def load_strategy(path: str | Path) -> StrategyDefinition:
    """
    Load and validate a strategy YAML file.

    Reads the YAML, extracts the top-level 'strategy' key, and parses the
    result into a StrategyDefinition. Pydantic validation runs at parse time —
    structural errors, invalid field values, and mutually-exclusive-field
    violations are all raised here as ValidationError.

    Parameterized strategies (sweep params or explicit combinations) are accepted;
    the caller is responsible for dispatching to BacktestEngine (scalar) or
    StudyExpander (parameterized).
    """
    path = Path(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    strategy_data = raw.get("strategy")
    if strategy_data is None:
        raise ValueError(f"YAML file {path} must have a top-level 'strategy' key")

    definition = StrategyDefinition.model_validate(strategy_data)
    return definition


def parse_condition(expr: str) -> pl.Expr:
    """
    Parse a condition string into a Polars expression.

    Supported syntax (MVP):
        Simple comparisons:  "rsi_14 < 40"
        Boolean operators:   "rsi_14 < 40 and vix_close < 30"
        Not:                 "not rsi_14 > 60"
        Leg properties:      "short_put.delta > -0.30"  → pl.col("leg_short_put_delta")
                             "short_put.strike < 5800"  → pl.col("leg_short_put_strike_price")

    Identifiers are resolved to column names. Dotted names (leg.field) are
    converted to underscore form (leg_field) for use in the wide joined DataFrame.

    Raises ValueError for unsupported syntax.
    """
    import ast as _ast

    try:
        tree = _ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid condition syntax: {expr!r}") from exc

    return _ast_to_polars(tree.body, expr)


def _ast_to_polars(node, source: str) -> pl.Expr:
    """Recursively convert an AST node to a Polars expression."""
    import ast as _ast

    if isinstance(node, _ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError(f"Chained comparisons are not supported: {source!r}")
        left = _ast_to_polars(node.left, source)
        right = _ast_to_polars(node.comparators[0], source)
        op = node.ops[0]
        if isinstance(op, _ast.Lt):
            return left < right
        if isinstance(op, _ast.Gt):
            return left > right
        if isinstance(op, _ast.LtE):
            return left <= right
        if isinstance(op, _ast.GtE):
            return left >= right
        if isinstance(op, _ast.Eq):
            return left == right
        if isinstance(op, _ast.NotEq):
            return left != right
        raise ValueError(f"Unsupported comparison operator in: {source!r}")

    if isinstance(node, _ast.BoolOp):
        parts = [_ast_to_polars(v, source) for v in node.values]
        result = parts[0]
        if isinstance(node.op, _ast.And):
            for p in parts[1:]:
                result = result & p
        elif isinstance(node.op, _ast.Or):
            for p in parts[1:]:
                result = result | p
        else:
            raise ValueError(f"Unsupported boolean operator in: {source!r}")
        return result

    if isinstance(node, _ast.UnaryOp):
        if isinstance(node.op, _ast.Not):
            return ~_ast_to_polars(node.operand, source)
        if isinstance(node.op, _ast.USub):
            # Negative numeric literal: -0.30 → pl.lit(-0.30)
            if isinstance(node.operand, _ast.Constant) and isinstance(
                node.operand.value, (int, float)
            ):
                return pl.lit(-node.operand.value)
        raise ValueError(f"Unsupported unary operator in: {source!r}")

    if isinstance(node, _ast.Name):
        return pl.col(node.id)

    if isinstance(node, _ast.Attribute):
        # short_put.delta → col("leg_short_put_delta")
        # short_put.strike → col("leg_short_put_strike_price")  (alias)
        if isinstance(node.value, _ast.Name):
            attr = "strike_price" if node.attr == "strike" else node.attr
            return pl.col(f"leg_{node.value.id}_{attr}")
        raise ValueError(f"Nested attribute access not supported: {source!r}")

    if isinstance(node, _ast.Constant):
        return pl.lit(node.value)

    raise ValueError(f"Unsupported expression node {type(node).__name__!r} in: {source!r}")
