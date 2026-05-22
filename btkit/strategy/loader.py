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

    For MVP, also raises ValueError if the loaded strategy is parameterized
    (contains sweep params or explicit combinations). Matrix runs are deferred.
    """
    path = Path(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    strategy_data = raw.get("strategy")
    if strategy_data is None:
        raise ValueError(f"YAML file {path} must have a top-level 'strategy' key")

    definition = StrategyDefinition.model_validate(strategy_data)

    if definition.is_parameterized():
        raise ValueError(
            "Strategy contains sweep or combination parameters. "
            "Matrix runs are not supported in this version. "
            "Use scalar values for all leg and exit parameters."
        )

    return definition


def parse_condition(expr: str) -> pl.Expr:
    """
    Parse a condition string into a Polars expression.

    Supported syntax (MVP):
        Simple comparisons:  "rsi_14 < 40"
        Boolean operators:   "rsi_14 < 40 and vix_close < 30"
        Not:                 "not rsi_14 > 60"
        Leg properties:      "short_put.delta > -0.30"  (dot notation → col name)

    Identifiers containing a dot are treated as leg property references and
    resolved to column names of the form "<leg_name>_<field>". Identifiers
    without a dot are resolved directly as column names in the joined DataFrame
    (underlying bar columns or indicator columns).

    if/then syntax is not supported in this version.

    Raises ValueError for unsupported syntax or unrecognised operators.
    """
    raise NotImplementedError
