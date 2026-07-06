"""
StudyExpander — resolves a StudyDefinition into a flat list of
(combination_id, scalar StrategyDefinition) pairs.

For each strategy referenced by the study:
  - Scalar strategy        → one combination
  - Sweep-parameterized    → cartesian product of all swept fields
  - Explicit combinations  → one combination per entry (structured or table)

Combinations across strategies are concatenated (not crossed). combination_id
is 1-based and globally unique within the study.

Dot-path format for override keys:
  "{trade_name}.{leg_name}.{field}"   for leg parameters (delta, dte)
  "{trade_name}.exit.{field}"          for exit parameters
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import polars as pl

from btkit.strategy.definition import (
    SimpleDeltaConfig,
    SweepRange,
    StopLossConfig,
    StrategyDefinition,
    TableCombinations,
    TakeProfitConfig,
)
from btkit.strategy.loader import load_strategy
from btkit.study.definition import StudyDefinition


def _as_values(v: Any) -> list[Any] | None:
    """Return the list of values for a sweep field, or None if it is scalar."""
    if isinstance(v, list):
        return v
    if isinstance(v, SweepRange):
        return v.values()
    return None


class StudyExpander:
    """
    Expands a StudyDefinition into a flat list of (combination_id, StrategyDefinition).

    Usage:
        expander = StudyExpander(study, study_dir)
        for cid, defn in expander.combinations:
            engine = BacktestEngine(..., strategy=defn, combination_id=cid)
    """

    def __init__(
        self,
        study: StudyDefinition,
        study_dir: Path,
        max_combinations: int | None = None,
    ) -> None:
        self._study = study
        self._study_dir = study_dir
        # CLI --max-combinations overrides YAML study.max_combinations
        self._max_combinations = max_combinations or study.max_combinations
        self._combinations: list[tuple[int, StrategyDefinition]] | None = None
        self._params_rows: list[dict[str, Any]] = []

    @property
    def combinations(self) -> list[tuple[int, StrategyDefinition]]:
        """List of (combination_id, scalar StrategyDefinition). combination_id is 1-based."""
        if self._combinations is None:
            self._expand_all()
        return self._combinations  # type: ignore[return-value]

    @property
    def params_df(self) -> pl.DataFrame:
        """
        One row per combination: combination_id, strategy_name, plus one column
        per swept dot-path parameter. Scalar combinations have only the first
        two columns populated.
        """
        _ = self.combinations  # ensure expansion has run
        if not self._params_rows:
            return pl.DataFrame()
        return pl.DataFrame(self._params_rows)

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand_all(self) -> None:
        result: list[tuple[int, StrategyDefinition]] = []
        self._params_rows = []
        cid = 1

        for ref in self._study.strategies:
            strategy_path = self._study_dir / ref.path
            defn = load_strategy(strategy_path)

            if not defn.is_parameterized():
                result.append((cid, defn))
                self._params_rows.append({"combination_id": cid, "strategy_name": defn.name})
                cid += 1
            else:
                if defn.combinations is not None:
                    overrides_list = self._expand_combinations(defn)
                else:
                    overrides_list = self._expand_sweeps(defn)
                for overrides in overrides_list:
                    scalar = self._apply_overrides(defn, overrides)
                    result.append((cid, scalar))
                    row: dict[str, Any] = {
                        "combination_id": cid,
                        "strategy_name": defn.name,
                    }
                    row.update(overrides)
                    self._params_rows.append(row)
                    cid += 1

        total = len(result)
        if self._max_combinations is not None and total > self._max_combinations:
            raise ValueError(
                f"Study would produce {total:,} combinations but "
                f"max_combinations={self._max_combinations:,}. "
                "Reduce the sweep range or increase max_combinations."
            )

        self._combinations = result

    def _expand_sweeps(self, defn: StrategyDefinition) -> list[dict[str, Any]]:
        """
        Collect all swept fields as dot-paths → value lists, then return the
        cartesian product as a list of {dot_path: scalar_value} dicts.
        """
        sweep_axes: dict[str, list[Any]] = {}

        for trade in defn.trades:
            for leg in trade.legs:
                if isinstance(leg.delta, SimpleDeltaConfig):
                    vals = _as_values(leg.delta.target)
                    if vals is not None:
                        sweep_axes[f"{trade.name}.{leg.name}.delta.target"] = vals
                vals = _as_values(leg.dte)
                if vals is not None:
                    sweep_axes[f"{trade.name}.{leg.name}.dte"] = vals

            for fname in ("stop_loss", "take_profit", "take_profit_pct", "dte_exit"):
                v = getattr(trade.exit, fname)
                if isinstance(v, StopLossConfig):
                    v = v.price
                elif isinstance(v, TakeProfitConfig):
                    # Use a sub-path so _apply_overrides patches only the swept
                    # sub-field, preserving confirmation_bars and other attrs.
                    sub = "price" if v.price is not None else "pct"
                    v = v.price if v.price is not None else v.pct
                    if v is None:
                        continue
                    vals = _as_values(v)
                    if vals is not None:
                        sweep_axes[f"{trade.name}.exit.{fname}.{sub}"] = vals
                    continue
                if v is None:
                    continue
                vals = _as_values(v)
                if vals is not None:
                    sweep_axes[f"{trade.name}.exit.{fname}"] = vals

        if not sweep_axes:
            return [{}]

        keys = list(sweep_axes)
        return [dict(zip(keys, combo)) for combo in itertools.product(*[sweep_axes[k] for k in keys])]

    def _expand_combinations(self, defn: StrategyDefinition) -> list[dict[str, Any]]:
        """
        Convert explicit combinations (structured or table) to the same flat
        {dot_path: value} format produced by _expand_sweeps().
        """
        combs = defn.combinations

        if isinstance(combs, TableCombinations):
            # columns are already full dot-paths
            return [dict(zip(combs.columns, row)) for row in combs.rows]

        # Structured mode: list[dict[section_key, dict[field, value]]]
        # section_key is a leg name (globally unique across trades) or "exit".
        # Build leg→trade lookup for unambiguous resolution.
        leg_to_trade: dict[str, str] = {}
        for trade in defn.trades:
            for leg in trade.legs:
                if leg.name in leg_to_trade:
                    raise ValueError(
                        f"Leg name {leg.name!r} appears in multiple trades. "
                        "Use full dot-path columns in 'table' combination mode."
                    )
                leg_to_trade[leg.name] = trade.name

        result: list[dict[str, Any]] = []
        for entry in combs:  # type: ignore[union-attr]
            flat: dict[str, Any] = {}
            for section_key, field_values in entry.items():
                if section_key == "exit":
                    if len(defn.trades) > 1:
                        raise ValueError(
                            "Structured combination 'exit' key is ambiguous for multi-trade "
                            "strategies. Use '{trade_name}.exit.{field}' dot-path format."
                        )
                    trade_name = defn.trades[0].name
                    for field, value in field_values.items():
                        flat[f"{trade_name}.exit.{field}"] = value
                elif section_key in leg_to_trade:
                    trade_name = leg_to_trade[section_key]
                    for field, value in field_values.items():
                        flat[f"{trade_name}.{section_key}.{field}"] = value
                else:
                    # section_key may already be a compound "{trade}.{leg_or_exit}" path
                    parts = section_key.split(".", 1)
                    if len(parts) == 2:
                        for field, value in field_values.items():
                            flat[f"{parts[0]}.{parts[1]}.{field}"] = value
                    else:
                        raise ValueError(
                            f"Unknown section key {section_key!r} in structured combination. "
                            "Must be a leg name, 'exit', or '{trade}.{leg}' format."
                        )
            result.append(flat)
        return result

    def _apply_overrides(
        self,
        defn: StrategyDefinition,
        overrides: dict[str, Any],
    ) -> StrategyDefinition:
        """
        Deep-copy defn via model_dump(), apply dot-path overrides, then
        re-validate with model_validate(). Re-validation ensures the result
        is a fully-scalar StrategyDefinition (no remaining SweepRange/list fields).

        Dot-path format:
          "{trade_name}.{leg_name}.delta.target" → trade.legs[j].delta.target
          "{trade_name}.{leg_name}.dte"         → trade.legs[j].dte
          "{trade_name}.exit.{field}"           → trade.exit.{field}
        """
        raw = defn.model_dump()
        trade_by_name = {t["name"]: i for i, t in enumerate(raw["trades"])}

        for dot_path, value in overrides.items():
            parts = dot_path.split(".", 3)
            if len(parts) < 3:
                raise ValueError(
                    f"Override dot-path must have at least 3 parts (trade.section.field): {dot_path!r}"
                )
            trade_name, section = parts[0], parts[1]
            if trade_name not in trade_by_name:
                raise ValueError(f"Unknown trade {trade_name!r} in dot-path {dot_path!r}")
            ti = trade_by_name[trade_name]

            if section == "exit":
                if len(parts) == 4:
                    # 4-part path: trade.exit.field.subfield — patch inside a
                    # nested config object (e.g. take_profit.pct inside TakeProfitConfig).
                    # null sweep value disables the entire parent feature (take_profit=None).
                    field, sub_field = parts[2], parts[3]
                    if value is None:
                        raw["trades"][ti]["exit"][field] = None
                    else:
                        existing = raw["trades"][ti]["exit"].get(field)
                        if isinstance(existing, dict):
                            raw["trades"][ti]["exit"][field] = dict(existing, **{sub_field: value})
                        else:
                            raw["trades"][ti]["exit"][field] = {sub_field: value}
                else:
                    field = parts[2]
                    raw["trades"][ti]["exit"][field] = value
            else:
                leg_by_name = {
                    leg["name"]: j for j, leg in enumerate(raw["trades"][ti]["legs"])
                }
                if section not in leg_by_name:
                    raise ValueError(f"Unknown leg {section!r} in dot-path {dot_path!r}")
                lj = leg_by_name[section]
                if len(parts) == 4:
                    field, sub_field = parts[2], parts[3]
                    existing = raw["trades"][ti]["legs"][lj].get(field, {})
                    if isinstance(existing, dict):
                        raw["trades"][ti]["legs"][lj][field] = dict(existing, **{sub_field: value})
                    else:
                        raw["trades"][ti]["legs"][lj][field] = {sub_field: value}
                else:
                    raw["trades"][ti]["legs"][lj][parts[2]] = value

        return StrategyDefinition.model_validate(raw)
