"""
Pydantic models for strategy definition.

A StrategyDefinition contains one or more TradeDefinitions. Each trade is an
independent position structure with its own instrument, entry rules, legs, and
exit rules. Universe, costs, and matrix config are shared across all trades.

A StrategyDefinition is loaded from a YAML file by strategy.loader and passed
to BacktestEngine. For MVP all sweep fields (NumericSweep, IntSweep) must be
plain scalars — list or SweepRange values are rejected by the engine at run time.
"""

from __future__ import annotations

from datetime import date, time
from typing import Any, Literal

from pydantic import BaseModel, model_validator


# ---------------------------------------------------------------------------
# Sweep parameter types
# ---------------------------------------------------------------------------

class SweepRange(BaseModel):
    start: float
    stop: float
    step: float

    def values(self) -> list[float]:
        """Generates values from start to stop inclusive at the given step."""
        result = []
        v = self.start
        eps = abs(self.step) * 1e-9
        while (self.step > 0 and v <= self.stop + eps) or \
              (self.step < 0 and v >= self.stop - eps):
            result.append(round(v, 10))
            v += self.step
        return result


# Scalar, list, or range — valid for any numeric sweep field.
# For MVP all fields using these types must resolve to plain scalars.
NumericSweep = float | list[float] | SweepRange
IntSweep = int | list[int] | SweepRange


# ---------------------------------------------------------------------------
# Session / universe
# ---------------------------------------------------------------------------

class SessionConfig(BaseModel):
    timezone: str = "America/New_York"
    start_time: time = time(9, 30)
    end_time: time = time(16, 0)
    weekdays_only: bool = True
    skip_dates: list[date] = []


class UniverseConfig(BaseModel):
    start_date: date
    end_date: date
    session: SessionConfig = SessionConfig()


class InstrumentConfig(BaseModel):
    root_symbol: str
    asset_class: Literal["future", "equity", "etf"]


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

class EntryWindowConfig(BaseModel):
    start: time
    end: time

    @model_validator(mode="after")
    def start_before_end(self) -> EntryWindowConfig:
        if self.start >= self.end:
            raise ValueError("entry window start must be before end")
        return self


class EntryConfig(BaseModel):
    window: EntryWindowConfig
    conditions: list[str] = []
    min_credit: NumericSweep | None = None   # skip entry if open_mark < this
    max_debit: NumericSweep | None = None    # skip entry if open_mark > this


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

class LegConfig(BaseModel):
    name: str
    right: Literal["call", "put"]
    action: Literal["buy_to_open", "sell_to_open"]
    delta: NumericSweep
    dte: IntSweep
    quantity: int = 1


# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------

class ExitConfig(BaseModel):
    stop_loss: NumericSweep
    take_profit: NumericSweep
    dte_exit: IntSweep | None = None
    expiry_exit: bool = True
    conditions: list[str] = []   # OR logic — position closes if any condition is true


# ---------------------------------------------------------------------------
# Costs / matrix
# ---------------------------------------------------------------------------

class CostsConfig(BaseModel):
    slippage_pct: float = 0.0
    fee_per_contract: float = 0.0


class MatrixConfig(BaseModel):
    max_combinations: int = 100


# ---------------------------------------------------------------------------
# Explicit combinations (used by StrategyMatrix — deferred for MVP)
# ---------------------------------------------------------------------------

# Structured mode: leg/section name → parameter overrides
# e.g. {"short_put": {"delta": -0.20, "dte": 30}, "exit": {"stop_loss": 1.50}}
StructuredCombination = dict[str, dict[str, Any]]


class TableCombinations(BaseModel):
    mode: Literal["table"]
    columns: list[str]              # dot-path refs e.g. "short_put.delta"
    rows: list[list[float | int]]

    @model_validator(mode="after")
    def row_lengths_match_columns(self) -> TableCombinations:
        for i, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"row {i} has {len(row)} values but {len(self.columns)} columns defined"
                )
        return self


# ---------------------------------------------------------------------------
# Trade definition — one per independent position structure
# ---------------------------------------------------------------------------

class TradeDefinition(BaseModel):
    name: str
    instrument: InstrumentConfig
    entry: EntryConfig
    legs: list[LegConfig]
    exit: ExitConfig

    @model_validator(mode="after")
    def leg_names_unique(self) -> TradeDefinition:
        names = [leg.name for leg in self.legs]
        if len(names) != len(set(names)):
            raise ValueError("leg names must be unique within a trade")
        return self


# ---------------------------------------------------------------------------
# Top-level strategy definition
# ---------------------------------------------------------------------------

class StrategyDefinition(BaseModel):
    name: str
    version: str = "1.0"
    universe: UniverseConfig
    costs: CostsConfig = CostsConfig()
    matrix: MatrixConfig = MatrixConfig()
    trades: list[TradeDefinition]
    combinations: list[StructuredCombination] | TableCombinations | None = None

    @model_validator(mode="after")
    def trades_share_underlying(self) -> StrategyDefinition:
        """All trades must reference the same root_symbol."""
        symbols = {t.instrument.root_symbol for t in self.trades}
        if len(symbols) > 1:
            raise ValueError(
                f"all trades must share the same underlying; got {symbols}"
            )
        return self

    @model_validator(mode="after")
    def trade_names_unique(self) -> StrategyDefinition:
        names = [t.name for t in self.trades]
        if len(names) != len(set(names)):
            raise ValueError("trade names must be unique within a strategy")
        return self

    @model_validator(mode="after")
    def validate_combinations_vs_sweeps(self) -> StrategyDefinition:
        """Sweep parameters and explicit combinations are mutually exclusive."""
        if self.combinations is None:
            return self

        def is_sweep(v: Any) -> bool:
            return isinstance(v, (list, SweepRange))

        sweep_fields: list[str] = []
        for trade in self.trades:
            for leg in trade.legs:
                if is_sweep(leg.delta):
                    sweep_fields.append(f"trades[{trade.name}].legs[{leg.name}].delta")
                if is_sweep(leg.dte):
                    sweep_fields.append(f"trades[{trade.name}].legs[{leg.name}].dte")
            for fname in ("stop_loss", "take_profit", "dte_exit"):
                v = getattr(trade.exit, fname)
                if v is not None and is_sweep(v):
                    sweep_fields.append(f"trades[{trade.name}].exit.{fname}")

        if sweep_fields:
            raise ValueError(
                "combinations and sweep parameters are mutually exclusive. "
                f"Remove combinations or make these fields scalar: {sweep_fields}"
            )
        return self

    def is_parameterized(self) -> bool:
        """Returns True if the strategy contains any sweep or combination parameters."""
        if self.combinations is not None:
            return True

        def is_sweep(v: Any) -> bool:
            return isinstance(v, (list, SweepRange))

        for trade in self.trades:
            for leg in trade.legs:
                if is_sweep(leg.delta) or is_sweep(leg.dte):
                    return True
            for fname in ("stop_loss", "take_profit", "dte_exit"):
                v = getattr(trade.exit, fname)
                if v is not None and is_sweep(v):
                    return True
        return False
