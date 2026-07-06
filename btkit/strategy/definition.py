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
from typing import Any, Literal, Optional

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
        while (self.step > 0 and v <= self.stop + eps) or (self.step < 0 and v >= self.stop - eps):
            result.append(round(v, 10))
            v += self.step
        return result


# Scalar, list, or range — valid for any numeric sweep field.
# Lists may contain None to include "disabled" as one combination value,
# e.g. stop_loss: [null, 2.0] sweeps over {no stop loss, stop loss at 2.0}.
# For MVP all fields using these types must resolve to plain scalars.
NumericSweep = float | list[float | None] | SweepRange
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
    audit_filter: str | list[str] = "hard_errors_only"
    """
    Controls which option_audit flags cause an instrument to be excluded from
    entries.  Requires btkit audit to have been run against the input database
    first; silently ignored if the option_audit table is absent.

    Preset strings:
        "none"             — no filter; all instruments are eligible for entry.
        "hard_errors_only" — exclude instruments with any hard flag (default).
                             Hard flags: BARS_TRUNCATED, NEGATIVE_CLOSE,
                             NEGATIVE_DTE, ZOMBIE_BAR, DELTA_SIGN_ERROR,
                             DELTA_MAGNITUDE_ERROR.
        "strict"           — exclude instruments with any flag (hard or soft).

    Explicit list: e.g. ["BARS_TRUNCATED", "NEGATIVE_CLOSE"] — exclude only
    instruments flagged with those specific codes.
    """


class InstrumentConfig(BaseModel):
    root_symbol: str
    asset_class: Literal["future", "equity", "etf"]
    tick_size: float = 0.0  # minimum price increment; 0.0 = continuous (no rounding)
    expiry_close_time: time | None = (
        None  # local time after which expiry_exit triggers on expiration day
    )
    roll_days_before_expiry: int = (
        7  # futures only: roll to next contract this many days before front-month expiry
    )


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
    min_credit: NumericSweep | None = None  # skip entry if open_mark < this
    max_debit: NumericSweep | None = None  # skip entry if open_mark > this
    max_entries_per_day: int | None = None  # cap re-entries per calendar day (None = unlimited)
    no_reentry_after_loss: bool = False     # block same-day re-entry after a stop-loss exit
    time_tolerance: int = 0                 # seconds; how far an option greeks timestamp may differ from the candidate bar timestamp (0 = exact match)


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Delta configuration — simple or IV-stepped
# ---------------------------------------------------------------------------


class DeltaStep(BaseModel):
    """One bucket in an IV-stepped delta configuration."""
    below: float | None = None          # fire when iv_source < below; None = catch-all
    target: float                        # delta target for this bucket
    tolerance: float | None = None       # None → inherit SteppedDeltaConfig.tolerance


class SimpleDeltaConfig(BaseModel):
    """Fixed delta target, optionally sweepable."""
    target: NumericSweep                 # -0.25 scalar or [-0.20, -0.25] sweep list
    tolerance: float = 0.10             # ±band around target for candidate search


class SteppedDeltaConfig(BaseModel):
    """IV-conditioned delta: selects target/tolerance based on an indicator column."""
    step_source: str                     # indicator column name (e.g. "ves1d_close")
    tolerance: float = 0.10             # fallback tolerance for steps without explicit tolerance
    steps: list[DeltaStep]

    @model_validator(mode="after")
    def validate_steps(self) -> SteppedDeltaConfig:
        if not self.steps:
            raise ValueError("delta.steps must not be empty")
        catch_alls = [i for i, s in enumerate(self.steps) if s.below is None]
        if len(catch_alls) > 1:
            raise ValueError("at most one catch-all step (no below) is allowed in delta.steps")
        if catch_alls and catch_alls[0] != len(self.steps) - 1:
            raise ValueError("catch-all step (no below) must be last in delta.steps")
        return self


# ---------------------------------------------------------------------------
# Unified stepped leg configuration — indicator-conditioned (dte, delta) tuple
# ---------------------------------------------------------------------------


class SteppedStep(BaseModel):
    """
    One bucket in a unified stepped leg configuration.

    A step emits the full per-bucket selection tuple. `dte` and `delta` are
    required (each step is self-contained); tolerances are optional and fall
    back to defaults (delta_tolerance → 0.10; dte_tolerance → the leg's
    dte_tolerance).
    """
    below: float | None = None          # fire when source < below; None = catch-all
    dte: int                             # target days-to-expiry for this bucket
    delta: float                         # target delta for this bucket
    delta_tolerance: float | None = None  # None → 0.10 default
    dte_tolerance: int | None = None      # None → leg-level dte_tolerance (default 5)


class SteppedLegConfig(BaseModel):
    """
    Unified indicator-conditioned leg selection: one 1-D threshold selector that
    emits the full (dte, delta, tolerances) tuple per bucket from a single
    indicator source.

    Steps are evaluated in declaration order; the first whose `source < below`
    holds wins.  A single trailing step without `below` is the catch-all.  If no
    step matches and there is no catch-all, the entry is skipped.
    """
    source: str                          # indicator column name (e.g. "iv_percentile")
    steps: list[SteppedStep]

    @model_validator(mode="after")
    def validate_steps(self) -> SteppedLegConfig:
        if not self.steps:
            raise ValueError("stepped.steps must not be empty")
        catch_alls = [i for i, s in enumerate(self.steps) if s.below is None]
        if len(catch_alls) > 1:
            raise ValueError("at most one catch-all step (no below) is allowed in stepped.steps")
        if catch_alls and catch_alls[0] != len(self.steps) - 1:
            raise ValueError("catch-all step (no below) must be last in stepped.steps")
        return self


# ---------------------------------------------------------------------------
# Named conditional targets — multi-axis, priority-resolved leg selection
# ---------------------------------------------------------------------------


class LegTarget(BaseModel):
    """
    One named conditional selection for a leg.

    Non-`default` targets require both `condition` (a parse_condition string) and
    `priority` (unique across the map).  The reserved `default` target sets
    neither and is chosen only when no target's condition matches.

    `dte` and `delta` are required (each target is self-contained); tolerances
    fall back to defaults (delta_tolerance → 0.10; dte_tolerance → the leg's
    dte_tolerance).  `size_multiplier` is reserved for a future position-sizing
    feature — it is parsed and validated now but currently a no-op (a non-1.0
    value emits a warning).
    """
    priority: int | None = None          # required for non-default; forbidden for default
    condition: str | None = None         # required for non-default; forbidden for default
    dte: int                             # target days-to-expiry
    delta: float                         # target delta
    delta_tolerance: float | None = None  # None → 0.10 default
    dte_tolerance: int | None = None      # None → leg-level dte_tolerance (default 5)
    size_multiplier: float = 1.0          # reserved; scales quantity once sizing lands


# ---------------------------------------------------------------------------
# Leg
# ---------------------------------------------------------------------------


class LegConfig(BaseModel):
    name: str
    right: Literal["call", "put"]
    action: Literal["buy_to_open", "sell_to_open"]
    # dte is required for delta-targeted legs.
    # For offset legs it may be omitted (None) to inherit the reference leg's
    # expiration — the standard case for vertical spreads.  Specifying a value
    # reserves the leg for a future calendar-spread mode (not yet executed by
    # the engine, which currently always inherits the reference expiration).
    dte: IntSweep | None = None
    quantity: int = 1
    # Selection mode A: delta-targeted (standard or IV-stepped)
    delta: SimpleDeltaConfig | SteppedDeltaConfig | None = None
    dte_tolerance: int = 5  # ±band around target_dte for candidate search
    # Selection mode B: fixed strike offset from a reference leg
    strike_offset: float | None = None  # positive = above ref strike, negative = below
    reference_leg: str | None = None  # name of the leg whose strike is the origin
    # Selection mode C: unified indicator-conditioned (dte, delta) stepping
    stepped: SteppedLegConfig | None = None
    # Selection mode D: named, priority-resolved conditional targets
    targets: dict[str, LegTarget] | None = None

    @model_validator(mode="after")
    def validate_selection_mode(self) -> LegConfig:
        has_offset = self.strike_offset is not None
        has_delta = self.delta is not None
        has_stepped = self.stepped is not None
        has_targets = self.targets is not None
        if sum([has_offset, has_delta, has_stepped, has_targets]) == 0:
            raise ValueError("one of delta, strike_offset, stepped, or targets is required")
        if has_stepped and (has_delta or has_offset or has_targets):
            raise ValueError("stepped is mutually exclusive with delta and strike_offset")
        if has_targets and (has_delta or has_offset):
            raise ValueError("targets is mutually exclusive with delta and strike_offset")
        if has_stepped and self.dte is not None:
            raise ValueError(
                "stepped is mutually exclusive with leg-level dte; set dte on each step"
            )
        if has_targets and self.dte is not None:
            raise ValueError(
                "targets is mutually exclusive with leg-level dte; set dte on each target"
            )
        if has_offset and has_delta:
            raise ValueError("delta and strike_offset are mutually exclusive")
        if has_offset and self.reference_leg is None:
            raise ValueError("reference_leg is required when strike_offset is set")
        if not has_offset and not has_stepped and not has_targets and self.dte is None:
            raise ValueError("dte is required for delta-targeted legs")
        if has_delta and isinstance(self.delta, SteppedDeltaConfig) and has_offset:
            raise ValueError("iv_delta_steps cannot be used with strike_offset legs")
        return self

    @model_validator(mode="after")
    def validate_targets_map(self) -> LegConfig:
        if self.targets is None:
            return self
        if not self.targets:
            raise ValueError("targets must not be empty")
        priorities: list[int] = []
        n_conditional = 0
        for name, tgt in self.targets.items():
            if name == "default":
                if tgt.condition is not None or tgt.priority is not None:
                    raise ValueError(
                        "the reserved 'default' target must not set condition or priority"
                    )
                continue
            n_conditional += 1
            if tgt.condition is None:
                raise ValueError(f"target '{name}' requires a condition")
            if tgt.priority is None:
                raise ValueError(f"target '{name}' requires a priority")
            priorities.append(tgt.priority)
        if n_conditional == 0:
            raise ValueError("targets must define at least one conditional (non-default) target")
        if len(priorities) != len(set(priorities)):
            raise ValueError("target priorities must be unique across the map")
        return self


# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------


class StopLossConfig(BaseModel):
    price: NumericSweep
    condition: str | None = None  # AND-gated: SL only fires when this expression is also true


class TakeProfitConfig(BaseModel):
    price: NumericSweep | None = None  # fixed per-point offset from open_mark
    pct: NumericSweep | None = None  # fraction of open_mark to retain (e.g. 0.70 = exit at 70% profit)
    condition: str | None = None  # AND-gated: TP only fires when this expression is also true
    confirmation_bars: int = 1  # consecutive 1-min bars at/below TP level required before exit fires

    @model_validator(mode="after")
    def validate_tp_config(self) -> TakeProfitConfig:
        if self.price is None and self.pct is None:
            raise ValueError("one of price or pct is required in take_profit")
        if self.price is not None and self.pct is not None:
            raise ValueError("price and pct are mutually exclusive in take_profit")
        return self


class LiquidityConfig(BaseModel):
    """
    Controls execution realism for exit fills.  All fields default to None /
    "flat", which reproduces the original naïve-fill behaviour exactly.

    min_exit_volume:         Minimum cumulative option volume (contracts) over
                             the trailing ``lookback_minutes`` window required
                             before a price-triggered exit (TP/SL) is attempted.
                             Bars missing from the data count as 0 volume, so
                             gaps near expiry are naturally penalised.
                             None → volume gate disabled.

    lookback_minutes:        Time-based rolling window for the volume check.
                             Defaults to 3 minutes.

    pre_expiry_lock_minutes: Suppress all price-triggered exits (TP, SL, gap)
                             in the final N minutes before the instrument's
                             expiry close.  Expiry and DTE exits are unaffected.
                             None → lock disabled.

    max_leg_stale_minutes:   Suppress all price-triggered exits (TP, SL, gap)
                             when any leg's last real bar is older than this many
                             minutes.  The exit scanner forward-fills stale leg
                             prices, which creates artificially compressed spread
                             marks that fire spurious TP exits.  Setting this to
                             3–5 minutes blocks exits where the position mark is
                             not a reliable market observation.
                             None → staleness gate disabled.

    slippage_model:          "flat"   – use bar close as fill price (current
                                        behaviour, no extra cost).
                             "spread" – add half the bar's high-low range per
                                        leg to the fill price, modelling the
                                        cost of crossing the bid-ask spread.
    """

    min_exit_volume: Optional[int] = None
    lookback_minutes: int = 3
    pre_expiry_lock_minutes: Optional[int] = None
    max_leg_stale_minutes: Optional[int] = None
    slippage_model: Literal["flat", "spread"] = "flat"

    @property
    def needs_volume(self) -> bool:
        return self.min_exit_volume is not None

    @property
    def needs_spread(self) -> bool:
        return self.slippage_model == "spread"

    @property
    def needs_staleness(self) -> bool:
        return self.max_leg_stale_minutes is not None

    @property
    def is_default(self) -> bool:
        """True when no liquidity constraints are active (original engine behaviour)."""
        return (
            not self.needs_volume
            and not self.needs_spread
            and not self.needs_staleness
            and self.pre_expiry_lock_minutes is None
        )


class ExitConfig(BaseModel):
    stop_loss: NumericSweep | StopLossConfig | None = None
    take_profit: NumericSweep | TakeProfitConfig | None = None
    take_profit_pct: NumericSweep | None = None  # legacy top-level form; use take_profit.pct for new configs
    dte_exit: IntSweep | None = None
    vega_exit: NumericSweep | None = None  # exit when spread net vega < this value
    expiry_exit: bool = True
    conditions: list[str] = []  # OR logic — position closes if any condition is true
    liquidity: LiquidityConfig = LiquidityConfig()
    leg_out: bool = False  # when True, long leg(s) run to expiry after short leg exits at TP/SL
    allow_after_hours_exits: bool = False  # when True, TP/SL can trigger outside the session window
    on_sl_long_continuation: bool = False  # when True, long leg runs under trailing stop after SL
    long_trailing_stop_pct: float = 0.50   # trailing stop: exit if price pulls back this fraction from peak

    @model_validator(mode="after")
    def validate_take_profit(self) -> ExitConfig:
        has_tp = self.take_profit is not None
        has_tp_pct = self.take_profit_pct is not None
        if has_tp and has_tp_pct:
            raise ValueError("take_profit and take_profit_pct are mutually exclusive")
        return self

    @model_validator(mode="after")
    def validate_continuation(self) -> ExitConfig:
        if self.on_sl_long_continuation and self.leg_out:
            raise ValueError("on_sl_long_continuation and leg_out are mutually exclusive")
        if self.on_sl_long_continuation and self.stop_loss is None:
            raise ValueError("on_sl_long_continuation requires stop_loss to be configured")
        if self.on_sl_long_continuation and not (0.0 < self.long_trailing_stop_pct < 1.0):
            raise ValueError("long_trailing_stop_pct must be between 0 and 1 (exclusive)")
        return self


# ---------------------------------------------------------------------------
# Costs / matrix
# ---------------------------------------------------------------------------


class FeesConfig(BaseModel):
    """
    Structured per-contract fee model.

    Fees are applied per leg per contract at each event:
      - entry_fee_per_contract: charged when a position is opened
      - exit_fee_per_contract:  charged when closed via TP, SL, condition, or DTE exit
      - expiration_fee_per_contract: charged when the position expires (typically $0 at IBKR)

    Total fee = (entry_fee + exit_or_expiration_fee) × sum(leg.quantity)
    """

    entry_fee_per_contract: float = 0.0
    exit_fee_per_contract: float = 0.0
    expiration_fee_per_contract: float = 0.0


class CostsConfig(BaseModel):
    slippage_pct: float = 0.0
    fee_per_contract: float = 0.0  # legacy: split evenly as entry=fee/2, exit=fee/2
    fees: FeesConfig | None = None  # structured form; takes precedence over fee_per_contract

    @model_validator(mode="after")
    def validate_fee_fields(self) -> CostsConfig:
        if self.fee_per_contract != 0.0 and self.fees is not None:
            raise ValueError(
                "fee_per_contract and fees are mutually exclusive; "
                "use fees.entry_fee_per_contract / exit_fee_per_contract / expiration_fee_per_contract"
            )
        return self

    @property
    def effective_fees(self) -> FeesConfig:
        """Resolve the active fee model.

        Returns the explicit fees block if present. Falls back to splitting
        fee_per_contract evenly across entry and exit (legacy behaviour).
        """
        if self.fees is not None:
            return self.fees
        if self.fee_per_contract != 0.0:
            half = self.fee_per_contract / 2.0
            return FeesConfig(entry_fee_per_contract=half, exit_fee_per_contract=half)
        return FeesConfig()


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
    columns: list[str]  # dot-path refs e.g. "short_put.delta"
    rows: list[list[float | int | None]]

    @model_validator(mode="after")
    def row_lengths_match_columns(self) -> TableCombinations:
        for i, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"row {i} has {len(row)} values but {len(self.columns)} columns defined"
                )
        return self


# ---------------------------------------------------------------------------
# Roll block — close and re-open position when threshold is crossed
# ---------------------------------------------------------------------------


class RollConfig(BaseModel):
    window: EntryWindowConfig | None = None  # roll only within this time window; None = use trade entry window
    dte: IntSweep | None = None              # roll when remaining DTE <= this value
    vega: NumericSweep | None = None         # roll when spread net vega < this value
    conditions: list[str] = []              # roll when any condition expression is true (same namespace as exit.conditions)

    @model_validator(mode="after")
    def at_least_one_trigger(self) -> RollConfig:
        if self.dte is None and self.vega is None and not self.conditions:
            raise ValueError("roll requires at least one trigger: dte, vega, or conditions")
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
    roll: RollConfig | None = None

    @model_validator(mode="after")
    def leg_names_unique(self) -> TradeDefinition:
        names = [leg.name for leg in self.legs]
        if len(names) != len(set(names)):
            raise ValueError("leg names must be unique within a trade")
        return self

    @model_validator(mode="after")
    def at_most_one_targets_leg(self) -> TradeDefinition:
        """Position tagging attributes one target name per position, so only one
        leg per trade may use `targets`."""
        n = sum(1 for leg in self.legs if leg.targets is not None)
        if n > 1:
            raise ValueError("at most one leg per trade may use targets")
        return self

    @model_validator(mode="after")
    def reference_legs_valid(self) -> TradeDefinition:
        delta_leg_names = {leg.name for leg in self.legs if leg.strike_offset is None}
        for leg in self.legs:
            if leg.reference_leg is None:
                continue
            if leg.reference_leg not in delta_leg_names:
                raise ValueError(
                    f"leg '{leg.name}' references '{leg.reference_leg}' which must be "
                    "a delta-selected leg defined earlier in the legs list"
                )
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
    indicators: list[str] = []  # paths to indicator scripts (relative to CWD or absolute)
    trades: list[TradeDefinition]
    combinations: list[StructuredCombination] | TableCombinations | None = None

    @model_validator(mode="after")
    def trades_share_underlying(self) -> StrategyDefinition:
        """All trades must reference the same root_symbol."""
        symbols = {t.instrument.root_symbol for t in self.trades}
        if len(symbols) > 1:
            raise ValueError(f"all trades must share the same underlying; got {symbols}")
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
                if isinstance(leg.delta, SimpleDeltaConfig) and is_sweep(leg.delta.target):
                    sweep_fields.append(f"trades[{trade.name}].legs[{leg.name}].delta.target")
                if is_sweep(leg.dte):
                    sweep_fields.append(f"trades[{trade.name}].legs[{leg.name}].dte")
            for fname in ("stop_loss", "take_profit", "take_profit_pct", "dte_exit"):
                v = getattr(trade.exit, fname)
                if isinstance(v, StopLossConfig):
                    v = v.price
                elif isinstance(v, TakeProfitConfig):
                    v = v.price if v.price is not None else v.pct
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
                if (isinstance(leg.delta, SimpleDeltaConfig) and is_sweep(leg.delta.target)) or is_sweep(leg.dte):
                    return True
            for fname in ("stop_loss", "take_profit", "take_profit_pct", "dte_exit"):
                v = getattr(trade.exit, fname)
                if isinstance(v, StopLossConfig):
                    v = v.price
                elif isinstance(v, TakeProfitConfig):
                    v = v.price if v.price is not None else v.pct
                if v is not None and is_sweep(v):
                    return True
        return False
