"""
Audit schema: FlagCode / FlagSeverity enums, severity maps, DDL, and filter resolution.
"""

from __future__ import annotations

from enum import Enum

import polars as pl


class FlagCode(str, Enum):
    # Phase 1 — implied volatility
    IV_NAN = "IV_NAN"
    IV_SENTINEL = "IV_SENTINEL"
    IV_HIGH = "IV_HIGH"
    # Phase 2 — Black-76 delta consistency
    DELTA_INCONSISTENT = "DELTA_INCONSISTENT"
    # Phase 3 — bar coverage
    BARS_TRUNCATED = "BARS_TRUNCATED"
    BARS_SPARSE = "BARS_SPARSE"
    NO_EXPIRY_BARS = "NO_EXPIRY_BARS"
    # Phase 4 — basic integrity
    NEGATIVE_CLOSE = "NEGATIVE_CLOSE"
    NEGATIVE_DTE = "NEGATIVE_DTE"
    ZOMBIE_BAR = "ZOMBIE_BAR"
    DELTA_SIGN_ERROR = "DELTA_SIGN_ERROR"
    DELTA_MAGNITUDE_ERROR = "DELTA_MAGNITUDE_ERROR"


class FlagSeverity(str, Enum):
    HARD = "hard"
    SOFT = "soft"


HARD_FLAGS: frozenset[FlagCode] = frozenset({
    FlagCode.BARS_TRUNCATED,
    FlagCode.NEGATIVE_CLOSE,
    FlagCode.NEGATIVE_DTE,
    FlagCode.ZOMBIE_BAR,
    FlagCode.DELTA_SIGN_ERROR,
    FlagCode.DELTA_MAGNITUDE_ERROR,
})

SOFT_FLAGS: frozenset[FlagCode] = frozenset({
    FlagCode.IV_NAN,
    FlagCode.IV_SENTINEL,
    FlagCode.IV_HIGH,
    FlagCode.DELTA_INCONSISTENT,
    FlagCode.BARS_SPARSE,
    FlagCode.NO_EXPIRY_BARS,
})

FLAG_SEVERITY: dict[FlagCode, FlagSeverity] = {
    **{c: FlagSeverity.HARD for c in HARD_FLAGS},
    **{c: FlagSeverity.SOFT for c in SOFT_FLAGS},
}

_PRESET_FILTER_CODES: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "hard_errors_only": frozenset(c.value for c in HARD_FLAGS),
    "strict": frozenset(c.value for c in HARD_FLAGS | SOFT_FLAGS),
}


def resolve_audit_filter(filter_spec: str | list[str]) -> frozenset[str]:
    """
    Resolve a filter preset string or explicit code list to a frozenset of flag code strings.

    Preset strings:
        "none"             — no filter (backward compatible)
        "hard_errors_only" — exclude instruments with any hard flag (default)
        "strict"           — exclude instruments with any flag (hard or soft)

    Explicit list: any subset of FlagCode values, e.g. ["BARS_TRUNCATED", "NEGATIVE_CLOSE"].
    Unknown codes are passed through unchanged (forward compatible).
    """
    if isinstance(filter_spec, list):
        return frozenset(filter_spec)
    return _PRESET_FILTER_CODES.get(filter_spec, frozenset())


# ---------------------------------------------------------------------------
# Shared empty-DataFrame factory for rule modules
# ---------------------------------------------------------------------------

AUDIT_COLUMNS = {
    "instrument_id": pl.Int64,
    "ts_event": pl.Datetime("us", "UTC"),
    "flag_code": pl.Utf8,
    "flag_severity": pl.Utf8,
    "flag_value": pl.Float64,
    "threshold": pl.Float64,
}


def empty_audit_df() -> pl.DataFrame:
    """Return an empty DataFrame with the option_audit schema."""
    return pl.DataFrame({col: pl.Series([], dtype=dt) for col, dt in AUDIT_COLUMNS.items()})


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

OPTION_AUDIT_DDL = """\
CREATE TABLE IF NOT EXISTS option_audit (
    instrument_id  INTEGER     NOT NULL,
    ts_event       TIMESTAMPTZ NOT NULL,
    flag_code      VARCHAR     NOT NULL,
    flag_severity  VARCHAR     NOT NULL,
    flag_value     DOUBLE,
    threshold      DOUBLE,
    PRIMARY KEY (instrument_id, ts_event, flag_code)
);
CREATE INDEX IF NOT EXISTS idx_option_audit_instrument
    ON option_audit (instrument_id, flag_code);
"""
