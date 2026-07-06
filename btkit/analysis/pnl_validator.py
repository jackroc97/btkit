"""
PnL bounds validator for btkit output databases.

Checks each position for economically impossible or internally inconsistent
values. Catches data artifacts (stale forward-fill, sign inversion) that
produce phantom profits or impossible losses, even after the engine-level
fixes in the exit scanner.

Designed to be run post-hoc against any output database, either as a
sanity-check script or as part of an automated acceptance test.

Usage:
    from btkit.analysis.pnl_validator import validate_positions
    flags = validate_positions("path/to/output.db")
    if not flags.is_empty():
        print(flags)
"""

from __future__ import annotations

import duckdb
import polars as pl


def validate_positions(
    db_path: str,
    tolerance: float = 0.50,
) -> pl.DataFrame:
    """
    Scan every position in an output database for bound and consistency violations.

    Derives theoretical PnL bounds from the leg strike prices. For a vertical
    spread the bounds are exact; for more complex spreads they are conservative
    (worst-case across all STO/BTO pairs of the same option right).

    Six checks are applied:

    1. exit_mark sign
         Credit spread (open_mark > 0): exit_mark must be >= 0.
         Debit spread (open_mark < 0): exit_mark must be <= 0.
         Violation indicates a stale forward-fill artifact: the market mark for
         one leg was carried forward from a price level that no longer holds,
         creating an economically impossible position value.

    2. exit_mark magnitude
         |exit_mark| must not exceed spread_width.
         Exception: non-expiry exits with leg_out=True and an illiquid long leg
         (filled at 0) may legitimately have |exit_mark| up to the short leg's
         full intrinsic value. Flag these manually if leg_out is in use.

    3. gross_pnl ceiling
         gross_pnl = (open_mark - exit_mark) * multiplier must not exceed the
         theoretical maximum profit for the spread.

    4. gross_pnl floor
         gross_pnl must not be worse (more negative) than the theoretical
         maximum loss for the spread.

    5. open_mark integrity
         open_mark stored in position must equal the sum of (open_price * sign *
         qty) computed from position_leg. Catches upstream computation errors or
         DB write bugs.

    6. net_pnl integrity
         net_pnl must equal gross_pnl - slippage_cost - fee_cost within
         tolerance. Catches PnL formula bugs.

    Parameters
    ----------
    db_path:
        Path to a btkit output DuckDB database.
    tolerance:
        Absolute tolerance (in points) applied to all checks to absorb tick
        rounding and floating-point imprecision. Default: 0.50 points
        (10 ticks at ES tick_size=0.05).

    Returns
    -------
    pl.DataFrame with one row per flagged position, columns:
        position_id, backtest_id, trade_name, exit_reason
        open_mark, exit_mark, worst_mark, net_pnl
        gross_pnl, max_gross_pnl, min_gross_pnl
        spread_width, open_mark_expected, net_pnl_expected
        violation  (pipe-separated list of triggered check names)
    Sorted by abs(gross_pnl) descending so the largest anomalies come first.
    Empty DataFrame when no violations are found.
    """
    con = duckdb.connect(db_path, read_only=True)

    positions = con.execute("""
        SELECT
            p.id            AS position_id,
            p.backtest_id,
            p.trade_name,
            p.exit_reason,
            p.open_mark,
            p.exit_mark,
            p.worst_mark,
            p.slippage_cost,
            p.fee_cost,
            p.net_pnl
        FROM position p
        WHERE p.exit_mark IS NOT NULL
          AND p.net_pnl   IS NOT NULL
    """).pl()

    legs = con.execute("""
        SELECT
            pl.position_id,
            pl.action,
            pl.quantity,
            pl.multiplier,
            pl.strike_price,
            pl."right"      AS opt_right,
            pl.open_price
        FROM position_leg pl
    """).pl()

    con.close()

    if positions.is_empty():
        return pl.DataFrame()

    # ------------------------------------------------------------------
    # Spread width: max |STO_strike - BTO_strike| for matching option rights
    # per position. Computed per (position_id, right) so put and call wings
    # are handled separately — correct for iron condors and single-side spreads.
    # For each right, cross-join all STO/BTO pairs and take the maximum width,
    # then take the max across rights.
    # ------------------------------------------------------------------
    sto = (
        legs.filter(pl.col("action") == "STO")
        .select(["position_id", "opt_right", pl.col("strike_price").alias("sto_strike")])
    )
    bto = (
        legs.filter(pl.col("action") == "BTO")
        .select(["position_id", "opt_right", pl.col("strike_price").alias("bto_strike")])
    )
    spread_widths = (
        sto.join(bto, on=["position_id", "opt_right"], how="inner")
        .with_columns((pl.col("sto_strike") - pl.col("bto_strike")).abs().alias("wing_width"))
        .group_by("position_id")
        .agg(pl.col("wing_width").max().alias("spread_width"))
    )

    # ------------------------------------------------------------------
    # Multiplier: same for every leg in a position; take from any leg.
    # ------------------------------------------------------------------
    multipliers = (
        legs.group_by("position_id")
        .agg(pl.col("multiplier").first())
    )

    # ------------------------------------------------------------------
    # open_mark cross-check: Σ(open_price × sign × qty) per position.
    # STO legs contribute positively (seller receives premium),
    # BTO legs negatively (buyer pays premium).
    # ------------------------------------------------------------------
    open_mark_recomputed = (
        legs.with_columns(
            (
                pl.col("open_price")
                * pl.when(pl.col("action") == "STO").then(1.0).otherwise(-1.0)
                * pl.col("quantity")
            ).alias("_contrib")
        )
        .group_by("position_id")
        .agg(pl.col("_contrib").sum().alias("open_mark_expected"))
    )

    # ------------------------------------------------------------------
    # Assemble: join derived values onto positions
    # ------------------------------------------------------------------
    df = (
        positions
        .join(spread_widths, on="position_id", how="left")
        .join(multipliers, on="position_id", how="left")
        .join(open_mark_recomputed, on="position_id", how="left")
    )

    df = df.with_columns(
        ((pl.col("open_mark") - pl.col("exit_mark")) * pl.col("multiplier"))
        .alias("gross_pnl")
    )

    # Theoretical PnL bounds (before costs):
    #   credit spread (open_mark > 0):
    #     max profit = open_mark × mult          (exit_mark → 0)
    #     max loss   = (open_mark - width) × mult (exit_mark → width)
    #   debit spread (open_mark < 0):
    #     max profit = (open_mark + width) × mult (exit_mark → −width)
    #     max loss   = open_mark × mult            (exit_mark → 0)
    df = df.with_columns([
        pl.when(pl.col("open_mark") >= 0)
        .then(pl.col("open_mark") * pl.col("multiplier"))
        .otherwise((pl.col("open_mark") + pl.col("spread_width")) * pl.col("multiplier"))
        .alias("max_gross_pnl"),

        pl.when(pl.col("open_mark") >= 0)
        .then((pl.col("open_mark") - pl.col("spread_width")) * pl.col("multiplier"))
        .otherwise(pl.col("open_mark") * pl.col("multiplier"))
        .alias("min_gross_pnl"),

        (pl.col("gross_pnl") - pl.col("slippage_cost") - pl.col("fee_cost"))
        .alias("net_pnl_expected"),
    ])

    tol = float(tolerance)

    # ------------------------------------------------------------------
    # Individual check flags
    # ------------------------------------------------------------------
    df = df.with_columns([
        # 1. exit_mark sign
        (
            ((pl.col("open_mark") > 0) & (pl.col("exit_mark") < -tol))
            | ((pl.col("open_mark") < 0) & (pl.col("exit_mark") > tol))
        ).alias("_flag_sign"),

        # 2. exit_mark magnitude
        (pl.col("exit_mark").abs() > pl.col("spread_width") + tol).alias("_flag_magnitude"),

        # 3. gross_pnl ceiling: profit must not exceed theoretical max
        (pl.col("gross_pnl") > pl.col("max_gross_pnl") + tol * pl.col("multiplier"))
        .alias("_flag_ceiling"),

        # 4. gross_pnl floor: loss must not exceed theoretical max
        (pl.col("gross_pnl") < pl.col("min_gross_pnl") - tol * pl.col("multiplier"))
        .alias("_flag_floor"),

        # 5. open_mark vs leg sum
        ((pl.col("open_mark") - pl.col("open_mark_expected")).abs() > tol)
        .fill_null(False)
        .alias("_flag_open_mark"),

        # 6. net_pnl = gross_pnl - costs
        ((pl.col("net_pnl") - pl.col("net_pnl_expected")).abs() > tol * pl.col("multiplier"))
        .alias("_flag_net_pnl"),
    ])

    _flag_cols = [
        "_flag_sign", "_flag_magnitude", "_flag_ceiling",
        "_flag_floor", "_flag_open_mark", "_flag_net_pnl",
    ]
    _flag_labels = {
        "_flag_sign":       "exit_mark_sign",
        "_flag_magnitude":  "exit_mark_magnitude",
        "_flag_ceiling":    "gross_pnl_exceeds_max_profit",
        "_flag_floor":      "gross_pnl_exceeds_max_loss",
        "_flag_open_mark":  "open_mark_inconsistent",
        "_flag_net_pnl":    "net_pnl_inconsistent",
    }

    any_flag = pl.fold(
        acc=pl.lit(False),
        function=lambda acc, x: acc | x,
        exprs=[pl.col(c) for c in _flag_cols],
    )
    flagged = df.filter(any_flag)

    if flagged.is_empty():
        return pl.DataFrame()

    # Build violation string: pipe-separated names of triggered checks
    # Evaluate per-row in Python — only called on flagged rows (expected to be rare).
    flag_records = flagged.select(_flag_cols).to_dicts()
    violations = [
        " | ".join(label for col, label in _flag_labels.items() if rec[col])
        for rec in flag_records
    ]

    return (
        flagged
        .with_columns(pl.Series("violation", violations, dtype=pl.Utf8))
        .drop(_flag_cols)
        .select([
            "position_id", "backtest_id", "trade_name", "exit_reason",
            "open_mark", "exit_mark", "worst_mark", "net_pnl",
            "gross_pnl", "max_gross_pnl", "min_gross_pnl",
            "spread_width", "open_mark_expected", "net_pnl_expected",
            "violation",
        ])
        .sort(pl.col("gross_pnl").abs(), descending=True)
    )


def summarise_violations(flags: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate flagged positions by violation type.

    Returns a DataFrame with columns:
        violation_type, count, total_gross_pnl_impact
    Useful for understanding the scale and composition of data quality issues.
    """
    if flags.is_empty():
        return pl.DataFrame({
            "violation_type": pl.Series([], dtype=pl.Utf8),
            "count": pl.Series([], dtype=pl.Int32),
            "total_gross_pnl_impact": pl.Series([], dtype=pl.Float64),
        })

    # Explode the pipe-separated violation string into one row per type
    exploded = (
        flags
        .with_columns(pl.col("violation").str.split(" | ").alias("_types"))
        .explode("_types")
        .rename({"_types": "violation_type"})
    )
    return (
        exploded
        .group_by("violation_type")
        .agg([
            pl.len().alias("count"),
            pl.col("gross_pnl").sum().alias("total_gross_pnl_impact"),
        ])
        .sort("count", descending=True)
    )
