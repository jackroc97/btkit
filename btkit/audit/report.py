"""
Audit report formatting.

Supports three output formats:
    text  — column-aligned table (default)
    json  — structured dict
    csv   — flag_code breakdown only (suitable for spreadsheet import)

Entry point: format_report(result, output_format) → str

Quintile table: format_quintile_summary(df) → str
    Formats the DataFrame returned by AuditRunner.quintile_summary().
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from btkit.audit.runner import AuditResult


def format_report(result: "AuditResult", output_format: str = "text") -> str:
    """
    Format an AuditResult as text, JSON, or CSV.

    text  — human-readable aligned table printed to stdout as the audit progresses.
    json  — machine-readable dict with flag_counts and summary totals.
    csv   — flag_code, severity, row_count, instrument_count as CSV rows.
    """
    fmt = output_format.lower()
    if fmt == "json":
        return _format_json(result)
    if fmt == "csv":
        return _format_csv(result)
    return _format_text(result)


def format_quintile_summary(df: pl.DataFrame) -> str:
    """
    Format the quintile breakdown DataFrame as a text table.

    The DataFrame is expected to have columns:
        right, quintile, delta_min, delta_max, delta_median,
        total_instruments, flagged_instruments, pct_flagged
    """
    if df.is_empty():
        return "(no quintile data)"

    lines: list[str] = []

    for right in ["P", "C"]:
        label = "Puts" if right == "P" else "Calls"
        subset = df.filter(pl.col("right") == right).sort("quintile")
        if subset.is_empty():
            continue
        lines.append(f"\n{label}")
        lines.append(
            f"  {'Q':<4}  {'Delta range':<25}  {'Total':>8}  {'Flagged':>8}  {'%':>6}"
        )
        lines.append("  " + "-" * 58)
        for row in subset.iter_rows(named=True):
            q = row["quintile"]
            delta_range = f"{row['delta_min']:+.3f} → {row['delta_max']:+.3f}"
            pct = row["pct_flagged"] or 0.0
            lines.append(
                f"  {q:<4}  {delta_range:<25}  {row['total_instruments']:>8,}"
                f"  {row['flagged_instruments']:>8,}  {pct:>5.1f}%"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal formatters
# ---------------------------------------------------------------------------


def _format_text(result: "AuditResult") -> str:
    lines: list[str] = []
    db_name = result.db_path.split("/")[-1]
    lines.append(f"\n{'='*62}")
    lines.append(f"  btkit audit — {db_name}")
    lines.append(f"{'='*62}")
    lines.append(f"  Duration    : {result.elapsed_seconds:.1f}s")
    if result.dry_run:
        lines.append("  Mode        : dry-run (option_audit table NOT written)")
    else:
        lines.append("  Mode        : written to option_audit table")
    if result.phase2_skipped:
        lines.append("  Phase 2     : skipped (--skip-phase2)")
    lines.append("")

    if not result.flag_counts:
        lines.append("  No flags found. Database appears clean.")
        lines.append(f"{'='*62}\n")
        return "\n".join(lines)

    col_w = [25, 8, 12, 14]
    header = (
        f"  {'Flag':<{col_w[0]}}  {'Sev':<{col_w[1]}}"
        f"  {'Rows':>{col_w[2]}}  {'Instruments':>{col_w[3]}}"
    )
    sep = "  " + "-" * (sum(col_w) + 3 * 2)
    lines.append(header)
    lines.append(sep)

    for fc in result.flag_counts:
        lines.append(
            f"  {fc.flag_code:<{col_w[0]}}  {fc.severity:<{col_w[1]}}"
            f"  {fc.row_count:>{col_w[2]},}  {fc.instrument_count:>{col_w[3]},}"
        )

    lines.append(sep)
    total_rows = sum(fc.row_count for fc in result.flag_counts)
    lines.append(
        f"  {'TOTAL':<{col_w[0]}}  {'':^{col_w[1]}}"
        f"  {total_rows:>{col_w[2]},}  {result.total_flagged_instruments:>{col_w[3]},}"
    )
    lines.append(f"{'='*62}\n")

    return "\n".join(lines)


def _format_json(result: "AuditResult") -> str:
    return json.dumps(
        {
            "db_path": result.db_path,
            "dry_run": result.dry_run,
            "phase2_skipped": result.phase2_skipped,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "total_flagged_instruments": result.total_flagged_instruments,
            "flag_counts": [
                {
                    "flag_code": fc.flag_code,
                    "severity": fc.severity,
                    "row_count": fc.row_count,
                    "instrument_count": fc.instrument_count,
                }
                for fc in result.flag_counts
            ],
        },
        indent=2,
    )


def _format_csv(result: "AuditResult") -> str:
    rows = ["flag_code,severity,row_count,instrument_count"]
    for fc in result.flag_counts:
        rows.append(f"{fc.flag_code},{fc.severity},{fc.row_count},{fc.instrument_count}")
    return "\n".join(rows)
