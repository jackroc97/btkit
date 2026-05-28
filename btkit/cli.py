"""
btkit CLI — entry point for all pipeline commands.

Commands:
    btkit build         Build the input database from raw Databento files.
    btkit run           Run a backtest from a strategy YAML (single scalar run).
    btkit analyze       Compute metrics and print results to terminal.
    btkit pipeline      Full pipeline: build (if needed) → run → analyze.
    btkit serve         Launch the interactive Dash dashboard for a backtest run.
    btkit study run     Run a study: expand parameterized strategies and execute
                        all combinations in parallel.
    btkit db merge      Merge two or more output databases into one.

Usage:
    btkit build       --data-path DATA --db-path DB [--indicators SCRIPT ...]
    btkit run         --strategy YAML --input-db DB --output-db DB [--initial-equity N]
    btkit analyze     --output-db DB [--backtest-id N | --study-id N]
    btkit pipeline    --data-path DATA --strategy YAML --db-path DB --output-db DB
                      [--indicators SCRIPT ...] [--initial-equity N] [--rebuild]
    btkit serve       --output-db DB [--backtest-id N] [--port 8050]
    btkit study run   --study YAML --input-db DB --output-db DB
                      [--workers N] [--max-combinations N] [--initial-equity N]
    btkit db merge    --sources a.db b.db ... --target combined.db
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import typer

from btkit.analysis.metrics import PostProcessor
from btkit.backtest.engine import BacktestEngine
from btkit.db.input_db import InputDatabase
from btkit.db.output_db import OutputDatabase
from btkit.pipeline.builder import DatabaseBuilder
from btkit.pipeline.indicators import IndicatorRunner
from btkit.strategy.definition import StrategyDefinition
from btkit.strategy.loader import load_strategy

app = typer.Typer(
    name="btkit",
    help="Vectorized options backtesting framework.",
    add_completion=False,
)

study_app = typer.Typer(
    name="study",
    help="Study commands — multi-strategy / parameter-sweep backtests.",
    add_completion=False,
)
app.add_typer(study_app, name="study")

db_app = typer.Typer(
    name="db",
    help="Database utilities.",
    add_completion=False,
)
app.add_typer(db_app, name="db")


def _require_output_db(path: str) -> None:
    """Exit with a clear message if the output database file does not exist."""
    if not Path(path).exists():
        typer.echo(
            f"Error: output database not found: {path}\n"
            "Run 'btkit run' or 'btkit study run' first to create it.",
            err=True,
        )
        raise typer.Exit(code=1)


def _ensure_indicators(input_db_path: str, strategy: StrategyDefinition) -> None:
    """Run any indicator scripts listed in strategy.indicators that are not yet in the DB,
    or that are stale because new underlying bars have been added since they were last built."""
    if not strategy.indicators:
        return

    typer.echo("Checking strategy indicators...")
    con = duckdb.connect(input_db_path)
    try:
        underlyings = con.execute(
            "SELECT DISTINCT instrument_id FROM underlying_bars"
        ).fetchall()

        max_underlying_ts = con.execute(
            "SELECT MAX(ts_event) FROM underlying_bars"
        ).fetchone()[0]

        for script_str in strategy.indicators:
            script_path = Path(script_str)
            if not script_path.exists():
                typer.echo(
                    f"  WARNING: indicator script not found: {script_path}", err=True
                )
                continue

            script_source = script_path.read_text()
            row = con.execute(
                """
                SELECT MAX(ib.ts_event)
                FROM indicator_bars ib
                JOIN indicator_definition idef ON ib.indicator_id = idef.id
                WHERE idef.script_source = ?
                """,
                [script_source],
            ).fetchone()
            max_indicator_ts = row[0] if row else None

            if max_indicator_ts is None:
                typer.echo(f"  {script_path.name}: new indicator detected, building...")
            elif max_underlying_ts and max_indicator_ts < max_underlying_ts:
                typer.echo(
                    f"  {script_path.name}: underlying data extended "
                    f"({max_indicator_ts.date()} → {max_underlying_ts.date()}), recalculating..."
                )
            else:
                typer.echo(f"  {script_path.name}: up to date, skipping")
                continue

            runner = IndicatorRunner(con, script_path)
            for (underlying_id,) in underlyings:
                runner.run(underlying_id)
            typer.echo(f"  {script_path.name}: done")
    finally:
        con.close()


@app.command()
def build(
    data_path: str = typer.Option(..., help="Path to directory containing .dbn files."),
    db_path: str = typer.Option(..., help="Output path for the input database file."),
    indicators: list[str] = typer.Option(
        default=[],
        help="Paths to indicator scripts. May be repeated for multiple scripts.",
    ),
    append: bool = typer.Option(
        default=False,
        help=(
            "Append new data to an existing database instead of rebuilding. "
            "Skips OHLCV rows, greeks, and indicator bars that are already present."
        ),
    ),
) -> None:
    """Build the input database from raw Databento files."""
    typer.echo(f"{'Appending to' if append else 'Building'} database: {db_path}")
    builder = DatabaseBuilder(
        raw_data_path=data_path,
        db_path=db_path,
        indicator_scripts=indicators or None,
        append=append,
    )
    builder.build()
    typer.echo("Build complete.")


@app.command()
def run(
    strategy: str = typer.Option(..., help="Path to the strategy YAML file."),
    input_db: str = typer.Option(..., help="Path to the input database."),
    output_db: str = typer.Option(..., help="Path for the output database (created if absent)."),
    initial_equity: float = typer.Option(
        default=100_000.0,
        help="Starting account equity. Used for equity curve and minimum_equity filtering.",
    ),
    note: str = typer.Option(
        default=None,
        help="Free-text label stored with this run (shown in the dashboard index).",
    ),
) -> None:
    """Run a backtest from a strategy YAML (single scalar run)."""
    typer.echo(f"Loading strategy: {strategy}")
    definition = load_strategy(strategy)
    if definition.is_parameterized():
        typer.echo(
            "Error: strategy contains sweep parameters. "
            "Use 'btkit study run' for parameterized strategies.",
            err=True,
        )
        raise typer.Exit(code=1)
    _ensure_indicators(input_db, definition)

    with InputDatabase(input_db) as idb, OutputDatabase(output_db) as odb:
        odb.create_schema()
        engine = BacktestEngine(
            input_db=idb,
            output_db=odb,
            strategy=definition,
            initial_equity=initial_equity,
            note=note,
        )
        backtest_id = engine.run()

    typer.echo(f"Backtest complete. backtest_id={backtest_id}")


@app.command()
def analyze(
    output_db: str = typer.Option(..., help="Path to the output database."),
    backtest_id: int = typer.Option(
        default=None, help="Analyse a specific backtest run (defaults to most recent)."
    ),
    study_id: int = typer.Option(default=None, help="Analyse all runs from a study."),
    matrix_id: int = typer.Option(
        default=None, hidden=True, help="Deprecated alias for --study-id."
    ),
) -> None:
    """Compute metrics and print results to terminal."""
    # Support legacy --matrix-id as a hidden alias.
    effective_study_id = study_id or matrix_id

    if backtest_id is not None and effective_study_id is not None:
        typer.echo("Provide at most one of --backtest-id or --study-id.", err=True)
        raise typer.Exit(code=1)

    _require_output_db(output_db)
    with OutputDatabase(output_db) as odb:
        processor = PostProcessor(
            odb, backtest_id=backtest_id, study_id=effective_study_id
        )
        summary = processor.summarize(formatted=True)

    typer.echo(summary)


@app.command()
def pipeline(
    data_path: str = typer.Option(..., help="Path to directory containing .dbn files."),
    strategy: str = typer.Option(..., help="Path to the strategy YAML file."),
    db_path: str = typer.Option(..., help="Path for the input database."),
    output_db: str = typer.Option(..., help="Path for the output database."),
    indicators: list[str] = typer.Option(
        default=[],
        help="Paths to indicator scripts. May be repeated.",
    ),
    initial_equity: float = typer.Option(default=100_000.0),
    rebuild: bool = typer.Option(
        default=False,
        help="Force rebuild of the input database even if it already exists.",
    ),
) -> None:
    """Full pipeline: build (if needed) → run → analyze."""
    db_file = Path(db_path)

    if rebuild or not db_file.exists():
        typer.echo(f"Building database: {db_path}")
        DatabaseBuilder(
            raw_data_path=data_path,
            db_path=db_path,
            indicator_scripts=indicators or None,
        ).build()
    else:
        typer.echo(f"Input database already exists, skipping build: {db_path}")

    typer.echo(f"Loading strategy: {strategy}")
    definition = load_strategy(strategy)
    if definition.is_parameterized():
        typer.echo(
            "Error: strategy contains sweep parameters. "
            "Use 'btkit study run' for parameterized strategies.",
            err=True,
        )
        raise typer.Exit(code=1)
    _ensure_indicators(db_path, definition)

    with InputDatabase(db_path) as idb, OutputDatabase(output_db) as odb:
        odb.create_schema()
        engine = BacktestEngine(
            input_db=idb,
            output_db=odb,
            strategy=definition,
            initial_equity=initial_equity,
        )
        backtest_id = engine.run()

    typer.echo(f"Backtest complete. backtest_id={backtest_id}")

    with OutputDatabase(output_db) as odb:
        processor = PostProcessor(odb, backtest_id=backtest_id)
        summary = processor.summarize(formatted=True)

    typer.echo(summary)


@app.command()
def serve(
    output_db: str = typer.Option(..., help="Path to the output database."),
    input_db: str = typer.Option(
        default=None, help="Path to the input database (enables per-trade candle charts)."
    ),
    backtest_id: int = typer.Option(
        default=None, help="Backtest run to display (defaults to most recent)."
    ),
    port: int = typer.Option(default=8050, help="Port for the dashboard server."),
    debug: bool = typer.Option(default=False, help="Run Dash in debug mode (enables hot-reload)."),
) -> None:
    """Launch the interactive dashboard for a backtest run."""
    _require_output_db(output_db)
    try:
        from btkit.analysis.dashboard import run_dashboard
    except ImportError as exc:
        typer.echo(
            "Dashboard dependencies not installed. Run: pip install btkit[viz]",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    run_dashboard(
        output_db, input_db_path=input_db, backtest_id=backtest_id, port=port, debug=debug
    )


# ---------------------------------------------------------------------------
# btkit study …
# ---------------------------------------------------------------------------


@study_app.command("run")
def study_run(
    study: str = typer.Option(..., help="Path to the study YAML file."),
    input_db: str = typer.Option(..., help="Path to the input database."),
    output_db: str = typer.Option(..., help="Path for the output database (created if absent)."),
    workers: int = typer.Option(
        default=None, help="Worker process count (default: cpu_count)."
    ),
    max_combinations: int = typer.Option(
        default=None, help="Abort if expansion exceeds this many combinations."
    ),
    initial_equity: float = typer.Option(
        default=100_000.0,
        help="Starting account equity for each combination.",
    ),
    note: str = typer.Option(
        default=None,
        help="Free-text label stored with this study (shown in the dashboard index).",
    ),
) -> None:
    """
    Run a study: expand parameterized strategies and execute all combinations
    in parallel, then merge results into a single output database.
    """
    from btkit.study.loader import load_study
    from btkit.study.runner import StudyRunner

    study_path = Path(study)
    study_yaml_text = study_path.read_text()
    study_def, study_dir = load_study(study_path)

    typer.echo(f"Starting study '{study_def.name}'")

    runner = StudyRunner(
        study=study_def,
        study_dir=study_dir,
        study_yaml_text=study_yaml_text,
        input_db_path=input_db,
        output_db_path=output_db,
        max_workers=workers,
        max_combinations=max_combinations,
        initial_equity=initial_equity,
        note=note,
    )

    study_id, failed = runner.run()

    if failed:
        typer.echo(
            f"\nStudy complete with {len(failed)} failed combination(s):", err=True
        )
        for f in failed:
            cid = f["combination_id"]
            label = f"combination_id={cid}" if cid >= 0 else "runner error"
            typer.echo(f"  {label}: {f['error']}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Study complete. study_id={study_id}")


# ---------------------------------------------------------------------------
# btkit db …
# ---------------------------------------------------------------------------


@db_app.command("merge")
def db_merge(
    sources: list[str] = typer.Option(
        ..., help="One or more source output database paths to merge."
    ),
    target: str = typer.Option(..., help="Target output database path (created if absent)."),
) -> None:
    """
    Merge two or more output databases into one.

    Example:
        btkit db merge --sources jan.db feb.db mar.db --target q1.db
    """
    from btkit.study.merger import OutputMerger

    missing = [p for p in sources if not Path(p).exists()]
    if missing:
        for p in missing:
            typer.echo(f"Error: source database not found: {p}", err=True)
        raise typer.Exit(code=1)

    if Path(target).resolve() in [Path(p).resolve() for p in sources]:
        typer.echo("Error: --target must not be one of the --sources paths.", err=True)
        raise typer.Exit(code=1)

    with OutputDatabase(target) as odb:
        odb.create_schema()

    typer.echo(f"Merging {len(sources)} database(s) into {target} …")
    OutputMerger().merge(source_db_paths=sources, target_db_path=target)
    typer.echo("Merge complete.")


if __name__ == "__main__":
    app()
