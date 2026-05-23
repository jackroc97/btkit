"""
btkit CLI — entry point for all pipeline commands.

Commands:
    btkit build     Build the input database from raw Databento files.
    btkit run       Run a backtest from a strategy YAML (single run only for MVP).
    btkit analyze   Compute metrics and print results to terminal.
    btkit pipeline  Full pipeline: build (if needed) → run → analyze.
    btkit serve     Launch the interactive Dash dashboard for a backtest run.

Usage:
    btkit build   --data-path DATA --db-path DB [--indicators SCRIPT ...]
    btkit run     --strategy YAML --input-db DB --output-db DB [--initial-equity N]
    btkit analyze --output-db DB [--backtest-id N | --matrix-id N]
    btkit pipeline --data-path DATA --strategy YAML --db-path DB --output-db DB
                   [--indicators SCRIPT ...] [--initial-equity N] [--rebuild]
    btkit serve   --output-db DB [--backtest-id N] [--port 8050]
"""

from __future__ import annotations

from pathlib import Path

import typer

from btkit.backtest.engine import BacktestEngine
from btkit.db.input_db import InputDatabase
from btkit.db.output_db import OutputDatabase
from btkit.pipeline.builder import DatabaseBuilder
from btkit.strategy.loader import load_strategy
from btkit.analysis.metrics import PostProcessor

app = typer.Typer(
    name="btkit",
    help="Vectorized options backtesting framework.",
    add_completion=False,
)


@app.command()
def build(
    data_path: str = typer.Option(..., help="Path to directory containing .dbn files."),
    db_path: str = typer.Option(..., help="Output path for the input database file."),
    indicators: list[str] = typer.Option(
        default=[],
        help="Paths to indicator scripts. May be repeated for multiple scripts.",
    ),
) -> None:
    """Build the input database from raw Databento files."""
    typer.echo(f"Building database: {db_path}")
    builder = DatabaseBuilder(
        raw_data_path=data_path,
        db_path=db_path,
        indicator_scripts=indicators or None,
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
) -> None:
    """Run a backtest from a strategy YAML. Single-run only for this version."""
    typer.echo(f"Loading strategy: {strategy}")
    definition = load_strategy(strategy)

    with InputDatabase(input_db) as idb, OutputDatabase(output_db) as odb:
        odb.create_schema()
        engine = BacktestEngine(
            input_db=idb,
            output_db=odb,
            strategy=definition,
            initial_equity=initial_equity,
        )
        backtest_id = engine.run()

    typer.echo(f"Backtest complete. backtest_id={backtest_id}")


@app.command()
def analyze(
    output_db: str = typer.Option(..., help="Path to the output database."),
    backtest_id: int = typer.Option(default=None, help="Analyse a specific backtest run (defaults to most recent)."),
    matrix_id: int = typer.Option(default=None, help="Analyse all runs from a matrix expansion."),
) -> None:
    """Compute metrics and print results to terminal."""
    if backtest_id is not None and matrix_id is not None:
        typer.echo("Provide at most one of --backtest-id or --matrix-id.", err=True)
        raise typer.Exit(code=1)

    with OutputDatabase(output_db) as odb:
        processor = PostProcessor(odb, backtest_id=backtest_id, matrix_id=matrix_id)
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
    input_db: str = typer.Option(default=None, help="Path to the input database (enables per-trade candle charts)."),
    backtest_id: int = typer.Option(default=None, help="Backtest run to display (defaults to most recent)."),
    port: int = typer.Option(default=8050, help="Port for the dashboard server."),
    debug: bool = typer.Option(default=False, help="Run Dash in debug mode (enables hot-reload)."),
) -> None:
    """Launch the interactive dashboard for a backtest run."""
    try:
        from btkit.analysis.dashboard import run_dashboard
    except ImportError:
        typer.echo(
            "Dashboard dependencies not installed. Run: pip install btkit[viz]",
            err=True,
        )
        raise typer.Exit(code=1)

    run_dashboard(output_db, input_db_path=input_db,
                  backtest_id=backtest_id, port=port, debug=debug)


if __name__ == "__main__":
    app()
