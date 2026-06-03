"""
btkit CLI — entry point for all pipeline commands.

Commands:
    btkit build         Build the input database from raw Databento files.
    btkit run           Run a backtest from a strategy YAML (single scalar run).
    btkit analyze       Compute metrics and print results to terminal.
    btkit pipeline      Full pipeline: build (if needed) → run → analyze.
    btkit dashboard     Launch the interactive dashboard for a backtest run.
    btkit study run     Run a study: expand parameterized strategies and execute
                        all combinations in parallel.
    btkit db merge      Merge two or more output databases into one.

Usage:
    btkit build       --data-path DATA --db-path DB [--indicators SCRIPT ...]
    btkit run         --strategy YAML --input-db DB --output-db DB [--initial-equity N]
    btkit analyze     --output-db DB [--backtest-id N | --study-id N]
    btkit pipeline    --data-path DATA --strategy YAML --db-path DB --output-db DB
                      [--indicators SCRIPT ...] [--initial-equity N] [--rebuild]
    btkit dashboard   --output-db DB [--backtest-id N] [--port 8050]
                      [--background]  launch server in the background
                      [--kill]        stop a background server
    btkit study run   --study YAML --input-db DB --output-db DB
                      [--workers N] [--max-combinations N] [--initial-equity N]
    btkit db extend   --sources a.db b.db ... --target combined.db
"""

from __future__ import annotations

import os
import signal
import sys
import time
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


_DASHBOARD_PID_FILE = Path.home() / ".btkit" / "dashboard.pid"


def _pid_on_port(port: int) -> int | None:
    """Return the PID listening on *port*, or None if the port is free."""
    import subprocess
    try:
        r = subprocess.run(
            ["lsof", "-ti", f"TCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True,
        )
        pids = [int(p) for p in r.stdout.split() if p.strip().isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


def _kill_dashboard(port: int = 8050) -> None:
    """Stop a dashboard process — targets both the PID file and the port holder."""
    killed_any = False

    # Kill the process recorded in the PID file.
    if _DASHBOARD_PID_FILE.exists():
        pid = int(_DASHBOARD_PID_FILE.read_text().strip())
        _DASHBOARD_PID_FILE.unlink(missing_ok=True)
        try:
            os.kill(pid, signal.SIGTERM)
            typer.echo(f"Dashboard stopped (PID {pid}).")
            killed_any = True
        except ProcessLookupError:
            typer.echo(f"PID {pid} was not running — stale PID file removed.")
        except PermissionError:
            typer.echo(f"Error: permission denied stopping PID {pid}.", err=True)
            raise typer.Exit(code=1)

    # Also kill anything still holding the port (catches sessions started outside --background).
    port_pid = _pid_on_port(port)
    if port_pid:
        try:
            os.kill(port_pid, signal.SIGTERM)
            typer.echo(f"Killed stale process on port {port} (PID {port_pid}).")
            killed_any = True
        except (ProcessLookupError, PermissionError):
            pass

    if not killed_any:
        typer.echo("No running dashboard found.")


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    ms = seconds * 1000
    if ms < 1000:
        return f"{ms:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    minutes = minutes % 60
    return f"{hours}h {minutes}m {int(secs)}s"


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
        _t0 = time.perf_counter()
        backtest_id = engine.run()
        _elapsed = time.perf_counter() - _t0

    typer.echo(f"Backtest (id={backtest_id}) completed in {_fmt_duration(_elapsed)}")


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
        _t0 = time.perf_counter()
        backtest_id = engine.run()
        _elapsed = time.perf_counter() - _t0

    typer.echo(f"Backtest (id={backtest_id}) completed in {_fmt_duration(_elapsed)}")

    with OutputDatabase(output_db) as odb:
        processor = PostProcessor(odb, backtest_id=backtest_id)
        summary = processor.summarize(formatted=True)

    typer.echo(summary)


@app.command()
def dashboard(
    output_db: str = typer.Option(
        default=None, help="Path to the output database. Required unless --kill is used."
    ),
    input_db: str = typer.Option(
        default=None, help="Path to the input database (enables per-trade candle charts)."
    ),
    port: int = typer.Option(default=8050, help="Port for the dashboard server."),
    debug: bool = typer.Option(default=False, help="Enable auto-reload (development mode)."),
    background: bool = typer.Option(
        False, "--background", help="Start the server in the background and return immediately."
    ),
    kill: bool = typer.Option(
        False, "--kill", help="Stop a dashboard previously started with --background."
    ),
) -> None:
    """Launch the React dashboard for a backtest run."""
    if kill:
        _kill_dashboard(port)
        return

    if output_db is None:
        typer.echo("Error: --output-db is required.", err=True)
        raise typer.Exit(code=1)

    _require_output_db(output_db)

    try:
        import uvicorn
    except ImportError as exc:
        typer.echo("uvicorn is not installed. Run: pip install btkit[api]", err=True)
        raise typer.Exit(code=1) from exc

    # Set database paths before uvicorn imports the application module.
    os.environ["BTKIT_DB"] = str(Path(output_db).resolve())
    if input_db:
        os.environ["BTKIT_INPUT_DB"] = str(Path(input_db).resolve())

    # Refuse to start if the port is already bound — catches both background and
    # foreground cases before uvicorn tries and fails silently.
    existing_pid = _pid_on_port(port)
    if existing_pid:
        typer.echo(
            f"Error: port {port} is already in use by PID {existing_pid}. "
            f"Run 'btkit dashboard --kill --port {port}' to stop it.",
            err=True,
        )
        raise typer.Exit(code=1)

    if background:
        if not hasattr(os, "fork"):
            typer.echo("Error: --background is not supported on this platform.", err=True)
            raise typer.Exit(code=1)

        pid = os.fork()
        if pid != 0:
            # Parent: record the child PID and exit.
            # Use os._exit() — the only safe exit after fork(); raise/sys.exit
            # would run atexit handlers and flush shared file descriptors in
            # both processes, corrupting the child's stdio state.
            _DASHBOARD_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            _DASHBOARD_PID_FILE.write_text(str(pid))
            typer.echo(f"Dashboard started in background (PID {pid})")
            typer.echo(f"  http://localhost:{port}")
            typer.echo(f"  Stop with:  btkit dashboard --kill")
            os._exit(0)

        # Child: detach from the terminal and silence stdio.
        os.setsid()
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        for fd in (sys.stdin.fileno(), sys.stdout.fileno(), sys.stderr.fileno()):
            os.dup2(devnull_fd, fd)
        os.close(devnull_fd)

    uvicorn.run(
        "btkit.analysis.api.app:app",
        host="::",
        port=port,
        reload=debug,
        log_level="info" if debug else "warning",
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


@db_app.command("extend")
def db_extend(
    sources: list[str] = typer.Option(
        ..., help="One or more source output database paths to append."
    ),
    target: str = typer.Option(..., help="Target output database path (created if absent)."),
) -> None:
    """
    Extend a target database with the contents of one or more source databases.

    Example:
        btkit db extend --sources jan.db feb.db mar.db --target q1.db
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
