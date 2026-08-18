"""
StudyRunner — orchestrates parallel execution of all study combinations.

Sequence:
    1. StudyExpander resolves all combinations (raises if > max_combinations).
    2. A study row is inserted into the output DB to get study_id.
    3. Each combination is submitted to a ProcessPoolExecutor worker.
       Workers open their own DB connections and write to isolated temp files.
    4. tqdm tracks progress as futures complete; per-combination failures are
       recorded but do not abort the run.
    5. OutputMerger consolidates worker DBs into the final output DB.
    6. The study row is finalised (finished_at timestamp).

The module-level _run_combination_worker() function is defined outside the class
so it is picklable by ProcessPoolExecutor.
"""

from __future__ import annotations

import os
import signal
import tempfile
import traceback as _tb
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from btkit.db.output_db import OutputDatabase
from btkit.study.definition import StudyDefinition
from btkit.study.expander import StudyExpander
from btkit.study.merger import OutputMerger


class _StudyTerminatedError(RuntimeError):
    """Raised from a SIGTERM handler so `kill <pid>` unwinds through the same
    worker-cleanup path as Ctrl-C (KeyboardInterrupt) instead of the default
    "die immediately, leave child processes orphaned" behavior."""


def _terminate_pool_workers(pool: ProcessPoolExecutor) -> None:
    """
    Best-effort hard-kill of any still-running worker subprocesses.

    pool.shutdown(cancel_futures=True) only cancels futures that haven't
    started yet — it does not stop workers already executing a combination,
    which is exactly the case that matters if a worker is hung (or merely
    slow) when the run is interrupted. concurrent.futures has no public API
    for this, so this reaches into the (long-stable, widely relied upon)
    private _processes mapping; if that ever changes shape, we degrade to a
    no-op rather than raise, since this only runs during shutdown.
    """
    processes = getattr(pool, "_processes", None)
    if not processes:
        return
    for proc in list(processes.values()):
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker function (module-level for picklability)
# ---------------------------------------------------------------------------


def _run_combination_worker(
    combination_id: int,
    study_id: int,
    strategy_defn_dict: dict,
    input_db_path: str,
    worker_db_path: str,
    initial_equity: float,
) -> None:
    """
    Runs in a subprocess. Opens its own DuckDB connections (never shared).
    BacktestEngine records errors in the output DB before re-raising, so the
    ProcessPoolExecutor future captures the exception for the supervisor to log.
    """
    from btkit.backtest.engine import BacktestEngine
    from btkit.db.input_db import InputDatabase
    from btkit.strategy.definition import StrategyDefinition

    defn = StrategyDefinition.model_validate(strategy_defn_dict)

    with InputDatabase(input_db_path) as idb, OutputDatabase(worker_db_path) as odb:
        odb.create_schema()
        engine = BacktestEngine(
            input_db=idb,
            output_db=odb,
            strategy=defn,
            initial_equity=initial_equity,
            study_id=study_id,
            combination_id=combination_id,
        )
        engine.run()  # engine records error then re-raises; let it propagate


# ---------------------------------------------------------------------------
# StudyRunner
# ---------------------------------------------------------------------------


class StudyRunner:
    """
    Orchestrates parallel execution of all combinations in a study.

    Returns (study_id, failed_combinations) where failed_combinations is a
    list of {"combination_id": int, "error": str} dicts for any workers that
    raised exceptions.
    """

    def __init__(
        self,
        study: StudyDefinition,
        study_dir: Path,
        study_yaml_text: str,
        input_db_path: str,
        output_db_path: str,
        max_workers: int | None = None,
        max_combinations: int | None = None,
        initial_equity: float = 100_000.0,
        note: str | None = None,
    ) -> None:
        self._study = study
        self._study_dir = study_dir
        self._study_yaml_text = study_yaml_text
        self._input_db_path = input_db_path
        self._output_db_path = output_db_path
        self._max_workers = max_workers or study.workers or os.cpu_count()
        self._max_combinations = max_combinations or study.max_combinations
        self._initial_equity = initial_equity
        self._note = note

    def run(self) -> tuple[int, list[dict]]:
        """Execute the full study. Returns (study_id, failed_combinations)."""
        expander = StudyExpander(self._study, self._study_dir, self._max_combinations)
        combinations = expander.combinations  # raises ValueError if over limit

        # Insert study row first to get a stable study_id before any worker starts.
        with OutputDatabase(self._output_db_path) as odb:
            odb.create_schema()
            study_id = odb.write_study(
                name=self._study.name,
                strategy_yaml=self._study_yaml_text,
                total_combinations=len(combinations),
                note=self._note,
            )

        tmp_dir = tempfile.mkdtemp(prefix="btkit_study_")
        worker_db_paths: list[str] = []
        failed: list[dict] = []

        def _on_sigterm(signum: int, frame: object) -> None:
            raise _StudyTerminatedError(f"received signal {signum}")

        # SIGTERM (a plain `kill <pid>`) doesn't raise anything catchable by
        # default — it terminates the process immediately, leaving any
        # already-running ProcessPoolExecutor workers as orphans since Python
        # never unwinds far enough to shut the pool down. Route it through the
        # same interrupt-handling path as Ctrl-C (SIGINT / KeyboardInterrupt).
        previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
        try:
            pool = ProcessPoolExecutor(max_workers=self._max_workers)
            futures_map = {}
            try:
                for combination_id, defn in combinations:
                    worker_db = str(Path(tmp_dir) / f"worker_{combination_id}.db")
                    worker_db_paths.append(worker_db)
                    future = pool.submit(
                        _run_combination_worker,
                        combination_id=combination_id,
                        study_id=study_id,
                        strategy_defn_dict=defn.model_dump(mode="json"),
                        input_db_path=self._input_db_path,
                        worker_db_path=worker_db,
                        initial_equity=self._initial_equity,
                    )
                    futures_map[future] = combination_id

                with tqdm(
                    total=len(combinations),
                    desc=f"Study '{self._study.name}'",
                    unit="combo",
                ) as pbar:
                    for future in as_completed(futures_map):
                        cid = futures_map[future]
                        exc = future.exception()
                        if exc is not None:
                            failed.append(
                                {
                                    "combination_id": cid,
                                    "error": str(exc),
                                }
                            )
                        pbar.update(1)

                pool.shutdown(wait=True)
                OutputMerger().merge(
                    source_db_paths=worker_db_paths,
                    target_db_path=self._output_db_path,
                    cleanup=True,
                    tmp_dir=tmp_dir,
                )

            except (KeyboardInterrupt, Exception):
                # Interrupted (Ctrl-C / SIGTERM) or some other failure (e.g. the
                # merge itself). Don't wait on cancel_futures=True alone — it only
                # cancels futures that haven't started, not ones already running
                # (possibly hung) in a worker — hard-kill any live worker process
                # directly. Order matters here: shutdown() clears pool._processes
                # (sets it to None) as part of tearing the executor down, so the
                # hard-kill must happen BEFORE shutdown(), not after — doing it
                # after is a silent no-op with nothing left to terminate.
                _terminate_pool_workers(pool)
                pool.shutdown(wait=False, cancel_futures=True)
                completed_paths = [p for p in worker_db_paths if Path(p).exists()]
                if completed_paths:
                    try:
                        OutputMerger().merge(
                            source_db_paths=completed_paths,
                            target_db_path=self._output_db_path,
                        )
                    except Exception:
                        pass
                failed.append({"combination_id": -1, "error": _tb.format_exc()})
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)

        with OutputDatabase(self._output_db_path) as odb:
            odb.finalize_study(study_id)

        return study_id, failed
