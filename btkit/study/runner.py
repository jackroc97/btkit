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
import tempfile
import traceback as _tb
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from btkit.db.output_db import OutputDatabase
from btkit.study.definition import StudyDefinition
from btkit.study.expander import StudyExpander
from btkit.study.merger import OutputMerger


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
    ) -> None:
        self._study = study
        self._study_dir = study_dir
        self._study_yaml_text = study_yaml_text
        self._input_db_path = input_db_path
        self._output_db_path = output_db_path
        self._max_workers = max_workers or study.workers or os.cpu_count()
        self._max_combinations = max_combinations or study.max_combinations
        self._initial_equity = initial_equity

    def run(self) -> tuple[int, list[dict]]:
        """Execute the full study. Returns (study_id, failed_combinations)."""
        expander = StudyExpander(
            self._study, self._study_dir, self._max_combinations
        )
        combinations = expander.combinations  # raises ValueError if over limit

        # Insert study row first to get a stable study_id before any worker starts.
        with OutputDatabase(self._output_db_path) as odb:
            odb.create_schema()
            study_id = odb.write_study(
                name=self._study.name,
                strategy_yaml=self._study_yaml_text,
                total_combinations=len(combinations),
            )

        tmp_dir = tempfile.mkdtemp(prefix="btkit_study_")
        worker_db_paths: list[str] = []
        failed: list[dict] = []

        try:
            futures_map = {}
            with ProcessPoolExecutor(max_workers=self._max_workers) as pool:
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
                            failed.append({
                                "combination_id": cid,
                                "error": str(exc),
                            })
                        pbar.update(1)

            OutputMerger().merge(
                worker_db_paths=worker_db_paths,
                output_db_path=self._output_db_path,
                cleanup=True,
                tmp_dir=tmp_dir,
            )

        except Exception:
            # Attempt best-effort merge of whatever completed before the error.
            completed_paths = [p for p in worker_db_paths if Path(p).exists()]
            if completed_paths:
                try:
                    OutputMerger().merge(
                        worker_db_paths=completed_paths,
                        output_db_path=self._output_db_path,
                        cleanup=False,
                    )
                except Exception:
                    pass
            failed.append({"combination_id": -1, "error": _tb.format_exc()})

        with OutputDatabase(self._output_db_path) as odb:
            odb.finalize_study(study_id)

        return study_id, failed
