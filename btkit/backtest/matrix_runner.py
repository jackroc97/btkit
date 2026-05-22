"""
MatrixRunner — parallel execution of a StrategyMatrix.

DEFERRED: not part of the MVP. Each combination runs in an isolated worker
process with its own output database file. Results are merged into a single
output database by OutputMerger on completion.
"""

from __future__ import annotations

from btkit.strategy.matrix import StrategyMatrix


class MatrixRunner:
    """
    Dispatches StrategyMatrix combinations to a ProcessPoolExecutor.
    Each worker creates its own DuckDB connections (never shared across processes).
    Progress tracked with tqdm.

    DEFERRED — not implemented for MVP.
    """

    def __init__(
        self,
        matrix: StrategyMatrix,
        input_db_path: str,
        output_db_path: str,
        max_workers: int | None = None,
    ) -> None:
        raise NotImplementedError("MatrixRunner is deferred — not available in MVP")

    def run(self) -> None:
        """
        1. Assign each combination a worker-specific output DB path.
        2. Submit to ProcessPoolExecutor with max_workers.
        3. Track progress with tqdm as futures complete.
        4. On full completion, call OutputMerger to consolidate worker DBs.
        """
        raise NotImplementedError
