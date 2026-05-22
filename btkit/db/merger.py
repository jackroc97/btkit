"""
OutputMerger — consolidates parallel worker output databases into one.

DEFERRED: not part of the MVP. Used by MatrixRunner after all parallel worker
processes complete. Each worker writes to its own isolated DB file to avoid
concurrent write contention; this class merges them into a single output file.
"""

from __future__ import annotations


class OutputMerger:
    """
    Reads each worker DB in sequence, re-sequences primary keys to be globally
    unique, preserves matrix_id and combination_id, writes all rows to a single
    output database, and optionally deletes the worker files.

    DEFERRED — not implemented for MVP.
    """

    def merge(
        self,
        worker_db_paths: list[str],
        output_db_path: str,
        cleanup: bool = True,
    ) -> None:
        """
        Merge all worker databases into output_db_path.

        Args:
            worker_db_paths: Paths to per-worker output DB files.
            output_db_path:  Destination path for the merged database.
            cleanup:         If True, delete worker DB files after merging.
        """
        raise NotImplementedError("OutputMerger is deferred — not available in MVP")
