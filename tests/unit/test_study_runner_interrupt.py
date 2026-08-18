"""
Unit tests for StudyRunner's worker-cleanup helper, added so that
interrupting `btkit study run` (Ctrl-C or `kill <pid>`) terminates any
still-running ProcessPoolExecutor worker subprocesses instead of leaving them
as orphans. No existing test exercised StudyRunner directly before this.
"""

from __future__ import annotations

import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor

import pytest

from btkit.study.runner import _StudyTerminatedError, _terminate_pool_workers


def _sleep_worker(seconds: float) -> None:
    time.sleep(seconds)


class _FakeProcess:
    def __init__(self, alive: bool) -> None:
        self._alive = alive
        self.killed = False

    def is_alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        self.killed = True
        self._alive = False


class _FakePool:
    def __init__(self, processes: dict) -> None:
        self._processes = processes


class TestTerminatePoolWorkers:
    def test_kills_only_alive_processes(self):
        alive = _FakeProcess(alive=True)
        dead = _FakeProcess(alive=False)
        pool = _FakePool({1: alive, 2: dead})
        _terminate_pool_workers(pool)
        assert alive.killed is True
        assert dead.killed is False

    def test_no_processes_attribute_is_a_noop(self):
        class _NoProcessesPool:
            pass

        _terminate_pool_workers(_NoProcessesPool())  # must not raise

    def test_empty_processes_dict_is_a_noop(self):
        pool = _FakePool({})
        _terminate_pool_workers(pool)  # must not raise

    def test_kill_exception_is_swallowed(self):
        class _ExplodingProcess:
            def is_alive(self) -> bool:
                return True

            def kill(self) -> None:
                raise OSError("no such process")

        pool = _FakePool({1: _ExplodingProcess()})
        _terminate_pool_workers(pool)  # must not raise


class TestSigtermTerminatesRealWorkers:
    """
    End-to-end verification (real ProcessPoolExecutor, real subprocess, real
    SIGTERM) that a plain `kill <pid>` — with NO follow-up force-kill —
    actually terminates a still-running worker, mirroring StudyRunner.run()'s
    exact signal-handling shape. This is the scenario the original bug
    report's kill sequence couldn't isolate (it combined SIGTERM with an
    explicit follow-up pkill -9, so it couldn't tell which one did the work).
    """

    def test_sigterm_kills_running_worker_with_no_force_kill_followup(self):
        def _on_sigterm(signum, frame):
            raise _StudyTerminatedError(f"received signal {signum}")

        previous = signal.signal(signal.SIGTERM, _on_sigterm)
        pool = ProcessPoolExecutor(max_workers=1)
        try:
            future = pool.submit(_sleep_worker, 30)
            # Give the worker a moment to actually start before interrupting.
            time.sleep(0.5)
            assert not future.done(), "worker should still be sleeping"

            worker_proc = next(iter(pool._processes.values()))
            assert worker_proc.is_alive()

            try:
                os.kill(os.getpid(), signal.SIGTERM)
                pytest.fail("SIGTERM handler should have raised")
            except _StudyTerminatedError:
                # Mirrors StudyRunner.run()'s except block exactly — no
                # follow-up force-kill beyond what this code path does itself.
                # Order matters: shutdown() clears pool._processes, so the
                # hard-kill must happen first or there's nothing left to kill.
                _terminate_pool_workers(pool)
                pool.shutdown(wait=False, cancel_futures=True)

            deadline = time.monotonic() + 5.0
            while worker_proc.is_alive() and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not worker_proc.is_alive(), (
                "worker subprocess must be terminated by SIGTERM handling alone, "
                "with no external force-kill"
            )
        finally:
            signal.signal(signal.SIGTERM, previous)
            _terminate_pool_workers(pool)
            pool.shutdown(wait=False, cancel_futures=True)
