"""
DatabaseBuilder — orchestrates the full input database build from raw Databento files.

Build sequence:
    1. _ingest_definitions()  — read .dbn definition files, build instrument map
    2. _ingest_ohlcv()        — read .dbn OHLCV files, write underlying_bars and
                                option_bars (with definition metadata pre-joined)
    3. _compute_greeks()      — run GreeksCalculator over option_bars
    4. _run_indicators()      — run each user indicator script via IndicatorRunner

Called from cli.build() and cli.pipeline().
"""

from __future__ import annotations

from pathlib import Path

from btkit.db.input_db import InputDatabase
from btkit.pipeline.greeks import GreeksCalculator
from btkit.pipeline.indicators import IndicatorRunner


class DatabaseBuilder:
    def __init__(
        self,
        raw_data_path: str | Path,
        db_path: str | Path,
        indicator_scripts: list[str | Path] | None = None,
    ) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.db_path = Path(db_path)
        self.indicator_scripts = [Path(p) for p in (indicator_scripts or [])]

    def build(self) -> None:
        """
        Full build: ingest → greeks → indicators.
        Creates or overwrites the database at db_path.
        """
        self._ingest_definitions()
        self._ingest_ohlcv()
        self._compute_greeks()
        self._run_indicators()

    def _ingest_definitions(self) -> None:
        """
        Read .dbn definition files from raw_data_path. Build an internal
        instrument map (instrument_id → expiration, strike, right, multiplier)
        used when pre-joining metadata into option_bars during OHLCV ingest.
        """
        raise NotImplementedError

    def _ingest_ohlcv(self) -> None:
        """
        Read .dbn OHLCV-1m files from raw_data_path. Route each row to either
        underlying_bars (root instruments) or option_bars (options), pre-joining
        definition metadata into option_bars at write time so no joins are
        needed at backtest runtime.
        """
        raise NotImplementedError

    def _compute_greeks(self) -> None:
        """
        Instantiate GreeksCalculator and process all option_bars in batches,
        writing results to option_greeks.
        """
        raise NotImplementedError

    def _run_indicators(self) -> None:
        """
        Instantiate one IndicatorRunner per script in indicator_scripts and run
        each against the underlying_bars for the relevant underlying.
        Skipped if indicator_scripts is empty.
        """
        raise NotImplementedError
