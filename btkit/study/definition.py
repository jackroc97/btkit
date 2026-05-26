"""
Pydantic models for a study definition.

A study references one or more strategy YAML files and optional study-level
overrides for max_combinations and worker count. Strategy paths are relative
to the directory containing the study YAML.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class StrategyRef(BaseModel):
    path: str  # relative to the study YAML file's parent directory


class StudyDefinition(BaseModel):
    name: str
    strategies: list[StrategyRef]
    max_combinations: int | None = None  # overrides per-strategy matrix.max_combinations
    workers: int | None = None           # None → os.cpu_count()

    @model_validator(mode="after")
    def at_least_one_strategy(self) -> StudyDefinition:
        if not self.strategies:
            raise ValueError("study must list at least one strategy")
        return self
