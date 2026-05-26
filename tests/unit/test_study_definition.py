"""Unit tests for StudyDefinition and load_study()."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from btkit.study.definition import StrategyRef, StudyDefinition
from btkit.study.loader import load_study

STUDIES_DIR = Path(__file__).parent.parent / "fixtures" / "studies"


class TestStudyDefinition:
    def test_minimal_valid(self):
        d = StudyDefinition(name="s", strategies=[StrategyRef(path="a.yaml")])
        assert d.name == "s"
        assert len(d.strategies) == 1

    def test_multi_strategy(self):
        d = StudyDefinition(
            name="multi",
            strategies=[StrategyRef(path="a.yaml"), StrategyRef(path="b.yaml")],
        )
        assert len(d.strategies) == 2

    def test_empty_strategies_raises(self):
        with pytest.raises(ValidationError, match="at least one strategy"):
            StudyDefinition(name="s", strategies=[])

    def test_max_combinations_stored(self):
        d = StudyDefinition(
            name="s",
            strategies=[StrategyRef(path="a.yaml")],
            max_combinations=50,
        )
        assert d.max_combinations == 50

    def test_workers_stored(self):
        d = StudyDefinition(
            name="s",
            strategies=[StrategyRef(path="a.yaml")],
            workers=4,
        )
        assert d.workers == 4

    def test_defaults_are_none(self):
        d = StudyDefinition(name="s", strategies=[StrategyRef(path="a.yaml")])
        assert d.max_combinations is None
        assert d.workers is None


class TestLoadStudy:
    def test_loads_valid_study(self):
        defn, study_dir = load_study(STUDIES_DIR / "simple_study.yaml")
        assert defn.name == "simple_study"
        assert len(defn.strategies) == 1
        assert study_dir == STUDIES_DIR

    def test_missing_study_key_raises(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("name: oops\n")
        with pytest.raises(ValueError, match="top-level 'study' key"):
            load_study(f)

    def test_returns_parent_dir(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        f = subdir / "my_study.yaml"
        f.write_text("study:\n  name: s\n  strategies:\n    - path: x.yaml\n")
        _, study_dir = load_study(f)
        assert study_dir == subdir

    def test_invalid_structure_raises(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("study:\n  strategies: []\n")  # empty strategies
        with pytest.raises(ValidationError):
            load_study(f)
