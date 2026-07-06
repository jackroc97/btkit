"""Unit tests for StudyExpander."""

from __future__ import annotations

from pathlib import Path

import pytest

from btkit.study.expander import StudyExpander
from btkit.study.definition import StudyDefinition, StrategyRef

STRATEGIES_DIR = Path(__file__).parent.parent / "fixtures" / "strategies"
STUDIES_DIR = Path(__file__).parent.parent / "fixtures" / "studies"


def _study(paths: list[str], max_combinations: int | None = None) -> tuple[StudyDefinition, Path]:
    return (
        StudyDefinition(
            name="test",
            strategies=[StrategyRef(path=p) for p in paths],
            max_combinations=max_combinations,
        ),
        STRATEGIES_DIR,
    )


class TestScalarStrategy:
    def test_scalar_produces_one_combination(self):
        study, study_dir = _study(["short_put_spread.yaml"])
        expander = StudyExpander(study, study_dir)
        combos = expander.combinations
        assert len(combos) == 1
        cid, defn = combos[0]
        assert cid == 1
        assert defn.name == "short_put_spread_test"

    def test_combination_id_is_one(self):
        study, study_dir = _study(["short_put_spread.yaml"])
        expander = StudyExpander(study, study_dir)
        assert expander.combinations[0][0] == 1


class TestSweepExpansion:
    def test_list_sweep_produces_cartesian_product(self, tmp_path):
        yaml = """
strategy:
  name: sweep_test
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument:
        root_symbol: ES
        asset_class: future
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: short_put
          right: put
          action: sell_to_open
          dte: 21
          delta:
            target: [-0.20, -0.25, -0.30]
      exit:
        stop_loss: [2.0, 3.0]
        take_profit: 1.0
"""
        f = tmp_path / "sweep.yaml"
        f.write_text(yaml)
        study = StudyDefinition(
            name="s", strategies=[StrategyRef(path="sweep.yaml")]
        )
        expander = StudyExpander(study, tmp_path)
        combos = expander.combinations
        # 3 deltas × 2 stop_losses = 6 combinations
        assert len(combos) == 6

    def test_combination_ids_sequential_from_one(self, tmp_path):
        yaml = """
strategy:
  name: sweep_test
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument:
        root_symbol: ES
        asset_class: future
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: sp
          right: put
          action: sell_to_open
          dte: 21
          delta:
            target: [-0.20, -0.25]
      exit:
        stop_loss: 2.0
        take_profit: 1.0
"""
        f = tmp_path / "s.yaml"
        f.write_text(yaml)
        study = StudyDefinition(name="s", strategies=[StrategyRef(path="s.yaml")])
        expander = StudyExpander(study, tmp_path)
        ids = [cid for cid, _ in expander.combinations]
        assert ids == [1, 2]

    def test_sweep_range_expansion(self, tmp_path):
        yaml = """
strategy:
  name: range_test
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument:
        root_symbol: ES
        asset_class: future
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: sp
          right: put
          action: sell_to_open
          dte:
            start: 14
            stop: 28
            step: 7
          delta:
            target: -0.25
      exit:
        stop_loss: 2.0
        take_profit: 1.0
"""
        f = tmp_path / "range.yaml"
        f.write_text(yaml)
        study = StudyDefinition(name="s", strategies=[StrategyRef(path="range.yaml")])
        expander = StudyExpander(study, tmp_path)
        combos = expander.combinations
        # dte: 14, 21, 28 → 3 combinations
        assert len(combos) == 3
        dte_values = [defn.trades[0].legs[0].dte for _, defn in combos]
        assert dte_values == [14, 21, 28]

    def test_override_applied_correctly(self, tmp_path):
        yaml = """
strategy:
  name: override_test
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument:
        root_symbol: ES
        asset_class: future
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: sp
          right: put
          action: sell_to_open
          dte: 21
          delta:
            target: [-0.20, -0.30]
      exit:
        stop_loss: 2.0
        take_profit: 1.0
"""
        f = tmp_path / "o.yaml"
        f.write_text(yaml)
        study = StudyDefinition(name="s", strategies=[StrategyRef(path="o.yaml")])
        expander = StudyExpander(study, tmp_path)
        combos = expander.combinations
        delta_values = [defn.trades[0].legs[0].delta.target for _, defn in combos]
        assert delta_values == [-0.20, -0.30]

    def test_all_combinations_fully_scalar(self, tmp_path):
        yaml = """
strategy:
  name: scalar_check
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument:
        root_symbol: ES
        asset_class: future
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: sp
          right: put
          action: sell_to_open
          dte: 21
          delta:
            target: [-0.20, -0.25]
      exit:
        stop_loss: 2.0
        take_profit: 1.0
"""
        f = tmp_path / "sc.yaml"
        f.write_text(yaml)
        study = StudyDefinition(name="s", strategies=[StrategyRef(path="sc.yaml")])
        expander = StudyExpander(study, tmp_path)
        for _, defn in expander.combinations:
            assert not defn.is_parameterized()


class TestMultiStrategy:
    def test_multi_strategy_concatenated(self, tmp_path):
        # Two scalar strategies → 2 combinations
        import shutil
        shutil.copy(STRATEGIES_DIR / "short_put_spread.yaml", tmp_path / "a.yaml")
        shutil.copy(STRATEGIES_DIR / "short_put_spread.yaml", tmp_path / "b.yaml")
        study = StudyDefinition(
            name="s",
            strategies=[StrategyRef(path="a.yaml"), StrategyRef(path="b.yaml")],
        )
        expander = StudyExpander(study, tmp_path)
        combos = expander.combinations
        assert len(combos) == 2
        assert combos[0][0] == 1
        assert combos[1][0] == 2

    def test_combination_ids_cross_strategies(self, tmp_path):
        # Strategy A: 2 sweep combos, Strategy B: scalar → IDs 1,2,3
        sweep_yaml = """
strategy:
  name: sweep
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument: {root_symbol: ES, asset_class: future}
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: sp
          right: put
          action: sell_to_open
          dte: 21
          delta:
            target: [-0.20, -0.25]
      exit:
        stop_loss: 2.0
        take_profit: 1.0
"""
        import shutil
        (tmp_path / "sweep.yaml").write_text(sweep_yaml)
        shutil.copy(STRATEGIES_DIR / "short_put_spread.yaml", tmp_path / "scalar.yaml")
        study = StudyDefinition(
            name="s",
            strategies=[StrategyRef(path="sweep.yaml"), StrategyRef(path="scalar.yaml")],
        )
        expander = StudyExpander(study, tmp_path)
        ids = [cid for cid, _ in expander.combinations]
        assert ids == [1, 2, 3]


class TestMaxCombinations:
    def test_raises_before_any_runs(self, tmp_path):
        yaml = """
strategy:
  name: big
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument: {root_symbol: ES, asset_class: future}
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: sp
          right: put
          action: sell_to_open
          dte: 21
          delta:
            target: [-0.10, -0.20, -0.30, -0.40, -0.50]
      exit:
        stop_loss: [1.0, 2.0, 3.0, 4.0]
        take_profit: 1.0
"""
        f = tmp_path / "big.yaml"
        f.write_text(yaml)
        study = StudyDefinition(
            name="s",
            strategies=[StrategyRef(path="big.yaml")],
            max_combinations=10,
        )
        expander = StudyExpander(study, tmp_path)
        with pytest.raises(ValueError, match="20.*max_combinations=10"):
            _ = expander.combinations


class TestParamsDf:
    def test_params_df_has_combination_id_and_strategy_name(self):
        study, study_dir = _study(["short_put_spread.yaml"])
        expander = StudyExpander(study, study_dir)
        df = expander.params_df
        assert "combination_id" in df.columns
        assert "strategy_name" in df.columns

    def test_params_df_includes_swept_values(self, tmp_path):
        yaml = """
strategy:
  name: s
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument: {root_symbol: ES, asset_class: future}
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: sp
          right: put
          action: sell_to_open
          dte: 21
          delta:
            target: [-0.20, -0.25]
      exit:
        stop_loss: 2.0
        take_profit: 1.0
"""
        f = tmp_path / "s.yaml"
        f.write_text(yaml)
        study = StudyDefinition(name="s", strategies=[StrategyRef(path="s.yaml")])
        expander = StudyExpander(study, tmp_path)
        df = expander.params_df
        assert "t1.sp.delta.target" in df.columns
        assert df["t1.sp.delta.target"].to_list() == [-0.20, -0.25]


class TestExplicitCombinations:
    def test_table_combinations(self, tmp_path):
        yaml = """
strategy:
  name: table_test
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument: {root_symbol: ES, asset_class: future}
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: sp
          right: put
          action: sell_to_open
          dte: 21
          delta:
            target: -0.25
      exit:
        stop_loss: 2.0
        take_profit: 1.0
  combinations:
    mode: table
    columns: ["t1.sp.delta.target", "t1.exit.stop_loss"]
    rows:
      - [-0.20, 2.0]
      - [-0.25, 3.0]
      - [-0.30, 4.0]
"""
        f = tmp_path / "t.yaml"
        f.write_text(yaml)
        study = StudyDefinition(name="s", strategies=[StrategyRef(path="t.yaml")])
        expander = StudyExpander(study, tmp_path)
        combos = expander.combinations
        assert len(combos) == 3
        deltas = [defn.trades[0].legs[0].delta.target for _, defn in combos]
        assert deltas == [-0.20, -0.25, -0.30]

    def test_unknown_trade_raises(self, tmp_path):
        yaml = """
strategy:
  name: bad
  universe:
    start_date: "2026-01-01"
    end_date: "2026-03-31"
  trades:
    - name: t1
      instrument: {root_symbol: ES, asset_class: future}
      entry:
        window: {start: "10:00", end: "12:00"}
      legs:
        - name: sp
          right: put
          action: sell_to_open
          dte: 21
          delta:
            target: -0.25
      exit:
        stop_loss: 2.0
        take_profit: 1.0
  combinations:
    mode: table
    columns: ["no_such_trade.sp.delta"]
    rows:
      - [-0.20]
"""
        f = tmp_path / "bad.yaml"
        f.write_text(yaml)
        study = StudyDefinition(name="s", strategies=[StrategyRef(path="bad.yaml")])
        expander = StudyExpander(study, tmp_path)
        with pytest.raises(ValueError, match="Unknown trade"):
            _ = expander.combinations
