"""Unit tests for None-valued sweep parameters.

Covers the case where a user wants to sweep over "disabled" as one of the
values, e.g. stop_loss: [null, 2.0] or take_profit_pct: [null, 0.70].
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from btkit.study.definition import StudyDefinition, StrategyRef
from btkit.study.expander import StudyExpander


def _write_strategy(tmp_path: Path, yaml: str) -> StudyExpander:
    f = tmp_path / "s.yaml"
    f.write_text(dedent(yaml))
    study = StudyDefinition(name="s", strategies=[StrategyRef(path="s.yaml")])
    return StudyExpander(study, tmp_path)


# ---------------------------------------------------------------------------
# stop_loss: [null, value]
# ---------------------------------------------------------------------------


class TestNoneStopLoss:
    _YAML = """\
        strategy:
          name: sl_sweep
          universe:
            start_date: "2026-01-01"
            end_date: "2026-03-31"
          trades:
            - name: t1
              instrument: {{root_symbol: ES, asset_class: future}}
              entry:
                window: {{start: "10:00", end: "12:00"}}
              legs:
                - name: sp
                  right: put
                  action: sell_to_open
                  dte: 21
                  delta:
                    target: -0.25
              exit:
                stop_loss: {stop_loss}
                take_profit: 0.50
        """

    def test_none_and_value_produces_two_combinations(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(stop_loss="[null, 2.0]")
        )
        assert len(expander.combinations) == 2

    def test_none_combination_has_null_stop_loss(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(stop_loss="[null, 2.0]")
        )
        combos = expander.combinations
        sl_values = [defn.trades[0].exit.stop_loss for _, defn in combos]
        assert sl_values[0] is None
        assert float(sl_values[1]) == 2.0  # type: ignore[arg-type]

    def test_three_values_including_none(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(stop_loss="[null, 1.5, 2.0]")
        )
        combos = expander.combinations
        assert len(combos) == 3
        sl_values = [defn.trades[0].exit.stop_loss for _, defn in combos]
        assert sl_values[0] is None
        assert float(sl_values[1]) == 1.5  # type: ignore[arg-type]
        assert float(sl_values[2]) == 2.0  # type: ignore[arg-type]

    def test_is_parameterized_true(self, tmp_path):
        from btkit.strategy.loader import load_strategy

        f = tmp_path / "s.yaml"
        f.write_text(dedent(self._YAML.format(stop_loss="[null, 2.0]")))
        defn = load_strategy(f)
        assert defn.is_parameterized()

    def test_combination_ids_sequential(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(stop_loss="[null, 2.0]")
        )
        ids = [cid for cid, _ in expander.combinations]
        assert ids == [1, 2]

    def test_all_combinations_fully_scalar(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(stop_loss="[null, 2.0]")
        )
        for _, defn in expander.combinations:
            assert not defn.is_parameterized()


# ---------------------------------------------------------------------------
# take_profit: [null, value]
# ---------------------------------------------------------------------------


class TestNoneTakeProfit:
    _YAML = """\
        strategy:
          name: tp_sweep
          universe:
            start_date: "2026-01-01"
            end_date: "2026-03-31"
          trades:
            - name: t1
              instrument: {{root_symbol: ES, asset_class: future}}
              entry:
                window: {{start: "10:00", end: "12:00"}}
              legs:
                - name: sp
                  right: put
                  action: sell_to_open
                  dte: 21
                  delta:
                    target: -0.25
              exit:
                stop_loss: 2.0
                take_profit: {take_profit}
        """

    def test_none_and_value_produces_two_combinations(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(take_profit="[null, 0.50]")
        )
        assert len(expander.combinations) == 2

    def test_none_combination_has_null_take_profit(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(take_profit="[null, 0.50]")
        )
        combos = expander.combinations
        tp_values = [defn.trades[0].exit.take_profit for _, defn in combos]
        assert tp_values[0] is None
        assert float(tp_values[1]) == 0.50  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# take_profit_pct: [null, value]  (pct-based sweep)
# ---------------------------------------------------------------------------


class TestNoneTakeProfitPct:
    _YAML = """\
        strategy:
          name: tp_pct_sweep
          universe:
            start_date: "2026-01-01"
            end_date: "2026-03-31"
          trades:
            - name: t1
              instrument: {{root_symbol: ES, asset_class: future}}
              entry:
                window: {{start: "10:00", end: "12:00"}}
              legs:
                - name: sp
                  right: put
                  action: sell_to_open
                  dte: 21
                  delta:
                    target: -0.25
              exit:
                stop_loss: 2.0
                take_profit_pct: {take_profit_pct}
        """

    def test_none_and_pct_produces_two_combinations(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(take_profit_pct="[null, 0.70]")
        )
        assert len(expander.combinations) == 2

    def test_none_combination_has_null_pct(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(take_profit_pct="[null, 0.70]")
        )
        combos = expander.combinations
        pct_values = [defn.trades[0].exit.take_profit_pct for _, defn in combos]
        assert pct_values[0] is None
        assert float(pct_values[1]) == 0.70  # type: ignore[arg-type]

    def test_pct_values_correct_across_three(self, tmp_path):
        expander = _write_strategy(
            tmp_path, self._YAML.format(take_profit_pct="[null, 0.50, 0.70]")
        )
        combos = expander.combinations
        assert len(combos) == 3
        pct_values = [defn.trades[0].exit.take_profit_pct for _, defn in combos]
        assert pct_values[0] is None
        assert float(pct_values[1]) == 0.50  # type: ignore[arg-type]
        assert float(pct_values[2]) == 0.70  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cross-sweep with None: [null, 2.0] × [null, 0.70]
# ---------------------------------------------------------------------------


def _cross_sweep_expander(tmp_path: Path) -> StudyExpander:
    yaml = """\
        strategy:
          name: cross_sweep
          universe:
            start_date: "2026-01-01"
            end_date: "2026-03-31"
          trades:
            - name: t1
              instrument:
                root_symbol: ES
                asset_class: future
              entry:
                window:
                  start: "10:00"
                  end: "12:00"
              legs:
                - name: sp
                  right: put
                  action: sell_to_open
                  dte: 21
                  delta:
                    target: -0.25
              exit:
                stop_loss: [null, 2.0]
                take_profit_pct: [null, 0.70]
        """
    return _write_strategy(tmp_path, yaml)


class TestCrossNoneSweep:
    def test_cartesian_product_four_combinations(self, tmp_path):
        expander = _cross_sweep_expander(tmp_path)
        # [None, 2.0] × [None, 0.70] = 4
        assert len(expander.combinations) == 4

    def test_no_sl_no_tp_combination_exists(self, tmp_path):
        expander = _cross_sweep_expander(tmp_path)
        combos = [(defn.trades[0].exit.stop_loss, defn.trades[0].exit.take_profit_pct)
                  for _, defn in expander.combinations]
        assert (None, None) in combos

    def test_all_sl_tp_pairs_present(self, tmp_path):
        expander = _cross_sweep_expander(tmp_path)
        combos = {
            (defn.trades[0].exit.stop_loss, defn.trades[0].exit.take_profit_pct)
            for _, defn in expander.combinations
        }
        assert combos == {(None, None), (None, 0.70), (2.0, None), (2.0, 0.70)}


# ---------------------------------------------------------------------------
# Table combinations with None values
# ---------------------------------------------------------------------------


class TestTableCombinationsWithNone:
    def test_table_row_with_none(self, tmp_path):
        yaml = """\
            strategy:
              name: table_none
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
                    take_profit: 0.50
              combinations:
                mode: table
                columns: ["t1.exit.stop_loss", "t1.exit.take_profit"]
                rows:
                  - [null, 0.50]
                  - [2.0, null]
                  - [2.0, 0.50]
        """
        expander = _write_strategy(tmp_path, yaml)
        combos = expander.combinations
        assert len(combos) == 3
        results = [
            (defn.trades[0].exit.stop_loss, defn.trades[0].exit.take_profit)
            for _, defn in combos
        ]
        assert results[0] == (None, 0.50)
        assert results[1] == (2.0, None)
        assert results[2] == (2.0, 0.50)
