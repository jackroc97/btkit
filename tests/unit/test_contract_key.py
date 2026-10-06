"""
Unit tests for the synthetic contract_id.

The id must be stable (reproducible across processes and runs, so separately-built
databases agree and `btkit study merge` works), positive and BIGINT-sized, and
sensitive to every field that distinguishes a contract — in particular the
expiration, because CME symbols recycle on a 10-year cycle.

See docs/data_duplication_fix.md §7.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date

from btkit.db.contract_key import contract_id


class TestDeterminism:
    def test_same_inputs_give_same_id(self):
        a = contract_id("EW1Q4 P5000", date(2024, 8, 16))
        b = contract_id("EW1Q4 P5000", date(2024, 8, 16))
        assert a == b

    def test_stable_across_processes(self):
        """Guards against using the built-in hash(), whose string hashing is
        randomised per process unless PYTHONHASHSEED is pinned."""
        code = (
            "from datetime import date;"
            "from btkit.db.contract_key import contract_id;"
            "print(contract_id('EW1Q4 P5000', date(2024, 8, 16)))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PATH": ""},
            ).stdout.strip()
            for seed in ("0", "1", "random")
        }
        assert len(runs) == 1
        assert runs != {""}

    def test_value_is_pinned(self):
        """A literal regression value: changing the hash would silently invalidate
        every contract_id already written to a database."""
        assert contract_id("EW1Q4 P5000", date(2024, 8, 16)) == 5562283201705170097


class TestRange:
    def test_positive(self):
        assert contract_id("EW1Q4 P5000", date(2024, 8, 16)) > 0

    def test_fits_signed_bigint(self):
        assert contract_id("EW1Q4 P5000", date(2024, 8, 16)) < 2**63

    def test_many_symbols_stay_in_range(self):
        for strike in range(4000, 4200, 5):
            for right in ("P", "C"):
                value = contract_id(f"EW1Q4 {right}{strike}", date(2024, 8, 16))
                assert 0 < value < 2**63


class TestDistinctness:
    def test_expiration_changes_the_id(self):
        """The CME 10-year symbol cycle: the same symbol with different expirations
        is two different contracts."""
        assert contract_id("EW2H5 P2100", date(2015, 3, 13)) != contract_id(
            "EW2H5 P2100", date(2025, 3, 14)
        )

    def test_symbol_changes_the_id(self):
        assert contract_id("EW1Q4 P5000", date(2024, 8, 16)) != contract_id(
            "EW1Q4 P5005", date(2024, 8, 16)
        )

    def test_right_changes_the_id_via_symbol(self):
        assert contract_id("EW1Q4 P5000", date(2024, 8, 16)) != contract_id(
            "EW1Q4 C5000", date(2024, 8, 16)
        )

    def test_no_collisions_across_a_realistic_chain(self):
        ids = {
            contract_id(f"EW{week}{code} {right}{strike}", exp)
            for week in (1, 2, 3, 4)
            for code, exp in (("Q4", date(2024, 8, 16)), ("U4", date(2024, 9, 20)))
            for right in ("P", "C")
            for strike in range(4000, 4600, 5)
        }
        expected = 4 * 2 * 2 * len(range(4000, 4600, 5))
        assert len(ids) == expected

    def test_separator_is_not_ambiguous(self):
        """A symbol containing the separator must not collide with a different
        symbol/expiration split."""
        assert contract_id("A|2024-01-01", None) != contract_id("A", date(2024, 1, 1))


class TestNullExpiration:
    def test_null_expiration_is_allowed(self):
        assert contract_id("ESM4", None) > 0

    def test_null_differs_from_a_real_expiration(self):
        assert contract_id("ESM4", None) != contract_id("ESM4", date(2024, 6, 21))

    def test_null_is_deterministic(self):
        assert contract_id("ESM4", None) == contract_id("ESM4", None)
