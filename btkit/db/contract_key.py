"""
Synthetic contract identifiers.

`instrument_id` comes from the publisher (tag 48 SecurityID for GLBX.MDP3) and is
only guaranteed unique *within a day* — Databento recycles it, and 23% of the
contracts in the ES database share one with another contract. It therefore cannot
serve as a stable join key, and every consumer that uses it has to defend itself
against recycling.

`contract_id` is a stable surrogate derived from `(symbol, expiration)`, which was
validated against 74.5M ES option bars as uniquely determining a contract's
economics:

    (symbol, expiration) -> >1 (strike, right, multiplier)   0 violations
    (symbol, expiration) -> >1 underlying_id                 0 violations
    distinct contracts                                       665,028
    distinct instrument_ids                                  538,686

`symbol` alone is not enough: CME symbols recycle on a 10-year cycle, so 303 option
symbols and every futures symbol span more than one expiration (`EW2H5 P2100`
expires both 2015-03-13 and 2025-03-14). The expiration disambiguates them.

The id is a hash rather than a sequence so that it is reproducible: `btkit study
merge` combines separately-built databases, and a sequence would assign the same
contract different ids in each. blake2b is used rather than the built-in hash(),
whose string hashing is randomised per process by PYTHONHASHSEED.

See docs/data_duplication_fix.md §7.
"""

from __future__ import annotations

from datetime import date
from hashlib import blake2b

# 63 bits: fits DuckDB's signed BIGINT and stays positive, so the id survives
# round-tripping through tools that treat the high bit as a sign.
_MASK_63 = (1 << 63) - 1


def contract_id(symbol: str, expiration: date | None) -> int:
    """Return the stable synthetic id for one contract.

    Deterministic across processes, machines and runs. A null expiration (a
    continuous or malformed definition) hashes to its own bucket rather than
    raising, so ingest never fails on an odd vendor record.

    Collision risk is negligible — about 5e-8 for a million contracts in 2^63 — and
    a collision could not pass silently anyway: contract_id is the primary key of
    the `contract` table, so two different contracts hashing alike would fail the
    insert rather than merge.
    """
    exp = expiration.isoformat() if expiration is not None else ""
    digest = blake2b(f"{symbol}|{exp}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK_63
