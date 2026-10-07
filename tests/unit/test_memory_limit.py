"""
Tests for the query memory-limit input.

DuckDB's own default buffer pool is ~80% of physical RAM, which let a single
backtest drive a 17 GB host into swap — 5.3 GB of 6.1 GB used, with one backtest
spending 21 of 31 elapsed minutes paging rather than computing. The limit is now an
explicit input with a safe fallback, and the CLI reports which source applied so a
silent fallback to the default is visible rather than inferred from a slow run.
"""

from __future__ import annotations

import duckdb
import pytest

from btkit.db.input_db import (
    DEFAULT_QUERY_MEMORY_LIMIT,
    INPUT_SCHEMA_SQL,
    MEMORY_LIMIT_ARGUMENT,
    MEMORY_LIMIT_DEFAULT,
    MEMORY_LIMIT_ENV,
    InputDatabase,
    describe_memory_limit,
    resolve_memory_limit,
)


@pytest.fixture()
def db_path(tmp_path):
    p = tmp_path / "in.db"
    con = duckdb.connect(str(p))
    con.execute(INPUT_SCHEMA_SQL)
    con.close()
    return str(p)


class TestResolution:
    def test_explicit_argument_wins(self, monkeypatch):
        monkeypatch.setenv(MEMORY_LIMIT_ENV, "12GB")
        assert resolve_memory_limit("8GB") == ("8GB", MEMORY_LIMIT_ARGUMENT)

    def test_environment_used_when_no_argument(self, monkeypatch):
        monkeypatch.setenv(MEMORY_LIMIT_ENV, "12GB")
        assert resolve_memory_limit(None) == ("12GB", MEMORY_LIMIT_ENV)

    def test_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv(MEMORY_LIMIT_ENV, raising=False)
        assert resolve_memory_limit(None) == (DEFAULT_QUERY_MEMORY_LIMIT, MEMORY_LIMIT_DEFAULT)

    def test_default_is_four_gb(self):
        """A safe ceiling for a 16-17 GB host; raising it is a deliberate choice."""
        assert DEFAULT_QUERY_MEMORY_LIMIT == "4GB"

    def test_empty_string_is_treated_as_unset(self, monkeypatch):
        monkeypatch.delenv(MEMORY_LIMIT_ENV, raising=False)
        assert resolve_memory_limit("") == (DEFAULT_QUERY_MEMORY_LIMIT, MEMORY_LIMIT_DEFAULT)

    def test_empty_env_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(MEMORY_LIMIT_ENV, "")
        assert resolve_memory_limit(None) == (DEFAULT_QUERY_MEMORY_LIMIT, MEMORY_LIMIT_DEFAULT)


class TestMessage:
    def test_default_is_called_out_explicitly(self):
        msg = describe_memory_limit(DEFAULT_QUERY_MEMORY_LIMIT, MEMORY_LIMIT_DEFAULT)
        assert "DEFAULT" in msg
        assert DEFAULT_QUERY_MEMORY_LIMIT in msg
        assert MEMORY_LIMIT_ENV in msg

    def test_argument_source_named(self):
        msg = describe_memory_limit("8GB", MEMORY_LIMIT_ARGUMENT)
        assert "8GB" in msg
        assert "--memory-limit" in msg
        assert "DEFAULT" not in msg

    def test_env_source_named(self):
        msg = describe_memory_limit("12GB", MEMORY_LIMIT_ENV)
        assert "12GB" in msg
        assert MEMORY_LIMIT_ENV in msg
        assert "DEFAULT" not in msg

    def test_message_is_one_line(self):
        for limit, source in (
            ("4GB", MEMORY_LIMIT_DEFAULT),
            ("8GB", MEMORY_LIMIT_ARGUMENT),
            ("12GB", MEMORY_LIMIT_ENV),
        ):
            assert "\n" not in describe_memory_limit(limit, source)


class TestApplied:
    def test_limit_reaches_the_connection(self, db_path, monkeypatch):
        monkeypatch.delenv(MEMORY_LIMIT_ENV, raising=False)
        with InputDatabase(db_path, memory_limit="2GB") as db:
            setting = db._con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            assert "1.8 GiB" in setting or "2.0 GiB" in setting
            assert db.memory_limit == "2GB"
            assert db.memory_limit_source == MEMORY_LIMIT_ARGUMENT

    def test_default_applied_when_unset(self, db_path, monkeypatch):
        monkeypatch.delenv(MEMORY_LIMIT_ENV, raising=False)
        with InputDatabase(db_path) as db:
            assert db.memory_limit == DEFAULT_QUERY_MEMORY_LIMIT
            assert db.memory_limit_source == MEMORY_LIMIT_DEFAULT

    def test_env_applied_when_unset(self, db_path, monkeypatch):
        monkeypatch.setenv(MEMORY_LIMIT_ENV, "3GB")
        with InputDatabase(db_path) as db:
            assert db.memory_limit == "3GB"
            assert db.memory_limit_source == MEMORY_LIMIT_ENV

    def test_temp_directory_is_set_so_spilling_has_somewhere_to_go(self, db_path):
        with InputDatabase(db_path) as db:
            temp = db._con.execute("SELECT current_setting('temp_directory')").fetchone()[0]
            assert temp
