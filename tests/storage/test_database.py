from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from maisonaccueil.contracts import SCHEMA_VERSION
from maisonaccueil.storage import Database, SchemaVersionError

EXPECTED_TABLES = {
    "schema_meta", "profiles", "collection_sessions", "reference_images", "embeddings",
    "settings", "arrival_log",
}


def test_open_applies_schema_wal_fk_and_version(tmp_path: Path) -> None:
    with Database(tmp_path / "a.sqlite") as db:
        tables = {
            r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert EXPECTED_TABLES <= tables
        assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert db.schema_version == SCHEMA_VERSION
        assert db.media_root == (tmp_path / "media").resolve()


def test_reopen_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "a.sqlite"
    Database(p).close()
    with Database(p) as db:
        n = db.conn.execute("SELECT COUNT(*) FROM schema_meta").fetchone()[0]
        assert n == 1 and db.schema_version == SCHEMA_VERSION


def _force_version(p: Path, v: str) -> None:
    raw = sqlite3.connect(p)
    raw.execute("UPDATE schema_meta SET value = ? WHERE key = 'schema_version'", (v,))
    raw.commit()
    raw.close()


def test_refuses_newer_schema(tmp_path: Path) -> None:
    p = tmp_path / "a.sqlite"
    Database(p).close()
    _force_version(p, str(SCHEMA_VERSION + 1))
    with pytest.raises(SchemaVersionError, match="mettre à jour"):
        Database(p)


def test_refuses_older_schema_without_migration(tmp_path: Path) -> None:
    p = tmp_path / "a.sqlite"
    Database(p).close()
    _force_version(p, str(SCHEMA_VERSION - 1))
    with pytest.raises(SchemaVersionError, match="migration"):
        Database(p)


def test_refuses_garbage_version(tmp_path: Path) -> None:
    p = tmp_path / "a.sqlite"
    Database(p).close()
    _force_version(p, "abc")
    with pytest.raises(SchemaVersionError):
        Database(p)


def test_memory_db_refused() -> None:
    with pytest.raises(ValueError):
        Database(":memory:")


def test_transaction_rolls_back(tmp_path: Path) -> None:
    with Database(tmp_path / "a.sqlite") as db:
        with pytest.raises(RuntimeError), db.transaction() as c:
            c.execute("INSERT INTO settings(key, value) VALUES ('k', '1')")
            raise RuntimeError("boom")
        assert db.conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0


def test_one_connection_per_thread_and_close(tmp_path: Path) -> None:
    db = Database(tmp_path / "a.sqlite")
    main_conn = db.conn
    seen: list[sqlite3.Connection] = []
    t = threading.Thread(target=lambda: seen.append(db.conn))
    t.start()
    t.join()
    assert seen and seen[0] is not main_conn
    assert db.conn is main_conn
    db.close()
    db.close()  # idempotent
    assert db.closed
    with pytest.raises(RuntimeError):
        _ = db.conn


def test_concurrent_writers(tmp_path: Path) -> None:
    db = Database(tmp_path / "a.sqlite")
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for j in range(25):
                with db.transaction() as c:
                    c.execute(
                        "INSERT INTO settings(key, value) VALUES (?, ?)", (f"k{i}-{j}", "0")
                    )
        except BaseException as exc:  # pragma: no cover - remonté par l'assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert db.conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 150
    db.close()
