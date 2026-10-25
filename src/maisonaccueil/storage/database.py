"""Ouverture de la base SQLite partagée (schéma ``contracts/schema.sql``).

Thread-safety : UNE connexion par thread (``threading.local``), créée à la demande.
Les écritures passent par ``Database.transaction()`` qui ouvre ``BEGIN IMMEDIATE`` :
SQLite (mode WAL) sérialise les écrivains entre connexions, avec ``busy_timeout``.
Aucune connexion n'est utilisée par un autre thread que celui qui l'a créée
(``close()`` excepté, à appeler quand plus aucun thread ne travaille).
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from maisonaccueil.contracts import SCHEMA_VERSION, schema_sql

SCHEMA_VERSION_KEY = "schema_version"


class SchemaVersionError(RuntimeError):
    """La base a été créée par une version du schéma que ce code ne sait pas gérer."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def to_utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Horodatage naïf refusé : fournir un datetime timezone-aware (UTC)")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class Database:
    """Base SQLite fichier.

    ``path``       : fichier de base (``:memory:`` refusé : une connexion par thread).
    ``media_root`` : dossier des images de référence (défaut : ``<dossier base>/media``).

    À l'ouverture : vérification de ``schema_meta.schema_version`` (refus d'une version
    supérieure, ou inférieure faute de migration), ``journal_mode=WAL``, application du
    schéma, écriture de la version. Chaque connexion active ``foreign_keys=ON``.
    """

    def __init__(
        self,
        path: str | Path,
        media_root: str | Path | None = None,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        if str(path) == ":memory:":
            raise ValueError("Base en mémoire non prise en charge (une connexion par thread)")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.media_root = (
            Path(media_root) if media_root is not None else self.path.parent / "media"
        ).resolve()
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._all_lock = threading.Lock()
        self._closed = False
        self._initialize()

    # -- connexions -----------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,  # autocommit ; transactions explicites
            check_same_thread=False,  # seulement pour close() global
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """Connexion propre au thread appelant."""
        if self._closed:
            raise RuntimeError("Database fermée")
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
            with self._all_lock:
                self._all.append(conn)
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Transaction d'écriture (``BEGIN IMMEDIATE``). Pas d'imbrication."""
        conn = self.conn
        if conn.in_transaction:
            raise RuntimeError("Transaction imbriquée non prise en charge")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def close(self) -> None:
        """Ferme toutes les connexions (idempotent)."""
        with self._all_lock:
            for conn in self._all:
                conn.close()
            self._all.clear()
        self._closed = True
        self._local = threading.local()

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- schéma ---------------------------------------------------------------------
    def _initialize(self) -> None:
        try:
            conn = self.conn
            existing = self._read_version(conn)
            if existing is not None and existing > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"Base en version de schéma {existing}, ce code gère la version "
                    f"{SCHEMA_VERSION} : mettre à jour l'application."
                )
            if existing is not None and existing < SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"Base en version de schéma {existing} < {SCHEMA_VERSION} : "
                    "aucune migration disponible."
                )
            mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RuntimeError(f"Impossible d'activer le mode WAL (obtenu : {mode})")
            conn.executescript(schema_sql())
            conn.execute("PRAGMA foreign_keys = ON")
            if existing is None:
                with self.transaction() as c:
                    c.execute(
                        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                        (SCHEMA_VERSION_KEY, str(SCHEMA_VERSION)),
                    )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _read_version(conn: sqlite3.Connection) -> int | None:
        has_meta = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'"
        ).fetchone()
        if not has_meta:
            return None
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = ?", (SCHEMA_VERSION_KEY,)
        ).fetchone()
        if row is None:
            return None
        try:
            return int(row[0])
        except ValueError as exc:
            raise SchemaVersionError(f"schema_version illisible : {row[0]!r}") from exc

    @property
    def schema_version(self) -> int | None:
        return self._read_version(self.conn)
