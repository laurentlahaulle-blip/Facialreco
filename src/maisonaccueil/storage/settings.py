"""Réglages applicatifs (table ``settings``, valeurs JSON)."""

from __future__ import annotations

import json
from typing import Any

from .database import Database


class SettingsRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, key: str, default: Any = None) -> Any:
        row = self._db.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        """Enregistre ``value`` (sérialisable JSON strict : pas de NaN/Inf)."""
        if not key:
            raise ValueError("Clé de réglage vide")
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        with self._db.transaction() as c:
            c.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, payload),
            )

    def set_many(self, values: dict[str, Any]) -> None:
        payloads = {
            k: json.dumps(v, ensure_ascii=False, allow_nan=False, sort_keys=True)
            for k, v in values.items()
        }
        if any(not k for k in payloads):
            raise ValueError("Clé de réglage vide")
        with self._db.transaction() as c:
            c.executemany(
                "INSERT INTO settings(key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                list(payloads.items()),
            )

    def delete(self, key: str) -> bool:
        with self._db.transaction() as c:
            return c.execute("DELETE FROM settings WHERE key = ?", (key,)).rowcount > 0

    def all(self) -> dict[str, Any]:
        rows = self._db.conn.execute("SELECT key, value FROM settings ORDER BY key")
        return {r[0]: json.loads(r[1]) for r in rows}
