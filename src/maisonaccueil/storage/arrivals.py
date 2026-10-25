"""Journal minimal des accueils (``arrival_log``), sans image, purgeable."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from maisonaccueil.contracts import ArrivalEvent, ArrivalKind, EventId, ProfileId

from .database import Database, from_iso, to_utc_iso


@dataclass(frozen=True, slots=True)
class ArrivalLogEntry:
    event_id: EventId
    kind: ArrivalKind
    occurred_at: datetime
    profile_id: ProfileId | None = None  # NULL aussi si le profil a été supprimé
    model_key: str | None = None
    policy: str | None = None


def _row(r: sqlite3.Row) -> ArrivalLogEntry:
    return ArrivalLogEntry(
        event_id=EventId(r["event_id"]),
        kind=ArrivalKind(r["kind"]),
        occurred_at=from_iso(r["occurred_at"]),
        profile_id=ProfileId(r["profile_id"]) if r["profile_id"] is not None else None,
        model_key=r["model_key"],
        policy=r["policy"],
    )


class ArrivalLogRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def record(
        self, event: ArrivalEvent, *, model_key: str | None = None, policy: str | None = None
    ) -> bool:
        """Enregistre l'événement (horodatage ``event.wall_time``). Idempotent :
        retourne False si ``event_id`` est déjà journalisé."""
        with self._db.transaction() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO arrival_log(event_id, kind, profile_id, occurred_at,"
                " model_key, policy) VALUES (?, ?, ?, ?, ?, ?)",
                (event.event_id, str(event.kind), event.profile_id,
                 to_utc_iso(event.wall_time), model_key, policy),
            )
            return cur.rowcount > 0

    def list_recent(
        self, limit: int = 100, *, profile_id: ProfileId | None = None
    ) -> list[ArrivalLogEntry]:
        sql = "SELECT * FROM arrival_log"
        args: list[object] = []
        if profile_id is not None:
            sql += " WHERE profile_id = ?"
            args.append(profile_id)
        sql += " ORDER BY occurred_at DESC, event_id DESC LIMIT ?"
        args.append(int(limit))
        return [_row(r) for r in self._db.conn.execute(sql, args)]

    def count(self) -> int:
        return int(self._db.conn.execute("SELECT COUNT(*) FROM arrival_log").fetchone()[0])

    def purge_older_than(self, cutoff: datetime) -> int:
        """Supprime les entrées strictement antérieures à ``cutoff`` (timezone-aware).

        Comparaison sur le texte ISO-8601 UTC : valable car toutes les dates sont écrites
        par ``to_utc_iso`` (même fuseau +00:00, même format)."""
        with self._db.transaction() as c:
            return int(
                c.execute(
                    "DELETE FROM arrival_log WHERE occurred_at < ?", (to_utc_iso(cutoff),)
                ).rowcount
            )

    def purge_all(self) -> int:
        with self._db.transaction() as c:
            return int(c.execute("DELETE FROM arrival_log").rowcount)
