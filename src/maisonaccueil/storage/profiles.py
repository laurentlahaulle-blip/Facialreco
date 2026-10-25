"""Profils des personnes enregistrées (volontaires, consentement obligatoire)."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from maisonaccueil.contracts import ProfileId, new_profile_id

from .database import Database, from_iso, to_utc_iso, utc_now_iso
from .media import remove_managed_files


class ProfileNotFoundError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class Profile:
    profile_id: ProfileId
    display_name: str
    consent_at: datetime
    created_at: datetime
    updated_at: datetime
    active: bool = True
    spoken_name: str | None = None
    welcome_text: str | None = None
    welcome_audio: str | None = None
    consent_note: str | None = None


@dataclass(frozen=True, slots=True)
class DeletionReport:
    """Bilan d'une suppression complète de profil."""

    profile_id: ProfileId
    references_deleted: int
    embeddings_deleted: int
    sessions_deleted: int
    arrivals_deleted: int
    files_deleted: int
    files_skipped: tuple[str, ...]  # hors media_root (non effacés) ou erreur d'effacement


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_UPDATABLE = ("display_name", "spoken_name", "welcome_text", "welcome_audio", "consent_note")


def _row_to_profile(r: sqlite3.Row) -> Profile:
    return Profile(
        profile_id=ProfileId(r["profile_id"]),
        display_name=r["display_name"],
        consent_at=from_iso(r["consent_at"]),
        created_at=from_iso(r["created_at"]),
        updated_at=from_iso(r["updated_at"]),
        active=bool(r["active"]),
        spoken_name=r["spoken_name"],
        welcome_text=r["welcome_text"],
        welcome_audio=r["welcome_audio"],
        consent_note=r["consent_note"],
    )


class ProfileRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        display_name: str,
        *,
        consent_at: datetime,
        consent_note: str | None = None,
        spoken_name: str | None = None,
        welcome_text: str | None = None,
        welcome_audio: str | None = None,
        profile_id: ProfileId | None = None,
    ) -> Profile:
        """Crée un profil. ``consent_at`` (timezone-aware) est obligatoire."""
        if not isinstance(consent_at, datetime):
            raise ValueError("Consentement requis : consent_at doit être un datetime")
        consent_iso = to_utc_iso(consent_at)
        name = display_name.strip()
        if not name:
            raise ValueError("display_name vide")
        pid = profile_id or new_profile_id()
        if not _SAFE_ID.match(pid):
            raise ValueError(f"profile_id invalide : {pid!r}")
        now = utc_now_iso()
        with self._db.transaction() as c:
            c.execute(
                "INSERT INTO profiles(profile_id, display_name, spoken_name, welcome_text,"
                " welcome_audio, consent_at, consent_note, active, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (pid, name, spoken_name, welcome_text, welcome_audio, consent_iso,
                 consent_note, now, now),
            )
        return self.get(pid)

    def get(self, profile_id: ProfileId) -> Profile:
        row = self._db.conn.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise ProfileNotFoundError(profile_id)
        return _row_to_profile(row)

    def find(self, profile_id: ProfileId) -> Profile | None:
        try:
            return self.get(profile_id)
        except ProfileNotFoundError:
            return None

    def list(self, *, include_inactive: bool = False) -> list[Profile]:
        sql = "SELECT * FROM profiles"
        if not include_inactive:
            sql += " WHERE active = 1"
        sql += " ORDER BY display_name COLLATE NOCASE, profile_id"
        return [_row_to_profile(r) for r in self._db.conn.execute(sql)]

    def update(self, profile_id: ProfileId, **fields: str | None) -> Profile:
        """Modifie ``display_name``, ``spoken_name``, ``welcome_text``, ``welcome_audio``,
        ``consent_note``. Le consentement se renouvelle via ``renew_consent``."""
        unknown = set(fields) - set(_UPDATABLE)
        if unknown:
            raise ValueError(f"Champs non modifiables : {sorted(unknown)}")
        if "display_name" in fields:
            dn = (fields["display_name"] or "").strip()
            if not dn:
                raise ValueError("display_name vide")
            fields["display_name"] = dn
        if not fields:
            return self.get(profile_id)
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._db.transaction() as c:
            cur = c.execute(
                f"UPDATE profiles SET {cols}, updated_at = ? WHERE profile_id = ?",
                (*fields.values(), utc_now_iso(), profile_id),
            )
            if cur.rowcount == 0:
                raise ProfileNotFoundError(profile_id)
        return self.get(profile_id)

    def renew_consent(
        self, profile_id: ProfileId, consent_at: datetime, note: str | None = None
    ) -> Profile:
        with self._db.transaction() as c:
            cur = c.execute(
                "UPDATE profiles SET consent_at = ?, consent_note = ?, updated_at = ?"
                " WHERE profile_id = ?",
                (to_utc_iso(consent_at), note, utc_now_iso(), profile_id),
            )
            if cur.rowcount == 0:
                raise ProfileNotFoundError(profile_id)
        return self.get(profile_id)

    def set_active(self, profile_id: ProfileId, active: bool) -> Profile:
        with self._db.transaction() as c:
            cur = c.execute(
                "UPDATE profiles SET active = ?, updated_at = ? WHERE profile_id = ?",
                (1 if active else 0, utc_now_iso(), profile_id),
            )
            if cur.rowcount == 0:
                raise ProfileNotFoundError(profile_id)
        return self.get(profile_id)

    def deactivate(self, profile_id: ProfileId) -> Profile:
        return self.set_active(profile_id, False)

    def activate(self, profile_id: ProfileId) -> Profile:
        return self.set_active(profile_id, True)

    def delete(self, profile_id: ProfileId, *, purge_arrivals: bool = True) -> DeletionReport:
        """Suppression COMPLÈTE : profil, séances, références, représentations, fichiers
        images locaux (sous ``media_root``) et, par défaut, lignes du journal d'arrivées
        de ce profil (sinon elles restent avec ``profile_id`` NULL, par le schéma).

        Les lignes sont supprimées dans une transaction ; les fichiers ensuite. Un fichier
        hors ``media_root`` n'est jamais effacé et est signalé dans ``files_skipped``.
        """
        with self._db.transaction() as c:
            exists = c.execute(
                "SELECT 1 FROM profiles WHERE profile_id = ?", (profile_id,)
            ).fetchone()
            if exists is None:
                raise ProfileNotFoundError(profile_id)
            paths = [
                r[0]
                for r in c.execute(
                    "SELECT file_path FROM reference_images WHERE profile_id = ?", (profile_id,)
                )
            ]
            n_emb = c.execute(
                "DELETE FROM embeddings WHERE profile_id = ?", (profile_id,)
            ).rowcount
            n_ref = c.execute(
                "DELETE FROM reference_images WHERE profile_id = ?", (profile_id,)
            ).rowcount
            n_ses = c.execute(
                "DELETE FROM collection_sessions WHERE profile_id = ?", (profile_id,)
            ).rowcount
            n_arr = 0
            if purge_arrivals:
                n_arr = c.execute(
                    "DELETE FROM arrival_log WHERE profile_id = ?", (profile_id,)
                ).rowcount
            c.execute("DELETE FROM profiles WHERE profile_id = ?", (profile_id,))
        removed, skipped = remove_managed_files(self._db.media_root, paths)
        profile_dir = self._db.media_root / profile_id
        if profile_dir.is_dir() and not any(profile_dir.iterdir()):
            profile_dir.rmdir()
        return DeletionReport(
            profile_id=profile_id,
            references_deleted=n_ref,
            embeddings_deleted=n_emb,
            sessions_deleted=n_ses,
            arrivals_deleted=n_arr,
            files_deleted=removed,
            files_skipped=tuple(skipped),
        )
