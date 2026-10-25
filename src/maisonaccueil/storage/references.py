"""Séances de collecte et images de référence (fichiers locaux sous ``media_root``)."""

from __future__ import annotations

import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from maisonaccueil.contracts import CameraId, ProfileId, SessionId, new_session_id

from .database import Database, from_iso, to_utc_iso
from .media import remove_managed_files, sha256_bytes, sha256_file, write_atomic

SPLITS = frozenset({"train", "tune", "test", "enroll"})
_ALLOWED_EXT = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


class ReferenceNotFoundError(KeyError):
    pass


class SessionNotFoundError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class CollectionSession:
    session_id: SessionId
    profile_id: ProfileId | None  # None = séance de personnes non enregistrées
    started_at: datetime
    camera_id: CameraId | None = None
    conditions: str | None = None
    split: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceImage:
    reference_id: int
    profile_id: ProfileId
    file_path: Path
    sha256: str
    captured_at: datetime
    session_id: SessionId | None = None
    width: int | None = None
    height: int | None = None
    quality_note: str | None = None


def _row_to_session(r: sqlite3.Row) -> CollectionSession:
    return CollectionSession(
        session_id=SessionId(r["session_id"]),
        profile_id=ProfileId(r["profile_id"]) if r["profile_id"] is not None else None,
        started_at=from_iso(r["started_at"]),
        camera_id=CameraId(r["camera_id"]) if r["camera_id"] is not None else None,
        conditions=r["conditions"],
        split=r["split"],
    )


def _row_to_reference(r: sqlite3.Row) -> ReferenceImage:
    return ReferenceImage(
        reference_id=int(r["reference_id"]),
        profile_id=ProfileId(r["profile_id"]),
        file_path=Path(r["file_path"]),
        sha256=r["sha256"],
        captured_at=from_iso(r["captured_at"]),
        session_id=SessionId(r["session_id"]) if r["session_id"] is not None else None,
        width=r["width"],
        height=r["height"],
        quality_note=r["quality_note"],
    )


def _check_split(split: str | None) -> None:
    if split is not None and split not in SPLITS:
        raise ValueError(f"split invalide : {split!r} (attendu : {sorted(SPLITS)} ou None)")


class ReferenceRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # -- séances --------------------------------------------------------------------
    def create_session(
        self,
        profile_id: ProfileId | None,
        *,
        started_at: datetime,
        split: str | None = None,
        camera_id: CameraId | None = None,
        conditions: str | None = None,
        session_id: SessionId | None = None,
    ) -> CollectionSession:
        _check_split(split)
        sid = session_id or new_session_id()
        with self._db.transaction() as c:
            c.execute(
                "INSERT INTO collection_sessions(session_id, profile_id, started_at,"
                " camera_id, conditions, split) VALUES (?, ?, ?, ?, ?, ?)",
                (sid, profile_id, to_utc_iso(started_at), camera_id, conditions, split),
            )
        return self.get_session(sid)

    def get_session(self, session_id: SessionId) -> CollectionSession:
        row = self._db.conn.execute(
            "SELECT * FROM collection_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        return _row_to_session(row)

    def list_sessions(
        self, *, profile_id: ProfileId | None = None, split: str | None = None
    ) -> list[CollectionSession]:
        _check_split(split)
        sql = "SELECT * FROM collection_sessions WHERE 1=1"
        args: list[object] = []
        if profile_id is not None:
            sql += " AND profile_id = ?"
            args.append(profile_id)
        if split is not None:
            sql += " AND split = ?"
            args.append(split)
        sql += " ORDER BY started_at, session_id"
        return [_row_to_session(r) for r in self._db.conn.execute(sql, args)]

    def set_session_split(self, session_id: SessionId, split: str | None) -> CollectionSession:
        """Affecte une séance COMPLÈTE à un découpage (unité de découpage = séance)."""
        _check_split(split)
        with self._db.transaction() as c:
            cur = c.execute(
                "UPDATE collection_sessions SET split = ? WHERE session_id = ?",
                (split, session_id),
            )
            if cur.rowcount == 0:
                raise SessionNotFoundError(session_id)
        return self.get_session(session_id)

    # -- images de référence --------------------------------------------------------
    def add_image_bytes(
        self,
        profile_id: ProfileId,
        data: bytes,
        *,
        extension: str = ".png",
        captured_at: datetime,
        session_id: SessionId | None = None,
        width: int | None = None,
        height: int | None = None,
        quality_note: str | None = None,
    ) -> ReferenceImage:
        """Écrit l'image encodée (PNG/JPEG…) sous ``media_root/<profile_id>/`` et
        l'enregistre avec son sha256. Le fichier est retiré si l'insertion échoue."""
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        if ext not in _ALLOWED_EXT:
            raise ValueError(f"Extension non prise en charge : {extension!r}")
        if not data:
            raise ValueError("Image vide")
        digest = sha256_bytes(data)
        target = self._target_path(profile_id, ext)
        write_atomic(target, data)
        return self._insert(
            profile_id, target, digest, captured_at, session_id, width, height, quality_note
        )

    def import_file(
        self,
        profile_id: ProfileId,
        source: str | Path,
        *,
        captured_at: datetime,
        session_id: SessionId | None = None,
        width: int | None = None,
        height: int | None = None,
        quality_note: str | None = None,
        move: bool = False,
    ) -> ReferenceImage:
        """Copie (ou déplace si ``move``) un fichier image local dans ``media_root``."""
        src = Path(source)
        if not src.is_file():
            raise FileNotFoundError(src)
        ext = src.suffix.lower()
        if ext not in _ALLOWED_EXT:
            raise ValueError(f"Extension non prise en charge : {src.suffix!r}")
        target = self._target_path(profile_id, ext)
        target.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), target)
        else:
            shutil.copyfile(src, target)
        digest = sha256_file(target)
        return self._insert(
            profile_id, target, digest, captured_at, session_id, width, height, quality_note
        )

    def _target_path(self, profile_id: ProfileId, ext: str) -> Path:
        if not profile_id or "/" in profile_id or "\\" in profile_id or ".." in profile_id:
            raise ValueError(f"profile_id invalide : {profile_id!r}")
        return self._db.media_root / profile_id / f"ref-{uuid.uuid4().hex}{ext}"

    def _insert(
        self,
        profile_id: ProfileId,
        path: Path,
        digest: str,
        captured_at: datetime,
        session_id: SessionId | None,
        width: int | None,
        height: int | None,
        quality_note: str | None,
    ) -> ReferenceImage:
        try:
            captured_iso = to_utc_iso(captured_at)
            with self._db.transaction() as c:
                if session_id is not None:
                    row = c.execute(
                        "SELECT profile_id FROM collection_sessions WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    if row is None:
                        raise SessionNotFoundError(session_id)
                    if row[0] is not None and row[0] != profile_id:
                        raise ValueError("La séance appartient à un autre profil")
                cur = c.execute(
                    "INSERT INTO reference_images(profile_id, session_id, file_path, sha256,"
                    " captured_at, width, height, quality_note)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (profile_id, session_id, str(path), digest, captured_iso, width, height,
                     quality_note),
                )
                rid = int(cur.lastrowid or 0)
        except BaseException:
            remove_managed_files(self._db.media_root, [str(path)])
            raise
        return self.get(rid)

    def get(self, reference_id: int) -> ReferenceImage:
        row = self._db.conn.execute(
            "SELECT * FROM reference_images WHERE reference_id = ?", (reference_id,)
        ).fetchone()
        if row is None:
            raise ReferenceNotFoundError(reference_id)
        return _row_to_reference(row)

    def list_for_profile(self, profile_id: ProfileId) -> list[ReferenceImage]:
        rows = self._db.conn.execute(
            "SELECT * FROM reference_images WHERE profile_id = ?"
            " ORDER BY captured_at, reference_id",
            (profile_id,),
        )
        return [_row_to_reference(r) for r in rows]

    def list_for_session(self, session_id: SessionId) -> list[ReferenceImage]:
        rows = self._db.conn.execute(
            "SELECT * FROM reference_images WHERE session_id = ? ORDER BY reference_id",
            (session_id,),
        )
        return [_row_to_reference(r) for r in rows]

    def verify(self, reference_id: int) -> bool:
        """Vrai si le fichier existe et que son sha256 correspond à la base."""
        ref = self.get(reference_id)
        return ref.file_path.is_file() and sha256_file(ref.file_path) == ref.sha256

    def delete(self, reference_id: int) -> bool:
        """Supprime une référence, ses représentations (cascade) et son fichier local."""
        with self._db.transaction() as c:
            row = c.execute(
                "SELECT file_path FROM reference_images WHERE reference_id = ?",
                (reference_id,),
            ).fetchone()
            if row is None:
                return False
            c.execute("DELETE FROM reference_images WHERE reference_id = ?", (reference_id,))
        remove_managed_files(self._db.media_root, [row[0]])
        return True


__all__ = [
    "SPLITS",
    "CollectionSession",
    "ReferenceImage",
    "ReferenceNotFoundError",
    "ReferenceRepository",
    "SessionNotFoundError",
]
