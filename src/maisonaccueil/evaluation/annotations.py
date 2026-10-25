"""Format d'annotation des séances et des passages (JSON Lines).

Format ``maisonaccueil.passages`` version 1 — un objet JSON par ligne, champ ``record`` :

``header`` (première ligne, obligatoire) ::

    {"record": "header", "format": "maisonaccueil.passages", "version": 1,
     "data_kind": "real" | "synthetic", "created_at": "2026-11-02T18:00:00Z",
     "annotator": "texte libre (facultatif)"}

``session`` — une séance de collecte complète (unité de découpage) ::

    {"record": "session", "session_id": "ses-...", "split": "tune" | "test" | "train" | "enroll",
     "recorded_at": "2026-11-02T17:30:00Z", "camera_id": "entree",
     "duration_ms": 600000, "video_sha256": "<hex, facultatif>", "conditions": "texte libre"}

``passage`` — une entrée d'une personne dans le champ, en temps de séance (ms depuis le
début de la vidéo / de l'enregistrement de la séance) ::

    {"record": "passage", "passage_id": "p-0001", "session_id": "ses-...",
     "start_ms": 12000, "end_ms": 19500,
     "expected": "known" | "unregistered",
     "profile_id": "prf-..." (obligatoire si known, interdit si unregistered),
     "subject": "inconnu-A" (pseudonyme facultatif, pour regrouper les inconnus),
     "welcome_expected": true (false si la personne est déjà présente et ne doit pas
                               être accueillie de nouveau),
     "note": ""}

Aucun nom de personne n'est requis : seuls des ``ProfileId`` et des pseudonymes.
Le fichier ne contient ni image ni représentation faciale.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from maisonaccueil.contracts import ProfileId, SessionId

from .jsonl import (
    FormatError,
    optional,
    parse_utc,
    read_jsonl,
    require,
    utc_now_iso,
    write_jsonl,
)

PASSAGES_FORMAT = "maisonaccueil.passages"
PASSAGES_VERSION = 1
SPLITS = ("enroll", "train", "tune", "test")


class DataKind(enum.StrEnum):
    REAL = "real"
    SYNTHETIC = "synthetic"  # tests logiciels uniquement : ne mesure rien de biométrique


class Expected(enum.StrEnum):
    KNOWN = "known"  # personne enregistrée (ProfileId attendu)
    UNREGISTERED = "unregistered"  # volontaire non enregistré : aucun nom ne doit être dit


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: SessionId
    split: str
    recorded_at: datetime | None = None
    camera_id: str = ""
    duration_ms: int | None = None
    video_sha256: str = ""
    conditions: str = ""

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split invalide {self.split!r} (attendu : {', '.join(SPLITS)})")


@dataclass(frozen=True, slots=True)
class Passage:
    passage_id: str
    session_id: SessionId
    start_ms: float
    end_ms: float
    expected: Expected
    profile_id: ProfileId | None = None
    subject: str = ""
    welcome_expected: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError(f"passage {self.passage_id}: end_ms < start_ms")
        if (self.expected == Expected.KNOWN) != (self.profile_id is not None):
            raise ValueError(
                f"passage {self.passage_id}: profile_id obligatoire pour 'known' "
                "et interdit pour 'unregistered'"
            )

    @property
    def person_key(self) -> str:
        """Clé de regroupement « par personne »."""
        if self.profile_id is not None:
            return str(self.profile_id)
        return f"inconnu:{self.subject}" if self.subject else "inconnu:(non nommé)"


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    data_kind: DataKind
    sessions: dict[SessionId, SessionRecord]
    passages: tuple[Passage, ...]
    sources: tuple[str, ...] = field(default=())

    def passages_in(self, splits: Iterable[str]) -> tuple[Passage, ...]:
        wanted = set(splits)
        return tuple(p for p in self.passages if self.sessions[p.session_id].split in wanted)

    def sessions_in(self, split: str) -> tuple[SessionId, ...]:
        return tuple(sorted(s for s, r in self.sessions.items() if r.split == split))

    @property
    def is_synthetic(self) -> bool:
        return self.data_kind == DataKind.SYNTHETIC


def _parse_session(obj: dict[str, Any], where: str) -> SessionRecord:
    recorded = optional(obj, "recorded_at", str, where, None)
    try:
        return SessionRecord(
            session_id=SessionId(require(obj, "session_id", str, where)),
            split=require(obj, "split", str, where),
            recorded_at=parse_utc(recorded, where) if recorded else None,
            camera_id=optional(obj, "camera_id", str, where, ""),
            duration_ms=optional(obj, "duration_ms", int, where, None),
            video_sha256=optional(obj, "video_sha256", str, where, ""),
            conditions=optional(obj, "conditions", str, where, ""),
        )
    except ValueError as exc:
        if isinstance(exc, FormatError):
            raise
        raise FormatError(f"{where}: {exc}") from None


def _parse_passage(obj: dict[str, Any], where: str) -> Passage:
    expected_raw = require(obj, "expected", str, where)
    try:
        expected = Expected(expected_raw)
    except ValueError:
        raise FormatError(f"{where}: 'expected' doit valoir 'known' ou 'unregistered'") from None
    profile = optional(obj, "profile_id", str, where, None)
    try:
        return Passage(
            passage_id=require(obj, "passage_id", str, where),
            session_id=SessionId(require(obj, "session_id", str, where)),
            start_ms=float(require(obj, "start_ms", (int, float), where)),
            end_ms=float(require(obj, "end_ms", (int, float), where)),
            expected=expected,
            profile_id=ProfileId(profile) if profile is not None else None,
            subject=optional(obj, "subject", str, where, ""),
            welcome_expected=optional(obj, "welcome_expected", bool, where, True),
            note=optional(obj, "note", str, where, ""),
        )
    except ValueError as exc:
        if isinstance(exc, FormatError):
            raise
        raise FormatError(f"{where}: {exc}") from None


def load_annotations(*paths: Path | str) -> AnnotationSet:
    """Charge et fusionne un ou plusieurs fichiers ``maisonaccueil.passages``.

    Erreurs (``FormatError``) : en-tête absent ou incompatible, ``data_kind`` différents
    entre fichiers, séance ou passage dupliqué, passage d'une séance non déclarée,
    passage hors de la durée déclarée de la séance."""
    if not paths:
        raise ValueError("au moins un fichier d'annotation est requis")
    kinds: set[DataKind] = set()
    sessions: dict[SessionId, SessionRecord] = {}
    passages: dict[str, tuple[Passage, str]] = {}
    for path in paths:
        header_seen = False
        for lineno, obj in read_jsonl(path):
            where = f"{path}:{lineno}"
            record = obj.get("record")
            if not header_seen:
                if record != "header":
                    raise FormatError(f"{where}: la première ligne doit être l'en-tête 'header'")
                if obj.get("format") != PASSAGES_FORMAT or obj.get("version") != PASSAGES_VERSION:
                    raise FormatError(
                        f"{where}: format attendu {PASSAGES_FORMAT} v{PASSAGES_VERSION}"
                    )
                try:
                    kinds.add(DataKind(require(obj, "data_kind", str, where)))
                except ValueError as exc:
                    if isinstance(exc, FormatError):
                        raise
                    raise FormatError(f"{where}: data_kind doit valoir real ou synthetic") from None
                header_seen = True
                continue
            if record == "session":
                s = _parse_session(obj, where)
                if s.session_id in sessions:
                    raise FormatError(f"{where}: séance {s.session_id} déclarée deux fois")
                sessions[s.session_id] = s
            elif record == "passage":
                p = _parse_passage(obj, where)
                if p.passage_id in passages:
                    raise FormatError(f"{where}: passage {p.passage_id} déclaré deux fois")
                passages[p.passage_id] = (p, where)
            else:
                raise FormatError(f"{where}: type d'enregistrement inconnu {record!r}")
        if not header_seen:
            raise FormatError(f"{path}: fichier vide (en-tête manquant)")
    if len(kinds) > 1:
        raise FormatError("mélange de fichiers 'real' et 'synthetic' interdit")
    for p, where in passages.values():
        sess = sessions.get(p.session_id)
        if sess is None:
            raise FormatError(f"{where}: séance {p.session_id} non déclarée")
        if sess.duration_ms is not None and p.end_ms > sess.duration_ms:
            raise FormatError(f"{where}: passage au-delà de la durée de la séance")
    ordered = sorted((p for p, _ in passages.values()), key=lambda p: (p.session_id, p.start_ms,
                                                                        p.passage_id))
    return AnnotationSet(kinds.pop(), sessions, tuple(ordered), tuple(str(p) for p in paths))


def annotation_records(
    data_kind: DataKind,
    sessions: Sequence[SessionRecord],
    passages: Sequence[Passage],
    *,
    annotator: str = "",
) -> list[dict[str, Any]]:
    """Enregistrements JSON d'un jeu d'annotations (pour ``write_jsonl``)."""
    out: list[dict[str, Any]] = [
        {
            "record": "header",
            "format": PASSAGES_FORMAT,
            "version": PASSAGES_VERSION,
            "data_kind": str(data_kind),
            "created_at": utc_now_iso(),
            "annotator": annotator,
        }
    ]
    for s in sessions:
        rec: dict[str, Any] = {"record": "session", "session_id": s.session_id, "split": s.split}
        if s.recorded_at is not None:
            rec["recorded_at"] = s.recorded_at.isoformat().replace("+00:00", "Z")
        if s.camera_id:
            rec["camera_id"] = s.camera_id
        if s.duration_ms is not None:
            rec["duration_ms"] = s.duration_ms
        if s.video_sha256:
            rec["video_sha256"] = s.video_sha256
        if s.conditions:
            rec["conditions"] = s.conditions
        out.append(rec)
    for p in passages:
        out.append(
            {
                "record": "passage",
                "passage_id": p.passage_id,
                "session_id": p.session_id,
                "start_ms": p.start_ms,
                "end_ms": p.end_ms,
                "expected": str(p.expected),
                "profile_id": p.profile_id,
                "subject": p.subject,
                "welcome_expected": p.welcome_expected,
                "note": p.note,
            }
        )
    return out


def save_annotations(
    path: Path | str,
    data_kind: DataKind,
    sessions: Sequence[SessionRecord],
    passages: Sequence[Passage],
    *,
    annotator: str = "",
) -> None:
    write_jsonl(path, annotation_records(data_kind, sessions, passages, annotator=annotator))
