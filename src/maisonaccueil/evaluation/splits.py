"""Découpage par séances ``train`` / ``tune`` / ``test`` et traçabilité du test.

Règles vérifiées par ``check_splits`` (erreurs bloquantes) :

- chaque séance appartient à exactement un découpage (garanti par le format) ;
- deux séances ne partagent pas la même vidéo (``video_sha256`` identique) : une même
  vidéo déclarée sous deux identifiants serait un chevauchement déguisé ;
- une séance déclarée à la fois dans les annotations et dans ``extra_sessions`` (ex. base
  SQLite) porte le même découpage des deux côtés.

Par construction, une évaluation ne retient que les passages des séances du découpage
demandé (``AnnotationSet.passages_in``) et ``match_runs`` refuse un journal d'une autre
séance.

Avertissements (non bloquants, affichés dans le rapport) :

- ``tune`` ou ``test`` sans passage d'inconnu non enregistré (rejet des inconnus non mesuré) ;
- séance ``tune``/``test`` enregistrée le même jour (UTC) qu'une séance ``enroll``/``train``
  pour la même personne (DATA_POLICY : séances d'évaluation d'autres jours).

Registre du test (``TestLedger``, JSON Lines, ``var/g/ledger.jsonl`` par défaut) ::

    {"record": "evaluation", "at": "...Z", "split": "test", "set_id": "<16 hex>",
     "sessions": ["ses-..."], "config_ids": ["engine@v#policy/seuil"], "note": ""}
    {"record": "tuning", "at": "...Z", "split": "tune", "set_id": "<16 hex>",
     "informed_by": ["<set_id>"], "config_before": "...", "config_after": "...", "note": ""}

Un ensemble de test (``set_id`` = empreinte des identifiants de séances triés) est
**brûlé** si :

1. il figure dans ``informed_by`` d'un réglage (des résultats de test ont guidé un réglage) ;
2. un réglage (``tuning``) a été enregistré directement sur cet ensemble ;
3. il est réévalué avec une configuration absente de sa **première** évaluation (la
   configuration a changé après avoir vu le test). Réévaluer exactement les mêmes
   configurations (reproduction) ou un sous-ensemble ne le brûle pas.

Un test brûlé exige un **nouvel ensemble de test indépendant** avant toute conclusion.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maisonaccueil.contracts import SessionId

from .annotations import AnnotationSet, Expected, SessionRecord
from .jsonl import FormatError, append_jsonl, parse_utc, read_jsonl, utc_now_iso


class SplitViolationError(ValueError):
    """Utilisation d'un passage ou d'une séance hors du découpage autorisé."""


class BurnedTestSetError(RuntimeError):
    """Ensemble de test déjà utilisé pour un réglage : un nouveau test est exigé."""


@dataclass(frozen=True, slots=True)
class SplitReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def set_id(sessions: Iterable[str]) -> str:
    """Empreinte stable d'un ensemble de séances (ordre indifférent)."""
    joined = "\n".join(sorted(set(sessions)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def check_splits(
    ann: AnnotationSet, extra_sessions: Sequence[SessionRecord] = ()
) -> SplitReport:
    """Contrôle d'absence de chevauchement. ``extra_sessions`` : séances connues par ailleurs
    (ex. ``enroll``/``train`` lues depuis la base) pour les contrôles croisés."""
    errors: list[str] = []
    warnings: list[str] = []
    sessions: dict[str, SessionRecord] = {}
    for s in list(extra_sessions) + list(ann.sessions.values()):
        prev = sessions.get(s.session_id)
        if prev is not None and prev.split != s.split:
            errors.append(
                f"séance {s.session_id} déclarée à la fois '{prev.split}' et '{s.split}'"
            )
        sessions[s.session_id] = s

    by_video: dict[str, list[str]] = defaultdict(list)
    for s in sessions.values():
        if s.video_sha256:
            by_video[s.video_sha256].append(s.session_id)
    for digest, ids in by_video.items():
        if len(ids) > 1:
            errors.append(
                f"même vidéo (sha256 {digest[:12]}…) déclarée sous plusieurs séances : "
                + ", ".join(sorted(ids))
            )

    counts: dict[str, dict[str, int]] = {}
    for split in ("tune", "test"):
        ps = [p for p in ann.passages if sessions[p.session_id].split == split]
        n_unreg = sum(1 for p in ps if p.expected == Expected.UNREGISTERED)
        counts[split] = {
            "sessions": sum(1 for s in ann.sessions.values() if s.split == split),
            "passages": len(ps),
            "known": len(ps) - n_unreg,
            "unregistered": n_unreg,
            "persons": len({p.person_key for p in ps}),
        }
        if ps and n_unreg == 0:
            warnings.append(
                f"'{split}' ne contient aucun passage d'inconnu non enregistré : "
                "le rejet des inconnus n'y est pas mesuré"
            )

    # Même jour que l'enrôlement/l'entraînement pour une même personne.
    train_days: dict[str, set[str]] = defaultdict(set)
    for p in ann.passages:
        s = sessions[p.session_id]
        if s.split in ("enroll", "train") and s.recorded_at and p.profile_id:
            train_days[p.profile_id].add(s.recorded_at.date().isoformat())
    for p in ann.passages:
        s = sessions[p.session_id]
        if s.split in ("tune", "test") and s.recorded_at and p.profile_id:
            day = s.recorded_at.date().isoformat()
            if day in train_days.get(p.profile_id, set()):
                warnings.append(
                    f"passage {p.passage_id} ({s.split}) le même jour qu'une séance "
                    "d'enrôlement/entraînement de la même personne"
                )
    return SplitReport(tuple(errors), tuple(dict.fromkeys(warnings)), counts)


def require_split(ann: AnnotationSet, session_ids: Iterable[str], split: str) -> None:
    """Lève ``SplitViolationError`` si une séance n'appartient pas à ``split``."""
    bad = sorted(
        s for s in set(session_ids) if s not in ann.sessions or ann.sessions[s].split != split
    )
    if bad:
        raise SplitViolationError(
            f"séances hors du découpage '{split}' : {', '.join(bad)}"
        )


def sessions_from_sqlite(db_path: Path | str) -> list[SessionRecord]:
    """Lit ``collection_sessions`` (schéma partagé des contrats) en lecture seule.

    Les séances sans ``split`` sont ignorées. Aucune écriture dans la base."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT session_id, split, started_at, camera_id, conditions "
            "FROM collection_sessions WHERE split IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    out: list[SessionRecord] = []
    for sid, split, started, camera, conditions in rows:
        try:
            recorded = parse_utc(started, f"collection_sessions[{sid}]") if started else None
        except FormatError:
            recorded = None
        out.append(
            SessionRecord(SessionId(sid), split, recorded, camera or "", None, "", conditions or "")
        )
    return out


@dataclass(frozen=True, slots=True)
class TestSetStatus:
    set_id: str
    burned: bool
    reasons: tuple[str, ...]
    first_configs: frozenset[str] | None  # configurations de la première évaluation
    evaluations: int


class TestLedger:
    """Registre append-only des évaluations de test et des réglages."""

    __test__ = False  # pas une classe de test pytest

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._records: list[dict[str, Any]] = []
        if self.path.exists():
            for lineno, obj in read_jsonl(self.path):
                if obj.get("record") not in ("evaluation", "tuning"):
                    raise FormatError(f"{self.path}:{lineno}: enregistrement de registre inconnu")
                self._records.append(obj)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    def _append(self, rec: dict[str, Any]) -> None:
        append_jsonl(self.path, rec)
        self._records.append(rec)

    def status(self, sid: str, config_ids: Iterable[str] = ()) -> TestSetStatus:
        """État d'un ensemble ; ``config_ids`` : configurations qu'on s'apprête à évaluer."""
        reasons: list[str] = []
        first: frozenset[str] | None = None
        n_eval = 0
        for rec in self._records:
            if rec["record"] == "tuning":
                if sid in rec.get("informed_by", []):
                    reasons.append(f"résultats utilisés pour un réglage le {rec.get('at')}")
                if rec.get("set_id") == sid:
                    reasons.append(f"réglage effectué directement sur cet ensemble le "
                                   f"{rec.get('at')}")
            elif rec["record"] == "evaluation" and rec.get("set_id") == sid:
                n_eval += 1
                cfgs = frozenset(rec.get("config_ids", []))
                if first is None:
                    first = cfgs
                elif not cfgs <= first:
                    reasons.append(
                        "réévalué avec une configuration absente de la première évaluation : "
                        + ", ".join(sorted(cfgs - first))
                    )
        wanted = frozenset(config_ids)
        if first is not None and not wanted <= first:
            reasons.append(
                "configuration(s) nouvelle(s) par rapport à la première évaluation : "
                + ", ".join(sorted(wanted - first))
            )
        return TestSetStatus(sid, bool(reasons), tuple(dict.fromkeys(reasons)), first, n_eval)

    def record_evaluation(
        self,
        sessions: Sequence[str],
        config_ids: Sequence[str],
        *,
        split: str = "test",
        allow_burned: bool = False,
        note: str = "",
    ) -> TestSetStatus:
        """Enregistre une évaluation. Pour ``test`` : lève ``BurnedTestSetError`` si
        l'ensemble est (ou deviendrait) brûlé, sauf ``allow_burned=True`` (le rapport le
        signale alors comme non concluant)."""
        sid = set_id(sessions)
        st = self.status(sid, config_ids) if split == "test" else TestSetStatus(
            sid, False, (), None, 0)
        if st.burned and not allow_burned:
            raise BurnedTestSetError(
                "ensemble de test brûlé — constituer un nouveau test indépendant : "
                + " ; ".join(st.reasons)
            )
        self._append(
            {
                "record": "evaluation",
                "at": utc_now_iso(),
                "split": split,
                "set_id": sid,
                "sessions": sorted(set(sessions)),
                "config_ids": sorted(set(config_ids)),
                "note": note,
            }
        )
        return st

    def record_tuning(
        self,
        sessions: Sequence[str],
        *,
        split: str = "tune",
        informed_by: Sequence[str] = (),
        config_before: str = "",
        config_after: str = "",
        note: str = "",
    ) -> None:
        """Enregistre un réglage. ``informed_by`` : ``set_id`` de TOUT ensemble dont les
        résultats ont été consultés pour décider (un test listé ici devient brûlé)."""
        self._append(
            {
                "record": "tuning",
                "at": utc_now_iso(),
                "split": split,
                "set_id": set_id(sessions),
                "informed_by": sorted(set(informed_by)),
                "config_before": config_before,
                "config_after": config_after,
                "note": note,
            }
        )
