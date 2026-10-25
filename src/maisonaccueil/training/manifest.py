"""Manifeste de données par séance (``SessionId``) et contrôle des fuites entre découpages.

Règles (docs/DATA_POLICY.md) :

- l'unité de découpage est la **séance complète**, jamais l'image ;
- une séance appartient à un seul découpage (``train``, ``tune``, ``test``, ``enroll``) ;
- un même fichier (chemin ou empreinte sha256) ne peut figurer que dans une seule séance ;
- l'apprentissage (lot E) n'utilise que ``train`` ; ``tune``/``test`` servent au réglage
  et à la mesure (lots F/G) et ``enroll`` à la galerie ;
- les séances sans profil (inconnus volontaires) ne peuvent pas être en ``train``
  (pas d'étiquette) : elles servent au rejet des inconnus dans ``tune``/``test``.

Le manifeste ne contient aucune image : seulement des identifiants, des chemins locaux
et des empreintes. Il vit dans ``var/e/`` ou ``runs/`` (ignorés par Git).
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from maisonaccueil.contracts import ProfileId, SessionId

MANIFEST_FORMAT = "maisonaccueil-e-manifest/1"
SPLITS = ("train", "tune", "test", "enroll")
RAW_PREPROCESS = "raw"  # images de référence non alignées (telles que stockées par C)


class ManifestError(ValueError):
    """Manifeste incohérent (référence inconnue, champ invalide…)."""


class SplitLeakError(ManifestError):
    """Chevauchement interdit entre découpages (séance ou fichier partagé)."""


@dataclass(frozen=True, slots=True)
class SessionEntry:
    session_id: SessionId
    profile_id: ProfileId | None  # None = séance d'inconnus volontaires
    split: str

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ManifestError(f"Découpage invalide {self.split!r} pour {self.session_id}")


@dataclass(frozen=True, slots=True)
class SampleEntry:
    session_id: SessionId
    path: str  # chemin local, relatif à la racine des données ou absolu
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Manifest:
    preprocess: str
    sessions: tuple[SessionEntry, ...]
    samples: tuple[SampleEntry, ...]
    _by_session: dict[str, SessionEntry] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        by_session: dict[str, SessionEntry] = {}
        for s in self.sessions:
            if s.session_id in by_session:
                raise SplitLeakError(f"Séance {s.session_id} déclarée plusieurs fois")
            by_session[s.session_id] = s
        object.__setattr__(self, "_by_session", by_session)
        self.validate()

    # -- contrôles ---------------------------------------------------------------------

    def validate(self) -> None:
        """Lève ``ManifestError``/``SplitLeakError`` si une règle est violée."""
        path_owner: dict[str, SessionEntry] = {}
        hash_owner: dict[str, SessionEntry] = {}
        for sample in self.samples:
            session = self._by_session.get(sample.session_id)
            if session is None:
                raise ManifestError(f"Image {sample.path!r} : séance inconnue {sample.session_id}")
            for key, owners, label in (
                (sample.path, path_owner, "chemin"),
                (sample.sha256, hash_owner, "empreinte"),
            ):
                if key is None:
                    continue
                previous = owners.get(key)
                if previous is not None and previous.session_id != session.session_id:
                    raise SplitLeakError(
                        f"Même {label} {key!r} dans les séances {previous.session_id} "
                        f"({previous.split}) et {session.session_id} ({session.split})"
                    )
                owners[key] = session
        for session in self.sessions:
            if session.split == "train" and session.profile_id is None:
                raise ManifestError(
                    f"Séance {session.session_id} sans profil ne peut pas être en 'train'"
                )

    def check_identity_disjoint(self, a: str, b: str) -> None:
        """Option plus stricte : aucune personne commune aux découpages ``a`` et ``b``
        (ex. modèle évalué sur des personnes jamais vues à l'apprentissage)."""
        pa = {s.profile_id for s in self.sessions if s.split == a and s.profile_id}
        pb = {s.profile_id for s in self.sessions if s.split == b and s.profile_id}
        common = pa & pb
        if common:
            raise SplitLeakError(f"Personnes communes à {a!r} et {b!r} : {sorted(common)}")

    # -- vues --------------------------------------------------------------------------

    def session(self, session_id: SessionId) -> SessionEntry:
        return self._by_session[session_id]

    def samples_for(self, split: str) -> list[tuple[SampleEntry, SessionEntry]]:
        if split not in SPLITS:
            raise ManifestError(f"Découpage invalide {split!r}")
        out = []
        for sample in self.samples:
            session = self._by_session[sample.session_id]
            if session.split == split:
                out.append((sample, session))
        return out

    def training_samples(self) -> tuple[list[str], list[int], list[ProfileId]]:
        """Chemins et étiquettes entières du découpage ``train`` uniquement.

        Renvoie ``(chemins, étiquettes, classes)`` où ``classes[i]`` est le profil de
        l'étiquette ``i`` (ordre trié, déterministe)."""
        pairs = self.samples_for("train")
        classes = sorted({s.profile_id for _, s in pairs if s.profile_id is not None})
        index = {p: i for i, p in enumerate(classes)}
        paths: list[str] = []
        labels: list[int] = []
        for sample, session in pairs:
            if session.profile_id is None:  # exclu par validate(), garde défensive
                raise ManifestError(f"Séance {session.session_id} sans profil en 'train'")
            paths.append(sample.path)
            labels.append(index[session.profile_id])
        return paths, labels, [ProfileId(p) for p in classes]

    def counts(self) -> dict[str, dict[str, int]]:
        """Effectifs par découpage : séances, images, personnes."""
        out: dict[str, dict[str, int]] = {}
        for split in SPLITS:
            sessions = [s for s in self.sessions if s.split == split]
            out[split] = {
                "sessions": len(sessions),
                "samples": len(self.samples_for(split)),
                "profiles": len({s.profile_id for s in sessions if s.profile_id}),
            }
        return out

    # -- sérialisation -----------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "format": MANIFEST_FORMAT,
            "preprocess": self.preprocess,
            "sessions": [
                {"session_id": s.session_id, "profile_id": s.profile_id, "split": s.split}
                for s in self.sessions
            ],
            "samples": [
                {"session_id": s.session_id, "path": s.path, "sha256": s.sha256}
                for s in self.samples
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), "utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Manifest:
        if data.get("format") != MANIFEST_FORMAT:
            raise ManifestError(f"Format de manifeste inattendu : {data.get('format')!r}")
        sessions_raw = data.get("sessions")
        samples_raw = data.get("samples")
        if not isinstance(sessions_raw, list) or not isinstance(samples_raw, list):
            raise ManifestError("Manifeste : 'sessions' et 'samples' doivent être des listes")
        sessions = tuple(
            SessionEntry(
                session_id=SessionId(str(s["session_id"])),
                profile_id=ProfileId(str(s["profile_id"])) if s.get("profile_id") else None,
                split=str(s["split"]),
            )
            for s in sessions_raw
        )
        samples = tuple(
            SampleEntry(
                session_id=SessionId(str(s["session_id"])),
                path=str(s["path"]),
                sha256=str(s["sha256"]) if s.get("sha256") else None,
            )
            for s in samples_raw
        )
        return cls(preprocess=str(data["preprocess"]), sessions=sessions, samples=samples)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.from_dict(json.loads(path.read_text("utf-8")))


def manifest_from_database(db_path: Path, include_unassigned: bool = False) -> Manifest:
    """Construit un manifeste en LECTURE SEULE depuis la base partagée (schéma des
    contrats : ``collection_sessions`` + ``reference_images``).

    Les images de référence ne sont pas alignées : ``preprocess = "raw"``. L'alignement
    exige les 5 points du détecteur (lot D) ; il n'est pas encore intégré.
    Les séances sans découpage (``split IS NULL``) sont refusées, sauf
    ``include_unassigned`` qui les ignore simplement.
    """
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT session_id, profile_id, split FROM collection_sessions ORDER BY session_id"
        ).fetchall()
        sessions = []
        for session_id, profile_id, split in rows:
            if split is None:
                if include_unassigned:
                    continue
                raise ManifestError(f"Séance {session_id} sans découpage (split NULL)")
            sessions.append(
                SessionEntry(
                    SessionId(session_id), ProfileId(profile_id) if profile_id else None, split
                )
            )
        known = {s.session_id for s in sessions}
        samples = []
        for session_id, profile_id, file_path, sha in conn.execute(
            "SELECT r.session_id, r.profile_id, r.file_path, r.sha256 FROM reference_images r "
            "WHERE r.session_id IS NOT NULL ORDER BY r.reference_id"
        ):
            if session_id not in known:
                continue
            session_profile = next(s.profile_id for s in sessions if s.session_id == session_id)
            if session_profile is not None and session_profile != profile_id:
                raise ManifestError(
                    f"Image {file_path!r} : profil {profile_id} ≠ profil de la séance {session_id}"
                )
            samples.append(SampleEntry(SessionId(session_id), str(file_path), str(sha)))
    finally:
        conn.close()
    return Manifest(preprocess=RAW_PREPROCESS, sessions=tuple(sessions), samples=tuple(samples))


def assign_splits(
    session_ids: Iterable[SessionId],
    fractions: Mapping[str, float],
    seed: int,
) -> dict[SessionId, str]:
    """Propose un découpage **par séance** (jamais par image), déterministe pour une graine.

    Ne modifie aucune base : le lot C/B enregistre le découpage choisi. Chaque découpage
    de fraction > 0 reçoit au moins une séance si le nombre de séances le permet.
    """
    unknown = set(fractions) - set(SPLITS)
    if unknown:
        raise ManifestError(f"Découpages inconnus : {sorted(unknown)}")
    if any(f < 0 for f in fractions.values()) or sum(fractions.values()) <= 0:
        raise ManifestError("Fractions invalides")
    ids: Sequence[SessionId] = sorted(set(session_ids))
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    total = sum(fractions.values())
    active = [(name, f / total) for name, f in fractions.items() if f > 0]
    n = len(shuffled)
    counts = {name: int(frac * n) for name, frac in active}
    if n >= len(active):
        for name, _ in active:
            if counts[name] == 0:
                counts[name] = 1
    # Ajuste le total en donnant/prenant au plus grand découpage.
    largest = max(active, key=lambda item: item[1])[0]
    counts[largest] += n - sum(counts.values())
    out: dict[SessionId, str] = {}
    cursor = 0
    for name, _ in active:
        for sid in shuffled[cursor : cursor + counts[name]]:
            out[sid] = name
        cursor += counts[name]
    return out
