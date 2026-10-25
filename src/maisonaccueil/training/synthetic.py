"""Données SYNTHÉTIQUES pour tester le logiciel d'entraînement (aucun visage).

Chaque « classe » est un motif aléatoire de bruit + bruit par image. Ces données ne
ressemblent pas à des visages, ne représentent aucune personne et ne permettent de juger
NI la qualité du modèle NI ses seuils : elles servent uniquement à vérifier que le code
s'exécute de bout en bout (formes, perte finie, journal, empreintes).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import ProfileId, SessionId

from .manifest import Manifest, SampleEntry, SessionEntry
from .preprocess import INPUT_SIZE, PREPROCESS_ID


def synthetic_aligned_images(
    n_classes: int, per_class: int, seed: int, size: int = INPUT_SIZE
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Renvoie ``(images (N, size, size, 3) uint8 BGR, étiquettes (N,))``."""
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 256, size=(n_classes, size, size, 3), dtype=np.int16)
    images = []
    labels = []
    for c in range(n_classes):
        for _ in range(per_class):
            noise = rng.integers(-20, 21, size=(size, size, 3), dtype=np.int16)
            images.append(np.clip(bases[c] + noise, 0, 255).astype(np.uint8))
            labels.append(c)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def write_synthetic_dataset(
    root: Path,
    n_classes: int,
    sessions_per_class: int,
    per_session: int,
    seed: int,
    split_of_session: dict[int, str] | None = None,
) -> Manifest:
    """Écrit des PNG synthétiques sous ``root`` et renvoie le manifeste correspondant.

    ``split_of_session`` associe l'indice de séance (0..sessions_per_class-1) à un
    découpage ; par défaut toutes les séances sont en ``train``. Les identifiants sont
    explicitement préfixés ``synthetic`` : aucune personne réelle ni inventée."""
    split_of_session = split_of_session or {}
    images, labels = synthetic_aligned_images(
        n_classes, sessions_per_class * per_session, seed
    )
    sessions: list[SessionEntry] = []
    samples: list[SampleEntry] = []
    root.mkdir(parents=True, exist_ok=True)
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        for s in range(sessions_per_class):
            sid = SessionId(f"ses-synthetic-c{c}-s{s}")
            sessions.append(
                SessionEntry(sid, ProfileId(f"prf-synthetic-{c}"), split_of_session.get(s, "train"))
            )
            for k in range(per_session):
                i = int(idx[s * per_session + k])
                rel = f"c{c}/s{s}/{k:03d}.png"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(root / rel), images[i]):
                    raise OSError(f"Écriture impossible : {root / rel}")
                samples.append(SampleEntry(sid, rel, None))
    return Manifest(preprocess=PREPROCESS_ID, sessions=tuple(sessions), samples=tuple(samples))
