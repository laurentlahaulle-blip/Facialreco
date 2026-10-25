"""Fichiers de poids tiers du lot D : emplacement, empreinte attendue, vérification.

Les poids ne sont jamais committés. Ils sont cherchés dans ``MAISONACCUEIL_MODELS_DIR``
puis dans ``<racine du dépôt>/models/d``. Voir docs/LICENSES-D.md (licences et sources).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


class ModelFileMissingError(FileNotFoundError):
    pass


class ModelIntegrityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelFile:
    filename: str
    sha256: str
    size: int
    license: str  # licence des POIDS
    source_url: str

    @property
    def short_hash(self) -> str:
        return self.sha256[:8]


YUNET_2023MAR = ModelFile(
    filename="face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size=232_589,
    license="MIT",
    source_url="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
)

SFACE_2021DEC = ModelFile(
    filename="face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    size=38_696_353,
    license="Apache-2.0",
    source_url="https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface",
)


def default_models_dir() -> Path:
    env = os.environ.get("MAISONACCUEIL_MODELS_DIR")
    if env:
        return Path(env)
    # src/maisonaccueil/vision/model_files.py -> racine du dépôt
    return Path(__file__).resolve().parents[3] / "models" / "d"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate(model: ModelFile, models_dir: Path | None = None) -> Path | None:
    """Chemin du fichier s'il existe (sans vérifier l'empreinte), sinon None."""
    path = (models_dir or default_models_dir()) / model.filename
    return path if path.is_file() else None


def resolve_verified(model: ModelFile, models_dir: Path | None = None) -> Path:
    """Chemin vérifié (taille + sha256) ; lève une erreur explicite sinon."""
    base = models_dir or default_models_dir()
    path = base / model.filename
    if not path.is_file():
        raise ModelFileMissingError(
            f"Poids absents : {path}. Voir docs/LICENSES-D.md pour la source ({model.source_url})."
        )
    if path.stat().st_size != model.size:
        raise ModelIntegrityError(f"Taille inattendue pour {path}")
    digest = sha256_file(path)
    if digest != model.sha256:
        raise ModelIntegrityError(f"Empreinte sha256 inattendue pour {path}: {digest}")
    return path
