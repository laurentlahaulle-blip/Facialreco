"""Lecture des visages déjà alignés (sans torch).

Le jeu d'apprentissage est constitué d'images ALIGNÉES ``PREPROCESS_ID`` (112×112 BGR),
produites hors ligne à partir des images de référence et des 5 points du détecteur
(lot D). Un manifeste dont ``preprocess`` diffère est refusé.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from .manifest import Manifest, ManifestError
from .preprocess import INPUT_SIZE, PREPROCESS_ID, to_input_array


def load_aligned(path: Path) -> npt.NDArray[np.float32]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Image illisible : {path}")
    if image.shape != (INPUT_SIZE, INPUT_SIZE, 3):
        raise ValueError(
            f"{path} : visage non aligné {image.shape} (attendu {(INPUT_SIZE, INPUT_SIZE, 3)} "
            f"selon {PREPROCESS_ID})"
        )
    return to_input_array(image)


@dataclass(frozen=True, slots=True)
class AlignedTrainingSet:
    """Découpage ``train`` d'un manifeste, prêt à être lu image par image."""

    root: Path
    paths: tuple[str, ...]
    labels: tuple[int, ...]
    classes: tuple[str, ...]

    @classmethod
    def from_manifest(cls, manifest: Manifest, root: Path) -> AlignedTrainingSet:
        if manifest.preprocess != PREPROCESS_ID:
            raise ManifestError(
                f"Manifeste prétraité {manifest.preprocess!r}, attendu {PREPROCESS_ID!r} "
                "(aligner d'abord les images avec les 5 points du détecteur)"
            )
        manifest.validate()
        paths, labels, classes = manifest.training_samples()
        if len(classes) < 2:
            raise ManifestError("Il faut au moins 2 personnes dans 'train'")
        return cls(root, tuple(paths), tuple(labels), tuple(classes))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[npt.NDArray[np.float32], int]:
        rel = Path(self.paths[index])
        path = rel if rel.is_absolute() else self.root / rel
        return load_aligned(path), self.labels[index]
