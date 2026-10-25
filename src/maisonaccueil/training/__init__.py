"""Lot E — NOTRE IA : modèle de représentation faciale entraîné de zéro.

Ce paquet s'importe SANS PyTorch : ``net``, ``trainer`` et ``export`` importent ``torch``
paresseusement (extra ``training``). Voir docs/lots/E-training.md.
"""

from .config import ExperimentConfig
from .embedder import ScratchFaceEmbedder
from .manifest import (
    Manifest,
    ManifestError,
    SampleEntry,
    SessionEntry,
    SplitLeakError,
    assign_splits,
    manifest_from_database,
)
from .preprocess import PREPROCESS_ID, align_face, to_input_array
from .runs import RunDir, build_model_ref, device_guard
from .trainer import TorchUnavailableError, torch_available

__all__ = [
    "PREPROCESS_ID",
    "ExperimentConfig",
    "Manifest",
    "ManifestError",
    "RunDir",
    "SampleEntry",
    "ScratchFaceEmbedder",
    "SessionEntry",
    "SplitLeakError",
    "TorchUnavailableError",
    "align_face",
    "assign_splits",
    "build_model_ref",
    "device_guard",
    "manifest_from_database",
    "to_input_array",
    "torch_available",
]
