"""Lot D — vision : détection, alignement, suivi, qualité (voir docs/lots/D-recognition.md)."""

from .alignment import (
    ALIGNED_SIZE,
    ARCFACE_TEMPLATE_112,
    PREPROCESS_ID,
    AlignmentError,
    align_face,
    landmarks_from_bbox,
)
from .detection import YuNetDetector
from .fakes import FakeBlobDetector, FakeScriptedDetector, synthetic_blob_image
from .model_files import (
    SFACE_2021DEC,
    YUNET_2023MAR,
    ModelFile,
    ModelFileMissingError,
    ModelIntegrityError,
    default_models_dir,
    locate,
    resolve_verified,
)
from .quality import QUALITY_POLICY_ID, QualityConfig, assess_quality
from .tracking import IouTracker

__all__ = [
    "ALIGNED_SIZE",
    "ARCFACE_TEMPLATE_112",
    "PREPROCESS_ID",
    "QUALITY_POLICY_ID",
    "SFACE_2021DEC",
    "YUNET_2023MAR",
    "AlignmentError",
    "FakeBlobDetector",
    "FakeScriptedDetector",
    "IouTracker",
    "ModelFile",
    "ModelFileMissingError",
    "ModelIntegrityError",
    "QualityConfig",
    "YuNetDetector",
    "align_face",
    "assess_quality",
    "default_models_dir",
    "landmarks_from_bbox",
    "locate",
    "resolve_verified",
    "synthetic_blob_image",
]
