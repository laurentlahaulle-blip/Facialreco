"""Lot D — reconnaissance de référence : représentation, galerie, chaîne complète."""

from .embedders import SFACE_MODEL_REF, SFaceEmbedder, l2_normalize
from .enrollment import (
    AddedReference,
    EnrollmentConfig,
    EnrollmentReport,
    ReferenceEnroller,
    ReferenceItem,
    SkippedReference,
    SkipReason,
    build_reference_enroller,
    read_image,
)
from .fakes import (
    FAKE_COLOR_MODEL_REF,
    FakeColorEmbedder,
    FakeEmbeddingStore,
    FakeIndexedEmbeddingStore,
    FakeLeakyEmbeddingStore,
    FakeRandomEmbedder,
)
from .gallery import InMemoryGallery
from .recognizer import RecognizerConfig, ReferenceRecognizer, build_reference_recognizer

__all__ = [
    "AddedReference",
    "EnrollmentConfig",
    "EnrollmentReport",
    "FakeIndexedEmbeddingStore",
    "ReferenceEnroller",
    "ReferenceItem",
    "SkipReason",
    "SkippedReference",
    "build_reference_enroller",
    "read_image",
    "FAKE_COLOR_MODEL_REF",
    "SFACE_MODEL_REF",
    "FakeColorEmbedder",
    "FakeEmbeddingStore",
    "FakeLeakyEmbeddingStore",
    "FakeRandomEmbedder",
    "InMemoryGallery",
    "RecognizerConfig",
    "ReferenceRecognizer",
    "SFaceEmbedder",
    "build_reference_recognizer",
    "l2_normalize",
]
