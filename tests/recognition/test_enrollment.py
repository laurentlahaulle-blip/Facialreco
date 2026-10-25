"""Enrôlement sur images SYNTHÉTIQUES (rectangles colorés) écrites dans tmp_path.

Doubles Fake* sans poids ; un test de fumée avec les poids réels, ignoré s'ils sont
absents (zéro visage accepté). Aucune photo réelle, aucune mesure de précision.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from maisonaccueil.contracts import ProfileId
from maisonaccueil.recognition import (
    FAKE_COLOR_MODEL_REF,
    EnrollmentReport,
    FakeColorEmbedder,
    FakeEmbeddingStore,
    FakeIndexedEmbeddingStore,
    InMemoryGallery,
    ReferenceEnroller,
    ReferenceItem,
    SkipReason,
    read_image,
)
from maisonaccueil.vision import (
    SFACE_2021DEC,
    YUNET_2023MAR,
    FakeBlobDetector,
    synthetic_blob_image,
)
from tests.vision.helpers import make_frame, weights_path

RED, GREEN, BLUE = (30, 30, 220), (30, 220, 30), (220, 30, 30)
P_RED, P_GREEN = ProfileId("prf-red"), ProfileId("prf-green")


def write_png(path: Path, blobs, shape=(240, 320)) -> Path:
    ok, buf = cv2.imencode(".png", synthetic_blob_image(*shape, blobs))
    assert ok
    buf.tofile(str(path))  # comme l'écriture, gère les chemins non ASCII
    return path


def enroller() -> ReferenceEnroller:
    return ReferenceEnroller(FakeBlobDetector(), FakeColorEmbedder())


@pytest.fixture
def refs(tmp_path: Path) -> list[ReferenceItem]:
    return [
        ReferenceItem(P_RED, 1, write_png(tmp_path / "r1.png", [(100, 60, 90, 90, RED)])),
        ReferenceItem(P_RED, 2, write_png(tmp_path / "r2.png", [(40, 80, 80, 80, RED)])),
        ReferenceItem(P_GREEN, 3, write_png(tmp_path / "g1.png", [(120, 70, 90, 90, GREEN)])),
        ReferenceItem(P_GREEN, 4, write_png(tmp_path / "g2.png", [(60, 50, 100, 100, GREEN)])),
    ]


def test_enroll_adds_embeddings_bound_to_model(refs):
    store = FakeEmbeddingStore()
    report = enroller().enroll(refs, store)
    assert isinstance(report, EnrollmentReport)
    assert (report.processed, report.added_count, report.skipped_count) == (4, 4, 0)
    assert [a.reference_id for a in report.added] == [1, 2, 3, 4]
    assert report.model == FAKE_COLOR_MODEL_REF
    rows = list(store.load_embeddings(FAKE_COLOR_MODEL_REF))
    assert [p for p, _ in rows] == [P_RED, P_RED, P_GREEN, P_GREEN]
    assert all(e.model == FAKE_COLOR_MODEL_REF for _, e in rows)
    # la galerie alimentée par ce stockage n'est plus vide (câblage, pas une précision)
    gallery = InMemoryGallery(FAKE_COLOR_MODEL_REF, store)
    assert gallery.profile_ids == {P_RED, P_GREEN}


def test_idempotent_same_instance(refs):
    store, enr = FakeEmbeddingStore(), enroller()
    enr.enroll(refs, store)
    again = enr.enroll(refs, store)
    assert again.added_count == 0
    assert again.skipped_by_reason() == {SkipReason.ALREADY_REPRESENTED: 4}
    assert len(list(store.load_embeddings(FAKE_COLOR_MODEL_REF))) == 4


def test_idempotent_new_instance_with_caller_list_or_indexed_store(refs):
    store = FakeEmbeddingStore()
    enroller().enroll(refs[:2], store)
    r = enroller().enroll(refs, store, already_represented=[1, 2])
    assert [a.reference_id for a in r.added] == [3, 4]
    assert r.skipped_by_reason() == {SkipReason.ALREADY_REPRESENTED: 2}

    indexed = FakeIndexedEmbeddingStore()
    enroller().enroll(refs[:3], indexed)
    r2 = enroller().enroll(refs, indexed)  # méthode proposée détectée automatiquement
    assert [a.reference_id for a in r2.added] == [4]
    assert r2.skipped_by_reason() == {SkipReason.ALREADY_REPRESENTED: 3}


def test_bad_inputs_are_skipped_with_reason(tmp_path: Path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"pas une image")
    items = [
        ReferenceItem(P_RED, 10, write_png(tmp_path / "noface.png", [])),
        ReferenceItem(P_RED, 11, tmp_path / "absent.png"),
        ReferenceItem(P_RED, 12, corrupt),
        ReferenceItem(P_RED, 13, write_png(tmp_path / "small.png", [(10, 10, 20, 20, RED)])),
        ReferenceItem(
            P_RED, 14,
            write_png(tmp_path / "two.png", [(10, 40, 90, 90, RED), (200, 40, 90, 90, GREEN)]),
        ),
        ReferenceItem(
            P_RED, 15, write_png(tmp_path / "dark.png", [(100, 60, 90, 90, (35, 35, 35))])
        ),
    ]
    store = FakeEmbeddingStore()
    report = enroller().enroll(items, store)
    reasons = {s.reference_id: s.reason for s in report.skipped}
    assert reasons == {
        10: SkipReason.NO_FACE,
        11: SkipReason.FILE_MISSING,
        12: SkipReason.UNREADABLE,
        13: SkipReason.LOW_QUALITY,
        14: SkipReason.AMBIGUOUS_FACES,
        15: SkipReason.LOW_QUALITY,
    }
    assert "face_too_small" in next(s.detail for s in report.skipped if s.reference_id == 13)
    assert report.added_count == 0 and list(store.load_embeddings(FAKE_COLOR_MODEL_REF)) == []


def test_largest_face_is_used(tmp_path: Path):
    path = write_png(tmp_path / "main.png", [(20, 20, 150, 150, BLUE), (250, 20, 40, 40, RED)])
    store = FakeEmbeddingStore()
    report = enroller().enroll([(P_RED, 20, path)], store)  # tuple accepté
    assert report.added_count == 1
    stored = next(iter(store.load_embeddings(FAKE_COLOR_MODEL_REF)))[1]
    # même vecteur que celui du grand rectangle seul
    alone, *_ = enroller().embed_image(synthetic_blob_image(240, 320, [(20, 20, 150, 150, BLUE)]))
    assert alone is not None and np.allclose(stored.vector, alone.vector)


def test_duplicates_store_errors_and_summary(refs):
    class FakeRejectingStore(FakeEmbeddingStore):
        def add_embedding(self, profile_id, reference_id, embedding):
            if reference_id == 2:
                raise KeyError(f"reference_id inconnu : {reference_id}")
            return super().add_embedding(profile_id, reference_id, embedding)

    report = enroller().enroll([refs[0], refs[0], refs[1], refs[2]], FakeRejectingStore())
    assert report.processed == 4 and report.added_count == 2
    assert report.skipped_by_reason() == {
        SkipReason.DUPLICATE_IN_BATCH: 1,
        SkipReason.STORE_ERROR: 1,
    }
    text = report.summary()
    assert "4 référence(s) traitée(s)" in text and "%" not in text


def test_non_ascii_path_and_read_image(tmp_path: Path):
    path = write_png(tmp_path / "référence é.png", [(100, 60, 90, 90, GREEN)])
    img = read_image(path)
    assert img is not None and img.shape == (240, 320, 3)
    assert enroller().enroll([(P_GREEN, 30, path)], FakeEmbeddingStore()).added_count == 1


needs_weights = pytest.mark.skipif(
    weights_path(YUNET_2023MAR) is None or weights_path(SFACE_2021DEC) is None,
    reason="poids absents (models/d)",
)


@needs_weights
def test_reference_enroller_smoke_with_real_weights(tmp_path: Path):
    from maisonaccueil.recognition import SFACE_MODEL_REF, build_reference_enroller

    enr = build_reference_enroller()
    assert enr.model == SFACE_MODEL_REF
    rng = np.random.default_rng(0)
    noise = tmp_path / "noise.png"
    cv2.imencode(".png", rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))[1].tofile(
        str(noise)
    )
    items = [
        (P_RED, 1, noise),
        (P_RED, 2, write_png(tmp_path / "blank.png", [])),
        (P_RED, 3, write_png(tmp_path / "blob.png", [(100, 60, 90, 90, RED)])),
    ]
    store = FakeEmbeddingStore()
    report = enr.enroll(items, store)
    assert report.processed == 3
    assert report.added_count + report.skipped_count == 3  # zéro visage accepté
    for _p, e in store.load_embeddings(SFACE_MODEL_REF):
        assert e.vector.shape == (128,)
    # l'enrôleur réel accepte aussi un Frame-like via embed_image
    emb, flags, reason, _ = enr.embed_image(make_frame(0).image)
    assert emb is None and reason == SkipReason.NO_FACE
