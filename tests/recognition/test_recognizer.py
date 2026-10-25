"""Chaîne complète sur images SYNTHÉTIQUES (rectangles colorés) avec doubles Fake*.

Vérifie le câblage (suivi stable, candidats, indicateurs, pertes), pas une précision.
"""

import numpy as np
import pytest

from maisonaccueil.contracts import (
    BoundingBox,
    Detection,
    IncompatibleEmbeddingError,
    ProfileId,
    QualityFlag,
    TrackedFace,
    TrackId,
)
from maisonaccueil.contracts.interfaces import FaceEmbedder, Recognizer
from maisonaccueil.recognition import (
    FAKE_COLOR_MODEL_REF,
    FakeColorEmbedder,
    FakeEmbeddingStore,
    InMemoryGallery,
    ReferenceRecognizer,
)
from maisonaccueil.vision import FakeBlobDetector, IouTracker, synthetic_blob_image
from tests.vision.helpers import make_frame

RED, GREEN = (30, 30, 220), (30, 220, 30)


def _enrol(store: FakeEmbeddingStore, embedder: FakeColorEmbedder) -> None:
    for pid, color in [("prf-red", RED), ("prf-green", GREEN)]:
        img = synthetic_blob_image(240, 320, [(100, 80, 80, 80, color)])
        f = make_frame(0, img)
        d = Detection(f.frame_id, BoundingBox(100, 80, 80, 80), 1.0, "enrol")
        e = embedder.embed(f, TrackedFace(TrackId("t"), d, 0, 0))
        assert e is not None
        store.add_embedding(ProfileId(pid), None, e)


@pytest.fixture
def chain() -> tuple[ReferenceRecognizer, FakeColorEmbedder]:
    store, emb = FakeEmbeddingStore(), FakeColorEmbedder()
    _enrol(store, emb)
    gallery = InMemoryGallery(FAKE_COLOR_MODEL_REF, store)
    rec = ReferenceRecognizer(FakeBlobDetector(), IouTracker(max_missed_frames=2), emb, gallery)
    return rec, emb


def test_protocols(chain):
    rec, emb = chain
    assert isinstance(rec, Recognizer)
    assert isinstance(emb, FaceEmbedder)


def test_two_moving_faces_stable_tracks_and_candidates(chain):
    rec, _ = chain
    seen: dict[str, set[str]] = {"prf-red": set(), "prf-green": set()}
    for i in range(6):
        img = synthetic_blob_image(
            240, 320, [(10 + 4 * i, 40, 70, 70, RED), (200 - 3 * i, 120, 70, 70, GREEN)]
        )
        results = rec.process(make_frame(i, img))
        assert len(results) == 2
        for r in results:
            assert r.frame_id == i and r.model == FAKE_COLOR_MODEL_REF
            assert r.best is not None and r.quality_flags == frozenset()
            assert len(r.candidates) == 2
            seen[r.best.profile_id].add(r.track_id)
    assert all(len(v) == 1 for v in seen.values())  # un passage stable par rectangle
    assert seen["prf-red"] != seen["prf-green"]


def test_small_face_flagged_and_not_embedded(chain):
    rec, emb = chain
    img = synthetic_blob_image(240, 320, [(10, 10, 20, 20, RED)])
    results = rec.process(make_frame(0, img))
    assert len(results) == 1
    assert QualityFlag.FACE_TOO_SMALL in results[0].quality_flags
    assert results[0].candidates == ()
    assert emb.calls == 2  # seulement les 2 appels d'enrôlement


def test_disappearance_reports_lost_track(chain):
    rec, _ = chain
    img = synthetic_blob_image(240, 320, [(50, 50, 80, 80, GREEN)])
    tid = rec.process(make_frame(0, img))[0].track_id
    for i in range(1, 5):
        assert rec.process(make_frame(i)) == []
    assert rec.lost_tracks() == [tid]
    assert rec.lost_tracks() == []


def test_embedder_and_gallery_models_must_match():
    from maisonaccueil.contracts import Metric, ModelRef

    other = ModelRef("x", "v1", 24, Metric.COSINE, "p", "test-double")
    with pytest.raises(IncompatibleEmbeddingError):
        ReferenceRecognizer(
            FakeBlobDetector(), IouTracker(), FakeColorEmbedder(), InMemoryGallery(other)
        )


def test_no_gallery_entries_still_emits_results():
    rec = ReferenceRecognizer(
        FakeBlobDetector(), IouTracker(), FakeColorEmbedder(), InMemoryGallery(FAKE_COLOR_MODEL_REF)
    )
    img = synthetic_blob_image(240, 320, [(50, 50, 80, 80, GREEN)])
    out = rec.process(make_frame(0, img))
    assert len(out) == 1 and out[0].candidates == ()
    assert np.isfinite(out[0].monotonic_ns)
