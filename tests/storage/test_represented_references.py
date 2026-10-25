"""``SqliteEmbeddingStore.represented_reference_ids`` (proposition D-represented-references)
et idempotence de l'enrôlement du lot D sur le vrai stockage SQLite.

Images SYNTHÉTIQUES (rectangles colorés) et doubles Fake* du lot D : aucune photo réelle,
aucune mesure de précision.
"""

from __future__ import annotations

import cv2
import numpy as np

from maisonaccueil.contracts import Embedding
from maisonaccueil.recognition.enrollment import ReferenceEnroller, ReferenceItem, SkipReason
from maisonaccueil.recognition.fakes import FAKE_COLOR_MODEL_REF, FakeColorEmbedder
from maisonaccueil.storage import (
    Database,
    ProfileRepository,
    ReferenceRepository,
    SqliteEmbeddingStore,
)
from maisonaccueil.vision.fakes import FakeBlobDetector, synthetic_blob_image

from .conftest import CONSENT, FAKE_IMAGE, make_model

RED, GREEN = (30, 30, 220), (30, 220, 30)


def _vec(dim: int = 4) -> np.ndarray:
    return np.ones(dim, np.float32)


def test_strict_model_filtering(
    db: Database, profiles: ProfileRepository, refs: ReferenceRepository
) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    r1 = refs.add_image_bytes(p.profile_id, FAKE_IMAGE, captured_at=CONSENT)
    r2 = refs.add_image_bytes(p.profile_id, FAKE_IMAGE, captured_at=CONSENT)
    store = SqliteEmbeddingStore(db)
    m1, m2, m3 = make_model(), make_model(version="w2"), make_model(preprocess="p2")
    store.add_embedding(p.profile_id, r1.reference_id, Embedding(m1, _vec()))
    store.add_embedding(p.profile_id, r1.reference_id, Embedding(m1, _vec()))  # doublon
    store.add_embedding(p.profile_id, None, Embedding(m1, _vec()))  # sans référence
    store.add_embedding(p.profile_id, r2.reference_id, Embedding(m2, _vec()))
    assert store.represented_reference_ids(m1) == {r1.reference_id}
    assert store.represented_reference_ids(m2) == {r2.reference_id}
    assert store.represented_reference_ids(m3) == set()
    store.delete_embeddings_for_model(m1)
    assert store.represented_reference_ids(m1) == set()


def test_deleted_references_and_profiles_disappear(
    db: Database, profiles: ProfileRepository, refs: ReferenceRepository
) -> None:
    a = profiles.create("A", consent_at=CONSENT)
    b = profiles.create("B", consent_at=CONSENT)
    ra1 = refs.add_image_bytes(a.profile_id, FAKE_IMAGE, captured_at=CONSENT)
    ra2 = refs.add_image_bytes(a.profile_id, FAKE_IMAGE, captured_at=CONSENT)
    rb = refs.add_image_bytes(b.profile_id, FAKE_IMAGE, captured_at=CONSENT)
    store = SqliteEmbeddingStore(db)
    m = make_model()
    for pid, r in ((a.profile_id, ra1), (a.profile_id, ra2), (b.profile_id, rb)):
        store.add_embedding(pid, r.reference_id, Embedding(m, _vec()))
    assert store.represented_reference_ids(m) == {ra1.reference_id, ra2.reference_id,
                                                  rb.reference_id}
    refs.delete(ra1.reference_id)
    assert store.represented_reference_ids(m) == {ra2.reference_id, rb.reference_id}
    profiles.delete(b.profile_id)
    assert store.represented_reference_ids(m) == {ra2.reference_id}


def test_inactive_profiles_still_counted(
    db: Database, profiles: ProfileRepository, refs: ReferenceRepository
) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    r = refs.add_image_bytes(p.profile_id, FAKE_IMAGE, captured_at=CONSENT)
    m = make_model()
    store = SqliteEmbeddingStore(db)
    store.add_embedding(p.profile_id, r.reference_id, Embedding(m, _vec()))
    profiles.deactivate(p.profile_id)
    assert list(store.load_embeddings(m)) == []  # pas reconnu…
    # …mais toujours « déjà représenté », quel que soit include_inactive.
    assert store.represented_reference_ids(m) == {r.reference_id}
    inc = SqliteEmbeddingStore(db, include_inactive=True)
    assert inc.represented_reference_ids(m) == {r.reference_id}


def _png(blobs: list[tuple[int, int, int, int, tuple[int, int, int]]]) -> bytes:
    ok, buf = cv2.imencode(".png", synthetic_blob_image(240, 320, blobs))
    assert ok
    return buf.tobytes()


def test_reference_enroller_does_not_duplicate_with_sqlite_store(
    db: Database, profiles: ProfileRepository, refs: ReferenceRepository
) -> None:
    red = profiles.create("Rouge (synthétique)", consent_at=CONSENT)
    green = profiles.create("Vert (synthétique)", consent_at=CONSENT)
    stored = [
        refs.add_image_bytes(red.profile_id, _png([(100, 60, 90, 90, RED)]), captured_at=CONSENT),
        refs.add_image_bytes(red.profile_id, _png([(40, 80, 80, 80, RED)]), captured_at=CONSENT),
        refs.add_image_bytes(green.profile_id, _png([(120, 70, 90, 90, GREEN)]),
                             captured_at=CONSENT),
    ]
    items = [ReferenceItem(r.profile_id, r.reference_id, r.file_path) for r in stored]
    store = SqliteEmbeddingStore(db)
    model = FAKE_COLOR_MODEL_REF

    first = ReferenceEnroller(FakeBlobDetector(), FakeColorEmbedder()).enroll(items, store)
    assert first.added_count == 3, first.summary()
    assert store.count(model) == 3

    # Instance NEUVE (aucune mémoire) : seule la base évite les doublons.
    embedder = FakeColorEmbedder()
    second = ReferenceEnroller(FakeBlobDetector(), embedder).enroll(items, store)
    assert second.added_count == 0
    assert second.skipped_by_reason() == {SkipReason.ALREADY_REPRESENTED: 3}
    assert embedder.calls == 0  # rien n'a été recalculé
    assert store.count(model) == 3

    # Nouvelle image : seule elle est traitée.
    extra = refs.add_image_bytes(green.profile_id, _png([(60, 50, 100, 100, GREEN)]),
                                 captured_at=CONSENT)
    items.append(ReferenceItem(extra.profile_id, extra.reference_id, extra.file_path))
    third = ReferenceEnroller(FakeBlobDetector(), FakeColorEmbedder()).enroll(items, store)
    assert [a.reference_id for a in third.added] == [extra.reference_id]
    assert store.count(model) == 4
    assert store.represented_reference_ids(model) == {r.reference_id for r in [*stored, extra]}
