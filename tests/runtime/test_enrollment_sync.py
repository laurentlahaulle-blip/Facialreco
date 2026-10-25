"""Synchronisation enrôlement : seules les séances ``enroll`` de profils actifs sont
transmises à l'enrôleur. Images SYNTHÉTIQUES ; aucune mesure de qualité."""

from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from maisonaccueil.runtime.enrollment_sync import enroll_pending
from maisonaccueil.storage import (
    Database,
    ProfileRepository,
    ReferenceRepository,
    SqliteEmbeddingStore,
)


class RecordingEnroller:
    def __init__(self):
        self.items = []

    def enroll(self, items, store):
        self.items.extend(items)

        class R:
            added: list = []
            skipped: list = []

            def summary(self):
                return ""

        return R()


def _png() -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "s.sqlite")
    yield d
    d.close()


def _add(db, name, split, active=True):
    profiles, refs = ProfileRepository(db), ReferenceRepository(db)
    p = profiles.create(name, consent_at=datetime.now(UTC))
    s = refs.create_session(p.profile_id, started_at=datetime.now(UTC), split=split)
    refs.add_image_bytes(p.profile_id, _png(), captured_at=datetime.now(UTC),
                         session_id=s.session_id, extension=".png")
    if not active:
        profiles.deactivate(p.profile_id)
    return p.profile_id


def test_only_enroll_sessions_of_active_profiles(db):
    a = _add(db, "A", "enroll")
    _add(db, "B", "test")
    _add(db, "C", "enroll", active=False)
    rec = RecordingEnroller()
    enroll_pending(db, rec, SqliteEmbeddingStore(db))
    assert [i.profile_id for i in rec.items] == [a]
    assert rec.items[0].image_path.exists()


def test_real_enroller_on_synthetic_image_adds_nothing(db):
    from maisonaccueil.recognition.enrollment import build_reference_enroller

    try:
        enroller = build_reference_enroller()
    except Exception as exc:
        pytest.skip(f"poids de référence indisponibles : {exc}")
    _add(db, "A", "enroll")
    store = SqliteEmbeddingStore(db)
    report = enroll_pending(db, enroller, store)
    assert report.processed == 1 and not report.added  # image noire : aucun visage
