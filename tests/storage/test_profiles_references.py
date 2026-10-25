from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from maisonaccueil.contracts import (
    ArrivalEvent,
    ArrivalKind,
    Embedding,
    ProfileId,
    TrackId,
    new_event_id,
)
from maisonaccueil.storage import (
    ArrivalLogRepository,
    Database,
    ProfileNotFoundError,
    ProfileRepository,
    ReferenceNotFoundError,
    ReferenceRepository,
    SessionNotFoundError,
    SqliteEmbeddingStore,
)

from .conftest import CONSENT, FAKE_IMAGE, make_model


def test_create_requires_consent_and_name(profiles: ProfileRepository) -> None:
    with pytest.raises(TypeError):
        profiles.create("Personne A")  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        profiles.create("Personne A", consent_at=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        profiles.create("Personne A", consent_at=datetime(2026, 1, 1))  # naïf
    with pytest.raises(ValueError):
        profiles.create("   ", consent_at=CONSENT)
    with pytest.raises(ValueError):
        profiles.create("X", consent_at=CONSENT, profile_id=ProfileId("../evil"))


def test_crud_and_deactivate(profiles: ProfileRepository) -> None:
    p = profiles.create("Personne B", consent_at=CONSENT, spoken_name="Bé")
    assert p.profile_id.startswith("prf-")
    assert p.consent_at == CONSENT and p.active
    assert profiles.get(p.profile_id) == p
    q = profiles.create("personne a", consent_at=CONSENT)
    assert [x.display_name for x in profiles.list()] == ["personne a", "Personne B"]

    upd = profiles.update(p.profile_id, welcome_text="Bonjour", display_name=" B2 ")
    assert upd.welcome_text == "Bonjour" and upd.display_name == "B2"
    assert upd.updated_at >= p.updated_at
    with pytest.raises(ValueError):
        profiles.update(p.profile_id, consent_at="x")
    with pytest.raises(ValueError):
        profiles.update(p.profile_id, display_name="")

    profiles.deactivate(q.profile_id)
    assert [x.profile_id for x in profiles.list()] == [p.profile_id]
    assert len(profiles.list(include_inactive=True)) == 2
    assert profiles.activate(q.profile_id).active

    later = CONSENT + timedelta(days=30)
    assert profiles.renew_consent(p.profile_id, later, "renouvelé").consent_at == later

    with pytest.raises(ProfileNotFoundError):
        profiles.get(ProfileId("prf-missing"))
    with pytest.raises(ProfileNotFoundError):
        profiles.update(ProfileId("prf-missing"), spoken_name="x")
    with pytest.raises(ProfileNotFoundError):
        profiles.deactivate(ProfileId("prf-missing"))
    assert profiles.find(ProfileId("prf-missing")) is None


def test_sessions_and_split(profiles: ProfileRepository, refs: ReferenceRepository) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    s1 = refs.create_session(p.profile_id, started_at=CONSENT, split="enroll", camera_id=None)
    s2 = refs.create_session(None, started_at=CONSENT + timedelta(days=1), split="test")
    assert refs.get_session(s1.session_id) == s1
    assert s2.profile_id is None
    assert [s.session_id for s in refs.list_sessions(split="test")] == [s2.session_id]
    assert [s.session_id for s in refs.list_sessions(profile_id=p.profile_id)] == [s1.session_id]
    assert refs.set_session_split(s1.session_id, "train").split == "train"
    with pytest.raises(ValueError):
        refs.create_session(None, started_at=CONSENT, split="validation")
    with pytest.raises(SessionNotFoundError):
        refs.set_session_split("ses-missing", "test")  # type: ignore[arg-type]


def test_add_reference_bytes_and_import(
    tmp_path: Path, db: Database, profiles: ProfileRepository, refs: ReferenceRepository
) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    s = refs.create_session(p.profile_id, started_at=CONSENT, split="enroll")
    r = refs.add_image_bytes(
        p.profile_id, FAKE_IMAGE, captured_at=CONSENT, session_id=s.session_id,
        width=8, height=8,
    )
    assert r.file_path.is_file()
    assert r.file_path.read_bytes() == FAKE_IMAGE
    assert db.media_root in r.file_path.parents
    assert len(r.sha256) == 64 and refs.verify(r.reference_id)
    assert refs.list_for_session(s.session_id) == [r]

    src = tmp_path / "outside.jpg"
    src.write_bytes(b"fake-jpeg-synthetic")
    r2 = refs.import_file(p.profile_id, src, captured_at=CONSENT)
    assert src.exists() and r2.file_path != src and refs.verify(r2.reference_id)
    assert [x.reference_id for x in refs.list_for_profile(p.profile_id)] == [
        r.reference_id, r2.reference_id,
    ]

    r2.file_path.write_bytes(b"tampered")
    assert not refs.verify(r2.reference_id)

    with pytest.raises(ValueError):
        refs.add_image_bytes(p.profile_id, FAKE_IMAGE, extension=".exe", captured_at=CONSENT)
    with pytest.raises(ValueError):
        refs.add_image_bytes(p.profile_id, b"", captured_at=CONSENT)


def test_failed_insert_removes_written_file(
    db: Database, profiles: ProfileRepository, refs: ReferenceRepository
) -> None:
    other = profiles.create("Autre", consent_at=CONSENT)
    p = profiles.create("P", consent_at=CONSENT)
    s = refs.create_session(other.profile_id, started_at=CONSENT)
    with pytest.raises(ValueError, match="autre profil"):
        refs.add_image_bytes(p.profile_id, FAKE_IMAGE, captured_at=CONSENT,
                             session_id=s.session_id)
    with pytest.raises(Exception):  # noqa: B017 - FK : profil inexistant
        refs.add_image_bytes(ProfileId("prf-missing"), FAKE_IMAGE, captured_at=CONSENT)
    files = [f for f in db.media_root.rglob("*") if f.is_file()] if db.media_root.exists() else []
    assert files == []


def test_delete_single_reference_cascades_embeddings(
    db: Database, profiles: ProfileRepository, refs: ReferenceRepository
) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    r = refs.add_image_bytes(p.profile_id, FAKE_IMAGE, captured_at=CONSENT)
    store = SqliteEmbeddingStore(db)
    m = make_model()
    store.add_embedding(p.profile_id, r.reference_id, Embedding(m, np.ones(4, np.float32)))
    assert refs.delete(r.reference_id)
    assert not r.file_path.exists()
    assert store.count(m) == 0
    assert not refs.delete(r.reference_id)
    with pytest.raises(ReferenceNotFoundError):
        refs.get(r.reference_id)


def test_full_profile_deletion(
    tmp_path: Path, db: Database, profiles: ProfileRepository, refs: ReferenceRepository
) -> None:
    keep = profiles.create("Garde", consent_at=CONSENT)
    p = profiles.create("Supprime", consent_at=CONSENT)
    s = refs.create_session(p.profile_id, started_at=CONSENT, split="enroll")
    r1 = refs.add_image_bytes(p.profile_id, FAKE_IMAGE, captured_at=CONSENT,
                              session_id=s.session_id)
    r2 = refs.add_image_bytes(p.profile_id, FAKE_IMAGE + b"2", captured_at=CONSENT)
    rk = refs.add_image_bytes(keep.profile_id, FAKE_IMAGE, captured_at=CONSENT)
    # Référence pointant HORS media_root (insérée à la main) : ne doit jamais être effacée.
    outside = tmp_path / "ne_pas_effacer.png"
    outside.write_bytes(b"x")
    db.conn.execute(
        "INSERT INTO reference_images(profile_id, file_path, sha256, captured_at)"
        " VALUES (?, ?, ?, ?)",
        (p.profile_id, str(outside), "0" * 64, CONSENT.isoformat()),
    )
    store = SqliteEmbeddingStore(db)
    m = make_model()
    store.add_embedding(p.profile_id, r1.reference_id, Embedding(m, np.ones(4, np.float32)))
    store.add_embedding(p.profile_id, None, Embedding(m, np.ones(4, np.float32)))
    store.add_embedding(keep.profile_id, rk.reference_id, Embedding(m, np.ones(4, np.float32)))
    log = ArrivalLogRepository(db)
    log.record(ArrivalEvent(new_event_id(), ArrivalKind.KNOWN, TrackId("trk-1"), 0,
                            datetime.now(UTC), p.profile_id))

    report = profiles.delete(p.profile_id)

    assert report.references_deleted == 3
    assert report.embeddings_deleted == 2
    assert report.sessions_deleted == 1
    assert report.arrivals_deleted == 1
    assert report.files_deleted == 2
    assert report.files_skipped == (str(outside),)
    assert outside.exists()
    assert not r1.file_path.exists() and not r2.file_path.exists()
    assert not (db.media_root / p.profile_id).exists()
    assert rk.file_path.exists()
    for table in ("profiles", "reference_images", "embeddings", "collection_sessions",
                  "arrival_log"):
        n = db.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE profile_id = ?", (p.profile_id,)
        ).fetchone()[0]
        assert n == 0, table
    assert [pid for pid, _ in store.load_embeddings(m)] == [keep.profile_id]
    with pytest.raises(ProfileNotFoundError):
        profiles.delete(p.profile_id)


def test_delete_keeps_anonymised_arrivals_when_asked(
    db: Database, profiles: ProfileRepository
) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    log = ArrivalLogRepository(db)
    log.record(ArrivalEvent(new_event_id(), ArrivalKind.KNOWN, TrackId("trk-1"), 0,
                            datetime.now(UTC), p.profile_id))
    profiles.delete(p.profile_id, purge_arrivals=False)
    [entry] = log.list_recent()
    assert entry.profile_id is None
