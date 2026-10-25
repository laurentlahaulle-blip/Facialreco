from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from maisonaccueil.contracts import (
    ArrivalEvent,
    ArrivalKind,
    Embedding,
    EventId,
    Metric,
    ProfileId,
    TrackId,
    interfaces,
)
from maisonaccueil.storage import (
    ArrivalLogRepository,
    CorruptEmbeddingError,
    Database,
    ProfileRepository,
    ReferenceRepository,
    SettingsRepository,
    SqliteEmbeddingStore,
)

from .conftest import CONSENT, FAKE_IMAGE, make_model


def test_implements_protocol(db: Database) -> None:
    assert isinstance(SqliteEmbeddingStore(db), interfaces.EmbeddingStore)


def test_roundtrip_float32_exact(db: Database, profiles: ProfileRepository) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    store = SqliteEmbeddingStore(db)
    m = make_model()
    v = np.array([0.1, -2.5, 3.25, 1e-7], np.float32)
    eid = store.add_embedding(p.profile_id, None, Embedding(m, v))
    assert eid > 0
    [(pid, e)] = list(store.load_embeddings(m))
    assert pid == p.profile_id and e.model == m
    assert e.vector.dtype == np.float32 and np.array_equal(e.vector, v)
    e.vector[0] = 9.0  # copie modifiable, n'affecte pas la base
    [(_, e2)] = list(store.load_embeddings(m))
    assert e2.vector[0] == np.float32(0.1)
    blob = db.conn.execute("SELECT vector FROM embeddings").fetchone()[0]
    assert blob == v.astype("<f4").tobytes()


def test_strict_model_filtering_and_purge(db: Database, profiles: ProfileRepository) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    store = SqliteEmbeddingStore(db)
    m1 = make_model()
    variants = [
        make_model(version="w2"),
        make_model(preprocess="p2"),
        make_model(metric=Metric.L2),
        make_model(name="other"),
        make_model(embedding_dim=8),
    ]
    store.add_embedding(p.profile_id, None, Embedding(m1, np.ones(4, np.float32)))
    store.add_embedding(p.profile_id, None, Embedding(m1, np.zeros(4, np.float32)))
    for m in variants:
        dim = m.embedding_dim
        store.add_embedding(p.profile_id, None, Embedding(m, np.ones(dim, np.float32)))
    assert len(list(store.load_embeddings(m1))) == 2
    for m in variants:
        loaded = list(store.load_embeddings(m))
        assert len(loaded) == 1 and loaded[0][1].model == m
    assert store.model_keys()[m1.key] == 2
    assert store.delete_embeddings_for_model(m1) == 2
    assert list(store.load_embeddings(m1)) == []
    assert store.count() == len(variants)
    assert store.delete_embeddings_for_model(m1) == 0


def test_rejects_inconsistent_vectors(db: Database, profiles: ProfileRepository) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    store = SqliteEmbeddingStore(db)
    m = make_model()
    # Embedding valide au départ, puis contourné (object.__setattr__) pour simuler un
    # appelant qui fabrique un objet incohérent.
    bad = Embedding(m, np.ones(4, np.float32))
    object.__setattr__(bad, "vector", np.ones(5, np.float32))
    with pytest.raises(ValueError, match="incohérente"):
        store.add_embedding(p.profile_id, None, bad)
    bad64 = Embedding(m, np.ones(4, np.float32))
    object.__setattr__(bad64, "vector", np.ones(4, np.float64))
    with pytest.raises(ValueError):
        store.add_embedding(p.profile_id, None, bad64)
    nan = np.ones(4, np.float32)
    nan[1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        store.add_embedding(p.profile_id, None, Embedding(m, nan))
    assert store.count() == 0


def test_reference_must_belong_to_profile(
    db: Database, profiles: ProfileRepository, refs: ReferenceRepository
) -> None:
    a = profiles.create("A", consent_at=CONSENT)
    b = profiles.create("B", consent_at=CONSENT)
    ra = refs.add_image_bytes(a.profile_id, FAKE_IMAGE, captured_at=CONSENT)
    store = SqliteEmbeddingStore(db)
    e = Embedding(make_model(), np.ones(4, np.float32))
    with pytest.raises(ValueError, match="autre profil"):
        store.add_embedding(b.profile_id, ra.reference_id, e)
    with pytest.raises(KeyError):
        store.add_embedding(a.profile_id, 999, e)
    with pytest.raises(Exception):  # noqa: B017 - FK profil inexistant
        store.add_embedding(ProfileId("prf-missing"), None, e)


def test_corrupt_rows_detected(db: Database, profiles: ProfileRepository) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    store = SqliteEmbeddingStore(db)
    m = make_model()
    store.add_embedding(p.profile_id, None, Embedding(m, np.ones(4, np.float32)))
    db.conn.execute("UPDATE embeddings SET vector = ?", (b"\x00" * 12,))
    with pytest.raises(CorruptEmbeddingError):
        list(store.load_embeddings(m))
    db.conn.execute("UPDATE embeddings SET vector = ?, model_version = 'w9'", (b"\x00" * 16,))
    with pytest.raises(CorruptEmbeddingError):
        list(store.load_embeddings(m))


def test_inactive_profiles_excluded_by_default(
    db: Database, profiles: ProfileRepository
) -> None:
    a = profiles.create("A", consent_at=CONSENT)
    b = profiles.create("B", consent_at=CONSENT)
    store = SqliteEmbeddingStore(db)
    m = make_model()
    for pid in (a.profile_id, b.profile_id):
        store.add_embedding(pid, None, Embedding(m, np.ones(4, np.float32)))
    store.delete_embeddings_for_profile(b.profile_id, make_model(version="other"))
    profiles.deactivate(b.profile_id)
    assert [pid for pid, _ in store.load_embeddings(m)] == [a.profile_id]
    all_store = SqliteEmbeddingStore(db, include_inactive=True)
    assert len(list(all_store.load_embeddings(m))) == 2
    assert store.delete_embeddings_for_profile(b.profile_id) == 1


def test_settings_json(db: Database) -> None:
    s = SettingsRepository(db)
    assert s.get("absent", 42) == 42
    s.set("camera", {"kind": "usb", "location": "0", "fps": 15.0})
    s.set("volume", 0.5)
    s.set("volume", 0.7)
    assert s.get("camera") == {"kind": "usb", "location": "0", "fps": 15.0}
    assert s.get("volume") == 0.7
    s.set_many({"a": [1, 2], "b": None, "nom": "é"})
    assert s.all() == {
        "a": [1, 2], "b": None, "camera": {"fps": 15.0, "kind": "usb", "location": "0"},
        "nom": "é", "volume": 0.7,
    }
    assert s.delete("a") and not s.delete("a")
    with pytest.raises(ValueError):
        s.set("x", math.nan)
    with pytest.raises(TypeError):
        s.set("x", object())
    with pytest.raises(ValueError):
        s.set("", 1)


def test_arrival_log_record_list_purge(db: Database, profiles: ProfileRepository) -> None:
    p = profiles.create("P", consent_at=CONSENT)
    log = ArrivalLogRepository(db)
    t0 = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
    events = [
        ArrivalEvent(EventId(f"evt-{i}"), ArrivalKind.KNOWN, TrackId(f"trk-{i}"), i,
                     t0 + timedelta(days=i), p.profile_id)
        for i in range(3)
    ]
    events.append(ArrivalEvent(EventId("evt-u"), ArrivalKind.UNKNOWN, TrackId("trk-u"), 9,
                               t0 + timedelta(days=5, microseconds=500)))
    for e in events:
        assert log.record(e, model_key=make_model().key, policy="temporal-v1")
    assert not log.record(events[0])  # idempotent
    recent = log.list_recent(limit=2)
    assert [r.event_id for r in recent] == ["evt-u", "evt-2"]
    assert recent[0].kind == ArrivalKind.UNKNOWN and recent[0].profile_id is None
    assert recent[1].occurred_at == t0 + timedelta(days=2)
    assert len(log.list_recent(profile_id=p.profile_id)) == 3
    assert log.purge_older_than(t0 + timedelta(days=1)) == 1
    assert log.count() == 3
    with pytest.raises(ValueError):
        log.purge_older_than(datetime(2026, 1, 1))
    assert log.purge_all() == 3 and log.count() == 0
