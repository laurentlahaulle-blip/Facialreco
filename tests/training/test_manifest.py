from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from maisonaccueil.contracts import ProfileId, SessionId, schema_sql
from maisonaccueil.training.manifest import (
    RAW_PREPROCESS,
    Manifest,
    ManifestError,
    SampleEntry,
    SessionEntry,
    SplitLeakError,
    assign_splits,
    manifest_from_database,
)
from maisonaccueil.training.preprocess import PREPROCESS_ID


def S(sid: str, prof: str | None, split: str) -> SessionEntry:  # noqa: N802
    return SessionEntry(SessionId(sid), ProfileId(prof) if prof else None, split)


def X(sid: str, path: str, sha: str | None = None) -> SampleEntry:  # noqa: N802
    return SampleEntry(SessionId(sid), path, sha)


def _manifest() -> Manifest:
    return Manifest(
        PREPROCESS_ID,
        (
            S("ses-a1", "prf-a", "train"),
            S("ses-b1", "prf-b", "train"),
            S("ses-a2", "prf-a", "test"),
            S("ses-u1", None, "tune"),
        ),
        (
            X("ses-a1", "a1/0.png", "h1"),
            X("ses-a1", "a1/1.png", "h2"),
            X("ses-b1", "b1/0.png", "h3"),
            X("ses-a2", "a2/0.png", "h4"),
            X("ses-u1", "u1/0.png", "h5"),
        ),
    )


def test_training_samples_use_only_train_split() -> None:
    paths, labels, classes = _manifest().training_samples()
    assert paths == ["a1/0.png", "a1/1.png", "b1/0.png"]
    assert labels == [0, 0, 1]
    assert classes == ["prf-a", "prf-b"]


def test_counts() -> None:
    counts = _manifest().counts()
    assert counts["train"] == {"sessions": 2, "samples": 3, "profiles": 2}
    assert counts["tune"] == {"sessions": 1, "samples": 1, "profiles": 0}


def test_duplicate_session_is_refused() -> None:
    with pytest.raises(SplitLeakError):
        Manifest(PREPROCESS_ID, (S("ses-a", "prf-a", "train"), S("ses-a", "prf-a", "test")), ())


def test_same_file_in_two_sessions_is_refused() -> None:
    sessions = (S("ses-a1", "prf-a", "train"), S("ses-a2", "prf-a", "test"))
    with pytest.raises(SplitLeakError, match="chemin"):
        Manifest(PREPROCESS_ID, sessions, (X("ses-a1", "p.png"), X("ses-a2", "p.png")))
    with pytest.raises(SplitLeakError, match="empreinte"):
        Manifest(PREPROCESS_ID, sessions, (X("ses-a1", "1.png", "h"), X("ses-a2", "2.png", "h")))


def test_unknown_session_and_unlabelled_train_are_refused() -> None:
    with pytest.raises(ManifestError):
        Manifest(PREPROCESS_ID, (S("ses-a", "prf-a", "train"),), (X("ses-zz", "p.png"),))
    with pytest.raises(ManifestError):
        Manifest(PREPROCESS_ID, (S("ses-u", None, "train"),), ())
    with pytest.raises(ManifestError):
        S("ses-a", "prf-a", "validation")


def test_identity_disjoint_option() -> None:
    m = _manifest()
    with pytest.raises(SplitLeakError):
        m.check_identity_disjoint("train", "test")
    m.check_identity_disjoint("train", "tune")


def test_json_roundtrip(tmp_path: Path) -> None:
    m = _manifest()
    m.save(tmp_path / "m.json")
    assert Manifest.load(tmp_path / "m.json") == m


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "e.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(schema_sql())
    now = "2026-01-01T00:00:00+00:00"
    for prof in ("prf-synthetic-a", "prf-synthetic-b"):
        conn.execute(
            "INSERT INTO profiles (profile_id, display_name, consent_at, created_at, updated_at)"
            " VALUES (?, 'Synthétique', ?, ?, ?)",
            (prof, now, now, now),
        )
    sessions = [
        ("ses-1", "prf-synthetic-a", "train"),
        ("ses-2", "prf-synthetic-b", "train"),
        ("ses-3", "prf-synthetic-a", "test"),
        ("ses-4", None, "tune"),
    ]
    for sid, prof, split in sessions:
        conn.execute(
            "INSERT INTO collection_sessions (session_id, profile_id, started_at, split)"
            " VALUES (?, ?, ?, ?)",
            (sid, prof, now, split),
        )
    for i, (sid, prof) in enumerate([("ses-1", "prf-synthetic-a"), ("ses-2", "prf-synthetic-b"),
                                     ("ses-3", "prf-synthetic-a")]):
        conn.execute(
            "INSERT INTO reference_images (profile_id, session_id, file_path, sha256, captured_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (prof, sid, f"ref/{i}.png", f"sha{i}", now),
        )
    conn.commit()
    conn.close()
    return path


def test_manifest_from_database_is_read_only_and_per_session(tmp_path: Path) -> None:
    db = _db(tmp_path)
    m = manifest_from_database(db)
    assert m.preprocess == RAW_PREPROCESS
    assert {s.session_id: s.split for s in m.sessions} == {
        "ses-1": "train", "ses-2": "train", "ses-3": "test", "ses-4": "tune"
    }
    assert len(m.samples_for("train")) == 2


def test_manifest_from_database_refuses_unassigned(tmp_path: Path) -> None:
    db = _db(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO collection_sessions (session_id, profile_id, started_at, split)"
        " VALUES ('ses-5', NULL, 'x', NULL)"
    )
    conn.commit()
    conn.close()
    with pytest.raises(ManifestError):
        manifest_from_database(db)
    assert len(manifest_from_database(db, include_unassigned=True).sessions) == 4


def test_assign_splits_is_per_session_and_deterministic() -> None:
    ids = [SessionId(f"ses-{i:02d}") for i in range(10)]
    fractions = {"train": 0.6, "tune": 0.2, "test": 0.2}
    a = assign_splits(ids, fractions, seed=1)
    assert a == assign_splits(reversed(ids), fractions, seed=1)
    assert set(a) == set(ids)
    values = list(a.values())
    assert values.count("train") == 6 and values.count("tune") == 2
    assert values.count("test") == 2


def test_assign_splits_gives_each_split_a_session_when_possible() -> None:
    ids = [SessionId(f"ses-{i}") for i in range(3)]
    a = assign_splits(ids, {"train": 0.8, "tune": 0.1, "test": 0.1}, seed=0)
    assert sorted(a.values()) == ["test", "train", "tune"]
    with pytest.raises(ManifestError):
        assign_splits(ids, {"bogus": 1.0}, seed=0)
