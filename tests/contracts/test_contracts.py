import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from maisonaccueil import contracts as c
from maisonaccueil.contracts import interfaces


def _model(**kw: object) -> c.ModelRef:
    base = dict(
        name="m", version="w1", embedding_dim=4, metric=c.Metric.COSINE,
        preprocess="p1", origin="test-double",
    )
    base.update(kw)
    return c.ModelRef(**base)  # type: ignore[arg-type]


def test_frame_requires_aware_time_and_bgr():
    img = np.zeros((4, 6, 3), np.uint8)
    f = c.Frame(c.FrameId(1), c.CameraId("cam"), 0, datetime.now(UTC), img)
    assert (f.width, f.height) == (6, 4)
    with pytest.raises(ValueError):
        c.Frame(c.FrameId(1), c.CameraId("cam"), 0, datetime.now(), img)
    with pytest.raises(ValueError):
        c.Frame(c.FrameId(1), c.CameraId("cam"), 0, datetime.now(UTC), np.zeros((4, 6), np.uint8))


def test_ids_are_distinct_namespaces():
    assert c.new_profile_id().startswith("prf-")
    assert c.new_track_id().startswith("trk-")
    assert c.new_event_id().startswith("evt-")


def test_bbox_iou():
    a = c.BoundingBox(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(c.BoundingBox(20, 20, 5, 5)) == 0.0
    with pytest.raises(ValueError):
        c.BoundingBox(0, 0, 0, 1)


def test_embedding_bound_to_model_and_incompatible_models_rejected():
    m1, m2 = _model(), _model(version="w2")
    e = c.Embedding(m1, np.ones(4, np.float32))
    assert e.model.key != m2.key
    with pytest.raises(ValueError):
        c.Embedding(m1, np.ones(3, np.float32))
    with pytest.raises(ValueError):
        c.Embedding(m1, np.ones(4, np.float64))
    c.ensure_compatible(m1, _model())
    with pytest.raises(c.IncompatibleEmbeddingError):
        c.ensure_compatible(m1, m2)
    with pytest.raises(ValueError):
        _model(origin="maybe")


def test_raw_score_ordering_by_metric():
    hi, lo = c.RawScore(0.9, c.Metric.COSINE), c.RawScore(0.1, c.Metric.COSINE)
    assert hi.is_better_than(lo)
    assert c.RawScore(0.1, c.Metric.L2).is_better_than(c.RawScore(0.9, c.Metric.L2))
    with pytest.raises(ValueError):
        hi.is_better_than(c.RawScore(0.1, c.Metric.L2))


def test_decision_profile_only_when_confirmed():
    common = dict(track_id=c.TrackId("trk-1"), monotonic_ns=0, model=_model(),
                  policy="t", observations=3, window_ms=500)
    c.Decision(status=c.DecisionStatus.CONFIRMED, profile_id=c.ProfileId("prf-1"), **common)
    c.Decision(status=c.DecisionStatus.UNKNOWN, **common)
    with pytest.raises(ValueError):
        c.Decision(status=c.DecisionStatus.CONFIRMED, **common)
    with pytest.raises(ValueError):
        c.Decision(status=c.DecisionStatus.UNCERTAIN, profile_id=c.ProfileId("p"), **common)


def test_arrival_and_audio_invariants():
    now = datetime.now(UTC)
    c.ArrivalEvent(c.new_event_id(), c.ArrivalKind.UNKNOWN, c.TrackId("t"), 0, now)
    with pytest.raises(ValueError):
        c.ArrivalEvent(c.new_event_id(), c.ArrivalKind.KNOWN, c.TrackId("t"), 0, now)
    c.AudioRequest(c.new_event_id(), c.EventId("e"), 0, "k", text="Bonjour")
    with pytest.raises(ValueError):
        c.AudioRequest(c.new_event_id(), c.EventId("e"), 0, "k")
    with pytest.raises(ValueError):
        c.AudioRequest(c.new_event_id(), c.EventId("e"), 0, "k", text="a", audio_path="b")


def test_schema_applies_on_fresh_sqlite():
    con = sqlite3.connect(":memory:")
    con.executescript(c.schema_sql())
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert {"profiles", "embeddings", "reference_images", "collection_sessions",
            "settings", "arrival_log"} <= tables


def test_interfaces_are_protocols():
    for name in ["FrameSource", "FaceDetector", "FaceTracker", "FaceEmbedder", "Gallery",
                 "Recognizer", "DecisionEngine", "ArrivalTracker", "WelcomeComposer",
                 "AudioPlayer", "EmbeddingStore"]:
        assert getattr(interfaces, name)._is_protocol
