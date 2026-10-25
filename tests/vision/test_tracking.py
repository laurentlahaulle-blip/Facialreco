import pytest

from maisonaccueil.contracts import BoundingBox, Detection, FrameId
from maisonaccueil.contracts.interfaces import FaceTracker
from maisonaccueil.vision import IouTracker
from tests.vision.helpers import make_frame


def det(i: int, x: float, y: float, w: float = 40, h: float = 40) -> Detection:
    return Detection(FrameId(i), BoundingBox(x, y, w, h), 0.9, "test")


def test_protocol_and_validation():
    assert isinstance(IouTracker(), FaceTracker)
    with pytest.raises(ValueError):
        IouTracker(iou_threshold=0)


def test_moving_box_keeps_track_id_and_age_increases():
    tr = IouTracker()
    ids, ages = set(), []
    for i in range(10):
        out = tr.update(make_frame(i), [det(i, 10 + 3 * i, 20)])
        assert len(out) == 1
        ids.add(out[0].track_id)
        ages.append(out[0].track_age_frames)
    assert len(ids) == 1
    assert ages == list(range(10))
    assert tr.lost_tracks() == []


def test_two_faces_get_distinct_stable_ids():
    tr = IouTracker()
    first = tr.update(make_frame(0), [det(0, 0, 0), det(0, 200, 0)])
    a, b = first[0].track_id, first[1].track_id
    assert a != b
    # ordre des détections inversé : l'association suit la géométrie, pas l'ordre
    second = tr.update(make_frame(1), [det(1, 202, 1), det(1, 2, 1)])
    assert [t.track_id for t in second] == [b, a]


def test_short_loss_keeps_id_long_loss_reports_once():
    tr = IouTracker(max_missed_frames=3, max_missed_ms=10_000)
    tid = tr.update(make_frame(0), [det(0, 50, 50)])[0].track_id
    for i in range(1, 4):  # 3 images sans détection : toléré
        assert tr.update(make_frame(i), []) == []
    assert tr.lost_tracks() == []
    back = tr.update(make_frame(4), [det(4, 52, 50)])
    assert back[0].track_id == tid
    for i in range(5, 10):
        tr.update(make_frame(i), [])
    assert tr.lost_tracks() == [tid]
    assert tr.lost_tracks() == []  # renvoyé une seule fois
    # réapparition après perte : nouveau passage
    again = tr.update(make_frame(10), [det(10, 52, 50)])
    assert again[0].track_id != tid


def test_time_based_loss():
    tr = IouTracker(max_missed_frames=100, max_missed_ms=500)
    tid = tr.update(make_frame(0, t_ms=0), [det(0, 50, 50)])[0].track_id
    tr.update(make_frame(1, t_ms=400), [])
    assert tr.lost_tracks() == []
    tr.update(make_frame(2, t_ms=600), [])
    assert tr.lost_tracks() == [tid]


def test_jump_creates_new_track_and_old_one_expires():
    tr = IouTracker(max_missed_frames=1)
    a = tr.update(make_frame(0), [det(0, 0, 0)])[0].track_id
    b = tr.update(make_frame(1), [det(1, 250, 150)])[0].track_id
    assert a != b
    tr.update(make_frame(2), [det(2, 250, 150)])
    assert tr.lost_tracks() == [a]


def test_reset_ends_all_tracks():
    tr = IouTracker()
    out = tr.update(make_frame(0), [det(0, 0, 0), det(0, 100, 100)])
    tr.reset()
    assert set(tr.lost_tracks()) == {o.track_id for o in out}
    assert tr.active_track_ids == ()


def test_custom_id_factory():
    n = iter(range(100))
    tr = IouTracker(new_track_id=lambda: f"trk-test-{next(n)}")
    out = tr.update(make_frame(0), [det(0, 0, 0)])
    assert out[0].track_id == "trk-test-0"
