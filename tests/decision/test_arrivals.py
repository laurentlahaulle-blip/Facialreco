from datetime import UTC, datetime

import pytest

from maisonaccueil.contracts import (
    ArrivalKind,
    Decision,
    DecisionStatus,
    ProfileId,
    TrackId,
)
from maisonaccueil.contracts.interfaces import ArrivalTracker
from maisonaccueil.decision import ArrivalPolicy, PresenceArrivalTracker

from ._helpers import MODEL, MS, A, B

WALL = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def dec(track: str, t_ms: int, status=DecisionStatus.CONFIRMED, pid: ProfileId | None = A):
    return Decision(
        track_id=TrackId(track), status=status, monotonic_ns=t_ms * MS, model=MODEL,
        policy="temporal-v1", observations=3, window_ms=200,
        profile_id=pid if status == DecisionStatus.CONFIRMED else None,
    )


def unknown(track: str, t_ms: int):
    return dec(track, t_ms, DecisionStatus.UNKNOWN, None)


def make(**kw):
    return PresenceArrivalTracker(ArrivalPolicy(**kw), wall_clock=lambda: WALL)


def test_implements_protocol():
    assert isinstance(make(), ArrivalTracker)


def test_one_arrival_per_presence():
    tr = make()
    ev = tr.on_decision(dec("t1", 0))
    assert ev is not None and ev.kind == ArrivalKind.KNOWN and ev.profile_id == A
    assert ev.wall_time == WALL and ev.track_id == TrackId("t1") and ev.monotonic_ns == 0
    assert tr.on_decision(dec("t1", 100)) is None
    assert tr.present_profiles == {A}


def test_track_id_change_of_same_profile_keeps_presence():
    tr = make(absence_timeout_ms=30_000, track_stale_ms=5_000)
    assert tr.on_decision(dec("t1", 0)) is not None
    tr.track_lost(TrackId("t1"), 1_000 * MS)
    tr.tick(3_000 * MS)
    assert tr.on_decision(dec("t2", 3_000)) is None  # nouveau passage, même personne
    assert set(tr.presence(A).tracks) == {TrackId("t2")}


def test_presence_survives_temporary_loss_then_expires():
    tr = make(absence_timeout_ms=10_000, track_stale_ms=2_000)
    tr.on_decision(dec("t1", 0))
    tr.track_lost(TrackId("t1"), 500 * MS)
    tr.tick(9_000 * MS)
    assert tr.present_profiles == {A}
    tr.tick(10_500 * MS)
    assert tr.present_profiles == frozenset()
    ev = tr.on_decision(dec("t9", 11_000))
    assert ev is not None and ev.kind == ArrivalKind.KNOWN  # nouvelle présence


def test_active_track_keeps_presence_alive():
    tr = make(absence_timeout_ms=1_000, track_stale_ms=2_000)
    tr.on_decision(dec("t1", 0))
    for t in range(0, 60_000, 500):  # la personne reste 60 s devant la caméra
        tr.note_track_activity(TrackId("t1"), t * MS)
        tr.tick(t * MS)
    assert tr.present_profiles == {A}


def test_stale_track_without_lost_notification_eventually_expires():
    tr = make(absence_timeout_ms=1_000, track_stale_ms=2_000)
    tr.on_decision(dec("t1", 0))
    tr.tick(1_500 * MS)
    assert tr.present_profiles == {A}  # passage encore considéré actif
    tr.tick(2_000 * MS)  # passage périmé, absence déjà > 1 s
    assert tr.present_profiles == frozenset()


def test_unknown_at_most_once_per_track_and_never_named():
    tr = make(unknown_cooldown_ms=0)
    ev = tr.on_decision(unknown("u1", 0))
    assert ev is not None and ev.kind == ArrivalKind.UNKNOWN and ev.profile_id is None
    assert tr.on_decision(unknown("u1", 100)) is None
    assert tr.on_decision(unknown("u2", 200)) is not None


def test_unknown_cooldown_absorbs_track_fragmentation():
    tr = make(unknown_cooldown_ms=5_000)
    assert tr.on_decision(unknown("u1", 0)) is not None
    assert tr.on_decision(unknown("u2", 1_000)) is None
    assert tr.on_decision(unknown("u3", 6_000)) is not None


def test_unknown_on_known_track_is_suppressed():
    tr = make(unknown_cooldown_ms=0)
    tr.on_decision(dec("t1", 0))
    assert tr.on_decision(unknown("t1", 100)) is None


def test_two_people_simultaneous_arrivals():
    tr = make()
    e1 = tr.on_decision(dec("t1", 0, pid=A))
    e2 = tr.on_decision(dec("t2", 10, pid=B))
    assert e1 and e2 and e1.event_id != e2.event_id
    assert tr.present_profiles == {A, B}


def test_track_swap_moves_track_between_presences():
    tr = make()
    tr.on_decision(dec("t1", 0, pid=A))
    ev = tr.on_decision(dec("t1", 500, pid=B))
    assert ev is not None and ev.profile_id == B
    assert TrackId("t1") not in tr.presence(A).tracks


def test_uncertain_is_only_a_sign_of_life():
    tr = make(absence_timeout_ms=1_000, track_stale_ms=1_000)
    tr.on_decision(dec("t1", 0))
    assert tr.on_decision(dec("t1", 900, DecisionStatus.UNCERTAIN, None)) is None
    tr.tick(1_500 * MS)
    assert tr.present_profiles == {A}


def test_policy_validation():
    with pytest.raises(ValueError):
        ArrivalPolicy(track_stale_ms=0)
