"""Scénarios de bout en bout du lot F, horloge injectée, données synthétiques.

Chaîne : RecognitionResult → TemporalDecisionEngine → PresenceArrivalTracker →
TemplateWelcomeComposer → QueuedAudioPlayer(FakeAudioBackend). Aucun son réel.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import pytest

from maisonaccueil.contracts import ArrivalKind, RecognitionResult, TrackId
from maisonaccueil.decision import (
    ArrivalPolicy,
    DecisionPolicy,
    PresenceArrivalTracker,
    TemporalDecisionEngine,
)
from maisonaccueil.welcome import (
    FakeAudioBackend,
    ProfileGreeting,
    QueuedAudioPlayer,
    TemplateWelcomeComposer,
    UnknownPolicy,
    WelcomeConfig,
    WinsoundBackend,
)

from ._helpers import MS, A, B, FakeClock, result

PROFILES = {A: ProfileGreeting("Alpha"), B: ProfileGreeting("Bravo")}


class Chain:
    def __init__(self, unknown_policy=UnknownPolicy.SILENT) -> None:
        self.clock = FakeClock()
        self.engine = TemporalDecisionEngine(DecisionPolicy())
        self.tracker = PresenceArrivalTracker(
            ArrivalPolicy(absence_timeout_ms=30_000, track_stale_ms=5_000,
                          unknown_cooldown_ms=5_000),
            wall_clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )
        self.composer = TemplateWelcomeComposer(
            PROFILES.get, WelcomeConfig(unknown_policy=unknown_policy)
        )
        self.backend = FakeAudioBackend(clock=self.clock)
        self.player = QueuedAudioPlayer(self.backend, clock=self.clock)
        self.events = []

    def push(self, r: RecognitionResult) -> None:
        self.clock.now = r.monotonic_ns
        self.tracker.note_track_activity(r.track_id, r.monotonic_ns)
        decision = self.engine.observe(r)
        if decision is not None:
            event = self.tracker.on_decision(decision)
            if event is not None:
                self.events.append(event)
                request = self.composer.compose(event)
                if request is not None:
                    self.player.submit(request)
                    # Lecture simulée instantanée : on attend le thread du lecteur pour
                    # que l'horloge factice n'avance pas pendant la mise en file.
                    assert self.player.wait_idle(2)
        self.tracker.tick(r.monotonic_ns)

    def lose(self, track: str, t_ms: int) -> None:
        self.engine.track_lost(TrackId(track), t_ms * MS)
        self.tracker.track_lost(TrackId(track), t_ms * MS)

    def idle(self, t_ms: int) -> None:
        self.clock.set_ms(t_ms)
        self.tracker.tick(t_ms * MS)

    def spoken(self) -> list[str]:
        assert self.player.wait_idle(2)
        return [r.request.text or r.request.audio_path or "" for r in self.backend.played]

    def close(self) -> None:
        self.player.close()


@pytest.fixture
def chain():
    c = Chain()
    yield c
    c.close()


def walk(chain, track, start_ms, end_ms, *cands, step=100):
    for t in range(start_ms, end_ms, step):
        chain.push(result(track, t, *cands))


def test_known_person_greeted_once_despite_track_loss_and_new_track(chain):
    walk(chain, "t1", 0, 2_000, (A, 0.8))
    chain.lose("t1", 2_000)  # occultation
    walk(chain, "t2", 4_000, 8_000, (A, 0.8))  # nouveau TrackId, même personne
    chain.lose("t2", 8_000)
    walk(chain, "t3", 20_000, 21_000, (A, 0.8))  # revient avant le délai d'absence
    assert [e.kind for e in chain.events] == [ArrivalKind.KNOWN]
    assert chain.spoken() == ["Bonjour Alpha !"]


def test_repeated_visits_after_absence_are_greeted_again(chain):
    walk(chain, "t1", 0, 1_000, (A, 0.8))
    chain.lose("t1", 1_000)
    chain.idle(40_000)  # absence > 30 s
    walk(chain, "t2", 40_000, 41_000, (A, 0.8))
    assert len(chain.events) == 2
    assert chain.spoken() == ["Bonjour Alpha !", "Bonjour Alpha !"]


def test_simultaneous_arrivals_are_chained(chain):
    for t in range(0, 1_000, 100):
        chain.push(result("t1", t, (A, 0.8), (B, 0.3)))
        chain.push(result("t2", t + 10, (B, 0.8), (A, 0.3)))
    assert {e.profile_id for e in chain.events} == {A, B}
    assert chain.spoken() == ["Bonjour Alpha !", "Bonjour Bravo !"]


def test_unknown_is_silent_by_default_and_reported_once(chain):
    walk(chain, "u1", 0, 3_000, (A, 0.2), (B, 0.1))
    assert [e.kind for e in chain.events] == [ArrivalKind.UNKNOWN]
    assert chain.events[0].profile_id is None
    assert chain.spoken() == []


def test_unknown_with_text_policy_says_no_name():
    c = Chain(UnknownPolicy.TEXT)
    try:
        walk(c, "u1", 0, 3_000, (A, 0.2))
        assert c.spoken() == ["Bonjour, bienvenue."]
    finally:
        c.close()


def test_oscillation_between_two_identities_greets_nobody(chain):
    for i in range(40):
        chain.push(result("t1", i * 100, (A, 0.8) if i % 2 else (B, 0.8)))
    assert chain.events == []
    assert chain.spoken() == []


def test_blurry_frames_delay_but_do_not_prevent_greeting(chain):
    from maisonaccueil.contracts import QualityFlag

    for t in range(0, 1_000, 100):
        chain.push(result("t1", t, (B, 0.9), flags=frozenset({QualityFlag.BLURRY})))
    assert chain.events == []
    walk(chain, "t1", 1_000, 1_500, (A, 0.8))
    assert [e.profile_id for e in chain.events] == [A]


@pytest.mark.skipif(sys.platform == "win32", reason="vérifie l'absence de winsound hors Windows")
def test_winsound_backend_refused_outside_windows():
    with pytest.raises(RuntimeError):
        WinsoundBackend()
