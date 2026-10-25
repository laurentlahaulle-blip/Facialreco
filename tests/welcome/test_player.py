"""Tests du lecteur avec FakeAudioBackend (aucun son réel n'est émis ni entendu)."""

import threading

import pytest

from maisonaccueil.contracts import AudioPriority, AudioRequest, EventId
from maisonaccueil.contracts.interfaces import AudioPlayer
from maisonaccueil.welcome import (
    FakeAudioBackend,
    LoggingAudioBackend,
    QueuedAudioPlayer,
    RoutingAudioBackend,
    UnsupportedAudioRequest,
    create_default_backend,
)

MS = 1_000_000


class FakeClock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def req(key: str, t_ns: int = 0, prio=AudioPriority.NORMAL, expires_ms=10_000, text=None):
    return AudioRequest(
        request_id=EventId(f"req-{key}-{t_ns}"), arrival_event_id=EventId(f"evt-{key}"),
        monotonic_ns=t_ns, dedupe_key=key, priority=prio,
        text=text or f"Bonjour {key}", expires_after_ms=expires_ms,
    )


@pytest.fixture
def gated():
    clock = FakeClock()
    gate = threading.Event()
    backend = FakeAudioBackend(clock=clock, gate=gate)
    player = QueuedAudioPlayer(backend, clock=clock, max_queue=3)
    yield clock, gate, backend, player
    gate.set()
    player.close()


def test_implements_protocol():
    p = QueuedAudioPlayer(FakeAudioBackend())
    try:
        assert isinstance(p, AudioPlayer)
    finally:
        p.close()


def test_plays_in_background_and_submit_does_not_block(gated):
    clock, gate, backend, player = gated
    assert player.submit(req("a"))
    assert backend.started.wait(2)  # le thread joue, submit est déjà revenu
    assert backend.played == []
    gate.set()
    assert player.wait_idle(2)
    assert backend.played_keys == ["a"]
    assert player.stats.played == 1


def test_deduplication_pending_and_playing(gated):
    clock, gate, backend, player = gated
    assert player.submit(req("a"))
    assert backend.started.wait(2)
    assert not player.submit(req("a", 1))  # identique à la lecture en cours
    assert player.submit(req("b"))
    assert not player.submit(req("b", 2))  # identique à une demande en file
    gate.set()
    assert player.wait_idle(2)
    assert backend.played_keys == ["a", "b"]
    assert player.stats.deduplicated == 2


def test_priority_order_then_fifo(gated):
    clock, gate, backend, player = gated
    player.submit(req("first"))
    assert backend.started.wait(2)
    player.submit(req("low", prio=AudioPriority.LOW))
    player.submit(req("n1"))
    player.submit(req("high", prio=AudioPriority.HIGH))
    assert [r.dedupe_key for r in player.pending] == ["high", "n1", "low"]
    gate.set()
    assert player.wait_idle(2)
    assert backend.played_keys == ["first", "high", "n1", "low"]


def test_expired_on_submit_and_while_waiting(gated):
    clock, gate, backend, player = gated
    clock.now = 20_000 * MS
    assert not player.submit(req("old", t_ns=0, expires_ms=10_000))
    player.submit(req("a", t_ns=clock.now))
    assert backend.started.wait(2)
    player.submit(req("b", t_ns=clock.now, expires_ms=1_000))
    clock.now += 5_000 * MS  # "a" dure longtemps : "b" devient périmé dans la file
    gate.set()
    assert player.wait_idle(2)
    assert backend.played_keys == ["a"]
    assert player.stats.expired == 2


def test_full_queue_evicts_lower_priority_or_rejects(gated):
    clock, gate, backend, player = gated
    player.submit(req("playing"))
    assert backend.started.wait(2)
    for k in ("l1", "l2", "l3"):
        assert player.submit(req(k, prio=AudioPriority.LOW))
    assert not player.submit(req("l4", prio=AudioPriority.LOW))
    assert player.submit(req("n", prio=AudioPriority.NORMAL))  # évince l3
    assert [r.dedupe_key for r in player.pending] == ["n", "l1", "l2"]
    s = player.stats
    assert (s.rejected, s.evicted) == (1, 1)


def test_backend_error_does_not_kill_thread():
    backend = FakeAudioBackend(fail_on=lambda r: r.dedupe_key == "bad")
    player = QueuedAudioPlayer(backend)
    try:
        player.submit(req("bad", t_ns=_now()))
        player.submit(req("good", t_ns=_now()))
        assert player.wait_idle(2)
        assert backend.played_keys == ["good"]
        assert player.stats.errors == 1 and player.is_alive
    finally:
        player.close()


def test_close_is_idempotent_and_rejects_afterwards(gated):
    clock, gate, backend, player = gated
    player.submit(req("a"))
    assert backend.started.wait(2)
    player.submit(req("b"))
    player.close()  # interrompt la lecture (FakeAudioBackend.stop lève la barrière)
    player.close()
    assert not player.is_alive
    assert backend.stopped == 1
    assert player.stats.dropped_on_close == 1
    assert not player.submit(req("c"))


def test_simultaneous_arrivals_are_chained_not_overlapping():
    backend = FakeAudioBackend(duration_ms=20)  # durée simulée, aucun son
    player = QueuedAudioPlayer(backend)
    try:
        t = _now()
        assert player.submit(req("arrival:A", t_ns=t))
        assert player.submit(req("arrival:B", t_ns=t))
        assert player.wait_idle(2)
        first, second = backend.played
        assert [first.request.dedupe_key, second.request.dedupe_key] == [
            "arrival:A", "arrival:B"]
        assert second.started_ns >= first.finished_ns  # jamais en parallèle
    finally:
        player.close()


def test_on_played_callback():
    seen = []
    player = QueuedAudioPlayer(FakeAudioBackend(), on_played=seen.append)
    try:
        player.submit(req("a", t_ns=_now()))
        assert player.wait_idle(2)
        assert [r.dedupe_key for r in seen] == ["a"]
    finally:
        player.close()


def test_routing_backend():
    files, texts = FakeAudioBackend(), FakeAudioBackend()
    r = RoutingAudioBackend(files, texts)
    r.play(req("t"))
    wav = AudioRequest(EventId("r"), EventId("e"), 0, "w", audio_path="x.wav")
    r.play(wav)
    assert texts.played_keys == ["t"] and files.played_keys == ["w"]
    with pytest.raises(UnsupportedAudioRequest):
        RoutingAudioBackend(files, None).play(req("t2"))


def test_default_backend_on_non_windows_is_silent(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    assert isinstance(create_default_backend(), LoggingAudioBackend)


def _now() -> int:
    import time

    return time.monotonic_ns()
