"""Doubles de test d'intégration. Aucune valeur ne représente un résultat réel."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import numpy as np

from maisonaccueil import contracts as c

FAKE_MODEL = c.ModelRef("fake", "0", 4, c.Metric.COSINE, "none", "test-double")


def make_frame(i: int, t_ms: int) -> c.Frame:
    return c.Frame(c.FrameId(i), c.CameraId("fake"), t_ms * 1_000_000, datetime.now(UTC),
                   np.zeros((8, 8, 3), np.uint8))


class FakeSource:
    """Pousse une liste d'images depuis un thread, puis s'arrête."""

    def __init__(self, frames: Sequence[c.Frame]) -> None:
        self._frames = frames
        self._state = c.CaptureState(c.CameraId("fake"), c.CaptureStatus.STOPPED, 0)
        self._thread: threading.Thread | None = None
        self.done = threading.Event()

    @property
    def state(self) -> c.CaptureState:
        return self._state

    def start(self, on_frame: Callable[[c.Frame], None],
              on_state: Callable[[c.CaptureState], None]) -> None:
        def run() -> None:
            self._state = c.CaptureState(c.CameraId("fake"), c.CaptureStatus.RUNNING, 0)
            on_state(self._state)
            for f in self._frames:
                on_frame(f)
            self.done.set()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._thread:
            self._thread.join(timeout_s)
        self._state = c.CaptureState(c.CameraId("fake"), c.CaptureStatus.STOPPED, 0)


class FakeRecognizer:
    """Scénario : chaque image renvoie le profil indiqué pour la piste indiquée."""

    def __init__(self, script: dict[int, list[tuple[str, str, float]]],
                 lost: dict[int, list[str]] | None = None) -> None:
        self._script, self._lost, self._pending_lost = script, lost or {}, []

    def process(self, frame: c.Frame) -> Sequence[c.RecognitionResult]:
        self._pending_lost = [c.TrackId(t) for t in self._lost.get(int(frame.frame_id), [])]
        out = []
        for track, profile, score in self._script.get(int(frame.frame_id), []):
            cand = c.Candidate(c.ProfileId(profile), c.RawScore(score, c.Metric.COSINE), 1)
            out.append(c.RecognitionResult(frame.frame_id, c.TrackId(track), frame.monotonic_ns,
                                           FAKE_MODEL, (cand,)))
        return out

    def lost_tracks(self) -> Sequence[c.TrackId]:
        lost, self._pending_lost = self._pending_lost, []
        return lost


class FakeDecision:
    """Confirme au 2e résultat d'une piste au-dessus de 0.5 (politique de test)."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.lost: list[str] = []

    def observe(self, r: c.RecognitionResult) -> c.Decision | None:
        best = r.best
        if best is None or best.score.value < 0.5:
            return None
        n = self.counts[r.track_id] = self.counts.get(r.track_id, 0) + 1
        if n != 2:
            return None
        return c.Decision(r.track_id, c.DecisionStatus.CONFIRMED, r.monotonic_ns, r.model,
                          "fake", n, 0, profile_id=best.profile_id, best_score=best.score)

    def track_lost(self, track_id: c.TrackId, monotonic_ns: int) -> None:
        self.lost.append(track_id)
        self.counts.pop(track_id, None)


class FakeArrivals:
    def __init__(self) -> None:
        self.present: set[str] = set()
        self.ticks = 0

    def on_decision(self, d: c.Decision) -> c.ArrivalEvent | None:
        if d.profile_id is None or d.profile_id in self.present:
            return None
        self.present.add(d.profile_id)
        return c.ArrivalEvent(c.new_event_id(), c.ArrivalKind.KNOWN, d.track_id, d.monotonic_ns,
                              datetime.now(UTC), profile_id=d.profile_id)

    def tick(self, monotonic_ns: int) -> None:
        self.ticks += 1

    def note_track_activity(self, track_id: c.TrackId, monotonic_ns: int) -> None:
        pass

    def track_lost(self, track_id: c.TrackId, monotonic_ns: int) -> None:
        pass


class FakeComposer:
    def compose(self, e: c.ArrivalEvent) -> c.AudioRequest | None:
        return c.AudioRequest(c.new_event_id(), e.event_id, e.monotonic_ns,
                              f"welcome:{e.profile_id}", text=f"Bonjour {e.profile_id}",
                              profile_id=e.profile_id)


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[c.AudioRequest] = []

    def submit(self, request: c.AudioRequest) -> bool:
        self.played.append(request)
        return True

    def close(self) -> None:
        pass
