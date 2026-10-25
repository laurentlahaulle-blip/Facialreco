"""Doubles de test du lot G. AUCUNE valeur produite ici ne mesure quoi que ce soit de
biométrique : scénarios scriptés pour vérifier les calculs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from maisonaccueil import contracts as c
from maisonaccueil.evaluation import (
    AnnotationSet,
    DataKind,
    Expected,
    Passage,
    RunHeader,
    RunLogWriter,
    SessionRecord,
)

FAKE_MODEL = c.ModelRef("fake-eval", "0", 4, c.Metric.COSINE, "none", "test-double")
FAKE_MODEL_B = c.ModelRef("fake-eval-b", "0", 4, c.Metric.COSINE, "none", "test-double")
FPS = 10.0  # 1 image = 100 ms


def frame(i: int, *, fps: float = FPS) -> c.Frame:
    """Image vide n° i (FrameId = i, temps = (i-1)/fps)."""
    t_ns = round((i - 1) * 1e9 / fps)
    return c.Frame(c.FrameId(i), c.CameraId("fake"), t_ns, datetime(2026, 1, 1, tzinfo=UTC),
                   np.zeros((2, 2, 3), np.uint8))


class FakeRecognizer:
    """``script[frame_id] = [(track_id, profile_id, score), ...]`` ;
    ``lost[frame_id] = [track_id, ...]``."""

    def __init__(
        self,
        script: dict[int, list[tuple[str, str, float]]],
        lost: dict[int, list[str]] | None = None,
        model: c.ModelRef = FAKE_MODEL,
    ) -> None:
        self.script, self.lost, self.model = script, lost or {}, model
        self._pending: list[c.TrackId] = []

    def process(self, f: c.Frame) -> Sequence[c.RecognitionResult]:
        self._pending = [c.TrackId(t) for t in self.lost.get(int(f.frame_id), [])]
        out = []
        for track, profile, score in self.script.get(int(f.frame_id), []):
            cand = c.Candidate(c.ProfileId(profile), c.RawScore(score, c.Metric.COSINE), 3)
            out.append(c.RecognitionResult(f.frame_id, c.TrackId(track), f.monotonic_ns,
                                           self.model, (cand,)))
        return out

    def lost_tracks(self) -> Sequence[c.TrackId]:
        lost, self._pending = self._pending, []
        return lost


class FakeDecisionEngine:
    """Politique de test : CONFIRMED au ``n``-ième résultat ≥ ``threshold`` d'une piste ;
    UNKNOWN au ``n``-ième résultat < ``threshold``. Une décision par piste."""

    def __init__(self, threshold: float = 0.5, n: int = 2) -> None:
        self.threshold, self.n = threshold, n
        self.hi: dict[str, int] = defaultdict(int)
        self.lo: dict[str, int] = defaultdict(int)
        self.done: set[str] = set()

    def observe(self, r: c.RecognitionResult) -> c.Decision | None:
        best = r.best
        if best is None or r.track_id in self.done:
            return None
        if best.score.value >= self.threshold:
            self.hi[r.track_id] += 1
            if self.hi[r.track_id] >= self.n:
                self.done.add(r.track_id)
                return c.Decision(r.track_id, c.DecisionStatus.CONFIRMED, r.monotonic_ns,
                                  r.model, "fake-v1", self.hi[r.track_id], 0,
                                  profile_id=best.profile_id, best_score=best.score)
        else:
            self.lo[r.track_id] += 1
            if self.lo[r.track_id] >= self.n:
                self.done.add(r.track_id)
                return c.Decision(r.track_id, c.DecisionStatus.UNKNOWN, r.monotonic_ns,
                                  r.model, "fake-v1", self.lo[r.track_id], 0)
        return None

    def track_lost(self, track_id: c.TrackId, monotonic_ns: int) -> None:
        self.hi.pop(track_id, None)
        self.lo.pop(track_id, None)
        self.done.discard(track_id)


class FakeArrivalTracker:
    """Une arrivée par décision CONFIRMED/UNKNOWN, sauf profil déjà présent (contrats 1.2 :
    ``note_track_activity`` et ``track_lost`` présents)."""

    def __init__(self, suppress_present: bool = True) -> None:
        self.present: set[str] = set()
        self.suppress = suppress_present
        self.activity: list[str] = []
        self.lost: list[str] = []

    def on_decision(self, d: c.Decision) -> c.ArrivalEvent | None:
        if d.status == c.DecisionStatus.UNCERTAIN:
            return None
        if d.status == c.DecisionStatus.UNKNOWN:
            return c.ArrivalEvent(c.new_event_id(), c.ArrivalKind.UNKNOWN, d.track_id,
                                  d.monotonic_ns, datetime(2026, 1, 1, tzinfo=UTC))
        assert d.profile_id is not None
        if self.suppress and d.profile_id in self.present:
            return None
        self.present.add(d.profile_id)
        return c.ArrivalEvent(c.new_event_id(), c.ArrivalKind.KNOWN, d.track_id, d.monotonic_ns,
                              datetime(2026, 1, 1, tzinfo=UTC), profile_id=d.profile_id)

    def tick(self, monotonic_ns: int) -> None:
        return None

    def note_track_activity(self, track_id: c.TrackId, monotonic_ns: int) -> None:
        self.activity.append(track_id)

    def track_lost(self, track_id: c.TrackId, monotonic_ns: int) -> None:
        self.lost.append(track_id)


class FakeWelcomeComposer:
    def compose(self, e: c.ArrivalEvent) -> c.AudioRequest | None:
        if e.kind != c.ArrivalKind.KNOWN:
            return None
        return c.AudioRequest(c.new_event_id(), e.event_id, e.monotonic_ns,
                              f"welcome:{e.profile_id}", text="(texte de test)",
                              profile_id=e.profile_id)


def header(session: str, config: str = "cfg-A", *, messages_from: str = "welcome",
           model: c.ModelRef = FAKE_MODEL) -> RunHeader:
    return RunHeader(c.SessionId(session), config, model, DataKind.SYNTHETIC,
                     source="fake", messages_from=messages_from)


@dataclass
class LogBuilder:
    """Construit un journal à la main (temps en ms de séance)."""

    session: str
    config: str = "cfg-A"
    messages_from: str = "welcome"
    model: c.ModelRef = FAKE_MODEL
    writer: RunLogWriter = field(init=False)

    def __post_init__(self) -> None:
        self.writer = RunLogWriter(header(self.session, self.config,
                                          messages_from=self.messages_from, model=self.model))

    def result(self, t_ms: float, track: str, profile: str, score: float,
               frame_id: int = 1) -> LogBuilder:
        cand = c.Candidate(c.ProfileId(profile), c.RawScore(score, self.model.metric), 1)
        self.writer.result(c.RecognitionResult(c.FrameId(frame_id), c.TrackId(track),
                                               int(t_ms * 1e6), self.model, (cand,)))
        return self

    def arrival(self, t_ms: float, profile: str | None, track: str = "trk-x",
                event_id: str | None = None) -> str:
        kind = c.ArrivalKind.KNOWN if profile else c.ArrivalKind.UNKNOWN
        eid = c.EventId(event_id or c.new_event_id())
        self.writer.arrival(c.ArrivalEvent(eid, kind, c.TrackId(track), int(t_ms * 1e6),
                                           datetime(2026, 1, 1, tzinfo=UTC),
                                           profile_id=c.ProfileId(profile) if profile else None))
        return eid

    def welcome(self, t_ms: float, arrival_id: str, profile: str | None) -> LogBuilder:
        self.writer.welcome(c.AudioRequest(c.new_event_id(), c.EventId(arrival_id),
                                           int(t_ms * 1e6), f"w:{profile}", text="x",
                                           profile_id=c.ProfileId(profile) if profile else None))
        return self

    def greet(self, t_ms: float, profile: str | None, track: str = "trk-x") -> LogBuilder:
        """Arrivée + accueil au même instant."""
        eid = self.arrival(t_ms, profile, track)
        if profile is not None:
            self.welcome(t_ms, eid, profile)
        return self

    def build(self) -> object:
        return self.writer.to_runlog()


def known(pid: str, session: str, start: float, end: float, profile: str,
          **kw: object) -> Passage:
    return Passage(pid, c.SessionId(session), start, end, Expected.KNOWN,
                   c.ProfileId(profile), **kw)  # type: ignore[arg-type]


def unreg(pid: str, session: str, start: float, end: float, subject: str = "") -> Passage:
    return Passage(pid, c.SessionId(session), start, end, Expected.UNREGISTERED, None, subject)


def sess(sid: str, split: str, **kw: object) -> SessionRecord:
    return SessionRecord(c.SessionId(sid), split, **kw)  # type: ignore[arg-type]


def annset(sessions: Sequence[SessionRecord], passages: Sequence[Passage],
           kind: DataKind = DataKind.SYNTHETIC) -> AnnotationSet:
    return AnnotationSet(kind, {s.session_id: s for s in sessions}, tuple(passages))
