"""Fabrique de RecognitionResult SYNTHÉTIQUES (aucune donnée biométrique)."""

from __future__ import annotations

from maisonaccueil.contracts import (
    Candidate,
    FrameId,
    Metric,
    ModelRef,
    ProfileId,
    QualityFlag,
    RawScore,
    RecognitionResult,
    TrackId,
)

MS = 1_000_000
MODEL = ModelRef("fake-engine", "w0", 4, Metric.COSINE, "none", "test-double")
MODEL_L2 = ModelRef("fake-engine", "w0", 4, Metric.L2, "none", "test-double")
A = ProfileId("prf-a")
B = ProfileId("prf-b")
C = ProfileId("prf-c")


class FakeClock:
    """Horloge monotone manuelle (ns)."""

    def __init__(self, start_ns: int = 0) -> None:
        self.now = start_ns

    def __call__(self) -> int:
        return self.now

    def set_ms(self, ms: int) -> None:
        self.now = ms * MS


def result(
    track: str,
    t_ms: int,
    *cands: tuple[ProfileId, float],
    flags: frozenset[QualityFlag] = frozenset(),
    model: ModelRef = MODEL,
    frame: int = 0,
) -> RecognitionResult:
    return RecognitionResult(
        frame_id=FrameId(frame),
        track_id=TrackId(track),
        monotonic_ns=t_ms * MS,
        model=model,
        candidates=tuple(
            Candidate(pid, RawScore(score, model.metric), 3) for pid, score in cands
        ),
        quality_flags=flags,
    )
