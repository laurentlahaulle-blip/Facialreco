"""Fixtures du lot B. Données SYNTHÉTIQUES uniquement (aucun visage réel)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from maisonaccueil.contracts import (
    Candidate,
    FrameId,
    Metric,
    ModelRef,
    ProfileId,
    RawScore,
    RecognitionResult,
    TrackId,
)


class FakeClock:
    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self.now = start_ns

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += int(ms * 1_000_000)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fixed_utc() -> datetime:
    return datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def model() -> ModelRef:
    return ModelRef("fake-engine", "t1", 4, Metric.COSINE, "none", "test-double")


@pytest.fixture
def other_model() -> ModelRef:
    return ModelRef("fake-engine", "t2", 4, Metric.COSINE, "none", "test-double")


def make_result(
    model: ModelRef,
    profile_id: str | None = "prf-a",
    value: float = 0.612,
    track: str = "trk-1",
) -> RecognitionResult:
    cands: tuple[Candidate, ...] = ()
    if profile_id is not None:
        cands = (Candidate(ProfileId(profile_id), RawScore(value, model.metric), 3),)
    return RecognitionResult(FrameId(1), TrackId(track), 10, model, cands)
