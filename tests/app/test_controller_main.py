"""Contrôleur + raccordement à ``pipeline.VisionLoop`` (avec doubles ``Fake*``) + point
d'entrée. Aucun de ces tests n'ouvre de fenêtre."""

from __future__ import annotations

import queue
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

from maisonaccueil.app import main as app_main
from maisonaccueil.app.controller import AppController
from maisonaccueil.app.fakes import FakeFrameSource
from maisonaccueil.app.ipc import UiInbox, VisionPublisher
from maisonaccueil.app.messages import ENGINE_STATUS_UNAVAILABLE, EngineStatus, EnrollmentSample
from maisonaccueil.app.viewmodels import ENGINE_UNAVAILABLE, EnrollmentPhase
from maisonaccueil.app.vision_worker import PreviewTap
from maisonaccueil.app.wiring import demo_services, optional_callable
from maisonaccueil.contracts import (
    ArrivalEvent,
    ArrivalKind,
    AudioRequest,
    CameraId,
    CaptureState,
    CaptureStatus,
    Decision,
    DecisionStatus,
    Frame,
    FrameId,
    ModelRef,
    RecognitionResult,
    TrackId,
    new_event_id,
)
from maisonaccueil.pipeline import VisionLoop, VisionOutput

from .conftest import make_result


def _ctrl() -> tuple[AppController, queue.Queue[object], queue.Queue[object]]:
    events: queue.Queue[object] = queue.Queue()
    preview: queue.Queue[object] = queue.Queue()
    ctrl = AppController(demo_services(), UiInbox(events, preview))
    ctrl.start()
    return ctrl, events, preview


def test_controller_without_engine_never_shows_results(model: ModelRef) -> None:
    ctrl, events, _ = _ctrl()
    events.put(ENGINE_STATUS_UNAVAILABLE)
    events.put(VisionOutput(1, 1, 2.0, results=(make_result(model),)))
    ctrl.poll()
    assert ctrl.status.face_lines == [ENGINE_UNAVAILABLE]
    assert ctrl.status.engine_text.startswith(ENGINE_UNAVAILABLE)


def test_controller_routes_enrollment_samples() -> None:
    ctrl, events, _ = _ctrl()
    from maisonaccueil.app.viewmodels import ProfileForm

    row = ctrl.profiles.create(ProfileForm("Personne Test", consent_given=True))
    events.put(CaptureState(CameraId("fake"), CaptureStatus.RUNNING, 1))
    ctrl.poll()
    assert ctrl.enrollment.camera_available and not ctrl.enrollment.engine_available
    ctrl.enrollment.select_profile(row.profile_id)
    ctrl.enrollment.start()
    events.put(EnrollmentSample(1, b"png-synthetique"))
    ctrl.poll()
    assert ctrl.enrollment.collected == 1
    assert ctrl.enrollment.phase == EnrollmentPhase.CAPTURING
    events.put(CaptureState(CameraId("fake"), CaptureStatus.FAILED, 2, detail="débranchée"))
    ctrl.poll()
    assert not ctrl.enrollment.camera_available
    assert "échec — débranchée" in ctrl.status.camera_text


# -- raccordement à VisionLoop avec des doubles (aucune vraie reconnaissance) -----------


class FakeRecognizer:
    def __init__(self, model: ModelRef) -> None:
        self.model = model

    def process(self, frame: Frame) -> Sequence[RecognitionResult]:
        return [make_result(self.model, "prf-x", 0.5)]

    def lost_tracks(self) -> Sequence[TrackId]:
        return []


class FakeDecision:
    def observe(self, result: RecognitionResult) -> Decision | None:
        return Decision(result.track_id, DecisionStatus.UNKNOWN, result.monotonic_ns,
                        result.model, "fake-policy", 1, 0)

    def track_lost(self, track_id: TrackId, monotonic_ns: int) -> None:
        pass


class FakeArrivals:
    def on_decision(self, decision: Decision) -> ArrivalEvent | None:
        return ArrivalEvent(new_event_id(), ArrivalKind.UNKNOWN, decision.track_id,
                            decision.monotonic_ns, datetime.now(UTC))

    def tick(self, monotonic_ns: int) -> None:
        pass

    def note_track_activity(self, track_id: TrackId, monotonic_ns: int) -> None:
        pass

    def track_lost(self, track_id: TrackId, monotonic_ns: int) -> None:
        pass


class FakeComposer:
    def compose(self, event: ArrivalEvent) -> AudioRequest | None:
        return None


class FakePlayer:
    def submit(self, request: AudioRequest) -> bool:
        return True

    def close(self) -> None:
        pass


def test_vision_loop_outputs_reach_view_models(model: ModelRef) -> None:
    ctrl, events, preview = _ctrl()
    publisher = VisionPublisher(events, preview)
    loop = VisionLoop(FakeFrameSource(), FakeRecognizer(model), FakeDecision(),  # type: ignore[arg-type]
                      FakeArrivals(), FakeComposer(), FakePlayer(),
                      on_output=publisher.on_output, on_state=publisher.on_state)
    frame = Frame(FrameId(1), CameraId("c"), 5, datetime.now(UTC),
                  np.zeros((240, 320, 3), dtype=np.uint8))
    PreviewTap(publisher, max_fps=1000)(frame)
    publisher.on_output(loop.process_frame(frame))
    ctrl.poll()
    # Moteur non déclaré : résultats ignorés, mais l'arrivée « inconnu » est journalisée.
    assert ctrl.status.face_lines == [ENGINE_UNAVAILABLE]
    assert [r.label for r in ctrl.arrivals.rows] == ["Visage inconnu"]
    assert ctrl.preview.image is not None
    # Moteur déclaré (double de test) : résultats affichés en score technique.
    events.put(EngineStatus(True, model, detail="double de test"))
    publisher.on_output(loop.process_frame(frame))
    ctrl.poll()
    lines = ctrl.status.face_lines
    assert any("score technique (non calibré) : 0.500" in line for line in lines)
    assert any("inconnu" in line for line in lines)
    assert all("%" not in line for line in lines)
    assert len(ctrl.arrivals.rows) == 2


# -- point d'entrée ---------------------------------------------------------------------


def test_main_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert app_main.main(["--version"]) == 0
    assert "MaisonAccueil" in capsys.readouterr().out


def test_main_without_qt(monkeypatch: pytest.MonkeyPatch,
                         capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(app_main, "qt_available", lambda: (False, "libEGL.so.1 absent"))
    assert app_main.main(["--check"]) == 2
    out = capsys.readouterr().out
    assert "NON disponible (libEGL.so.1 absent)" in out and "moteur non disponible" in out
    assert app_main.main(["--demo"]) == 2
    assert "Qt Widgets ne se charge pas" in capsys.readouterr().err


def test_qt_available_reports_real_state() -> None:
    ok, reason = app_main.qt_available()
    assert ok or reason  # pas d'exception, raison fournie si indisponible


def test_optional_callable() -> None:
    assert optional_callable("maisonaccueil.module_absent:f") is None
    assert optional_callable("maisonaccueil.app.fakes:absent") is None
    assert optional_callable("maisonaccueil.app.fakes:fake_source_factory") is not None
