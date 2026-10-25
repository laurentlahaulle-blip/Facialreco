from __future__ import annotations

from datetime import datetime

import pytest

from maisonaccueil.app.messages import EngineStatus, PreviewImage
from maisonaccueil.app.viewmodels import (
    ENGINE_UNAVAILABLE,
    PreviewViewModel,
    SystemStatusViewModel,
    format_model_ref,
    format_raw_score,
    format_utc,
)
from maisonaccueil.contracts import (
    CameraId,
    CaptureState,
    CaptureStatus,
    Decision,
    DecisionStatus,
    Metric,
    ProfileId,
    RawScore,
    TrackId,
)
from maisonaccueil.pipeline import VisionOutput

from .conftest import make_result

NAMES = {ProfileId("prf-a"): "Personne A"}


def vm(clock):  # type: ignore[no-untyped-def]
    return SystemStatusViewModel(NAMES.get, clock=clock)


def test_score_format_is_technical_never_percent() -> None:
    text = format_raw_score(RawScore(0.612, Metric.COSINE))
    assert "score technique (non calibré)" in text
    assert "0.612" in text
    assert "%" not in text
    assert "plus bas" in format_raw_score(RawScore(0.9, Metric.L2))


def test_format_utc_rejects_naive(fixed_utc: datetime) -> None:
    assert format_utc(fixed_utc) == "2026-10-25 12:00:00 UTC"
    with pytest.raises(ValueError):
        format_utc(datetime(2026, 1, 1))


def test_model_text(model) -> None:  # type: ignore[no-untyped-def]
    assert format_model_ref(None) == ENGINE_UNAVAILABLE
    text = format_model_ref(model)
    assert "fake-engine vt1" in text and "DOUBLE DE TEST" in text and "cosine" in text


def test_no_engine_shows_unavailable_and_ignores_results(clock, model) -> None:  # type: ignore[no-untyped-def]
    s = vm(clock)
    assert s.engine_text == ENGINE_UNAVAILABLE
    assert s.face_lines == [ENGINE_UNAVAILABLE]
    s.apply_vision_output(VisionOutput(1, 1, 5.0, results=(make_result(model),)))
    assert s.face_lines == [ENGINE_UNAVAILABLE]
    assert s.ignored_results == 1
    s.apply_engine_status(EngineStatus(False, detail="lot D non intégré"))
    assert s.engine_text == f"{ENGINE_UNAVAILABLE} (lot D non intégré)"
    assert s.model_text == ENGINE_UNAVAILABLE


def test_engine_available_shows_technical_scores(clock, model) -> None:  # type: ignore[no-untyped-def]
    s = vm(clock)
    s.apply_engine_status(EngineStatus(True, model))
    s.apply_vision_output(VisionOutput(1, 1, 5.0, results=(make_result(model),)))
    (line,) = s.face_lines
    assert "Personne A" in line
    assert "pas une décision" in line
    assert "score technique (non calibré)" in line
    assert "%" not in line
    assert "fake-engine" in s.model_text


def test_result_from_other_model_is_ignored(clock, model, other_model) -> None:  # type: ignore[no-untyped-def]
    s = vm(clock)
    s.apply_engine_status(EngineStatus(True, model))
    s.apply_vision_output(VisionOutput(1, 1, 5.0, results=(make_result(other_model),)))
    assert s.face_lines == []
    assert s.ignored_results == 1


def test_decisions_and_unknown_profile(clock, model) -> None:  # type: ignore[no-untyped-def]
    s = vm(clock)
    s.apply_engine_status(EngineStatus(True, model))
    confirmed = Decision(TrackId("trk-1"), DecisionStatus.CONFIRMED, 1, model, "temporal-v1",
                         5, 800, profile_id=ProfileId("prf-a"))
    unknown = Decision(TrackId("trk-2"), DecisionStatus.UNKNOWN, 1, model, "temporal-v1", 5, 800)
    out = VisionOutput(
        1, 1, 5.0,
        results=(make_result(model), make_result(model, None, track="trk-3")),
        decisions=(confirmed, unknown),
    )
    s.apply_vision_output(out)
    lines = s.face_lines
    assert any("confirmé — Personne A" in line and "trk-1" in line for line in lines)
    assert any("aucune référence compatible" in line for line in lines)
    assert any(line.startswith("Décision (temporal-v1) : inconnu") for line in lines)
    assert not any("Personne A" in line for line in lines if "inconnu" in line)


def test_unknown_profile_id_is_not_named(clock, model) -> None:  # type: ignore[no-untyped-def]
    s = vm(clock)
    s.apply_engine_status(EngineStatus(True, model))
    s.apply_vision_output(VisionOutput(1, 1, 5.0, results=(make_result(model, "prf-zzz"),)))
    assert "introuvable" in s.face_lines[0]


def test_face_lines_expire(clock, model) -> None:  # type: ignore[no-untyped-def]
    s = vm(clock)
    s.apply_engine_status(EngineStatus(True, model))
    s.apply_vision_output(VisionOutput(1, 1, 5.0, results=(make_result(model),)))
    clock.advance_ms(2_000)
    s.tick()
    assert len(s.face_lines) == 1
    clock.advance_ms(2_000)
    s.tick()
    assert s.face_lines == []


def test_capture_and_counters(clock) -> None:  # type: ignore[no-untyped-def]
    s = vm(clock)
    assert s.camera_text == "caméra non disponible"
    assert not s.camera_running
    s.apply_capture_state(CaptureState(CameraId("camera-0"), CaptureStatus.RECONNECTING, 1,
                                       detail="coupure", frames_captured=10, frames_dropped=3))
    assert s.camera_text == "camera-0 : reconnexion en cours — coupure"
    s.apply_vision_output(VisionOutput(1, 1, 12.4, frames_dropped=7))
    assert "abandonnées (capture) : 3" in s.dropped_text
    assert "abandonnées (vision) : 7" in s.dropped_text
    assert s.processing_text == "traitement : 12 ms par image"


def test_engine_status_validation(model) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        EngineStatus(True)
    with pytest.raises(ValueError):
        EngineStatus(False, model)


def test_notifications(clock) -> None:  # type: ignore[no-untyped-def]
    s = vm(clock)
    calls: list[int] = []
    unsubscribe = s.subscribe(lambda: calls.append(1))
    s.subscribe(lambda: 1 / 0)  # un abonné défaillant ne casse rien
    s.apply_engine_status(EngineStatus(False))
    unsubscribe()
    s.apply_engine_status(EngineStatus(False))
    assert calls == [1]


def _preview(n: int = 1) -> PreviewImage:
    return PreviewImage(n, "cam", 1, 4, 2, 640, 320, bytes(4 * 2 * 3))


def test_preview_vm(clock) -> None:  # type: ignore[no-untyped-def]
    p = PreviewViewModel(stale_after_ms=1_000, clock=clock)
    assert p.image is None and p.caption == "aucune image reçue"
    p.apply_preview(_preview(7))
    assert p.image is not None and "image 7" in p.caption and "640×320" in p.caption
    clock.advance_ms(1_500)
    p.tick()
    assert p.image is None
    assert "interrompu" in p.caption


def test_preview_image_validation() -> None:
    with pytest.raises(ValueError):
        PreviewImage(1, "cam", 1, 4, 2, 4, 2, b"\x00" * 5)
