from __future__ import annotations

import time

import numpy as np
import pytest

from maisonaccueil.capture import SyntheticSource, create_source, to_bgr_uint8
from maisonaccueil.contracts import (
    CameraId,
    CaptureStatus,
    SourceConfig,
    SourceKind,
)
from maisonaccueil.contracts.interfaces import FrameSource

from .fakes import Recorder


def test_conforms_to_frame_source() -> None:
    assert isinstance(SyntheticSource(CameraId("syn")), FrameSource)


def test_initial_state_is_stopped() -> None:
    src = SyntheticSource(CameraId("syn"))
    assert src.state.status == CaptureStatus.STOPPED
    assert src.state.frames_captured == 0


def test_frames_are_ordered_timestamped_and_bgr() -> None:
    rec = Recorder()
    src = SyntheticSource(CameraId("syn"), width=64, height=48, fps=None, max_frames=20)
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.STOPPED)
    src.stop()
    assert len(rec.frames) == 20
    ids = [f.frame_id for f in rec.frames]
    assert ids == list(range(1, 21))
    mono = [f.monotonic_ns for f in rec.frames]
    assert mono == sorted(mono)
    for f in rec.frames:
        assert f.camera_id == "syn"
        assert f.wall_time.utcoffset() is not None and f.wall_time.utcoffset().total_seconds() == 0
        assert f.image.shape == (48, 64, 3) and f.image.dtype == np.uint8
        assert SyntheticSource.decode_counter(f.image) == f.frame_id
    assert rec.distinct_statuses() == [
        CaptureStatus.CONNECTING, CaptureStatus.RUNNING, CaptureStatus.STOPPED,
    ]
    assert rec.states[-1].detail == "fin du flux"
    assert rec.states[-1].frames_captured == 20


def test_start_is_non_blocking_and_fps_respected() -> None:
    rec = Recorder()
    src = SyntheticSource(CameraId("syn"), fps=50.0)
    t0 = time.monotonic()
    src.start(rec.on_frame, rec.on_state)
    assert time.monotonic() - t0 < 0.2
    time.sleep(0.5)
    src.stop()
    # ~25 images attendues ; borne large pour un environnement partagé
    assert 5 <= len(rec.frames) <= 40


@pytest.mark.parametrize("pattern", ["bars", "checker", "gradient"])
def test_patterns_render_deterministically(pattern: str) -> None:
    src = SyntheticSource(CameraId("syn"), pattern=pattern)
    a, b = src.render(7), src.render(7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, src.render(8))


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        SyntheticSource(CameraId("syn"), pattern="faces")
    with pytest.raises(ValueError):
        SyntheticSource(CameraId("syn"), fps=0)
    with pytest.raises(ValueError):
        SyntheticSource(CameraId("syn"), width=4)


def test_factory_builds_synthetic() -> None:
    cfg = SourceConfig(CameraId("s"), SourceKind.SYNTHETIC, "checker", width=40, height=30)
    src = create_source(cfg, fps=None, max_frames=3)
    assert isinstance(src, SyntheticSource)
    assert (src.width, src.height, src.pattern) == (40, 30, "checker")
    with pytest.raises(ValueError):
        create_source(SourceConfig(CameraId("s"), SourceKind.SYNTHETIC, "nope"))


def test_to_bgr_uint8_conversions() -> None:
    gray = np.zeros((4, 5), np.uint8)
    assert to_bgr_uint8(gray).shape == (4, 5, 3)
    bgra = np.zeros((4, 5, 4), np.uint8)
    out = to_bgr_uint8(bgra)
    assert out.shape == (4, 5, 3) and out.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        to_bgr_uint8(np.zeros((4, 5, 3), np.float32))
    with pytest.raises(ValueError):
        to_bgr_uint8(np.zeros((4, 5, 2), np.uint8))
