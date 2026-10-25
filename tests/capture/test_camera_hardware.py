"""Tests sur caméra physique : marqués ``camera`` et ignorés par défaut.

Exécution volontaire uniquement :
``MAISONACCUEIL_CAMERA_TESTS=1 uv run pytest -m camera tests/capture``
La caméra est réservée via ``ResourceLock("camera-<index>")`` par la source elle-même.
Aucune image n'est conservée.
"""

from __future__ import annotations

import os

import pytest

from maisonaccueil.capture import OpenCVSource, list_usb_cameras
from maisonaccueil.contracts import CameraId, CaptureStatus, SourceConfig, SourceKind

from .fakes import Recorder

pytestmark = [
    pytest.mark.camera,
    pytest.mark.skipif(
        os.environ.get("MAISONACCUEIL_CAMERA_TESTS") != "1",
        reason="test matériel : définir MAISONACCUEIL_CAMERA_TESTS=1",
    ),
]

INDEX = int(os.environ.get("MAISONACCUEIL_CAMERA_INDEX", "0"))


def test_real_usb_camera_produces_frames() -> None:
    src = OpenCVSource(
        SourceConfig(CameraId(f"usb{INDEX}"), SourceKind.USB, str(INDEX)),
        lock_owner="tests-lot-A",
    )
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    try:
        assert rec.wait_frames(10, timeout=15), rec.statuses()
    finally:
        src.stop()
    assert src.state.status == CaptureStatus.STOPPED
    rec.frames.clear()  # aucune image conservée


def test_real_enumeration_runs() -> None:
    found = list_usb_cameras(4)
    assert all(c.index >= 0 for c in found)
