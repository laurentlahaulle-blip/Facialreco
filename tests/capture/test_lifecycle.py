"""Cycle d'états, arrêt idempotent, arrêt sous charge, abandon via LatestQueue."""

from __future__ import annotations

import threading
import time

import pytest

from maisonaccueil.capture import SyntheticSource
from maisonaccueil.contracts import CameraId, CaptureState, CaptureStatus, Frame
from maisonaccueil.pipeline import LatestQueue

from .fakes import Recorder


def test_state_cycle_start_stop() -> None:
    rec = Recorder()
    src = SyntheticSource(CameraId("syn"), fps=200.0)
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.RUNNING)
    assert src.is_running
    src.stop()
    assert not src.is_running
    assert rec.distinct_statuses() == [
        CaptureStatus.CONNECTING, CaptureStatus.RUNNING, CaptureStatus.STOPPED,
    ]
    assert src.state.status == CaptureStatus.STOPPED
    assert src.state.detail == "arrêt demandé"


def test_stop_is_idempotent_and_safe_before_start() -> None:
    src = SyntheticSource(CameraId("syn"))
    src.stop()  # jamais démarrée
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    src.stop()
    src.stop()
    src.stop(timeout_s=0.0)
    assert rec.statuses().count(CaptureStatus.STOPPED) == 1


def test_double_start_rejected_and_restart_allowed() -> None:
    rec = Recorder()
    src = SyntheticSource(CameraId("syn"), fps=100.0)
    src.start(rec.on_frame, rec.on_state)
    with pytest.raises(RuntimeError):
        src.start(rec.on_frame, rec.on_state)
    assert rec.wait_frames(2)
    src.stop()
    last_id = rec.frames[-1].frame_id
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_frames(len(rec.frames) + 2)
    src.stop()
    assert rec.frames[-1].frame_id > last_id  # numérotation toujours croissante


def test_stop_under_load_with_latest_queue() -> None:
    queue: LatestQueue[Frame] = LatestQueue(maxsize=2)
    states: list[CaptureState] = []
    src = SyntheticSource(
        CameraId("syn"), fps=None, drop_counter=lambda: queue.dropped, stats_interval_s=0.05
    )
    src.start(queue.put, states.append)
    time.sleep(0.3)  # producteur à pleine vitesse, aucun consommateur
    t0 = time.monotonic()
    src.stop(timeout_s=2.0)
    assert time.monotonic() - t0 < 2.0
    assert not src.is_running
    assert len(queue) <= 2
    assert queue.dropped > 0
    final = states[-1]
    assert final.status == CaptureStatus.STOPPED
    assert final.frames_captured >= queue.dropped
    assert final.frames_dropped == queue.dropped
    # des états RUNNING périodiques remontent les compteurs
    running = [s for s in states if s.status == CaptureStatus.RUNNING]
    assert len(running) >= 2
    assert running[-1].frames_dropped > 0


def test_slow_consumer_gets_recent_frames() -> None:
    """Le consommateur lent reçoit des images récentes : les anciennes sont abandonnées."""
    queue: LatestQueue[Frame] = LatestQueue(maxsize=2)
    src = SyntheticSource(CameraId("syn"), fps=500.0, drop_counter=lambda: queue.dropped)
    src.start(queue.put, lambda s: None)
    consumed: list[Frame] = []
    try:
        for _ in range(5):
            f = queue.get(timeout=2.0)
            assert f is not None
            consumed.append(f)
            time.sleep(0.05)  # traitement lent
    finally:
        src.stop()
    gaps = [b.frame_id - a.frame_id for a, b in zip(consumed, consumed[1:], strict=False)]
    assert max(gaps) > 1
    assert queue.dropped > 0
    assert src.state.frames_dropped == queue.dropped


def test_on_frame_exception_does_not_kill_capture() -> None:
    calls = {"n": 0}
    ok = threading.Event()

    def flaky(frame: Frame) -> None:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("consommateur défaillant")
        if calls["n"] >= 7:
            ok.set()

    src = SyntheticSource(CameraId("syn"), fps=None)
    src.start(flaky, lambda s: None)
    assert ok.wait(3.0)
    src.stop()
    assert src.callback_errors >= 3
    assert src.state.status == CaptureStatus.STOPPED


def test_stop_from_state_callback_does_not_deadlock() -> None:
    src = SyntheticSource(CameraId("syn"), fps=100.0)
    done = threading.Event()

    def on_state(state: CaptureState) -> None:
        if state.status == CaptureStatus.RUNNING:
            src.stop()
        if state.status == CaptureStatus.STOPPED:
            done.set()

    src.start(lambda f: None, on_state)
    assert done.wait(3.0)
    src.stop()
