"""Séparation des processus : files bornées, aperçu réduit côté vision, processus fils."""

from __future__ import annotations

import pickle
import queue
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from maisonaccueil.app.fakes import FakeFrameSource, fake_source_factory
from maisonaccueil.app.ipc import UiInbox, VisionPublisher
from maisonaccueil.app.messages import (
    ENGINE_STATUS_UNAVAILABLE,
    EngineStatus,
    EnrollmentSample,
    PreviewImage,
)
from maisonaccueil.app.vision_worker import (
    InProcessVision,
    PreviewTap,
    PreviewThrottle,
    VisionProcess,
    make_preview,
    resolve_factory,
    run_preview_worker,
)
from maisonaccueil.contracts import (
    ArrivalEvent,
    ArrivalKind,
    CameraId,
    CaptureState,
    CaptureStatus,
    EventId,
    Frame,
    FrameId,
    SourceConfig,
    SourceKind,
    TrackId,
)
from maisonaccueil.pipeline import ResourceLock, VisionOutput

from .conftest import make_result

SYNTH = SourceConfig(CameraId("fake-synthetic"), SourceKind.SYNTHETIC, "demo", 640, 480, 30.0)


def _frame(w: int = 640, h: int = 480) -> Frame:
    return Frame(FrameId(3), CameraId("cam"), 123, datetime.now(UTC),
                 np.full((h, w, 3), 50, dtype=np.uint8))


def wait_for(cond: Callable[[], bool], timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def test_make_preview_downscales() -> None:
    p = make_preview(_frame(), max_width=320)
    assert (p.width, p.height, p.source_width, p.source_height) == (320, 240, 640, 480)
    assert len(p.data) == 320 * 240 * 3
    small = make_preview(_frame(100, 80), max_width=320)
    assert (small.width, small.height) == (100, 80)


def test_throttle() -> None:
    t = PreviewThrottle(max_fps=10)
    assert t.accept(0)
    assert not t.accept(50_000_000)
    assert t.accept(100_000_000)


def test_messages_pickle_roundtrip(model) -> None:  # type: ignore[no-untyped-def]
    event = ArrivalEvent(EventId("evt-1"), ArrivalKind.UNKNOWN, TrackId("t"), 1,
                         datetime.now(UTC))
    out = VisionOutput(1, 2, 3.0, results=(make_result(model),), arrivals=(event,))
    for msg in (out, EngineStatus(True, model), ENGINE_STATUS_UNAVAILABLE,
                make_preview(_frame()), EnrollmentSample(1, b"png"),
                CaptureState(CameraId("c"), CaptureStatus.RUNNING, 1)):
        again = pickle.loads(pickle.dumps(msg))
        assert again == msg


def test_publisher_never_blocks_and_counts_losses() -> None:
    events: queue.Queue[object] = queue.Queue(maxsize=1)
    preview: queue.Queue[object] = queue.Queue(maxsize=1)
    pub = VisionPublisher(events, preview, arrival_timeout_s=0.05)
    assert pub.publish(ENGINE_STATUS_UNAVAILABLE)
    t0 = time.monotonic()
    assert not pub.publish(VisionOutput(1, 1, 1.0))
    event = ArrivalEvent(EventId("e"), ArrivalKind.UNKNOWN, TrackId("t"), 1, datetime.now(UTC))
    assert not pub.publish(VisionOutput(2, 1, 1.0, arrivals=(event,)))  # court délai puis abandon
    assert time.monotonic() - t0 < 1.0
    assert pub.events_lost == 2
    p = make_preview(_frame())
    assert pub.publish_preview(p) and not pub.publish_preview(p)
    assert pub.previews_lost == 1


def test_inbox_keeps_latest_preview_and_filters() -> None:
    events: queue.Queue[object] = queue.Queue()
    preview: queue.Queue[object] = queue.Queue()
    for i in range(3):
        preview.put(PreviewImage(i, "c", i, 1, 1, 1, 1, b"\x00\x00\x00"))
    events.put(ENGINE_STATUS_UNAVAILABLE)
    events.put("intrus")
    batch = UiInbox(events, preview).drain()
    assert batch.preview is not None and batch.preview.frame_id == 2
    assert batch.events == [ENGINE_STATUS_UNAVAILABLE]
    assert batch.unexpected == 1
    empty = UiInbox(events, preview).drain()
    assert empty.events == [] and empty.preview is None


def test_preview_tap_forwards_downstream() -> None:
    got: list[Frame] = []
    preview: queue.Queue[object] = queue.Queue()
    tap = PreviewTap(VisionPublisher(queue.Queue(), preview), max_fps=1000,
                     downstream=got.append)
    tap(_frame())
    assert len(got) == 1 and isinstance(preview.get_nowait(), PreviewImage)


def test_in_process_worker_publishes_states_and_previews() -> None:
    events: queue.Queue[object] = queue.Queue()
    preview: queue.Queue[object] = queue.Queue()
    worker = InProcessVision(fake_source_factory, SYNTH, events, preview, max_preview_fps=100)
    worker.start()
    try:
        assert wait_for(lambda: preview.qsize() >= 2)
    finally:
        worker.stop()
    received = []
    while not events.empty():
        received.append(events.get_nowait())
    assert received[0] == ENGINE_STATUS_UNAVAILABLE
    statuses = [m.status for m in received if isinstance(m, CaptureState)]
    assert CaptureStatus.RUNNING in statuses and statuses[-1] == CaptureStatus.STOPPED
    img = preview.get_nowait()
    assert isinstance(img, PreviewImage) and img.width == 320


def test_worker_reports_bad_factory() -> None:
    events: queue.Queue[object] = queue.Queue()
    stop = threading.Event()
    stop.set()
    run_preview_worker("maisonaccueil.app.fakes:absent", SYNTH, events, queue.Queue(), stop)
    events.get_nowait()
    state = events.get_nowait()
    assert isinstance(state, CaptureState) and state.status == CaptureStatus.FAILED
    assert "source non disponible" in state.detail


def test_worker_optional_camera_lock_busy(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAISONACCUEIL_LOCK_DIR", str(tmp_path))
    usb = SourceConfig(CameraId("camera-0"), SourceKind.USB, "0")
    events: queue.Queue[object] = queue.Queue()
    stop = threading.Event()
    stop.set()
    with ResourceLock("camera-0", owner="autre-session", lock_dir=tmp_path):
        run_preview_worker(lambda c: FakeFrameSource(), usb, events, queue.Queue(), stop,
                           lock_camera=True)
    events.get_nowait()
    state = events.get_nowait()
    assert state.status == CaptureStatus.FAILED and "réservée" in state.detail


def test_resolve_factory_errors() -> None:
    assert resolve_factory("maisonaccueil.app.fakes:fake_source_factory") is fake_source_factory
    with pytest.raises(ValueError):
        resolve_factory("sans-deux-points")


@pytest.mark.slow
def test_separate_process_with_spawn() -> None:
    """Vrai processus fils (spawn) : l'interface ne reçoit que des messages sérialisés."""
    proc = VisionProcess("maisonaccueil.app.fakes:fake_source_factory", SYNTH,
                         max_preview_fps=20)
    inbox = UiInbox(proc.events, proc.preview)
    proc.start()
    seen_preview = False
    seen_running = False
    try:
        end = time.monotonic() + 30
        while time.monotonic() < end and not (seen_preview and seen_running):
            batch = inbox.drain()
            seen_preview |= batch.preview is not None
            seen_running |= any(isinstance(e, CaptureState) and e.status == CaptureStatus.RUNNING
                                for e in batch.events)
            time.sleep(0.05)
    finally:
        proc.stop()
    assert seen_preview and seen_running
    assert not proc.alive
