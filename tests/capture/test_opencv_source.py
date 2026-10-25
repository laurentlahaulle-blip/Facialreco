"""OpenCVSource avec un faux VideoCapture injectable (aucune caméra requise)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from maisonaccueil.capture import (
    OpenCVSource,
    ReconnectPolicy,
    create_source,
    is_local_network_url,
    redact_url,
)
from maisonaccueil.capture.opencv_source import resolve_backend
from maisonaccueil.contracts import (
    CameraId,
    CaptureStatus,
    SourceConfig,
    SourceKind,
)
from maisonaccueil.contracts.interfaces import FrameSource
from maisonaccueil.pipeline import ResourceBusyError, ResourceLock

from .fakes import FakeCaptureFactory, FakeVideoCapture, Recorder

USB0 = SourceConfig(CameraId("cam0"), SourceKind.USB, "0", width=640, height=480, fps=15)


def usb_source(
    factory: FakeCaptureFactory, policy: ReconnectPolicy, lock_dir: Path, **kw: object
) -> OpenCVSource:
    return OpenCVSource(
        USB0, capture_factory=factory, policy=policy, lock_dir=lock_dir, **kw  # type: ignore[arg-type]
    )


def test_conforms_and_configures_device(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    cap = FakeVideoCapture()
    factory = FakeCaptureFactory([cap])
    src = usb_source(factory, fast_policy, isolated_lock_dir, backend="dshow")
    assert isinstance(src, FrameSource)
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_frames(3)
    src.stop()
    target, api, params = factory.calls[0]
    assert target == 0 and api == cv2.CAP_DSHOW and params == []
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert cap.props[cv2.CAP_PROP_FPS] == 15
    assert cap.props[cv2.CAP_PROP_BUFFERSIZE] == 1
    assert cap.released
    assert rec.frames[0].image.shape == (24, 32, 3)


def test_reconnects_after_open_failures(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    factory = FakeCaptureFactory(
        [FakeVideoCapture(opened=False), FakeVideoCapture(opened=False), FakeVideoCapture()]
    )
    src = usb_source(factory, fast_policy, isolated_lock_dir)
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_frames(2)
    src.stop()
    assert rec.distinct_statuses() == [
        CaptureStatus.CONNECTING, CaptureStatus.RECONNECTING, CaptureStatus.RUNNING,
        CaptureStatus.STOPPED,
    ]
    assert all(c.released for c in factory.created)


def test_reconnects_after_stream_loss_and_keeps_frame_ids(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    # 3 images puis coupure (2 lectures ratées) ; la capture suivante fonctionne.
    first = FakeVideoCapture(script=[True, True, True], tail=False)
    second = FakeVideoCapture()
    factory = FakeCaptureFactory([first, second])
    src = usb_source(factory, fast_policy, isolated_lock_dir)
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_frames(6)
    src.stop()
    assert first.released and second.released
    ids = [f.frame_id for f in rec.frames]
    assert ids == list(range(1, len(ids) + 1))
    assert rec.distinct_statuses()[:4] == [
        CaptureStatus.CONNECTING, CaptureStatus.RUNNING, CaptureStatus.RECONNECTING,
        CaptureStatus.RUNNING,
    ]


def test_fails_after_max_attempts_and_releases_lock(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    factory = FakeCaptureFactory([FakeVideoCapture(opened=False)])
    src = usb_source(factory, fast_policy, isolated_lock_dir)
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.FAILED)
    src.stop()
    assert len(factory.created) == fast_policy.max_attempts
    assert rec.statuses().count(CaptureStatus.RECONNECTING) == fast_policy.max_attempts - 1
    assert src.state.status == CaptureStatus.FAILED
    assert "4 tentatives" in src.state.detail
    assert not rec.frames
    assert not (isolated_lock_dir / "camera-0.lock").exists()


def test_repeated_stream_loss_without_frames_fails(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    # Ouverture OK mais aucune image lisible : compte comme tentative échouée.
    factory = FakeCaptureFactory([FakeVideoCapture(tail=False)])
    src = usb_source(factory, fast_policy, isolated_lock_dir)
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.FAILED)
    src.stop()
    assert CaptureStatus.RUNNING not in rec.statuses()


def test_factory_exception_is_an_open_failure(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    def broken(target: object, api: int, params: object) -> FakeVideoCapture:
        raise cv2.error("backend indisponible")

    src = OpenCVSource(
        USB0, capture_factory=broken, policy=fast_policy, lock_dir=isolated_lock_dir
    )
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.FAILED)
    src.stop()


def test_backoff_is_exponential_and_bounded() -> None:
    p = ReconnectPolicy(initial_delay_s=0.5, max_delay_s=4.0, factor=2.0, max_attempts=7)
    assert list(p.delays()) == [0.5, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0]
    assert p.delay_for(0) == 0.0
    with pytest.raises(ValueError):
        ReconnectPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        ReconnectPolicy(initial_delay_s=5, max_delay_s=1)


def test_stop_interrupts_backoff_wait(isolated_lock_dir: Path) -> None:
    import time

    slow = ReconnectPolicy(initial_delay_s=30.0, max_delay_s=30.0, max_attempts=3)
    factory = FakeCaptureFactory([FakeVideoCapture(opened=False)])
    src = OpenCVSource(USB0, capture_factory=factory, policy=slow, lock_dir=isolated_lock_dir)
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.RECONNECTING)
    t0 = time.monotonic()
    src.stop(timeout_s=2.0)
    assert time.monotonic() - t0 < 1.0
    assert src.state.status == CaptureStatus.STOPPED


def test_usb_lock_is_exclusive_and_released(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    a = usb_source(FakeCaptureFactory([FakeVideoCapture()]), fast_policy, isolated_lock_dir)
    b = usb_source(FakeCaptureFactory([FakeVideoCapture()]), fast_policy, isolated_lock_dir)
    rec = Recorder()
    a.start(rec.on_frame, rec.on_state)
    lock_file = isolated_lock_dir / "camera-0.lock"
    assert lock_file.exists()
    with pytest.raises(ResourceBusyError):
        b.start(rec.on_frame, rec.on_state)
    assert not b.is_running
    a.stop()
    assert not lock_file.exists()
    b.start(rec.on_frame, rec.on_state)  # la caméra est de nouveau libre
    b.stop()
    assert not lock_file.exists()


def test_usb_busy_from_other_session(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    factory = FakeCaptureFactory([FakeVideoCapture()])
    src = usb_source(factory, fast_policy, isolated_lock_dir)
    with ResourceLock("camera-0", owner="autre-session", lock_dir=isolated_lock_dir):
        with pytest.raises(ResourceBusyError):
            src.start(lambda f: None, lambda s: None)
    assert factory.calls == []  # la caméra n'a jamais été ouverte


def test_lock_uses_env_lock_dir_by_default(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    src = OpenCVSource(
        SourceConfig(CameraId("c2"), SourceKind.USB, "2"),
        capture_factory=FakeCaptureFactory([FakeVideoCapture()]),
        policy=fast_policy,
    )
    src.start(lambda f: None, lambda s: None)
    assert (isolated_lock_dir / "camera-2.lock").exists()
    src.stop()
    assert not (isolated_lock_dir / "camera-2.lock").exists()


def test_grayscale_frames_are_converted(
    fast_policy: ReconnectPolicy, isolated_lock_dir: Path
) -> None:
    factory = FakeCaptureFactory([FakeVideoCapture(shape=(10, 12))])
    src = usb_source(factory, fast_policy, isolated_lock_dir)
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_frames(1)
    src.stop()
    assert rec.frames[0].image.shape == (10, 12, 3)


# ---------------------------------------------------------------- réseau
@pytest.mark.parametrize(
    "url",
    [
        "rtsp://192.168.1.20:554/stream1",
        "rtsp://user:secret@10.0.0.5/live",
        "http://172.16.3.4/video.mjpg",
        "http://127.0.0.1:8080/feed",
        "rtsp://camera-entree.local/stream",
        "http://camera/stream",
        "http://[fe80::1]/stream",
    ],
)
def test_local_urls_accepted(url: str) -> None:
    assert is_local_network_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://8.8.8.8/stream",
        "http://example.com/cam",
        "ftp://192.168.1.2/x",
        "file:///etc/passwd",
        "rtsp:///nohost",
    ],
)
def test_non_local_urls_rejected(url: str) -> None:
    assert not is_local_network_url(url)
    with pytest.raises(ValueError):
        OpenCVSource(SourceConfig(CameraId("n"), SourceKind.NETWORK, url))


def test_redact_url_hides_credentials() -> None:
    red = redact_url("rtsp://user:secret@192.168.1.20:554/stream?token=abc")
    assert "secret" not in red and "token" not in red and "user" not in red
    assert red == "rtsp://192.168.1.20:554/stream"


def test_network_source_uses_timeouts_and_never_exposes_secret(
    fast_policy: ReconnectPolicy,
) -> None:
    url = "rtsp://user:secret@192.168.1.20/stream"
    factory = FakeCaptureFactory([FakeVideoCapture(opened=False)])
    src = OpenCVSource(
        SourceConfig(CameraId("net"), SourceKind.NETWORK, url),
        capture_factory=factory, policy=fast_policy,
    )
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.FAILED)
    src.stop()
    target, api, params = factory.calls[0]
    assert target == url and api == cv2.CAP_FFMPEG
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        assert cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in params
    assert all("secret" not in s.detail for s in rec.states)


# ---------------------------------------------------------------- fichier
def test_file_source_end_of_stream(fast_policy: ReconnectPolicy, tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    path.write_bytes(b"")  # le contenu est lu par le faux VideoCapture
    factory = FakeCaptureFactory([FakeVideoCapture(script=[True] * 5, tail=False)])
    src = OpenCVSource(
        SourceConfig(CameraId("f"), SourceKind.FILE, str(path)),
        capture_factory=factory, policy=fast_policy, realtime=False,
    )
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.STOPPED)
    src.stop()
    assert len(rec.frames) == 5
    assert src.state.detail == "fin du flux"
    assert CaptureStatus.RECONNECTING not in rec.statuses()


def test_missing_file_fails(fast_policy: ReconnectPolicy, tmp_path: Path) -> None:
    factory = FakeCaptureFactory([FakeVideoCapture()])
    src = OpenCVSource(
        SourceConfig(CameraId("f"), SourceKind.FILE, str(tmp_path / "absent.avi")),
        capture_factory=factory, policy=fast_policy,
    )
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.FAILED)
    src.stop()
    assert factory.calls == []


def test_real_video_file_roundtrip(tmp_path: Path) -> None:
    """Fichier vidéo synthétique écrit dans un dossier temporaire, relu par OpenCV."""
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 25.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("Encodeur MJPG indisponible dans cet environnement")
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    src = create_source(
        SourceConfig(CameraId("f"), SourceKind.FILE, str(path)), realtime=False
    )
    assert isinstance(src, OpenCVSource)
    rec = Recorder()
    src.start(rec.on_frame, rec.on_state)
    assert rec.wait_status(CaptureStatus.STOPPED, timeout=10)
    src.stop()
    assert len(rec.frames) == 10
    assert rec.frames[0].image.shape == (48, 64, 3)


def test_invalid_configs() -> None:
    with pytest.raises(ValueError):
        OpenCVSource(SourceConfig(CameraId("u"), SourceKind.USB, "abc"))
    with pytest.raises(ValueError):
        OpenCVSource(SourceConfig(CameraId("u"), SourceKind.USB, "-1"))
    with pytest.raises(ValueError):
        OpenCVSource(SourceConfig(CameraId("s"), SourceKind.SYNTHETIC, ""))
    with pytest.raises(ValueError):
        resolve_backend("quicktime")
    assert resolve_backend("MSMF") == cv2.CAP_MSMF
    assert resolve_backend(None) == cv2.CAP_ANY
