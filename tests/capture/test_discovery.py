from __future__ import annotations

from pathlib import Path
from typing import Any

from maisonaccueil.capture import UsbProbeStatus, list_usb_cameras
from maisonaccueil.pipeline import ResourceLock

from .fakes import FakeVideoCapture


class FakeProbeFactory:
    def __init__(self, present: set[int]) -> None:
        self.present = present
        self.opened: list[int] = []
        self.caps: list[FakeVideoCapture] = []

    def __call__(self, target: Any, api: int, params: Any) -> FakeVideoCapture:
        self.opened.append(int(target))
        cap = FakeVideoCapture(opened=int(target) in self.present)
        self.caps.append(cap)
        return cap


def test_lists_present_cameras_and_releases(isolated_lock_dir: Path) -> None:
    factory = FakeProbeFactory({0, 2})
    found = list_usb_cameras(4, capture_factory=factory, lock_dir=isolated_lock_dir)
    assert [(c.index, c.status) for c in found] == [
        (0, UsbProbeStatus.AVAILABLE), (2, UsbProbeStatus.AVAILABLE),
    ]
    assert found[1].location == "2"
    assert all(c.released for c in factory.caps)
    assert not list(isolated_lock_dir.glob("*.lock"))  # aucune réservation laissée


def test_busy_index_is_not_opened(isolated_lock_dir: Path) -> None:
    factory = FakeProbeFactory({0, 1})
    with ResourceLock("camera-0", owner="lot-B", lock_dir=isolated_lock_dir):
        found = list_usb_cameras(2, capture_factory=factory, lock_dir=isolated_lock_dir)
        assert 0 not in factory.opened
        assert found[0].status == UsbProbeStatus.BUSY and "lot-B" in found[0].detail
        assert found[1].status == UsbProbeStatus.AVAILABLE
        only_free = list_usb_cameras(
            2, capture_factory=factory, lock_dir=isolated_lock_dir, include_busy=False
        )
        assert [c.index for c in only_free] == [1]
    assert (isolated_lock_dir / "camera-0.lock").exists() is False


def test_probe_errors_are_ignored(isolated_lock_dir: Path) -> None:
    def broken(target: Any, api: int, params: Any) -> FakeVideoCapture:
        raise RuntimeError("pilote absent")

    assert list_usb_cameras(3, capture_factory=broken, lock_dir=isolated_lock_dir) == []
