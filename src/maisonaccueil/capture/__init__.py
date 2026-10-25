"""Lot A — Capture : sources d'images conformes à ``contracts.FrameSource``.

Usage typique (le consommateur dépose dans une ``LatestQueue``) ::

    queue = LatestQueue[Frame](maxsize=2)
    source = create_source(config, drop_counter=lambda: queue.dropped)
    source.start(queue.put, on_state)
    ...
    source.stop()
"""

from .base import (
    DeviceOpenError,
    EndOfStream,
    ReconnectPolicy,
    ThreadedFrameSource,
    to_bgr_uint8,
)
from .discovery import UsbCameraInfo, UsbProbeStatus, list_usb_cameras
from .factory import create_source
from .opencv_source import (
    BACKENDS,
    OpenCVSource,
    VideoCaptureLike,
    is_local_network_url,
    redact_url,
)
from .synthetic import PATTERNS, SyntheticSource

__all__ = [
    "BACKENDS",
    "PATTERNS",
    "DeviceOpenError",
    "EndOfStream",
    "OpenCVSource",
    "ReconnectPolicy",
    "SyntheticSource",
    "ThreadedFrameSource",
    "UsbCameraInfo",
    "UsbProbeStatus",
    "VideoCaptureLike",
    "create_source",
    "is_local_network_url",
    "list_usb_cameras",
    "redact_url",
    "to_bgr_uint8",
]
