"""Source OpenCV : webcam USB, flux réseau local (RTSP/HTTP) ou fichier vidéo.

- USB : ``SourceConfig.location`` = index (ex. ``"0"``). Réservation obligatoire
  ``ResourceLock("camera-<index>")`` prise dans ``start`` (``ResourceBusyError`` si occupée)
  et libérée à la fin du thread de capture.
- Réseau : URL ``rtsp://``, ``rtsps://``, ``http://`` ou ``https://`` vers une adresse du
  réseau local uniquement (privée, boucle locale, lien local, ``*.local`` / ``*.lan`` /
  ``*.home.arpa`` ou nom sans domaine). Les identifiants de l'URL ne sont jamais journalisés.
- Fichier : lecture cadencée à la fréquence du fichier (``realtime=True``) ; fin de fichier
  = fin de flux (ou relecture si ``loop_file=True``).

Le ``VideoCapture`` est injectable (``capture_factory``) pour les tests (``Fake*``).
Aucune image n'est écrite sur disque.
"""

from __future__ import annotations

import ipaddress
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import cv2
import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import SourceConfig, SourceKind
from maisonaccueil.pipeline import ResourceLock

from .base import (
    DeviceOpenError,
    DropCounter,
    EndOfStream,
    ReconnectPolicy,
    ThreadedFrameSource,
)

logger = logging.getLogger(__name__)


class VideoCaptureLike(Protocol):
    """Sous-ensemble de ``cv2.VideoCapture`` utilisé par la source."""

    def isOpened(self) -> bool: ...  # noqa: N802 - nom imposé par OpenCV

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def get(self, prop_id: int) -> float: ...


# (cible, backend API, paramètres d'ouverture [prop, valeur, prop, valeur, ...])
CaptureFactory = Callable[[int | str, int, Sequence[int]], VideoCaptureLike]

BACKENDS: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
    "ffmpeg": cv2.CAP_FFMPEG,
    "gstreamer": cv2.CAP_GSTREAMER,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}

NETWORK_SCHEMES = ("rtsp", "rtsps", "http", "https")
LOCAL_SUFFIXES = (".local", ".lan", ".home.arpa", ".localdomain")


def default_capture_factory(
    target: int | str, api: int, params: Sequence[int]
) -> VideoCaptureLike:
    cap: Any = (
        cv2.VideoCapture(target, api, list(params)) if params else cv2.VideoCapture(target, api)
    )
    result: VideoCaptureLike = cap
    return result


def resolve_backend(backend: str | int | None) -> int:
    """``None`` = choix par défaut d'OpenCV. Accepte un nom (``"msmf"``, ``"dshow"``…)
    ou une constante ``cv2.CAP_*``."""
    if backend is None:
        return int(cv2.CAP_ANY)
    if isinstance(backend, int):
        return backend
    try:
        return int(BACKENDS[backend.lower()])
    except KeyError:
        raise ValueError(
            f"Backend inconnu {backend!r} ; choix : {', '.join(sorted(BACKENDS))}"
        ) from None


def default_usb_backend() -> str:
    """Backend USB conseillé : MSMF sous Windows (DSHOW en repli), V4L2 sous Linux."""
    if sys.platform == "win32":
        return "msmf"
    if sys.platform.startswith("linux"):
        return "v4l2"
    return "any"


def parse_usb_index(location: str) -> int:
    try:
        index = int(location.strip())
    except ValueError:
        raise ValueError(f"Index USB invalide : {location!r}") from None
    if index < 0:
        raise ValueError(f"Index USB invalide : {location!r}")
    return index


def is_local_network_url(url: str) -> bool:
    """Vrai si l'URL vise une caméra du réseau local (aucun service distant)."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in NETWORK_SCHEMES:
        return False
    host = (parts.hostname or "").lower()
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or "." not in host or host.endswith(LOCAL_SUFFIXES)
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local)


def redact_url(url: str) -> str:
    """Retire identifiants et paramètres d'une URL pour l'affichage."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}{parts.path}"


class OpenCVSource(ThreadedFrameSource):
    """``FrameSource`` adossée à ``cv2.VideoCapture``."""

    def __init__(
        self,
        config: SourceConfig,
        *,
        backend: str | int | None = None,
        policy: ReconnectPolicy | None = None,
        drop_counter: DropCounter | None = None,
        capture_factory: CaptureFactory | None = None,
        lock_owner: str = "maisonaccueil-capture",
        lock_dir: Path | None = None,
        realtime: bool = True,
        loop_file: bool = False,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        stats_interval_s: float = 1.0,
    ) -> None:
        if config.kind == SourceKind.SYNTHETIC:
            raise ValueError("Utiliser SyntheticSource pour une source synthétique")
        self.config = config
        self._factory: CaptureFactory = capture_factory or default_capture_factory
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self.realtime = realtime
        self.loop_file = loop_file
        self._cap: VideoCaptureLike | None = None
        self._lock: ResourceLock | None = None
        self._period_ns = 0
        self._next_due_ns = 0

        self._target: int | str
        if config.kind == SourceKind.USB:
            self.usb_index: int | None = parse_usb_index(config.location)
            self._target = self.usb_index
            self._api = resolve_backend(backend if backend is not None else default_usb_backend())
            self._lock = ResourceLock(
                f"camera-{self.usb_index}", owner=lock_owner, lock_dir=lock_dir
            )
        elif config.kind == SourceKind.NETWORK:
            self.usb_index = None
            if not is_local_network_url(config.location):
                raise ValueError(
                    "Flux réseau refusé : seules les caméras du réseau local sont autorisées "
                    f"({redact_url(config.location)})"
                )
            self._target = config.location
            self._api = resolve_backend(backend if backend is not None else "ffmpeg")
        elif config.kind == SourceKind.FILE:
            self.usb_index = None
            self._target = config.location
            self._api = resolve_backend(backend)
        else:  # pragma: no cover - énumération fermée
            raise ValueError(f"Type de source non pris en charge : {config.kind}")

        super().__init__(
            config.camera_id,
            policy=policy,
            drop_counter=drop_counter,
            stats_interval_s=stats_interval_s,
            thread_name=f"capture-{config.camera_id}",
        )

    # ------------------------------------------------------------------ hooks
    def _describe(self) -> str:
        kind = self.config.kind
        if kind == SourceKind.USB:
            return f"caméra USB {self.usb_index} ({self.camera_id})"
        if kind == SourceKind.NETWORK:
            return f"flux réseau {redact_url(self.config.location)} ({self.camera_id})"
        return f"fichier {Path(self.config.location).name} ({self.camera_id})"

    def _before_start(self) -> None:
        if self._lock is not None:
            self._lock.acquire()  # ResourceBusyError si une autre session l'utilise

    def _after_thread(self) -> None:
        if self._lock is not None:
            self._lock.release()

    def _open_params(self) -> list[int]:
        if self.config.kind != SourceKind.NETWORK:
            return []
        params: list[int] = []
        for name, value in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", self._open_timeout_ms),
            ("CAP_PROP_READ_TIMEOUT_MSEC", self._read_timeout_ms),
        ):
            prop = getattr(cv2, name, None)
            if prop is not None and value > 0:
                params += [int(prop), int(value)]
        return params

    def _open_device(self) -> None:
        if self.config.kind == SourceKind.FILE and not Path(self.config.location).is_file():
            raise DeviceOpenError("fichier introuvable")
        try:
            cap = self._factory(self._target, self._api, self._open_params())
        except Exception as exc:  # OpenCV peut lever cv2.error
            raise DeviceOpenError(f"ouverture impossible : {exc}") from exc
        self._cap = cap
        if not cap.isOpened():
            raise DeviceOpenError("source indisponible")
        cfg = self.config
        if cfg.kind != SourceKind.FILE:
            if cfg.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.width))
            if cfg.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.height))
            if cfg.fps:
                cap.set(cv2.CAP_PROP_FPS, float(cfg.fps))
            # Tampon minimal : on veut l'image la plus récente, pas un retard accumulé.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
        self._period_ns = 0
        if cfg.kind == SourceKind.FILE and self.realtime:
            fps = cfg.fps or float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps > 0:
                self._period_ns = int(1e9 / fps)
        self._next_due_ns = time.monotonic_ns()

    def _close_device(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    def _read_image(self) -> npt.NDArray[np.uint8] | None:
        cap = self._cap
        if cap is None:
            return None
        if self._period_ns:
            wait_ns = self._next_due_ns - time.monotonic_ns()
            if wait_ns > 0 and self._wait(wait_ns / 1e9):
                return None
            self._next_due_ns = max(self._next_due_ns + self._period_ns, time.monotonic_ns())
        try:
            ok, image = cap.read()
        except Exception:  # cv2.error : traité comme une lecture ratée
            logger.debug("Lecture en erreur sur %s", self._describe(), exc_info=True)
            ok, image = False, None
        if ok and image is not None and image.size > 0:
            return np.asarray(image, dtype=np.uint8)
        if self.config.kind == SourceKind.FILE:
            if self.loop_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                ok, image = cap.read()
                if ok and image is not None and image.size > 0:
                    return np.asarray(image, dtype=np.uint8)
            raise EndOfStream
        return None
