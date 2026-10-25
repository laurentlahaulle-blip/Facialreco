"""Socle commun des sources d'images : thread de capture, états, reconnexion.

Une sous-classe fournit seulement trois opérations exécutées DANS le thread de capture :
``_open_device`` (ouvrir), ``_read_image`` (lire une image) et ``_close_device`` (libérer).
Le socle gère :

- le thread dédié (``start`` non bloquant, ``stop`` idempotent avec délai) ;
- la publication des ``CaptureState`` via ``on_state`` ;
- la reconnexion automatique avec attente exponentielle bornée, puis ``FAILED`` après
  ``ReconnectPolicy.max_attempts`` tentatives consécutives ;
- la numérotation croissante des ``FrameId`` (conservée à travers les reconnexions),
  l'horodatage monotone (``time.monotonic_ns``) et UTC ;
- la normalisation des images en BGR uint8 (H, W, 3).

Aucune image n'est écrite sur disque.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import CameraId, CaptureState, CaptureStatus, Frame, FrameId

logger = logging.getLogger(__name__)

OnFrame = Callable[[Frame], None]
OnState = Callable[[CaptureState], None]
DropCounter = Callable[[], int]


class EndOfStream(Exception):  # noqa: N818 - signal de contrôle, pas une erreur
    """Levée par ``_read_image`` quand la source est terminée (fin de fichier, quota)."""


class DeviceOpenError(Exception):
    """Levée par ``_open_device`` quand la source ne peut pas être ouverte."""


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    """Politique de reconnexion.

    ``max_attempts`` : tentatives d'ouverture consécutives échouées avant ``FAILED``
    (le compteur repart à zéro dès qu'une image est lue).
    ``max_consecutive_read_failures`` : lectures ratées d'affilée avant de considérer
    la source comme coupée.
    """

    initial_delay_s: float = 0.5
    max_delay_s: float = 10.0
    factor: float = 2.0
    max_attempts: int = 8
    max_consecutive_read_failures: int = 5

    def __post_init__(self) -> None:
        if self.initial_delay_s < 0 or self.max_delay_s < self.initial_delay_s:
            raise ValueError("Délais de reconnexion invalides")
        if self.factor < 1.0:
            raise ValueError("factor doit être >= 1")
        if self.max_attempts < 1 or self.max_consecutive_read_failures < 1:
            raise ValueError("max_attempts et max_consecutive_read_failures doivent être >= 1")

    def delay_for(self, attempt: int) -> float:
        """Attente avant la tentative ``attempt`` (1 = première nouvelle tentative)."""
        if attempt <= 0:
            return 0.0
        return float(min(self.max_delay_s, self.initial_delay_s * self.factor ** (attempt - 1)))

    def delays(self) -> Iterator[float]:
        for attempt in range(1, self.max_attempts + 1):
            yield self.delay_for(attempt)


def to_bgr_uint8(image: npt.NDArray[np.generic]) -> npt.NDArray[np.uint8]:
    """Convertit une image (gris, BGR, BGRA) en BGR uint8 contigu (H, W, 3)."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Type d'image non pris en charge : {arr.dtype}")
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif not (arr.ndim == 3 and arr.shape[2] == 3):
        raise ValueError(f"Forme d'image non prise en charge : {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.uint8)


class ThreadedFrameSource(ABC):
    """Implémentation de ``contracts.FrameSource`` fondée sur un thread de capture."""

    def __init__(
        self,
        camera_id: CameraId,
        *,
        policy: ReconnectPolicy | None = None,
        drop_counter: DropCounter | None = None,
        stats_interval_s: float = 1.0,
        thread_name: str | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.policy = policy or ReconnectPolicy()
        self._drop_counter = drop_counter
        self._stats_interval_ns = int(stats_interval_s * 1e9)
        self._thread_name = thread_name or f"capture-{camera_id}"

        self._lifecycle = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_frame: OnFrame | None = None
        self._on_state: OnState | None = None

        self._next_frame_id = 1
        self._frames_captured = 0
        self._callback_errors = 0
        self._state = CaptureState(camera_id, CaptureStatus.STOPPED, time.monotonic_ns())

    # ------------------------------------------------------------------ hooks
    def _before_start(self) -> None:  # noqa: B027 - hook optionnel
        """Appelé dans le thread de l'appelant avant le démarrage (ex. réservation)."""

    def _after_thread(self) -> None:  # noqa: B027 - hook optionnel
        """Appelé à la fin du thread de capture (ex. libération de la réservation)."""

    @abstractmethod
    def _open_device(self) -> None:
        """Ouvre la source ; lève ``DeviceOpenError`` en cas d'échec."""

    @abstractmethod
    def _read_image(self) -> npt.NDArray[np.generic] | None:
        """Lit une image. ``None`` = lecture ratée (transitoire) ; ``EndOfStream`` = fin."""

    @abstractmethod
    def _close_device(self) -> None:
        """Libère la source. Doit être sans effet si elle n'est pas ouverte."""

    def _describe(self) -> str:
        """Description lisible de la source (sans secret) pour les messages d'état."""
        return str(self.camera_id)

    # ------------------------------------------------------------ public API
    @property
    def state(self) -> CaptureState:
        with self._state_lock:
            return self._state

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    @property
    def callback_errors(self) -> int:
        """Nombre d'exceptions levées par ``on_frame`` (ignorées et journalisées)."""
        return self._callback_errors

    def start(self, on_frame: OnFrame, on_state: OnState) -> None:
        """Démarre le thread de capture et rend la main immédiatement."""
        with self._lifecycle:
            if self.is_running:
                raise RuntimeError(f"La source {self._describe()} est déjà démarrée")
            self._before_start()
            self._on_frame = on_frame
            self._on_state = on_state
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name=self._thread_name, daemon=True
            )
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                self._after_thread()
                raise

    def stop(self, timeout_s: float = 5.0) -> None:
        """Demande l'arrêt et attend la fin du thread (idempotent).

        Appelé depuis un rappel (``on_frame``/``on_state``, donc dans le thread de
        capture), il demande l'arrêt sans attendre."""
        self._stop_event.set()
        if self._thread is threading.current_thread():
            return
        with self._lifecycle:
            self._stop_event.set()
            thread = self._thread
            if thread is None:
                return
            thread.join(timeout_s)
            if thread.is_alive():
                logger.warning(
                    "Le thread de capture %s ne s'est pas arrêté en %.1f s "
                    "(lecture bloquée ?) ; il s'arrêtera à la prochaine image.",
                    self._describe(),
                    timeout_s,
                )
                return
            self._thread = None

    # -------------------------------------------------------------- internals
    def _dropped(self) -> int:
        if self._drop_counter is None:
            return 0
        try:
            return int(self._drop_counter())
        except Exception:  # pragma: no cover - compteur défaillant : on n'interrompt pas
            return 0

    def _publish(self, status: CaptureStatus, detail: str = "") -> None:
        state = CaptureState(
            camera_id=self.camera_id,
            status=status,
            monotonic_ns=time.monotonic_ns(),
            detail=detail,
            frames_captured=self._frames_captured,
            frames_dropped=self._dropped(),
        )
        with self._state_lock:
            self._state = state
        cb = self._on_state
        if cb is not None:
            try:
                cb(state)
            except Exception:
                logger.exception("on_state a levé une exception (ignorée)")

    def _emit(self, raw: npt.NDArray[np.generic]) -> None:
        now_ns = time.monotonic_ns()
        image = to_bgr_uint8(raw)
        frame = Frame(
            frame_id=FrameId(self._next_frame_id),
            camera_id=self.camera_id,
            monotonic_ns=now_ns,
            wall_time=datetime.now(UTC),
            image=image,
        )
        self._next_frame_id += 1
        self._frames_captured += 1
        cb = self._on_frame
        if cb is None:
            return
        try:
            cb(frame)
        except Exception:
            self._callback_errors += 1
            logger.exception("on_frame a levé une exception (image ignorée)")

    def _wait(self, seconds: float) -> bool:
        """Attend en restant interruptible. Retourne True si un arrêt est demandé."""
        return self._stop_event.wait(seconds) if seconds > 0 else self._stop_event.is_set()

    def _safe_close(self) -> None:
        try:
            self._close_device()
        except Exception:
            logger.exception("Erreur lors de la libération de %s", self._describe())

    def _run(self) -> None:
        final_status = CaptureStatus.STOPPED
        final_detail = "arrêt demandé"
        try:
            final_status, final_detail = self._loop()
        except Exception as exc:  # erreur inattendue : jamais de thread qui meurt en silence
            logger.exception("Erreur inattendue dans la capture %s", self._describe())
            final_status, final_detail = CaptureStatus.FAILED, f"erreur inattendue : {exc}"
        finally:
            self._safe_close()
            try:
                self._after_thread()
            finally:
                self._publish(final_status, final_detail)

    def _loop(self) -> tuple[CaptureStatus, str]:
        policy = self.policy
        failures = 0  # tentatives consécutives sans image
        self._publish(CaptureStatus.CONNECTING, f"ouverture de {self._describe()}")
        while not self._stop_event.is_set():
            try:
                self._open_device()
            except DeviceOpenError as exc:
                self._safe_close()
                failures += 1
                if failures >= policy.max_attempts:
                    return CaptureStatus.FAILED, (
                        f"abandon après {failures} tentatives : {exc}"
                    )
                delay = policy.delay_for(failures)
                self._publish(
                    CaptureStatus.RECONNECTING,
                    f"échec d'ouverture ({exc}) ; nouvelle tentative "
                    f"{failures + 1}/{policy.max_attempts} dans {delay:.1f} s",
                )
                if self._wait(delay):
                    break
                continue

            outcome = self._stream()
            self._safe_close()
            if outcome == "stopped":
                break
            if outcome == "eos":
                return CaptureStatus.STOPPED, "fin du flux"
            # Coupure : si au moins une image a été lue pendant ce cycle, on repart à 0.
            if outcome == "lost_after_frames":
                failures = 0
            failures += 1
            if failures >= policy.max_attempts:
                return CaptureStatus.FAILED, f"abandon après {failures} tentatives : flux coupé"
            delay = policy.delay_for(failures)
            self._publish(
                CaptureStatus.RECONNECTING,
                f"flux coupé ; nouvelle tentative {failures + 1}/{policy.max_attempts} "
                f"dans {delay:.1f} s",
            )
            if self._wait(delay):
                break
        return CaptureStatus.STOPPED, "arrêt demandé"

    def _stream(self) -> str:
        """Lit des images jusqu'à arrêt, fin de flux ou coupure.

        Retourne ``"stopped"``, ``"eos"``, ``"lost_after_frames"`` ou ``"lost"``.
        """
        read_failures = 0
        got_frame = False
        last_stats = time.monotonic_ns()
        while not self._stop_event.is_set():
            try:
                raw = self._read_image()
            except EndOfStream:
                return "eos"
            if self._stop_event.is_set():
                return "stopped"
            if raw is None:
                read_failures += 1
                if read_failures >= self.policy.max_consecutive_read_failures:
                    return "lost_after_frames" if got_frame else "lost"
                continue
            read_failures = 0
            if not got_frame:
                got_frame = True
                self._publish(CaptureStatus.RUNNING, f"capture active : {self._describe()}")
                last_stats = time.monotonic_ns()
            self._emit(raw)
            now = time.monotonic_ns()
            if self._stats_interval_ns > 0 and now - last_stats >= self._stats_interval_ns:
                last_stats = now
                self._publish(CaptureStatus.RUNNING, "statistiques")
        return "stopped"
