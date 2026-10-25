"""Côté PROCESSUS VISION : capture et aperçu réduit, publiés vers l'interface.

L'interface ne fait aucun calcul d'image : la réduction de l'aperçu est faite ici.
Ce module n'importe pas Qt.

Deux usages :

1. ``run_preview_worker`` : boucle minimale « source → aperçu + états », utilisée tant
   qu'aucun moteur n'est intégré (l'interface affiche alors « moteur non disponible »).
2. ``VisionPublisher.on_output`` / ``on_state`` (``app.ipc``) : à brancher sur
   ``pipeline.VisionLoop`` par l'intégration quand la chaîne complète existe ;
   ``PreviewTap`` permet d'intercaler l'aperçu sur le flux d'images.

``VisionProcess`` lance un travailleur (``worker``, par défaut ``run_preview_worker`` ;
ex. ``maisonaccueil.runtime.engine:run_engine_worker`` pour la chaîne complète) dans un
processus séparé (contexte ``spawn``), désigné par nom ``module:fonction`` avec des
arguments sérialisables.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import pickle
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from maisonaccueil.contracts import (
    CaptureState,
    CaptureStatus,
    Frame,
    SourceConfig,
    SourceKind,
)
from maisonaccueil.contracts.interfaces import FrameSource
from maisonaccueil.pipeline import ResourceBusyError, ResourceLock

from .ipc import EVENTS_MAXSIZE, PREVIEW_MAXSIZE, VisionPublisher
from .messages import ENGINE_STATUS_UNAVAILABLE, EngineStatus, PreviewImage

log = logging.getLogger(__name__)

SourceFactory = Callable[[SourceConfig], FrameSource]


def make_preview(frame: Frame, max_width: int = 320) -> PreviewImage:
    """Réduit l'image (INTER_AREA) à ``max_width`` pixels de large au plus."""
    h, w = frame.height, frame.width
    if w > max_width:
        new_h = max(1, round(h * max_width / w))
        small = cv2.resize(frame.image, (max_width, new_h), interpolation=cv2.INTER_AREA)
    else:
        small = frame.image
    small = np.ascontiguousarray(small, dtype=np.uint8)
    return PreviewImage(
        frame_id=int(frame.frame_id),
        camera_id=str(frame.camera_id),
        monotonic_ns=frame.monotonic_ns,
        width=int(small.shape[1]),
        height=int(small.shape[0]),
        source_width=w,
        source_height=h,
        data=small.tobytes(),
    )


class PreviewThrottle:
    """Limite la fréquence des aperçus (horloge monotone de l'image)."""

    def __init__(self, max_fps: float = 10.0) -> None:
        self._min_interval_ns = int(1e9 / max_fps) if max_fps > 0 else 0
        self._last_ns: int | None = None

    def accept(self, monotonic_ns: int) -> bool:
        if self._last_ns is not None and monotonic_ns - self._last_ns < self._min_interval_ns:
            return False
        self._last_ns = monotonic_ns
        return True


class PreviewTap:
    """Callable ``on_frame`` : publie un aperçu réduit (limité en fréquence), puis
    transmet l'image à ``downstream`` (ex. la file de ``VisionLoop``) si fourni."""

    def __init__(
        self,
        publisher: VisionPublisher,
        *,
        max_fps: float = 10.0,
        max_width: int = 320,
        downstream: Callable[[Frame], None] | None = None,
    ) -> None:
        self._publisher = publisher
        self._throttle = PreviewThrottle(max_fps)
        self._max_width = max_width
        self._downstream = downstream

    def __call__(self, frame: Frame) -> None:
        if self._throttle.accept(frame.monotonic_ns):
            try:
                self._publisher.publish_preview(make_preview(frame, self._max_width))
            except Exception:
                log.exception("Aperçu impossible pour l'image %s", frame.frame_id)
        if self._downstream is not None:
            self._downstream(frame)


def requires_camera_lock(config: SourceConfig) -> bool:
    """Une webcam physique (USB) se réserve via ``ResourceLock`` (CLAUDE.md, règle 8).

    La source du lot A (``OpenCVSource``) prend elle-même ce verrou dans ``start`` : le
    travailleur ne le prend donc que sur demande (``lock_camera=True``) pour une source
    qui ne le ferait pas (un double verrou dans le même processus échouerait)."""
    return config.kind == SourceKind.USB


def camera_resource_name(config: SourceConfig) -> str:
    loc = config.location.strip()
    return f"camera-{loc}" if loc.isdigit() else "camera-0"


def resolve_callable(spec: str, what: str = "fonction") -> Callable[..., Any]:
    """``"paquet.module:fonction"`` → objet appelable (importé dans le processus appelant)."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"{what} invalide : {spec!r} (attendu module:fonction)")
    target = getattr(importlib.import_module(module_name), attr)
    if not callable(target):
        raise TypeError(f"{spec!r} n'est pas appelable")
    return target  # type: ignore[no-any-return]


def resolve_factory(spec: str) -> SourceFactory:
    """``"paquet.module:fonction"`` → fabrique de ``FrameSource`` (doit être importable
    dans le processus vision, ce qui exige une fonction de niveau module)."""
    return resolve_callable(spec, "Fabrique de source")


def run_preview_worker(
    factory: SourceFactory | str,
    config: SourceConfig,
    events: Any,
    preview: Any,
    stop_event: Any,
    *,
    engine_status: EngineStatus = ENGINE_STATUS_UNAVAILABLE,
    max_preview_fps: float = 10.0,
    preview_width: int = 320,
    lock_owner: str = "lot-B-interface",
    lock_camera: bool = False,
) -> None:
    """Boucle du processus vision sans moteur : source → aperçus + états.

    Bloque jusqu'à ``stop_event.set()``. Publie d'abord ``engine_status``."""
    publisher = VisionPublisher(events, preview)
    publisher.publish(engine_status)
    lock: ResourceLock | None = None
    try:
        if lock_camera and requires_camera_lock(config):
            lock = ResourceLock(camera_resource_name(config), owner=lock_owner)
            try:
                lock.acquire()
            except ResourceBusyError as exc:
                lock = None
                publisher.publish(_failed(config, f"caméra réservée par une autre session : {exc}"))
                stop_event.wait()
                return
        try:
            source_factory = resolve_factory(factory) if isinstance(factory, str) else factory
            source = source_factory(config)
        except Exception as exc:
            publisher.publish(_failed(config, f"source non disponible : {exc}"))
            stop_event.wait()
            return
        tap = PreviewTap(publisher, max_fps=max_preview_fps, max_width=preview_width)
        source.start(on_frame=tap, on_state=publisher.on_state)
        try:
            stop_event.wait()
        finally:
            source.stop(5.0)
            publisher.publish(source.state)
    finally:
        if lock is not None:
            lock.release()


def _failed(config: SourceConfig, detail: str) -> CaptureState:
    return CaptureState(
        camera_id=config.camera_id,
        status=CaptureStatus.FAILED,
        monotonic_ns=time.monotonic_ns(),
        detail=detail,
    )


PREVIEW_WORKER = "maisonaccueil.app.vision_worker:run_preview_worker"

WorkerTarget = Callable[..., None] | str
"""Travailleur vision : ``f(factory, config, events, preview, stop_event, **kwargs)``,
bloquant jusqu'à ``stop_event.set()``. Ex. ``run_preview_worker`` (aperçu seul) ou
``maisonaccueil.runtime.engine:run_engine_worker`` (chaîne complète, coordinateur)."""


def run_named_worker(
    worker: WorkerTarget,
    factory: SourceFactory | str,
    config: SourceConfig,
    events: Any,
    preview: Any,
    stop_event: Any,
    kwargs: dict[str, Any],
) -> None:
    """Point d'entrée du processus fils : résout ``worker`` (``module:fonction``) DANS le
    fils puis l'exécute. Si la cible est introuvable, se replie sur l'aperçu seul en
    publiant « moteur non disponible » avec la raison (aucun résultat inventé)."""
    try:
        target = resolve_callable(worker, "Travailleur vision") if isinstance(worker, str) \
            else worker
    except Exception as exc:
        log.warning("Travailleur vision %r indisponible : %s", worker, exc)
        run_preview_worker(
            factory, config, events, preview, stop_event,
            engine_status=EngineStatus(False, detail=f"travailleur indisponible : {exc}"),
            max_preview_fps=float(kwargs.get("max_preview_fps", 10.0)),
        )
        return
    target(factory, config, events, preview, stop_event, **kwargs)


def check_picklable(value: Any, label: str) -> None:
    """Vérifie dans le processus parent qu'une valeur peut être transmise au fils
    ``spawn`` (erreur claire plutôt qu'un échec au démarrage du fils)."""
    try:
        pickle.dumps(value)
    except Exception as exc:
        raise ValueError(f"{label} non sérialisable pour le processus vision : {exc}") from exc


class VisionProcess:
    """Processus vision séparé (contexte ``spawn``, identique sous Windows et Linux).

    ``factory`` et ``worker`` doivent être des chaînes ``module:fonction`` (ou des
    fonctions de niveau module) ; ``worker_kwargs`` doit être sérialisable (pickle)."""

    def __init__(
        self,
        factory: SourceFactory | str,
        config: SourceConfig,
        *,
        worker: WorkerTarget = PREVIEW_WORKER,
        worker_kwargs: dict[str, Any] | None = None,
        max_preview_fps: float = 10.0,
    ) -> None:
        kwargs = {"max_preview_fps": max_preview_fps, **(worker_kwargs or {})}
        check_picklable((worker, factory, config, kwargs), "Paramètres du travailleur")
        self.worker = worker
        self.worker_kwargs = kwargs
        self._ctx = mp.get_context("spawn")
        self.events: Any = self._ctx.Queue(EVENTS_MAXSIZE)
        self.preview: Any = self._ctx.Queue(PREVIEW_MAXSIZE)
        self._stop = self._ctx.Event()
        self._process = self._ctx.Process(
            target=run_named_worker,
            args=(worker, factory, config, self.events, self.preview, self._stop, kwargs),
            name="maisonaccueil-vision",
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._process.pid is not None:
            self._process.join(timeout_s)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(1.0)
        for q in (self.events, self.preview):
            q.cancel_join_thread()
            q.close()


class InProcessVision:
    """Variante dans un thread (tests, diagnostic) : même travailleur, files ``queue``."""

    def __init__(
        self,
        factory: SourceFactory | str,
        config: SourceConfig,
        events: Any,
        preview: Any,
        *,
        worker: WorkerTarget = PREVIEW_WORKER,
        **kwargs: Any,
    ) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=run_named_worker,
            args=(worker, factory, config, events, preview, self._stop, kwargs),
            name="vision-worker",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout_s)
