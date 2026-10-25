"""Lecteur audio non bloquant : thread dédié, file à priorité, déduplication, expiration.

Arrivées simultanées : les messages sont ENCHAÎNÉS (pas regroupés). Ordre : priorité
décroissante puis ordre d'arrivée. Raisons :
- les messages personnalisés préenregistrés (WAV) ne se fusionnent pas ;
- ``AudioRequest`` ne porte pas le nom affichable, le lecteur ne peut donc pas
  construire « Bonjour A et B » sans recomposer le texte (voir
  docs/contract-changes/F-audio-grouping.md).
Une demande dont le délai ``expires_after_ms`` (mesuré depuis ``request.monotonic_ns``)
est dépassé au moment d'être jouée est abandonnée : un accueil périmé n'est pas dit.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from maisonaccueil.contracts import AudioRequest

from .backends import AudioBackend

log = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000


@dataclass(frozen=True, slots=True)
class PlayerStats:
    submitted: int = 0
    played: int = 0
    deduplicated: int = 0
    expired: int = 0
    rejected: int = 0  # file pleine ou lecteur fermé
    evicted: int = 0  # retirées de la file au profit d'une demande plus prioritaire
    errors: int = 0
    dropped_on_close: int = 0


class QueuedAudioPlayer:
    """Implémentation de ``contracts.AudioPlayer``. ``submit`` est thread-safe et ne
    bloque jamais sur la lecture."""

    def __init__(
        self,
        backend: AudioBackend,
        clock: Callable[[], int] = time.monotonic_ns,
        max_queue: int = 8,
        on_played: Callable[[AudioRequest], None] | None = None,
        name: str = "maisonaccueil-audio",
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue doit être >= 1")
        self._backend = backend
        self._clock = clock
        self._max_queue = max_queue
        self._on_played = on_played
        self._cond = threading.Condition()
        self._heap: list[tuple[int, int, AudioRequest]] = []
        self._seq = itertools.count()
        self._current: AudioRequest | None = None
        self._closed = False
        self._stats: dict[str, int] = dict.fromkeys(PlayerStats.__slots__, 0)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- interface AudioPlayer ----------------------------------------------------
    def submit(self, request: AudioRequest) -> bool:
        with self._cond:
            self._stats["submitted"] += 1
            if self._closed:
                self._stats["rejected"] += 1
                return False
            if self._is_expired(request):
                self._stats["expired"] += 1
                return False
            key = request.dedupe_key
            if (self._current is not None and self._current.dedupe_key == key) or any(
                r.dedupe_key == key for _, _, r in self._heap
            ):
                self._stats["deduplicated"] += 1
                return False
            if len(self._heap) >= self._max_queue and not self._evict_for(request):
                self._stats["rejected"] += 1
                return False
            heapq.heappush(self._heap, (-int(request.priority), next(self._seq), request))
            self._cond.notify_all()
            return True

    def close(self, timeout_s: float = 2.0, interrupt: bool = True) -> None:
        """Idempotent. Abandonne la file ; interrompt la lecture en cours si possible."""
        with self._cond:
            if not self._closed:
                self._closed = True
                self._stats["dropped_on_close"] += len(self._heap)
                self._heap.clear()
                self._cond.notify_all()
            playing = self._current is not None
        if interrupt and playing:
            try:
                self._backend.stop()
            except Exception:
                log.exception("Arrêt du backend audio impossible")
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout_s)

    # -- outils -------------------------------------------------------------------
    @property
    def stats(self) -> PlayerStats:
        with self._cond:
            return PlayerStats(**self._stats)

    @property
    def pending(self) -> list[AudioRequest]:
        with self._cond:
            return [r for _, _, r in sorted(self._heap)]

    def wait_idle(self, timeout_s: float = 5.0) -> bool:
        """Attend que la file soit vide et que rien ne soit en lecture."""
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._heap and self._current is None, timeout_s
            )

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # -- interne ------------------------------------------------------------------
    def _is_expired(self, request: AudioRequest) -> bool:
        age = self._clock() - request.monotonic_ns
        return age > request.expires_after_ms * _NS_PER_MS

    def _evict_for(self, request: AudioRequest) -> bool:
        # Victime : priorité la plus basse, puis la plus récente.
        victim = max(self._heap, key=lambda item: (item[0], item[1]))
        if -victim[0] >= int(request.priority):
            return False
        self._heap.remove(victim)
        heapq.heapify(self._heap)
        self._stats["evicted"] += 1
        return True

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._heap or self._closed)
                if self._closed:
                    return
                _, _, request = heapq.heappop(self._heap)
                if self._is_expired(request):
                    self._stats["expired"] += 1
                    self._cond.notify_all()
                    continue
                self._current = request
            ok = False
            try:
                self._backend.play(request)
                ok = True
            except Exception:
                log.exception("Lecture audio échouée (%s)", request.dedupe_key)
            with self._cond:
                self._current = None
                self._stats["played" if ok else "errors"] += 1
                self._cond.notify_all()
            if ok and self._on_played is not None:
                try:
                    self._on_played(request)
                except Exception:
                    log.exception("Callback on_played en erreur")
