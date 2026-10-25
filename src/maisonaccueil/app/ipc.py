"""Files bornées entre le processus vision et le processus interface.

Deux canaux :

- ``events``  : ``VisionOutput``, ``CaptureState``, ``EngineStatus``, ``EnrollmentSample``.
  Borné ; si plein, le message est abandonné et compté (la vision ne bloque jamais sur
  l'interface), sauf un ``VisionOutput`` portant des arrivées, qui bénéficie d'un court
  délai d'attente avant abandon.
- ``preview`` : ``PreviewImage`` uniquement, très petit (2) ; l'interface ne garde que
  l'image la plus récente.

Les files doivent offrir ``put(item, timeout=...)``, ``put_nowait`` et ``get_nowait``
(``multiprocessing.Queue`` ou ``queue.Queue``).
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from typing import Any, Protocol

from maisonaccueil.contracts import CaptureState
from maisonaccueil.pipeline import VisionOutput

from .messages import EngineStatus, EnrollmentSample, PreviewImage

UiEvent = VisionOutput | CaptureState | EngineStatus | EnrollmentSample

EVENTS_MAXSIZE = 256
PREVIEW_MAXSIZE = 2


class QueueLike(Protocol):
    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None: ...
    def put_nowait(self, item: Any) -> None: ...
    def get_nowait(self) -> Any: ...


class VisionPublisher:
    """Côté vision : publication NON bloquante vers l'interface."""

    def __init__(
        self, events: QueueLike, preview: QueueLike, *, arrival_timeout_s: float = 0.2
    ) -> None:
        self._events = events
        self._preview = preview
        self._arrival_timeout_s = arrival_timeout_s
        self.events_lost = 0
        self.previews_lost = 0

    def publish(self, message: UiEvent) -> bool:
        try:
            self._events.put_nowait(message)
            return True
        except queue.Full:
            pass
        if isinstance(message, VisionOutput) and message.arrivals:
            try:
                self._events.put(message, True, self._arrival_timeout_s)
                return True
            except queue.Full:
                pass
        self.events_lost += 1
        return False

    def publish_preview(self, image: PreviewImage) -> bool:
        try:
            self._preview.put_nowait(image)
            return True
        except queue.Full:
            self.previews_lost += 1
            return False

    # Adaptateurs directs pour ``VisionLoop(on_output=..., on_state=...)``.
    def on_output(self, output: VisionOutput) -> None:
        self.publish(output)

    def on_state(self, state: CaptureState) -> None:
        self.publish(state)


@dataclass(slots=True)
class InboxBatch:
    events: list[UiEvent] = field(default_factory=list)
    preview: PreviewImage | None = None
    unexpected: int = 0  # messages d'un type inattendu, ignorés


class UiInbox:
    """Côté interface : lecture non bloquante, appelée périodiquement (minuterie Qt)."""

    def __init__(self, events: QueueLike, preview: QueueLike) -> None:
        self._events = events
        self._preview = preview

    def drain(self, max_events: int = 200) -> InboxBatch:
        batch = InboxBatch()
        for _ in range(max_events):
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError, ValueError):  # file fermée / processus mort
                break
            if isinstance(item, VisionOutput | CaptureState | EngineStatus | EnrollmentSample):
                batch.events.append(item)
            else:
                batch.unexpected += 1
        while True:
            try:
                item = self._preview.get_nowait()
            except (queue.Empty, EOFError, OSError, ValueError):
                break
            if isinstance(item, PreviewImage):
                batch.preview = item
            else:
                batch.unexpected += 1
        return batch
