"""Boucle de traitement vidéo : relie les lots uniquement via leurs interfaces.

Capture (A) ─► LatestQueue ─► Recognizer (D/E) ─► DecisionEngine (F)
           ─► ArrivalTracker (F) ─► WelcomeComposer (F) ─► AudioPlayer (F)

La boucle tourne dans un thread dédié du processus vision. Les sorties destinées à
l'interface sont publiées via ``on_output`` (à relayer vers une file inter-processus
bornée par l'appelant).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from maisonaccueil.contracts import (
    ArrivalEvent,
    AudioRequest,
    CaptureState,
    Decision,
    Frame,
    RecognitionResult,
    TrackId,
)
from maisonaccueil.contracts.interfaces import (
    ArrivalTracker,
    AudioPlayer,
    DecisionEngine,
    FrameSource,
    Recognizer,
    WelcomeComposer,
)

from .latest_queue import LatestQueue

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VisionOutput:
    """Résumé d'une image traitée, sans pixels (l'aperçu est transmis séparément)."""

    frame_id: int
    monotonic_ns: int
    processing_ms: float
    results: tuple[RecognitionResult, ...] = ()
    decisions: tuple[Decision, ...] = ()
    arrivals: tuple[ArrivalEvent, ...] = ()
    audio: tuple[AudioRequest, ...] = field(default=())
    frames_dropped: int = 0
    lost_tracks: tuple[TrackId, ...] = ()  # passages terminés à cette image (évaluation)


class VisionLoop:
    def __init__(
        self,
        source: FrameSource,
        recognizer: Recognizer,
        decision: DecisionEngine,
        arrivals: ArrivalTracker,
        composer: WelcomeComposer,
        player: AudioPlayer,
        *,
        queue_size: int = 2,
        on_output: Callable[[VisionOutput], None] | None = None,
        on_state: Callable[[CaptureState], None] | None = None,
        on_frame: Callable[[Frame], None] | None = None,
        idle_tick_s: float = 0.5,
    ) -> None:
        self._source = source
        self._recognizer = recognizer
        self._decision = decision
        self._arrivals = arrivals
        self._composer = composer
        self._player = player
        self._queue: LatestQueue[Frame] = LatestQueue(queue_size)
        self._on_output = on_output
        self._on_state = on_state
        self._on_frame = on_frame  # ex. aperçu : appelé dans le thread de capture, doit être rapide
        self._idle_tick_s = idle_tick_s
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    @property
    def frames_dropped(self) -> int:
        return self._queue.dropped

    def process_frame(self, frame: Frame) -> VisionOutput:
        """Traite une image de façon synchrone (utilisé par la boucle et les tests)."""
        t0 = time.perf_counter()
        results = tuple(self._recognizer.process(frame))
        decisions: list[Decision] = []
        events: list[ArrivalEvent] = []
        audio: list[AudioRequest] = []
        for result in results:
            self._arrivals.note_track_activity(result.track_id, result.monotonic_ns)
            d = self._decision.observe(result)
            if d is None:
                continue
            decisions.append(d)
            event = self._arrivals.on_decision(d)
            if event is None:
                continue
            events.append(event)
            request = self._composer.compose(event)
            if request is not None and self._player.submit(request):
                audio.append(request)
        lost = tuple(self._recognizer.lost_tracks())
        for track_id in lost:
            self._decision.track_lost(track_id, frame.monotonic_ns)
            self._arrivals.track_lost(track_id, frame.monotonic_ns)
        self._arrivals.tick(frame.monotonic_ns)
        return VisionOutput(
            frame_id=int(frame.frame_id),
            monotonic_ns=frame.monotonic_ns,
            processing_ms=(time.perf_counter() - t0) * 1000.0,
            results=results,
            decisions=tuple(decisions),
            arrivals=tuple(events),
            audio=tuple(audio),
            frames_dropped=self._queue.dropped,
            lost_tracks=lost,
        )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("VisionLoop déjà démarrée")
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="vision-loop", daemon=True)
        self._thread.start()
        self._source.start(on_frame=self._handle_frame, on_state=self._handle_state)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._running.clear()
        try:
            self._source.stop(timeout_s)
        finally:
            self._queue.close()
            if self._thread is not None:
                self._thread.join(timeout_s)
                self._thread = None

    def _handle_frame(self, frame: Frame) -> None:
        if self._on_frame is not None:
            try:
                self._on_frame(frame)
            except Exception:
                log.exception("Erreur du rappel on_frame (image %s)", frame.frame_id)
        self._queue.put(frame)

    def _handle_state(self, state: CaptureState) -> None:
        if self._on_state is not None:
            self._on_state(state)

    def _run(self) -> None:
        while self._running.is_set():
            frame = self._queue.get(timeout=self._idle_tick_s)
            if frame is None:
                # Pas d'image : les présences doivent quand même expirer.
                self._arrivals.tick(time.monotonic_ns())
                continue
            try:
                output = self.process_frame(frame)
            except Exception:  # un module défaillant ne doit pas tuer la boucle
                log.exception("Erreur de traitement de l'image %s", frame.frame_id)
                continue
            if self._on_output is not None:
                self._on_output(output)
