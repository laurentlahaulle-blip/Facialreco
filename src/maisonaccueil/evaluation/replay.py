"""Rejeu hors ligne : ``Recognizer`` + ``DecisionEngine`` + ``ArrivalTracker`` (+ compositeur)
sur une vidéo de séance (fichier local, JAMAIS committé) ou sur un journal existant.

Le rejeu passe par ``pipeline.VisionLoop.process_frame`` (même enchaînement que l'appli),
sans source réelle ni lecture audio (``SilentPlayer`` accepte et n'émet aucun son).

Temps : en rejeu vidéo, ``Frame.monotonic_ns`` est **réhorodaté** sur le temps de la vidéo
(``(FrameId − 1) / fps``, origine 0). Ainsi les fenêtres temporelles de la décision et les
``t_ms`` du journal sont dans le repère des annotations, indépendamment de la vitesse de
calcul. Le rejeu lit toutes les images (aucun abandon) : il mesure la logique, pas la
latence temps réel ; ``processing_ms`` est journalisé à titre indicatif.
"""

from __future__ import annotations

import queue
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from maisonaccueil.contracts import (
    ArrivalEvent,
    AudioRequest,
    CameraId,
    CaptureState,
    CaptureStatus,
    Frame,
    FrameId,
    ModelRef,
    RecognitionResult,
    SessionId,
    SourceConfig,
    SourceKind,
    TrackId,
    ensure_compatible,
)
from maisonaccueil.contracts.interfaces import (
    ArrivalTracker,
    DecisionEngine,
    Recognizer,
    WelcomeComposer,
)
from maisonaccueil.pipeline.vision_loop import VisionLoop

from .annotations import DataKind
from .runlog import RunHeader, RunLog, RunLogWriter


class NullSource:
    """Source vide : le rejeu appelle ``process_frame`` directement."""

    def __init__(self) -> None:
        self._state = CaptureState(CameraId("replay"), CaptureStatus.STOPPED, 0)

    @property
    def state(self) -> CaptureState:
        return self._state

    def start(
        self, on_frame: Callable[[Frame], None], on_state: Callable[[CaptureState], None]
    ) -> None:
        raise RuntimeError("NullSource ne se démarre pas (rejeu synchrone)")

    def stop(self, timeout_s: float = 5.0) -> None:
        return None


class SilentPlayer:
    """``AudioPlayer`` de rejeu : accepte toutes les demandes, ne joue rien."""

    def __init__(self) -> None:
        self.submitted: list[AudioRequest] = []

    def submit(self, request: AudioRequest) -> bool:
        self.submitted.append(request)
        return True

    def close(self) -> None:
        return None


class NoWelcomeComposer:
    """Compositeur absent : aucune demande audio (les messages sont alors les arrivées)."""

    def compose(self, event: ArrivalEvent) -> AudioRequest | None:
        return None


class _LostTrackRecorder:
    """Enveloppe d'un ``Recognizer`` qui mémorise les passages perdus renvoyés."""

    def __init__(self, inner: Recognizer) -> None:
        self.inner = inner
        self.last_lost: list[TrackId] = []

    def process(self, frame: Frame) -> Sequence[RecognitionResult]:
        return self.inner.process(frame)

    def lost_tracks(self) -> Sequence[TrackId]:
        lost = list(self.inner.lost_tracks())
        self.last_lost = lost
        return lost


@dataclass(frozen=True, slots=True)
class ReplayStats:
    frames: int
    results: int
    decisions: int
    arrivals: int
    welcomes: int
    processing_ms_total: float


def replay_frames(
    frames: Iterable[Frame],
    *,
    recognizer: Recognizer,
    decision: DecisionEngine,
    arrivals: ArrivalTracker,
    writer: RunLogWriter,
    composer: WelcomeComposer | None = None,
) -> ReplayStats:
    """Traite ``frames`` dans l'ordre et journalise tout dans ``writer``.

    Chaque ``RecognitionResult`` doit porter le ``ModelRef`` de l'en-tête du journal
    (``IncompatibleEmbeddingError`` sinon)."""
    rec = _LostTrackRecorder(recognizer)
    player = SilentPlayer()
    loop = VisionLoop(NullSource(), rec, decision, arrivals, composer or NoWelcomeComposer(),
                      player)
    model: ModelRef = writer.header.model
    n = n_res = n_dec = n_arr = n_wel = 0
    total_ms = 0.0
    for frame in frames:
        out = loop.process_frame(frame)
        n += 1
        total_ms += out.processing_ms
        writer.frame(int(frame.frame_id), frame.monotonic_ns, out.processing_ms)
        for r in out.results:
            ensure_compatible(r.model, model)
            writer.result(r)
        for d in out.decisions:
            writer.decision(d)
        for e in out.arrivals:
            writer.arrival(e)
        for a in out.audio:
            writer.welcome(a)
        for t in rec.last_lost:
            writer.track_lost(t, frame.monotonic_ns)
        rec.last_lost = []
        n_res += len(out.results)
        n_dec += len(out.decisions)
        n_arr += len(out.arrivals)
        n_wel += len(out.audio)
    return ReplayStats(n, n_res, n_dec, n_arr, n_wel, total_ms)


def probe_fps(path: Path | str) -> float:
    """Fréquence déclarée par le conteneur vidéo (0 si inconnue)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    finally:
        cap.release()


def iter_video_frames(
    path: Path | str,
    *,
    camera_id: str = "replay",
    fps: float | None = None,
    max_frames: int | None = None,
    start_wall_time: datetime | None = None,
    timeout_s: float = 30.0,
) -> Iterator[Frame]:
    """Images d'un fichier vidéo via la source fichier du lot A (``create_source``),
    lecture non cadencée et SANS abandon (file bloquante), réhorodatées sur le temps
    vidéo : ``monotonic_ns = (FrameId − 1) · 1e9 / fps``.

    ``fps`` : fréquence à utiliser ; à défaut celle du fichier (``ValueError`` si inconnue).
    """
    from maisonaccueil.capture import ReconnectPolicy, create_source

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    rate = fps or probe_fps(p)
    if rate <= 0:
        raise ValueError(f"fréquence d'images inconnue pour {p.name} : préciser fps")
    period_ns = 1e9 / rate
    origin_wall = start_wall_time or datetime(1970, 1, 1, tzinfo=UTC)

    q: queue.Queue[Frame | CaptureState] = queue.Queue(maxsize=8)
    closing = threading.Event()

    def put(item: Frame | CaptureState) -> None:
        while not closing.is_set():
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def on_state(state: CaptureState) -> None:
        if state.status in (CaptureStatus.STOPPED, CaptureStatus.FAILED):
            put(state)

    cfg = SourceConfig(CameraId(camera_id), SourceKind.FILE, str(p))
    source = create_source(
        cfg,
        realtime=False,
        policy=ReconnectPolicy(initial_delay_s=0.0, max_delay_s=0.0, max_attempts=1),
        stats_interval_s=3600.0,
    )
    source.start(put, on_state)
    count = 0
    try:
        while max_frames is None or count < max_frames:
            try:
                item = q.get(timeout=timeout_s)
            except queue.Empty:
                raise TimeoutError(f"aucune image de {p.name} depuis {timeout_s} s") from None
            if isinstance(item, CaptureState):
                if item.status == CaptureStatus.FAILED:
                    raise RuntimeError(f"lecture de {p.name} impossible : {item.detail}")
                return
            index = int(item.frame_id) - 1
            t_ns = round(index * period_ns)
            count += 1
            yield Frame(
                frame_id=item.frame_id,
                camera_id=item.camera_id,
                monotonic_ns=t_ns,
                wall_time=origin_wall + timedelta(microseconds=t_ns / 1000),
                image=item.image,
            )
    finally:
        closing.set()
        source.stop(timeout_s=5.0)


def replay_video(
    video_path: Path | str,
    *,
    session_id: str,
    config_id: str,
    model: ModelRef,
    recognizer: Recognizer,
    decision: DecisionEngine,
    arrivals: ArrivalTracker,
    composer: WelcomeComposer | None = None,
    out_path: Path | str | None = None,
    data_kind: DataKind = DataKind.REAL,
    policy: str = "",
    fps: float | None = None,
    max_frames: int | None = None,
    note: str = "",
) -> tuple[RunLog, ReplayStats]:
    """Rejoue une séance vidéo et écrit le journal ``maisonaccueil.runlog``."""
    header = RunHeader(
        session_id=SessionId(session_id),
        config_id=config_id,
        model=model,
        data_kind=data_kind,
        policy=policy,
        source=f"replay:{Path(video_path).name}",
        origin_monotonic_ns=0,
        messages_from="welcome" if composer is not None else "arrival",
        note=note,
    )
    with RunLogWriter(header, out_path) as writer:
        stats = replay_frames(
            iter_video_frames(video_path, fps=fps, max_frames=max_frames),
            recognizer=recognizer,
            decision=decision,
            arrivals=arrivals,
            composer=composer,
            writer=writer,
        )
    return writer.to_runlog(), stats


class LoggedResultsRecognizer:
    """``Recognizer`` qui restitue les résultats d'un journal (pour rejouer la décision
    seule, ex. balayage de seuils). Les images doivent être fournies dans l'ordre."""

    def __init__(self, log: RunLog) -> None:
        self._by_frame: dict[int, list[RecognitionResult]] = defaultdict(list)
        for r in log.recognition_results():
            self._by_frame[int(r.frame_id)].append(r)
        origin = log.header.origin_monotonic_ns
        self._lost: dict[int, list[TrackId]] = defaultdict(list)
        self._frame_at: dict[int, int] = {}
        for e in log.of("frame"):
            self._frame_at[origin + round(float(e["t_ms"]) * 1e6)] = int(e["frame_id"])
        for e in log.of("track_lost"):
            t = origin + round(float(e["t_ms"]) * 1e6)
            fid = self._frame_at.get(t)
            key = fid if fid is not None else -t  # sans "frame" : rattaché au temps
            self._lost[key].append(TrackId(e["track_id"]))
        self._pending: list[TrackId] = []

    def process(self, frame: Frame) -> Sequence[RecognitionResult]:
        fid = int(frame.frame_id)
        self._pending = self._lost.pop(fid, []) + self._lost.pop(-frame.monotonic_ns, [])
        return list(self._by_frame.get(fid, []))

    def lost_tracks(self) -> Sequence[TrackId]:
        lost, self._pending = self._pending, []
        return lost


def frames_from_log(log: RunLog) -> list[Frame]:
    """Images vides (1×1) aux instants du journal : ``frame`` si présents, sinon instants
    des résultats et pertes de suivi (les ``tick`` ne sont alors appelés qu'à ces instants)."""
    origin = log.header.origin_monotonic_ns
    blank = np.zeros((1, 1, 3), np.uint8)
    wall0 = datetime(1970, 1, 1, tzinfo=UTC)
    points: dict[int, int] = {}  # monotonic_ns -> frame_id
    frames_ev = log.of("frame")
    if frames_ev:
        for e in frames_ev:
            points[origin + round(float(e["t_ms"]) * 1e6)] = int(e["frame_id"])
    else:
        for e in log.of("result"):
            points[origin + round(float(e["t_ms"]) * 1e6)] = int(e["frame_id"])
        for e in log.of("track_lost"):
            t = origin + round(float(e["t_ms"]) * 1e6)
            points.setdefault(t, -1)
    out: list[Frame] = []
    synthetic_id = 10**12
    for t, fid in sorted(points.items()):
        if fid < 0:
            synthetic_id += 1
            fid = synthetic_id
        out.append(Frame(FrameId(fid), CameraId("replay"), t,
                         wall0 + timedelta(microseconds=t / 1000), blank))
    return out


def rerun_decisions(
    log: RunLog,
    *,
    config_id: str,
    decision: DecisionEngine,
    arrivals: ArrivalTracker,
    composer: WelcomeComposer | None = None,
    policy: str = "",
    note: str = "",
) -> RunLog:
    """Rejoue décision + arrivées (+ accueil) sur les résultats d'un journal existant,
    sans vidéo ni moteur de vision. Produit un nouveau journal en mémoire."""
    h = log.header
    header = RunHeader(
        session_id=h.session_id,
        config_id=config_id,
        model=h.model,
        data_kind=h.data_kind,
        policy=policy,
        source=f"rerun:{h.run_id}",
        origin_monotonic_ns=h.origin_monotonic_ns,
        messages_from="welcome" if composer is not None else "arrival",
        note=note,
    )
    writer = RunLogWriter(header)
    replay_frames(frames_from_log(log), recognizer=LoggedResultsRecognizer(log),
                  decision=decision, arrivals=arrivals, composer=composer, writer=writer)
    return writer.to_runlog()
