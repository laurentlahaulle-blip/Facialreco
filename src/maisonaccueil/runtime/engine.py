"""Processus vision complet : capture (A) → reconnaissance de référence (D) → décision,
arrivées, accueil (F), avec stockage (C) et publication vers l'interface (B).

C'est la seule racine qui assemble les implémentations des lots. Si le moteur de
référence est indisponible (poids absents, licence non acceptée), le processus se
replie sur l'aperçu seul et publie « moteur non disponible » : aucun résultat inventé.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from maisonaccueil.app.ipc import VisionPublisher
from maisonaccueil.app.messages import EngineStatus
from maisonaccueil.app.vision_worker import PreviewTap, resolve_factory, run_preview_worker
from maisonaccueil.contracts import CaptureState, CaptureStatus, ProfileId, SourceConfig
from maisonaccueil.contracts.interfaces import AudioPlayer, FrameSource
from maisonaccueil.pipeline import VisionLoop, VisionOutput

log = logging.getLogger(__name__)

GALLERY_RELOAD_S = 30.0  # prise en compte des nouveaux profils / suppressions


def run_engine_worker(
    factory: Callable[[SourceConfig], FrameSource] | str,
    config: SourceConfig,
    events: Any,
    preview: Any,
    stop_event: Any,
    *,
    db_path: str | Path,
    models_dir: str | Path | None = None,
    player: AudioPlayer | None = None,
    max_preview_fps: float = 10.0,
    gallery_reload_s: float = GALLERY_RELOAD_S,
) -> None:
    """Bloque jusqu'à ``stop_event.set()``. Utilisable dans un processus ``spawn``
    (arguments sérialisables si ``factory`` est une chaîne et ``player`` None)."""
    from maisonaccueil.decision import PresenceArrivalTracker, TemporalDecisionEngine
    from maisonaccueil.recognition.recognizer import build_reference_recognizer
    from maisonaccueil.storage import (
        ArrivalLogRepository,
        Database,
        ProfileRepository,
        SqliteEmbeddingStore,
    )
    from maisonaccueil.welcome import (
        ProfileGreeting,
        QueuedAudioPlayer,
        TemplateWelcomeComposer,
        create_default_backend,
    )

    db = Database(db_path)
    try:
        try:
            recognizer = build_reference_recognizer(
                SqliteEmbeddingStore(db), Path(models_dir) if models_dir else None
            )
        except Exception as exc:
            log.warning("Moteur de référence indisponible : %s", exc)
            db.close()
            run_preview_worker(
                factory, config, events, preview, stop_event,
                engine_status=EngineStatus(False, detail=f"moteur de référence : {exc}"),
                max_preview_fps=max_preview_fps,
            )
            return

        model = recognizer.embedder.model
        profiles = ProfileRepository(db)
        arrival_log = ArrivalLogRepository(db)
        engine = TemporalDecisionEngine()

        def greeting(pid: ProfileId) -> ProfileGreeting | None:
            p = profiles.find(pid)
            if p is None or not p.active:
                return None
            return ProfileGreeting(display_name=p.spoken_name or p.display_name,
                                   audio_path=p.welcome_audio, text=p.welcome_text)

        own_player = player is None
        audio = player or QueuedAudioPlayer(create_default_backend())
        publisher = VisionPublisher(events, preview)
        last_reload = [time.monotonic()]

        def on_output(out: VisionOutput) -> None:
            for ev in out.arrivals:
                try:
                    arrival_log.record(ev, model_key=model.key,
                                       policy=engine.policy.policy_id)
                except Exception:
                    log.exception("Journal d'arrivée impossible")
            publisher.on_output(out)
            if time.monotonic() - last_reload[0] >= gallery_reload_s:
                last_reload[0] = time.monotonic()
                recognizer.gallery.reload()

        try:
            source_factory = resolve_factory(factory) if isinstance(factory, str) else factory
            source = source_factory(config)
        except Exception as exc:
            publisher.publish(CaptureState(config.camera_id, CaptureStatus.FAILED,
                                           time.monotonic_ns(), f"source non disponible : {exc}"))
            stop_event.wait()
            return

        publisher.publish(EngineStatus(True, model=model, detail="référence (YuNet + SFace)"))
        loop = VisionLoop(
            source, recognizer, engine, PresenceArrivalTracker(),
            TemplateWelcomeComposer(greeting), audio,
            on_output=on_output, on_state=publisher.on_state,
            on_frame=PreviewTap(publisher, max_fps=max_preview_fps),
        )
        loop.start()
        try:
            stop_event.wait()
        finally:
            loop.stop(5.0)
            publisher.publish(source.state)
            if own_player:
                audio.close()
    finally:
        db.close()
