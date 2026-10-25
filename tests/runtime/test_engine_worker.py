"""Processus vision complet (runtime) dans un thread, sur source SYNTHÉTIQUE.

Vérifie l'assemblage et le repli « moteur non disponible » ; aucune mesure de qualité.
"""

import queue
import threading
import time

import pytest

from maisonaccueil.app.messages import EngineStatus, PreviewImage
from maisonaccueil.contracts import CameraId, CaptureState, SourceConfig, SourceKind
from maisonaccueil.pipeline import VisionOutput
from maisonaccueil.runtime.engine import run_engine_worker
from maisonaccueil.vision import model_files

CFG = SourceConfig(CameraId("synthetic"), SourceKind.SYNTHETIC, "test", 320, 240, 30.0)


def synthetic_factory(config: SourceConfig):
    from maisonaccueil.capture.synthetic import SyntheticSource

    return SyntheticSource(config.camera_id, fps=config.fps)


def _run(tmp_path, models_dir, seconds=1.0):
    events: queue.Queue = queue.Queue(500)
    preview: queue.Queue = queue.Queue(5)
    stop = threading.Event()
    t = threading.Thread(target=run_engine_worker,
                         args=(synthetic_factory, CFG, events, preview, stop),
                         kwargs={"db_path": tmp_path / "rt.sqlite", "models_dir": models_dir},
                         daemon=True)
    t.start()
    time.sleep(seconds)
    stop.set()
    t.join(10)
    assert not t.is_alive()
    got = []
    while not events.empty():
        got.append(events.get_nowait())
    previews = []
    while not preview.empty():
        previews.append(preview.get_nowait())
    return got, previews


def test_falls_back_to_preview_when_weights_missing(tmp_path):
    got, previews = _run(tmp_path, tmp_path / "vide")
    status = [m for m in got if isinstance(m, EngineStatus)]
    assert status and status[0].available is False
    assert not any(isinstance(m, VisionOutput) for m in got)  # aucun résultat inventé
    assert previews and isinstance(previews[0], PreviewImage)


def test_full_engine_on_synthetic_frames(tmp_path):
    try:
        model_files.resolve_verified(model_files.SFACE_2021DEC, None)
        model_files.resolve_verified(model_files.YUNET_2023MAR, None)
    except Exception as exc:
        pytest.skip(f"poids de référence indisponibles : {exc}")
    got, previews = _run(tmp_path, None, seconds=1.5)
    status = [m for m in got if isinstance(m, EngineStatus)]
    assert status and status[0].available and status[0].model is not None
    outputs = [m for m in got if isinstance(m, VisionOutput)]
    assert outputs and all(not o.arrivals for o in outputs)  # pas de visage synthétique
    assert any(isinstance(m, CaptureState) for m in got)
    assert previews
