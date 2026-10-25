"""Chaîne réelle A + C + D + F dans VisionLoop, sur images SYNTHÉTIQUES.

Vérifie l'assemblage logiciel uniquement (aucun visage réel, aucune mesure de qualité).
Ignoré si les poids de référence (lot D) sont absents.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from maisonaccueil.capture.synthetic import SyntheticSource
from maisonaccueil.contracts import CameraId, Embedding, Metric, ModelRef, interfaces
from maisonaccueil.decision import PresenceArrivalTracker, TemporalDecisionEngine
from maisonaccueil.pipeline import VisionLoop
from maisonaccueil.storage import Database, ProfileRepository, SqliteEmbeddingStore
from maisonaccueil.welcome import FakeAudioBackend, QueuedAudioPlayer, TemplateWelcomeComposer

recognizer_mod = pytest.importorskip("maisonaccueil.recognition.recognizer")
model_files = pytest.importorskip("maisonaccueil.vision.model_files")


@pytest.fixture
def recognizer_factory():
    try:
        model_files.resolve_verified(model_files.SFACE_2021DEC, None)
        model_files.resolve_verified(model_files.YUNET_2023MAR, None)
    except Exception as exc:  # poids absents ou altérés
        pytest.skip(f"poids de référence indisponibles : {exc}")
    return recognizer_mod.build_reference_recognizer


def test_full_reference_chain_on_synthetic_frames(tmp_path, recognizer_factory):
    from maisonaccueil.recognition import SFACE_MODEL_REF

    db = Database(tmp_path / "chain.sqlite")
    profiles = ProfileRepository(db)
    store = SqliteEmbeddingStore(db)
    p = profiles.create("Profil test", consent_at=datetime.now(UTC))
    rng = np.random.default_rng(0)
    vec = rng.normal(size=SFACE_MODEL_REF.embedding_dim).astype(np.float32)
    store.add_embedding(p.profile_id, None, Embedding(SFACE_MODEL_REF, vec / np.linalg.norm(vec)))
    # Représentation d'un autre modèle : ne doit jamais être chargée par la galerie SFace.
    other = ModelRef("autre", "x", 4, Metric.COSINE, "p", "test-double")
    store.add_embedding(p.profile_id, None, Embedding(other, np.ones(4, np.float32)))

    recognizer = recognizer_factory(store)
    assert isinstance(recognizer, interfaces.Recognizer)
    backend = FakeAudioBackend()
    player = QueuedAudioPlayer(backend)
    source = SyntheticSource(CameraId("synthetic"), fps=None, max_frames=15)
    outputs = []
    loop = VisionLoop(source, recognizer, TemporalDecisionEngine(), PresenceArrivalTracker(),
                      TemplateWelcomeComposer(lambda pid: None), player,
                      on_output=outputs.append)
    frames = []
    source.start(on_frame=frames.append, on_state=lambda s: None)
    for _ in range(100):
        if len(frames) >= 15:
            break
        __import__("time").sleep(0.02)
    source.stop()
    for f in frames:
        outputs.append(loop.process_frame(f))

    assert len(outputs) == len(frames) >= 15
    # Images synthétiques sans visage : aucune décision, aucun accueil.
    assert all(not o.decisions and not o.arrivals for o in outputs)
    assert player.wait_idle(2) and backend.played == []
    player.close()
    db.close()
