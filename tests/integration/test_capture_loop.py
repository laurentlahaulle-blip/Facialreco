"""Intégration lot A (source synthétique réelle) + VisionLoop + doubles des lots D/F."""

import threading

from maisonaccueil.capture.synthetic import SyntheticSource
from maisonaccueil.contracts import CameraId, CaptureState, CaptureStatus
from maisonaccueil.pipeline import VisionLoop, VisionOutput

from .fakes import FakeArrivals, FakeComposer, FakeDecision, FakePlayer, FakeRecognizer


class SlowRecognizer(FakeRecognizer):
    """Plus lent que la caméra : force l'abandon d'images anciennes."""

    def process(self, frame):
        threading.Event().wait(0.01)
        return super().process(frame)


def test_synthetic_capture_through_vision_loop_drops_old_frames():
    holder: dict[str, VisionLoop] = {}
    source = SyntheticSource(CameraId("synthetic"), fps=None, max_frames=200,
                             drop_counter=lambda: holder["loop"].frames_dropped)
    outputs: list[VisionOutput] = []
    states: list[CaptureState] = []
    script = {i: [("trk-1", "prf-1", 0.9)] for i in range(0, 201)}
    loop = VisionLoop(source, SlowRecognizer(script), FakeDecision(), FakeArrivals(),
                      FakeComposer(), player := FakePlayer(), queue_size=2,
                      on_output=outputs.append, on_state=states.append, idle_tick_s=0.05)
    holder["loop"] = loop
    loop.start()
    deadline = threading.Event()
    for _ in range(100):
        if any(s.status in (CaptureStatus.STOPPED, CaptureStatus.FAILED) for s in states):
            break
        deadline.wait(0.05)
    loop.stop(timeout_s=3)

    assert outputs, "aucune image traitée"
    assert loop.frames_dropped > 0, "la file bornée aurait dû abandonner des images"
    ids = [o.frame_id for o in outputs]
    assert ids == sorted(ids)
    assert len(player.played) == 1  # un seul accueil pour la présence
    assert any(s.status == CaptureStatus.RUNNING for s in states)
