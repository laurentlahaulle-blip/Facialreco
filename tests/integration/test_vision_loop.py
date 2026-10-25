import threading

from maisonaccueil.contracts import interfaces
from maisonaccueil.pipeline.vision_loop import VisionLoop, VisionOutput

from .fakes import (
    FakeArrivals,
    FakeComposer,
    FakeDecision,
    FakePlayer,
    FakeRecognizer,
    FakeSource,
    make_frame,
)


def _loop(frames, script, lost=None, **kw):
    parts = dict(source=FakeSource(frames), recognizer=FakeRecognizer(script, lost),
                 decision=FakeDecision(), arrivals=FakeArrivals(), composer=FakeComposer(),
                 player=FakePlayer())
    return VisionLoop(**parts, **kw), parts


def test_fakes_satisfy_interfaces():
    _, p = _loop([], {})
    assert isinstance(p["source"], interfaces.FrameSource)
    assert isinstance(p["recognizer"], interfaces.Recognizer)
    assert isinstance(p["decision"], interfaces.DecisionEngine)
    assert isinstance(p["arrivals"], interfaces.ArrivalTracker)
    assert isinstance(p["composer"], interfaces.WelcomeComposer)
    assert isinstance(p["player"], interfaces.AudioPlayer)


def test_single_welcome_per_presence_and_simultaneous_arrivals():
    script = {i: [("trk-a", "prf-alice", 0.9), ("trk-b", "prf-bob", 0.8)] for i in range(1, 6)}
    # Alice perd son suivi à l'image 3 puis réapparaît avec une nouvelle piste.
    script[4] = [("trk-a2", "prf-alice", 0.9)]
    script[5] = [("trk-a2", "prf-alice", 0.9)]
    loop, p = _loop([], script, lost={3: ["trk-a"]})
    for i in range(1, 6):
        loop.process_frame(make_frame(i, i * 100))
    played = [r.profile_id for r in p["player"].played]
    assert sorted(played) == ["prf-alice", "prf-bob"]
    assert p["decision"].lost == ["trk-a"]


def test_low_scores_never_welcome():
    loop, p = _loop([], {i: [("trk-x", "prf-x", 0.2)] for i in range(1, 10)})
    for i in range(1, 10):
        loop.process_frame(make_frame(i, i * 100))
    assert p["player"].played == []


def test_threaded_loop_runs_and_stops_cleanly():
    frames = [make_frame(i, i * 40) for i in range(1, 50)]
    outputs: list[VisionOutput] = []
    got = threading.Event()

    def on_output(o: VisionOutput) -> None:
        outputs.append(o)
        got.set()

    loop, p = _loop(frames, {i: [("trk-a", "prf-alice", 0.9)] for i in range(1, 50)},
                    on_output=on_output, queue_size=2, idle_tick_s=0.05)
    loop.start()
    assert p["source"].done.wait(2) and got.wait(2)
    loop.stop(timeout_s=2)
    assert outputs and all(o.frames_dropped >= 0 for o in outputs)
    # Abandon des anciennes images possible, mais jamais plus d'un accueil.
    assert len(p["player"].played) <= 1
    assert len(outputs) + loop.frames_dropped <= len(frames)
