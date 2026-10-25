"""Intégration lot F réel (décision, présences, accueil, lecteur) dans VisionLoop.

Les scores sont scénarisés (FakeRecognizer) : ce test vérifie l'enchaînement logiciel,
pas la qualité de reconnaissance.
"""

from maisonaccueil.contracts import ProfileId, interfaces
from maisonaccueil.decision import PresenceArrivalTracker, TemporalDecisionEngine
from maisonaccueil.pipeline import VisionLoop
from maisonaccueil.welcome import (
    FakeAudioBackend,
    ProfileGreeting,
    QueuedAudioPlayer,
    TemplateWelcomeComposer,
)

from .fakes import FakeRecognizer, FakeSource, make_frame

PROFILES = {"prf-a": ProfileGreeting(display_name="Profil A"),
            "prf-b": ProfileGreeting(display_name="Profil B")}


def _chain(script, lost=None):
    now = {"ns": 0}
    clock = lambda: now["ns"]  # noqa: E731
    backend = FakeAudioBackend(clock=clock)
    player = QueuedAudioPlayer(backend, clock=clock)
    arrivals = PresenceArrivalTracker()
    composer = TemplateWelcomeComposer(lambda pid: PROFILES.get(pid))
    loop = VisionLoop(FakeSource([]), FakeRecognizer(script, lost), TemporalDecisionEngine(),
                      arrivals, composer, player)
    return loop, now, backend, player, arrivals, composer


def _run(loop, now, frames, step_ms=100):
    for i in range(1, frames + 1):
        now["ns"] = i * step_ms * 1_000_000
        loop.process_frame(make_frame(i, i * step_ms))


def test_real_lot_f_satisfies_interfaces():
    loop, _, _, player, arrivals, composer = _chain({})
    assert isinstance(arrivals, interfaces.ArrivalTracker)
    assert isinstance(composer, interfaces.WelcomeComposer)
    assert isinstance(player, interfaces.AudioPlayer)
    assert isinstance(TemporalDecisionEngine(), interfaces.DecisionEngine)
    player.close()


def test_simultaneous_arrivals_and_no_repeat_across_track_loss():
    script = {i: [("trk-a", "prf-a", 0.9), ("trk-b", "prf-b", 0.8)] for i in range(1, 11)}
    # A perd son suivi à l'image 11 et revient sous une nouvelle piste.
    for i in range(11, 31):
        script[i] = [("trk-a2", "prf-a", 0.9), ("trk-b", "prf-b", 0.8)]
    loop, now, backend, player, arrivals, _ = _chain(script, lost={11: ["trk-a"]})
    _run(loop, now, 30)
    assert player.wait_idle(3)
    welcomed = sorted(r.request.profile_id for r in backend.played)
    assert welcomed == [ProfileId("prf-a"), ProfileId("prf-b")]
    player.close()


def test_unknown_face_is_never_named():
    script = {i: [("trk-x", "prf-a", 0.1)] for i in range(1, 20)}
    loop, now, backend, player, _, _ = _chain(script)
    _run(loop, now, 19)
    assert player.wait_idle(3)
    assert backend.played == []  # politique par défaut : silence pour les inconnus
    player.close()
