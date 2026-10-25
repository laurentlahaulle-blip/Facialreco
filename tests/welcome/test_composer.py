from datetime import UTC, datetime

from maisonaccueil.contracts import (
    ArrivalEvent,
    ArrivalKind,
    AudioPriority,
    EventId,
    ProfileId,
    TrackId,
)
from maisonaccueil.contracts.interfaces import WelcomeComposer
from maisonaccueil.welcome import (
    ProfileGreeting,
    TemplateWelcomeComposer,
    UnknownPolicy,
    WelcomeConfig,
)

A = ProfileId("prf-a")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def known(pid=A):
    return ArrivalEvent(EventId("evt-1"), ArrivalKind.KNOWN, TrackId("t1"), 5_000, NOW, pid)


def unknown():
    return ArrivalEvent(EventId("evt-2"), ArrivalKind.UNKNOWN, TrackId("t2"), 7_000, NOW)


def composer(profiles, config=None, existing=frozenset()):
    return TemplateWelcomeComposer(
        profiles.get, config, file_exists=lambda p: p in existing
    )


def test_implements_protocol():
    assert isinstance(composer({}), WelcomeComposer)


def test_profile_audio_file_preferred():
    c = composer({A: ProfileGreeting("Prénom", audio_path="a.wav", text="Salut")},
                 existing={"a.wav"})
    req = c.compose(known())
    assert req.audio_path == "a.wav" and req.text is None
    assert req.profile_id == A and req.dedupe_key == f"arrival:{A}"
    assert req.arrival_event_id == "evt-1" and req.monotonic_ns == 5_000
    assert req.priority == AudioPriority.NORMAL and req.expires_after_ms == 10_000


def test_missing_audio_file_falls_back_to_profile_text():
    c = composer({A: ProfileGreeting("Prénom", audio_path="absent.wav", text="Salut toi")})
    req = c.compose(known())
    assert req.audio_path is None and req.text == "Salut toi"


def test_generic_template_with_name():
    req = composer({A: ProfileGreeting("Prénom")}).compose(known())
    assert req.text == "Bonjour Prénom !"


def test_unknown_profile_or_lookup_error_uses_fallback():
    assert composer({}).compose(known()).text == "Bonjour !"

    def broken(_pid):
        raise RuntimeError("base indisponible")

    req = TemplateWelcomeComposer(broken).compose(known())
    assert req.text == "Bonjour !"


def test_unknown_silent_by_default():
    assert composer({}).compose(unknown()) is None


def test_unknown_text_policy_never_names():
    cfg = WelcomeConfig(unknown_policy=UnknownPolicy.TEXT, unknown_text="Bienvenue.")
    req = composer({}, cfg).compose(unknown())
    assert req.text == "Bienvenue." and req.profile_id is None
    assert req.dedupe_key == "arrival:unknown" and req.priority == AudioPriority.LOW


def test_unknown_audio_policy():
    cfg = WelcomeConfig(unknown_policy=UnknownPolicy.AUDIO, unknown_audio_path="u.wav")
    assert composer({}, cfg, existing={"u.wav"}).compose(unknown()).audio_path == "u.wav"
    # fichier absent : repli sur le texte générique
    assert composer({}, cfg).compose(unknown()).text == cfg.unknown_text
