"""Lot F — composition des messages d'accueil et lecture audio locale."""

from .backends import (
    AudioBackend,
    FakeAudioBackend,
    LoggingAudioBackend,
    PlayedRecord,
    Pyttsx3Backend,
    RoutingAudioBackend,
    UnsupportedAudioRequest,
    WinsoundBackend,
    create_default_backend,
)
from .composer import (
    ProfileGreeting,
    ProfileGreetingLookup,
    TemplateWelcomeComposer,
    UnknownPolicy,
    WelcomeConfig,
)
from .player import PlayerStats, QueuedAudioPlayer

__all__ = [
    "AudioBackend",
    "FakeAudioBackend",
    "LoggingAudioBackend",
    "PlayedRecord",
    "PlayerStats",
    "ProfileGreeting",
    "ProfileGreetingLookup",
    "Pyttsx3Backend",
    "QueuedAudioPlayer",
    "RoutingAudioBackend",
    "TemplateWelcomeComposer",
    "UnknownPolicy",
    "UnsupportedAudioRequest",
    "WelcomeConfig",
    "WinsoundBackend",
    "create_default_backend",
]
