"""Composition du message d'accueil d'une arrivée.

Les données de profil sont fournies par un callable injecté (``ProfileGreetingLookup``),
par exemple adossé au stockage du lot C par l'intégration. Aucun nom n'est jamais
prononcé pour un inconnu.
"""

from __future__ import annotations

import enum
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from maisonaccueil.contracts import (
    ArrivalEvent,
    ArrivalKind,
    AudioPriority,
    AudioRequest,
    EventId,
    ProfileId,
    new_event_id,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProfileGreeting:
    """Ce que le composer a besoin de savoir d'un profil."""

    display_name: str | None = None
    audio_path: str | None = None  # message préenregistré local (WAV)
    text: str | None = None  # message texte personnalisé (synthèse locale)


ProfileGreetingLookup = Callable[[ProfileId], ProfileGreeting | None]


class UnknownPolicy(enum.StrEnum):
    SILENT = "silent"  # défaut : rien n'est dit
    TEXT = "text"  # message générique sans nom
    AUDIO = "audio"  # fichier local générique


@dataclass(frozen=True, slots=True)
class WelcomeConfig:
    known_template: str = "Bonjour {name} !"
    known_fallback_text: str = "Bonjour !"  # profil sans nom ni message
    unknown_policy: UnknownPolicy = UnknownPolicy.SILENT
    unknown_text: str = "Bonjour, bienvenue."
    unknown_audio_path: str | None = None
    known_priority: AudioPriority = AudioPriority.NORMAL
    unknown_priority: AudioPriority = AudioPriority.LOW
    expires_after_ms: int = 10_000
    check_audio_exists: bool = True  # fichier absent → repli sur le texte


class TemplateWelcomeComposer:
    """Implémentation de ``contracts.WelcomeComposer``.

    Ordre de préférence pour un profil connu : fichier audio du profil (s'il existe),
    texte personnalisé du profil, modèle générique avec le nom, texte de repli.
    L'horodatage de la demande est celui de l'arrivée : l'expiration mesure donc le
    délai depuis l'arrivée.
    """

    def __init__(
        self,
        lookup: ProfileGreetingLookup,
        config: WelcomeConfig | None = None,
        request_id_factory: Callable[[], EventId] = new_event_id,
        file_exists: Callable[[str], bool] = os.path.isfile,
    ) -> None:
        self._lookup = lookup
        self.config = config or WelcomeConfig()
        self._new_id = request_id_factory
        self._exists = file_exists

    def compose(self, event: ArrivalEvent) -> AudioRequest | None:
        if event.kind == ArrivalKind.KNOWN:
            assert event.profile_id is not None
            return self._compose_known(event, event.profile_id)
        return self._compose_unknown(event)

    def _compose_known(self, event: ArrivalEvent, pid: ProfileId) -> AudioRequest:
        cfg = self.config
        try:
            info = self._lookup(pid)
        except Exception:  # le stockage ne doit pas empêcher l'accueil
            log.exception("Lecture du profil %s impossible", pid)
            info = None
        info = info or ProfileGreeting()
        audio = info.audio_path
        if audio is not None and cfg.check_audio_exists and not self._exists(audio):
            log.warning("Fichier d'accueil introuvable pour %s, repli texte", pid)
            audio = None
        text: str | None = None
        if audio is None:
            if info.text:
                text = info.text
            elif info.display_name:
                text = cfg.known_template.format(name=info.display_name)
            else:
                text = cfg.known_fallback_text
        return AudioRequest(
            request_id=self._new_id(),
            arrival_event_id=event.event_id,
            monotonic_ns=event.monotonic_ns,
            dedupe_key=f"arrival:{pid}",
            priority=cfg.known_priority,
            audio_path=audio,
            text=text,
            profile_id=pid,
            expires_after_ms=cfg.expires_after_ms,
        )

    def _compose_unknown(self, event: ArrivalEvent) -> AudioRequest | None:
        cfg = self.config
        audio: str | None = None
        text: str | None = None
        if cfg.unknown_policy == UnknownPolicy.SILENT:
            return None
        if cfg.unknown_policy == UnknownPolicy.AUDIO and cfg.unknown_audio_path:
            if not cfg.check_audio_exists or self._exists(cfg.unknown_audio_path):
                audio = cfg.unknown_audio_path
        if audio is None:
            text = cfg.unknown_text
        return AudioRequest(
            request_id=self._new_id(),
            arrival_event_id=event.event_id,
            monotonic_ns=event.monotonic_ns,
            dedupe_key="arrival:unknown",
            priority=cfg.unknown_priority,
            audio_path=audio,
            text=text,
            profile_id=None,
            expires_after_ms=cfg.expires_after_ms,
        )
