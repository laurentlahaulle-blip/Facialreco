"""Backends de lecture audio LOCAUX. Aucun service cloud.

- ``WinsoundBackend`` : fichiers WAV via ``winsound`` (bibliothèque standard, Windows).
- ``Pyttsx3Backend`` : synthèse vocale locale OPTIONNELLE (SAPI5 sous Windows), si le
  paquet ``pyttsx3`` est installé. Dépendance NON déclarée dans pyproject.toml :
  voir docs/contract-changes/F-tts-pyttsx3.md.
- ``RoutingAudioBackend`` : fichier → backend fichier, texte → backend texte.
- ``LoggingAudioBackend`` : n'émet aucun son, journalise (poste sans carte son).
- ``FakeAudioBackend`` : double de test, enregistre les demandes.

Un backend ``play`` est BLOQUANT (le ``QueuedAudioPlayer`` l'appelle depuis son thread).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from maisonaccueil.contracts import AudioRequest

log = logging.getLogger(__name__)


class UnsupportedAudioRequest(RuntimeError):
    """Le backend ne sait pas lire ce type de demande (ex. texte sans synthèse)."""


@runtime_checkable
class AudioBackend(Protocol):
    def play(self, request: AudioRequest) -> None:
        """Lecture bloquante jusqu'à la fin du message."""
        ...

    def stop(self) -> None:
        """Interrompt si possible la lecture en cours (appelé depuis un autre thread)."""
        ...


class WinsoundBackend:
    """Lecture synchrone de fichiers WAV avec ``winsound`` (Windows uniquement)."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WinsoundBackend n'est disponible que sous Windows")
        import winsound

        self._winsound: Any = winsound

    def play(self, request: AudioRequest) -> None:
        if request.audio_path is None:
            raise UnsupportedAudioRequest("winsound ne lit que des fichiers WAV")
        ws = self._winsound
        ws.PlaySound(request.audio_path, ws.SND_FILENAME | ws.SND_NODEFAULT)

    def stop(self) -> None:
        self._winsound.PlaySound(None, 0)


class Pyttsx3Backend:
    """Synthèse vocale locale via ``pyttsx3`` (dépendance optionnelle non installée).

    Le moteur est créé paresseusement dans le thread de lecture (exigence COM/SAPI5).
    """

    def __init__(self, voice_id: str | None = None, rate: int | None = None) -> None:
        try:
            import pyttsx3  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dépend de l'environnement
            raise RuntimeError(
                "pyttsx3 n'est pas installé (voir docs/contract-changes/F-tts-pyttsx3.md)"
            ) from exc
        self._module: Any = pyttsx3
        self._engine: Any = None
        self._voice_id = voice_id
        self._rate = rate

    def play(self, request: AudioRequest) -> None:  # pragma: no cover - non testable ici
        if request.text is None:
            raise UnsupportedAudioRequest("Pyttsx3Backend ne lit que du texte")
        if self._engine is None:
            self._engine = self._module.init()
            if self._voice_id:
                self._engine.setProperty("voice", self._voice_id)
            if self._rate:
                self._engine.setProperty("rate", self._rate)
        self._engine.say(request.text)
        self._engine.runAndWait()

    def stop(self) -> None:  # pragma: no cover - non testable ici
        if self._engine is not None:
            self._engine.stop()


class LoggingAudioBackend:
    """Aucun son : journalise la demande (développement sans carte son)."""

    def play(self, request: AudioRequest) -> None:
        what = request.audio_path if request.audio_path is not None else request.text
        log.info("[audio simulé] %s", what)

    def stop(self) -> None:
        pass


class RoutingAudioBackend:
    """Aiguille les demandes fichier et texte vers deux backends distincts."""

    def __init__(
        self, file_backend: AudioBackend | None, text_backend: AudioBackend | None
    ) -> None:
        self._file = file_backend
        self._text = text_backend
        self._current: AudioBackend | None = None

    def play(self, request: AudioRequest) -> None:
        target = self._file if request.audio_path is not None else self._text
        if target is None:
            kind = "fichier" if request.audio_path is not None else "texte"
            raise UnsupportedAudioRequest(f"Aucun backend pour une demande {kind}")
        self._current = target
        try:
            target.play(request)
        finally:
            self._current = None

    def stop(self) -> None:
        current = self._current
        if current is not None:
            current.stop()


def create_default_backend() -> AudioBackend:
    """Windows : winsound pour les WAV + pyttsx3 si disponible pour le texte.
    Autres plateformes : ``LoggingAudioBackend`` (aucun son)."""
    if sys.platform != "win32":
        return LoggingAudioBackend()
    text: AudioBackend | None
    try:
        text = Pyttsx3Backend()
    except RuntimeError:
        log.warning("Synthèse vocale locale indisponible : les messages texte seront ignorés")
        text = None
    return RoutingAudioBackend(WinsoundBackend(), text)


@dataclass(frozen=True, slots=True)
class PlayedRecord:
    request: AudioRequest
    started_ns: int
    finished_ns: int


class FakeAudioBackend:
    """Double de test : n'émet aucun son, enregistre ce qui aurait été joué.

    ``duration_ms`` : durée simulée (``sleep`` injectable, réel par défaut).
    ``gate`` : si fourni, chaque lecture attend que l'événement soit levé (contrôle
    déterministe de l'enchaînement depuis le test).
    ``fail_on`` : prédicat déclenchant une erreur de lecture simulée.
    """

    def __init__(
        self,
        clock: Callable[[], int] = time.monotonic_ns,
        duration_ms: int = 0,
        sleep: Callable[[float], None] = time.sleep,
        gate: threading.Event | None = None,
        fail_on: Callable[[AudioRequest], bool] | None = None,
    ) -> None:
        self._clock = clock
        self._duration_ms = duration_ms
        self._sleep = sleep
        self._gate = gate
        self._fail_on = fail_on
        self._lock = threading.Lock()
        self._played: list[PlayedRecord] = []
        self.started = threading.Event()  # levé au début de chaque lecture
        self.stopped = 0

    def play(self, request: AudioRequest) -> None:
        start = self._clock()
        self.started.set()
        if self._gate is not None:
            self._gate.wait()
        if self._duration_ms:
            self._sleep(self._duration_ms / 1000)
        if self._fail_on is not None and self._fail_on(request):
            raise RuntimeError("échec de lecture simulé (FakeAudioBackend)")
        with self._lock:
            self._played.append(PlayedRecord(request, start, self._clock()))

    def stop(self) -> None:
        self.stopped += 1
        if self._gate is not None:
            self._gate.set()

    @property
    def played(self) -> list[PlayedRecord]:
        with self._lock:
            return list(self._played)

    @property
    def played_keys(self) -> list[str]:
        return [r.request.dedupe_key for r in self.played]
