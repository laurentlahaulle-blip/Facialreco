"""Observateur minimal : les widgets s'abonnent, les modèles de vue notifient."""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Saisie refusée ; le message (en français) est destiné à l'utilisateur."""


class Observable:
    def __init__(self) -> None:
        self._listeners: list[Callable[[], None]] = []

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Abonne ``listener`` ; retourne la fonction de désabonnement."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _notify(self) -> None:
        for listener in list(self._listeners):
            try:
                listener()
            except Exception:  # un widget défaillant ne doit pas casser le modèle
                log.exception("Erreur dans un abonné de %s", type(self).__name__)
