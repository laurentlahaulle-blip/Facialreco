"""Journal des arrivées : historique (dépôt injecté) + événements en direct.

Un visage inconnu n'est jamais nommé. L'interface n'écrit pas dans le journal : elle
affiche (l'enregistrement relève de la chaîne vision / décision).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from maisonaccueil.contracts import ArrivalEvent, ArrivalKind, EventId, ProfileId
from maisonaccueil.pipeline import VisionOutput

from ..ports import ArrivalHistory, ArrivalRecord
from .base import Observable
from .formatting import format_utc

NameLookup = Callable[[ProfileId], str | None]


@dataclass(frozen=True, slots=True)
class ArrivalRow:
    event_id: EventId
    kind: ArrivalKind
    time_text: str
    label: str
    live: bool  # True : reçu pendant cette exécution ; False : historique


class ArrivalLogViewModel(Observable):
    def __init__(
        self,
        name_lookup: NameLookup,
        history: ArrivalHistory | None = None,
        *,
        max_rows: int = 200,
    ) -> None:
        super().__init__()
        self._name_lookup = name_lookup
        self._history = history
        self._rows: deque[ArrivalRow] = deque(maxlen=max_rows)
        self._seen: set[EventId] = set()
        self.load_error: str | None = None

    def load_history(self, limit: int = 100) -> None:
        if self._history is None:
            return
        try:
            records: list[ArrivalRecord] = list(self._history.list_recent(limit))
        except Exception as exc:  # base indisponible : on l'affiche, on ne plante pas
            self.load_error = f"journal indisponible : {exc}"
            self._notify()
            return
        self.load_error = None
        # list_recent : plus récent d'abord ; on ajoute du plus ancien au plus récent.
        for r in reversed(records):
            self._add(r.event_id, r.kind, r.profile_id, format_utc(r.occurred_at), live=False)
        self._notify()

    def apply_vision_output(self, output: VisionOutput) -> None:
        if not output.arrivals:
            return
        for event in output.arrivals:
            self.apply_event(event, notify=False)
        self._notify()

    def apply_event(self, event: ArrivalEvent, *, notify: bool = True) -> None:
        self._add(event.event_id, event.kind, event.profile_id, format_utc(event.wall_time),
                  live=True)
        if notify:
            self._notify()

    @property
    def rows(self) -> list[ArrivalRow]:
        """Plus récent d'abord."""
        return list(reversed(self._rows))

    def _label(self, kind: ArrivalKind, profile_id: ProfileId | None) -> str:
        if kind == ArrivalKind.UNKNOWN:
            return "Visage inconnu"
        if profile_id is None:
            return "Personne enregistrée (profil supprimé)"
        name = self._name_lookup(profile_id)
        return f"Arrivée : {name}" if name else "Personne enregistrée (profil introuvable)"

    def _add(
        self,
        event_id: EventId,
        kind: ArrivalKind,
        profile_id: ProfileId | None,
        time_text: str,
        *,
        live: bool,
    ) -> None:
        if event_id in self._seen:
            return
        if self._rows.maxlen is not None and len(self._rows) == self._rows.maxlen:
            self._seen.discard(self._rows[0].event_id)
        self._seen.add(event_id)
        self._rows.append(ArrivalRow(event_id, kind, time_text, self._label(kind, profile_id),
                                     live))
