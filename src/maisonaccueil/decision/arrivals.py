"""Présences et événements d'arrivée.

Une *présence* regroupe tous les passages (``TrackId``) attribués à un même profil
tant que la personne n'est pas considérée partie. Une seule ``ArrivalEvent`` est émise
par présence.

Une présence se termine (dans ``tick``) quand AUCUN de ses passages n'est actif et que
le dernier signe de vie date d'au moins ``absence_timeout_ms``. Un passage est actif
jusqu'à ``track_lost`` ou jusqu'à ``track_stale_ms`` sans activité.

Les signes de vie proviennent des décisions et, si l'intégration les fournit, de
``note_track_activity`` (à appeler pour chaque ``RecognitionResult``) et de
``track_lost``. Ces deux méthodes ne font pas partie du Protocol ``ArrivalTracker`` :
voir docs/contract-changes/F-arrival-track-activity.md.

Inconnus : au plus un événement UNKNOWN par passage, jamais de nom ; un passage déjà
attribué à un profil n'émet pas d'UNKNOWN. ``unknown_cooldown_ms`` limite en plus la
fréquence globale des UNKNOWN (fragmentation du suivi d'un même inconnu).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from maisonaccueil.contracts import (
    ArrivalEvent,
    ArrivalKind,
    Decision,
    DecisionStatus,
    EventId,
    ProfileId,
    TrackId,
    new_event_id,
)

_NS_PER_MS = 1_000_000
_MAX_REMEMBERED_TRACKS = 1_024


@dataclass(frozen=True, slots=True)
class ArrivalPolicy:
    """Valeurs par défaut PROVISOIRES, à régler avec le lot G (« message répété »)."""

    absence_timeout_ms: int = 30_000  # PROVISOIRE
    track_stale_ms: int = 5_000  # PROVISOIRE
    unknown_cooldown_ms: int = 5_000  # PROVISOIRE ; 0 = désactivé

    def __post_init__(self) -> None:
        if self.absence_timeout_ms < 0 or self.track_stale_ms <= 0:
            raise ValueError("absence_timeout_ms >= 0 et track_stale_ms > 0 requis")
        if self.unknown_cooldown_ms < 0:
            raise ValueError("unknown_cooldown_ms doit être >= 0")


@dataclass(slots=True)
class Presence:
    profile_id: ProfileId
    event_id: EventId
    started_ns: int
    last_seen_ns: int
    tracks: dict[TrackId, int] = field(default_factory=dict)  # actifs → dernière activité


class PresenceArrivalTracker:
    """Implémentation de ``contracts.ArrivalTracker``. Non thread-safe."""

    def __init__(
        self,
        policy: ArrivalPolicy | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        event_id_factory: Callable[[], EventId] = new_event_id,
    ) -> None:
        self.policy = policy or ArrivalPolicy()
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._new_id = event_id_factory
        self._presences: dict[ProfileId, Presence] = {}
        self._track_owner: dict[TrackId, ProfileId] = {}
        # Passages ayant déjà produit un UNKNOWN (mémoire bornée).
        self._unknown_tracks: OrderedDict[TrackId, int] = OrderedDict()
        self._last_unknown_ns: int | None = None

    # -- interface ArrivalTracker -------------------------------------------------
    def on_decision(self, decision: Decision) -> ArrivalEvent | None:
        now = decision.monotonic_ns
        if decision.status == DecisionStatus.CONFIRMED:
            assert decision.profile_id is not None
            return self._on_confirmed(decision.track_id, decision.profile_id, now)
        if decision.status == DecisionStatus.UNKNOWN:
            return self._on_unknown(decision.track_id, now)
        # UNCERTAIN : simple signe de vie pour un passage déjà attribué.
        self.note_track_activity(decision.track_id, now)
        return None

    def tick(self, monotonic_ns: int) -> None:
        stale = self.policy.track_stale_ms * _NS_PER_MS
        absence = self.policy.absence_timeout_ms * _NS_PER_MS
        for pid, pres in list(self._presences.items()):
            for tid, seen in list(pres.tracks.items()):
                if monotonic_ns - seen >= stale:
                    del pres.tracks[tid]
            if not pres.tracks and monotonic_ns - pres.last_seen_ns >= absence:
                self._end_presence(pid)

    # -- signes de vie optionnels -------------------------------------------------
    def note_track_activity(self, track_id: TrackId, monotonic_ns: int) -> None:
        pid = self._track_owner.get(track_id)
        pres = self._presences.get(pid) if pid is not None else None
        if pres is None:
            return
        pres.tracks[track_id] = max(pres.tracks.get(track_id, 0), monotonic_ns)
        pres.last_seen_ns = max(pres.last_seen_ns, monotonic_ns)

    def track_lost(self, track_id: TrackId, monotonic_ns: int) -> None:
        pid = self._track_owner.get(track_id)
        pres = self._presences.get(pid) if pid is not None else None
        if pres is not None and track_id in pres.tracks:
            last = pres.tracks.pop(track_id)
            pres.last_seen_ns = max(pres.last_seen_ns, min(last, monotonic_ns))

    # -- consultation -------------------------------------------------------------
    @property
    def present_profiles(self) -> frozenset[ProfileId]:
        return frozenset(self._presences)

    def presence(self, profile_id: ProfileId) -> Presence | None:
        return self._presences.get(profile_id)

    # -- interne ------------------------------------------------------------------
    def _on_confirmed(self, track_id: TrackId, pid: ProfileId, now: int) -> ArrivalEvent | None:
        previous = self._track_owner.get(track_id)
        if previous is not None and previous != pid and previous in self._presences:
            # Échange de suivi : le passage quitte l'ancienne présence.
            old = self._presences[previous]
            old.tracks.pop(track_id, None)
        self._track_owner[track_id] = pid
        pres = self._presences.get(pid)
        if pres is not None:
            pres.tracks[track_id] = max(pres.tracks.get(track_id, 0), now)
            pres.last_seen_ns = max(pres.last_seen_ns, now)
            return None
        event = ArrivalEvent(
            event_id=self._new_id(),
            kind=ArrivalKind.KNOWN,
            track_id=track_id,
            monotonic_ns=now,
            wall_time=self._wall_clock(),
            profile_id=pid,
        )
        self._presences[pid] = Presence(pid, event.event_id, now, now, {track_id: now})
        return event

    def _on_unknown(self, track_id: TrackId, now: int) -> ArrivalEvent | None:
        if track_id in self._track_owner:
            self.note_track_activity(track_id, now)
            return None
        if track_id in self._unknown_tracks:
            return None
        self._unknown_tracks[track_id] = now
        while len(self._unknown_tracks) > _MAX_REMEMBERED_TRACKS:
            self._unknown_tracks.popitem(last=False)
        cooldown = self.policy.unknown_cooldown_ms * _NS_PER_MS
        if self._last_unknown_ns is not None and now - self._last_unknown_ns < cooldown:
            return None
        self._last_unknown_ns = now
        return ArrivalEvent(
            event_id=self._new_id(),
            kind=ArrivalKind.UNKNOWN,
            track_id=track_id,
            monotonic_ns=now,
            wall_time=self._wall_clock(),
        )

    def _end_presence(self, pid: ProfileId) -> None:
        del self._presences[pid]
        for tid in [t for t, p in self._track_owner.items() if p == pid]:
            del self._track_owner[tid]
