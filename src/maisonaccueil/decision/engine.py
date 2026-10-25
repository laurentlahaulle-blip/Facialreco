"""Décision temporelle par passage (``TrackId``).

Règles d'émission (``observe`` renvoie ``None`` quand il n'y a rien de nouveau) :

- CONFIRMED est émise une seule fois par passage et par profil. Une fois un passage
  confirmé, il reste confirmé (« collant ») : les UNCERTAIN/UNKNOWN ultérieurs ne sont
  pas émis. Seule une confirmation d'un AUTRE profil (échange de suivi) est émise.
- UNKNOWN est émise une fois par passage (tant qu'il n'est pas confirmé).
- UNCERTAIN n'est émise que pour une contradiction (plusieurs profils acceptés dans la
  fenêtre), une fois à chaque entrée dans cet état. L'état « pas encore assez
  d'observations » n'est jamais émis : c'est l'état implicite de départ.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field

from maisonaccueil.contracts import (
    Decision,
    DecisionStatus,
    ModelRef,
    ProfileId,
    RawScore,
    RecognitionResult,
    TrackId,
)

from .policy import DecisionPolicy

log = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000

REASON_INSUFFICIENT = "insufficient"
REASON_CONFLICT = "conflict"
REASON_AGREEMENT = "agreement"
REASON_NO_MATCH = "no-accepted-profile"


@dataclass(frozen=True, slots=True)
class _Observation:
    monotonic_ns: int
    accepted: ProfileId | None  # profil accepté (seuil + marge), sinon None
    best_score: RawScore | None  # meilleur score de l'observation (technique)


@dataclass(slots=True)
class _TrackState:
    model: ModelRef
    window: deque[_Observation] = field(default_factory=deque)
    last_seen_ns: int = 0
    last_key: tuple[DecisionStatus, ProfileId | None, str] = (
        DecisionStatus.UNCERTAIN,
        None,
        REASON_INSUFFICIENT,
    )
    confirmed: ProfileId | None = None


class TemporalDecisionEngine:
    """Implémentation de ``contracts.DecisionEngine`` (politique ``temporal-v1``).

    Non thread-safe : à appeler depuis un seul thread (celui du pipeline vision).
    """

    def __init__(self, policy: DecisionPolicy | None = None) -> None:
        self.policy = policy or DecisionPolicy()
        self._tracks: dict[TrackId, _TrackState] = {}

    # -- interface DecisionEngine -------------------------------------------------
    def observe(self, result: RecognitionResult) -> Decision | None:
        now = result.monotonic_ns
        self._drop_stale(now)
        state = self._tracks.get(result.track_id)
        if state is None or state.model != result.model:
            if state is not None:
                log.warning("Passage %s : changement de modèle, état réinitialisé",
                            result.track_id)
            state = _TrackState(model=result.model)
            self._tracks[result.track_id] = state
        state.last_seen_ns = max(state.last_seen_ns, now)

        if result.quality_flags & self.policy.blocking_flags:
            return None  # observation inexploitable : ignorée

        state.window.append(self._to_observation(result))
        self._prune_window(state)
        return self._evaluate(result.track_id, state, now)

    def track_lost(self, track_id: TrackId, monotonic_ns: int) -> None:
        self._tracks.pop(track_id, None)

    # -- outils -------------------------------------------------------------------
    @property
    def active_tracks(self) -> frozenset[TrackId]:
        return frozenset(self._tracks)

    def _to_observation(self, result: RecognitionResult) -> _Observation:
        cands = result.candidates
        if not cands:
            return _Observation(result.monotonic_ns, None, None)
        best = cands[0]
        second = cands[1].score if len(cands) > 1 else None
        ok = self.policy.is_accepted(best.score) and self.policy.has_margin(best.score, second)
        return _Observation(result.monotonic_ns, best.profile_id if ok else None, best.score)

    def _prune_window(self, state: _TrackState) -> None:
        newest = max(o.monotonic_ns for o in state.window)
        limit = newest - self.policy.window_ms * _NS_PER_MS
        while state.window and state.window[0].monotonic_ns < limit:
            state.window.popleft()

    def _drop_stale(self, now: int) -> None:
        limit = now - self.policy.stale_track_ms * _NS_PER_MS
        for tid in [t for t, s in self._tracks.items() if s.last_seen_ns < limit]:
            del self._tracks[tid]

    def _evaluate(self, track_id: TrackId, state: _TrackState, now: int) -> Decision | None:
        p = self.policy
        window = state.window
        counts = Counter(o.accepted for o in window if o.accepted is not None)
        profile: ProfileId | None = None
        if counts:
            top, top_n = counts.most_common(1)[0]
            others = sum(counts.values()) - top_n
            if others > p.max_conflicting:
                status, reason = DecisionStatus.UNCERTAIN, REASON_CONFLICT
            elif top_n >= p.min_agreeing:
                status, reason, profile = DecisionStatus.CONFIRMED, REASON_AGREEMENT, top
            else:
                status, reason = DecisionStatus.UNCERTAIN, REASON_INSUFFICIENT
        elif len(window) >= p.min_observations_unknown:
            status, reason = DecisionStatus.UNKNOWN, REASON_NO_MATCH
        else:
            status, reason = DecisionStatus.UNCERTAIN, REASON_INSUFFICIENT

        key = (status, profile, reason)
        if key == state.last_key:
            return None
        if state.confirmed is not None and status != DecisionStatus.CONFIRMED:
            return None  # passage déjà confirmé : pas de rétrogradation émise
        if status == DecisionStatus.UNCERTAIN and reason == REASON_INSUFFICIENT:
            state.last_key = key  # état implicite, jamais émis
            return None

        state.last_key = key
        if status == DecisionStatus.CONFIRMED:
            state.confirmed = profile
        relevant = [o for o in window if profile is None or o.accepted == profile]
        return Decision(
            track_id=track_id,
            status=status,
            monotonic_ns=now,
            model=state.model,
            policy=p.policy_id,
            observations=len(window),
            window_ms=(window[-1].monotonic_ns - window[0].monotonic_ns) // _NS_PER_MS,
            profile_id=profile,
            best_score=_best_score(relevant),
            reason=reason,
        )


def _best_score(observations: list[_Observation]) -> RawScore | None:
    best: RawScore | None = None
    for o in observations:
        if o.best_score is not None and (best is None or o.best_score.is_better_than(best)):
            best = o.best_score
    return best
