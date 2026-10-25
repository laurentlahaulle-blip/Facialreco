"""Métriques par passage et par personne, avec effectifs et intervalles de Wilson.

Aucune valeur n'est un résultat tant qu'elle n'est pas calculée sur des passages réels
d'un ensemble de test indépendant. Un taux n'est jamais présenté sans ``k/n``.

Définitions (dénominateurs) :

- bonne identification     = #correct / #passages known
- mauvaise identité        = #wrong_identity / #passages known
- connu non reconnu        = #missed / #passages known
- inconnu accepté à tort   = #false_accept / #passages unregistered
- message répété           = #passages avec repeated_messages > 0 / #passages (tous)
- délai jusqu'à l'accueil  = distribution des ``delay_ms`` (n, min, médiane, p90, max)

Intervalle de Wilson (score) à 95 % par défaut (z = 1,959964). Il suppose des passages
indépendants ; des passages d'une même personne sont corrélés, l'intervalle est donc
optimiste — d'où le tableau par personne et le nombre de personnes affiché.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .annotations import Expected
from .matching import MatchResult, Outcome, PassageResult

Z95 = 1.959963984540054


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Intervalle de score de Wilson pour une proportion k/n ; ``None`` si n = 0.

    centre = (p + z²/2n) / (1 + z²/n) ; demi-largeur = z·√(p(1−p)/n + z²/4n²) / (1 + z²/n).
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"effectifs invalides k={k}, n={n}")
    if n == 0:
        return None
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


@dataclass(frozen=True, slots=True)
class Proportion:
    k: int
    n: int
    ci: tuple[float, float] | None

    @classmethod
    def of(cls, k: int, n: int, z: float = Z95) -> Proportion:
        return cls(k, n, wilson_interval(k, n, z))

    @property
    def rate(self) -> float | None:
        return self.k / self.n if self.n else None

    def format(self, digits: int = 1) -> str:
        """« k/n (x % ; IC95 [a ; b]) » ou « 0/0 (non mesurable) »."""
        if self.n == 0 or self.ci is None:
            return f"{self.k}/{self.n} (non mesurable)"
        r = 100.0 * self.k / self.n
        lo, hi = 100.0 * self.ci[0], 100.0 * self.ci[1]
        return (f"{self.k}/{self.n} ({r:.{digits}f} % ; IC95 Wilson "
                f"[{lo:.{digits}f} ; {hi:.{digits}f}] %)")


def percentile_nearest_rank(values: Sequence[float], q: float) -> float:
    """Percentile « rang le plus proche » : plus petite valeur v telle qu'au moins q % des
    valeurs soient ≤ v (q dans ]0, 100])."""
    if not values:
        raise ValueError("aucune valeur")
    if not 0 < q <= 100:
        raise ValueError("q doit être dans ]0, 100]")
    s = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(s)))
    return s[rank - 1]


def median(values: Sequence[float]) -> float:
    s = sorted(values)
    if not s:
        raise ValueError("aucune valeur")
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


@dataclass(frozen=True, slots=True)
class DelayStats:
    n: int
    min_ms: float | None = None
    median_ms: float | None = None
    p90_ms: float | None = None
    max_ms: float | None = None

    @classmethod
    def of(cls, values: Iterable[float]) -> DelayStats:
        v = list(values)
        if not v:
            return cls(0)
        return cls(len(v), min(v), median(v), percentile_nearest_rank(v, 90), max(v))

    def format(self) -> str:
        if self.n == 0:
            return "n=0 (non mesurable)"
        return (f"n={self.n} ; min {self.min_ms:.0f} ms ; médiane {self.median_ms:.0f} ms ; "
                f"p90 {self.p90_ms:.0f} ms ; max {self.max_ms:.0f} ms")


@dataclass(frozen=True, slots=True)
class Summary:
    """Agrégat d'un ensemble de passages (toutes personnes, ou une personne)."""

    n_passages: int
    n_known: int
    n_unregistered: int
    n_persons: int
    correct: Proportion
    wrong_identity: Proportion
    missed: Proportion
    false_accept: Proportion
    repeated: Proportion
    delay: DelayStats
    repeated_messages_total: int


def summarize(results: Sequence[PassageResult], z: float = Z95) -> Summary:
    known = [r for r in results if r.passage.expected == Expected.KNOWN]
    unreg = [r for r in results if r.passage.expected == Expected.UNREGISTERED]
    n_k, n_u = len(known), len(unreg)

    def count(rs: Sequence[PassageResult], o: Outcome) -> int:
        return sum(1 for r in rs if r.outcome == o)

    return Summary(
        n_passages=len(results),
        n_known=n_k,
        n_unregistered=n_u,
        n_persons=len({r.passage.person_key for r in results}),
        correct=Proportion.of(count(known, Outcome.CORRECT), n_k, z),
        wrong_identity=Proportion.of(count(known, Outcome.WRONG_IDENTITY), n_k, z),
        missed=Proportion.of(count(known, Outcome.MISSED), n_k, z),
        false_accept=Proportion.of(count(unreg, Outcome.FALSE_ACCEPT), n_u, z),
        repeated=Proportion.of(sum(1 for r in results if r.message_repeated), len(results), z),
        delay=DelayStats.of(r.delay_ms for r in known if r.delay_ms is not None),
        repeated_messages_total=sum(r.repeated_messages for r in results),
    )


def per_person(results: Sequence[PassageResult], z: float = Z95) -> dict[str, Summary]:
    groups: dict[str, list[PassageResult]] = defaultdict(list)
    for r in results:
        groups[r.passage.person_key].append(r)
    return {k: summarize(v, z) for k, v in sorted(groups.items())}


@dataclass(frozen=True, slots=True)
class EngineEvaluation:
    config_id: str
    overall: Summary
    persons: dict[str, Summary]
    unattributed_known: int  # arrivées KNOWN hors de tout passage
    unattributed_total: int
    match: MatchResult


def evaluate_match(match: MatchResult, z: float = Z95) -> EngineEvaluation:
    unattr_known = sum(1 for u in match.unattributed if u.record == "arrival" and u.kind == "known")
    unattr_total = sum(1 for u in match.unattributed if u.record == "arrival")
    return EngineEvaluation(
        match.config_id,
        summarize(match.passages, z),
        per_person(match.passages, z),
        unattr_known,
        unattr_total,
        match,
    )
