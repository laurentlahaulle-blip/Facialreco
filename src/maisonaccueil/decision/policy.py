"""Politique de décision temporelle versionnée.

Toutes les valeurs par défaut sont PROVISOIRES : elles ne reposent sur aucune mesure.
Elles devront être fixées par le lot G sur l'ensemble de réglage (``tune``) puis
vérifiées sur un ensemble de test indépendant. Un score n'est pas une probabilité.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from maisonaccueil.contracts import Metric, QualityFlag, RawScore

POLICY_TEMPORAL_V1 = "temporal-v1"

DEFAULT_BLOCKING_FLAGS: frozenset[QualityFlag] = frozenset(
    {QualityFlag.FACE_TOO_SMALL, QualityFlag.BLURRY, QualityFlag.OCCLUDED}
)


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Paramètres de ``TemporalDecisionEngine``.

    ``accept_cosine`` / ``accept_l2`` : seuil d'acceptation d'un candidat selon la
        métrique du score (cosinus : score >= seuil ; L2 : distance <= seuil).
    ``min_margin`` : écart minimal entre le meilleur et le 2ᵉ candidat (en unités de la
        métrique). Sans 2ᵉ candidat, la marge est considérée suffisante.
    ``window_ms`` : fenêtre glissante d'accumulation par passage.
    ``min_agreeing`` : observations acceptées concordantes (même profil) pour CONFIRMED.
    ``max_conflicting`` : nombre maximal d'observations acceptées pour un AUTRE profil
        tolérées dans la fenêtre ; au-delà, contradiction → UNCERTAIN.
    ``min_observations_unknown`` : observations exploitables sans aucun profil accepté
        pour conclure UNKNOWN.
    ``blocking_flags`` : observations ignorées si elles portent l'un de ces drapeaux.
    ``stale_track_ms`` : un passage sans observation depuis ce délai est oublié (filet de
        sécurité si ``track_lost`` n'est pas appelé).
    """

    policy_id: str = POLICY_TEMPORAL_V1
    accept_cosine: float = 0.45  # PROVISOIRE
    accept_l2: float = 1.10  # PROVISOIRE
    min_margin: float = 0.05  # PROVISOIRE
    window_ms: int = 2_000  # PROVISOIRE
    min_agreeing: int = 3  # PROVISOIRE
    max_conflicting: int = 1  # PROVISOIRE
    min_observations_unknown: int = 5  # PROVISOIRE
    blocking_flags: frozenset[QualityFlag] = field(default=DEFAULT_BLOCKING_FLAGS)
    stale_track_ms: int = 10_000

    def __post_init__(self) -> None:
        if self.window_ms <= 0:
            raise ValueError("window_ms doit être > 0")
        if self.min_agreeing < 1 or self.min_observations_unknown < 1:
            raise ValueError("min_agreeing et min_observations_unknown doivent être >= 1")
        if self.max_conflicting < 0 or self.min_margin < 0:
            raise ValueError("max_conflicting et min_margin doivent être >= 0")
        if self.stale_track_ms <= 0:
            raise ValueError("stale_track_ms doit être > 0")

    def is_accepted(self, score: RawScore) -> bool:
        if score.metric == Metric.COSINE:
            return score.value >= self.accept_cosine
        return score.value <= self.accept_l2

    def has_margin(self, best: RawScore, second: RawScore | None) -> bool:
        if second is None:
            return True
        if best.metric != second.metric:
            raise ValueError("Candidats de métriques différentes")
        gap = best.value - second.value if best.metric == Metric.COSINE else (
            second.value - best.value
        )
        return gap >= self.min_margin
