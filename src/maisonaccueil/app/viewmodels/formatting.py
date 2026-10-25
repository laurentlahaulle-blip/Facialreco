"""Textes affichés. Règle : un score n'est JAMAIS présenté comme une probabilité ni en %."""

from __future__ import annotations

from datetime import UTC, datetime

from maisonaccueil.contracts import (
    CaptureStatus,
    DecisionStatus,
    Metric,
    ModelRef,
    RawScore,
)

ENGINE_UNAVAILABLE = "moteur non disponible"
CAMERA_UNAVAILABLE = "caméra non disponible"
SCORE_LABEL = "score technique (non calibré)"

CAPTURE_LABELS: dict[CaptureStatus, str] = {
    CaptureStatus.STOPPED: "arrêtée",
    CaptureStatus.CONNECTING: "connexion en cours",
    CaptureStatus.RUNNING: "en marche",
    CaptureStatus.RECONNECTING: "reconnexion en cours",
    CaptureStatus.FAILED: "échec",
}

DECISION_LABELS: dict[DecisionStatus, str] = {
    DecisionStatus.CONFIRMED: "confirmé",
    DecisionStatus.UNKNOWN: "inconnu",
    DecisionStatus.UNCERTAIN: "incertain",
}

_METRIC_LABELS: dict[Metric, str] = {
    Metric.COSINE: "cosinus, plus haut = plus proche",
    Metric.L2: "distance L2, plus bas = plus proche",
}

_ORIGIN_LABELS = {
    "pretrained": "poids préentraînés tiers",
    "scratch": "entraîné de zéro",
    "finetuned": "poids tiers adaptés",
    "test-double": "DOUBLE DE TEST",
}


def format_raw_score(score: RawScore) -> str:
    """Ex. « score technique (non calibré) : 0.612 (cosinus, plus haut = plus proche) »."""
    return f"{SCORE_LABEL} : {score.value:.3f} ({_METRIC_LABELS[score.metric]})"


def format_model_ref(model: ModelRef | None) -> str:
    if model is None:
        return ENGINE_UNAVAILABLE
    origin = _ORIGIN_LABELS.get(model.origin, model.origin)
    return (
        f"{model.name} v{model.version} — {origin} ; dimension {model.embedding_dim}, "
        f"métrique {model.metric}, prétraitement {model.preprocess}"
    )


def format_utc(value: datetime) -> str:
    """Horodatage affiché en UTC explicite (les datetimes naïfs sont refusés)."""
    if value.tzinfo is None:
        raise ValueError("Horodatage naïf : timezone-aware (UTC) attendu")
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def short_id(value: str, width: int = 16) -> str:
    return value if len(value) <= width else value[:width] + "…"
