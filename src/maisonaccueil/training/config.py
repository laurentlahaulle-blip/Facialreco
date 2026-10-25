"""Configuration d'expérience versionnée (lot E).

Une configuration est figée, sérialisée en JSON canonique et identifiée par son empreinte
(``config_hash``). Elle est recopiée dans ``runs/<id>/config.json``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .preprocess import INPUT_SIZE, PREPROCESS_ID

CONFIG_SCHEMA = "maisonaccueil-e-config/1"
LOSSES = ("cosface", "arcface")
DATA_KINDS = ("collected", "synthetic")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    seed: int = 0
    # Données
    data_kind: str = "collected"  # "synthetic" = test logiciel uniquement
    preprocess: str = PREPROCESS_ID
    input_size: int = INPUT_SIZE
    # Réseau de représentation (initialisé aléatoirement)
    embedding_dim: int = 128
    stage_channels: tuple[int, ...] = (32, 64, 128, 256)
    stage_blocks: tuple[int, ...] = (1, 2, 2, 1)
    dropout: float = 0.2
    # Perte à marge angulaire
    loss: str = "cosface"
    scale: float = 30.0
    margin: float = 0.35  # CosFace : marge additive sur le cosinus ; ArcFace : sur l'angle
    # Optimisation
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 5e-4
    max_steps: int | None = None  # borne dure (test de fumée)
    num_workers: int = 0
    hflip: bool = True  # augmentation : miroir horizontal aléatoire (visage aligné)
    device: str = "cpu"  # "cuda", "cuda:0"… → réservation ResourceLock("gpu-0")
    # Initialisation : None = aléatoire (origin "scratch"). Un chemin de poids tiers
    # ferait du modèle un "finetuned" — non utilisé dans ce lot.
    init_weights: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or any(c in self.name for c in "/\\ "):
            raise ValueError("ExperimentConfig.name : non vide, sans espace ni séparateur")
        if self.loss not in LOSSES:
            raise ValueError(f"Perte inconnue {self.loss!r} (attendu : {LOSSES})")
        if self.data_kind not in DATA_KINDS:
            raise ValueError(f"data_kind inconnu {self.data_kind!r}")
        if len(self.stage_channels) != len(self.stage_blocks) or not self.stage_channels:
            raise ValueError("stage_channels et stage_blocks doivent avoir la même longueur")
        if self.input_size % (2 ** len(self.stage_channels)) != 0:
            raise ValueError("input_size doit être divisible par 2^(nombre d'étages)")
        if self.embedding_dim <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("embedding_dim, batch_size et epochs doivent être > 0")
        if self.preprocess != PREPROCESS_ID:
            raise ValueError(
                f"Prétraitement {self.preprocess!r} non pris en charge (attendu {PREPROCESS_ID})"
            )

    @property
    def uses_cuda(self) -> bool:
        return self.device.startswith("cuda")

    @property
    def origin(self) -> str:
        """Origine honnête du modèle produit par cette configuration."""
        return "scratch" if self.init_weights is None else "finetuned"

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["stage_channels"] = list(self.stage_channels)
        data["stage_blocks"] = list(self.stage_blocks)
        return {"schema": CONFIG_SCHEMA, **data}

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        data = dict(data)
        schema = data.pop("schema", CONFIG_SCHEMA)
        if schema != CONFIG_SCHEMA:
            raise ValueError(f"Schéma de configuration inattendu : {schema!r}")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Champs de configuration inconnus : {sorted(unknown)}")
        for key in ("stage_channels", "stage_blocks"):
            if key in data:
                data[key] = tuple(int(v) for v in data[key])
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(path.read_text("utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), "utf-8")
