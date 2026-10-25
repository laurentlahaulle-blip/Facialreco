"""Journal d'expérience ``runs/<id>/`` et identité ``ModelRef`` du modèle produit.

Contenu d'un dossier d'expérience (jamais committé, ``runs/`` est ignoré par Git) :

- ``config.json``      : configuration complète (+ empreinte) ;
- ``manifest_counts.json`` : effectifs par découpage (aucune image) ;
- ``metrics.jsonl``    : une ligne JSON par pas (perte, taux d'apprentissage…) ;
- ``weights.pt``       : poids du réseau de représentation (sans la tête de perte) ;
- ``model_ref.json``   : ``ModelRef`` + empreintes (poids, configuration) ;
- ``summary.json``     : durée, nombre de pas, avertissements.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from maisonaccueil.contracts import Metric, ModelRef
from maisonaccueil.pipeline import ResourceLock

from .config import ExperimentConfig

MODEL_NAME = "maisonaccueil-scratch"
SMOKE_SUFFIX = "-smoke"  # modèle entraîné sur données synthétiques : inutilisable en vrai
GPU_RESOURCE = "gpu-0"
LOCK_OWNER = "lot-E"
DEFAULT_RUNS_ROOT = Path("runs")


def build_model_ref(config: ExperimentConfig, weights_sha256: str) -> ModelRef:
    """``ModelRef`` honnête : ``origin`` dérive de l'initialisation (aléatoire → "scratch",
    poids tiers → "finetuned") ; la version est l'empreinte des POIDS ; un modèle entraîné
    sur données synthétiques porte un nom distinct (``-smoke``) pour ne jamais être
    confondu avec un modèle réel dans la galerie."""
    if len(weights_sha256) < 12:
        raise ValueError("Empreinte des poids trop courte")
    name = MODEL_NAME + (SMOKE_SUFFIX if config.data_kind == "synthetic" else "")
    return ModelRef(
        name=name,
        version=f"{config.name}-{weights_sha256[:12]}",
        embedding_dim=config.embedding_dim,
        metric=Metric.COSINE,
        preprocess=config.preprocess,
        origin=config.origin,
    )


def model_ref_to_dict(ref: ModelRef) -> dict[str, Any]:
    return {
        "name": ref.name,
        "version": ref.version,
        "embedding_dim": ref.embedding_dim,
        "metric": str(ref.metric),
        "preprocess": ref.preprocess,
        "origin": ref.origin,
        "key": ref.key,
    }


def model_ref_from_dict(data: dict[str, Any]) -> ModelRef:
    ref = ModelRef(
        name=str(data["name"]),
        version=str(data["version"]),
        embedding_dim=int(data["embedding_dim"]),
        metric=Metric(str(data["metric"])),
        preprocess=str(data["preprocess"]),
        origin=str(data["origin"]),
    )
    if "key" in data and data["key"] != ref.key:
        raise ValueError("model_ref.json : clé incohérente avec les champs")
    return ref


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def device_guard(config: ExperimentConfig, lock_dir: Path | None = None) -> Iterator[None]:
    """Réserve ``ResourceLock("gpu-0")`` pendant tout l'entraînement si CUDA est utilisé.

    Lève ``ResourceBusyError`` si le GPU est déjà réservé (pas d'entraînements
    concurrents). Sur CPU, aucune ressource partagée n'est réservée."""
    if config.uses_cuda:
        with ResourceLock(GPU_RESOURCE, owner=LOCK_OWNER, lock_dir=lock_dir):
            yield
    else:
        yield


@dataclass(frozen=True, slots=True)
class RunDir:
    run_id: str
    path: Path

    @classmethod
    def create(cls, config: ExperimentConfig, root: Path | None = None) -> RunDir:
        root = root if root is not None else DEFAULT_RUNS_ROOT
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base = f"{stamp}-{config.name}-{config.config_hash[:8]}"
        run_id, n = base, 1
        while (root / run_id).exists():
            n += 1
            run_id = f"{base}-{n}"
        path = root / run_id
        path.mkdir(parents=True)
        config.save(path / "config.json")
        return cls(run_id=run_id, path=path)

    @property
    def weights_path(self) -> Path:
        return self.path / "weights.pt"

    @property
    def onnx_path(self) -> Path:
        return self.path / "model.onnx"

    def write_json(self, name: str, data: Any) -> Path:
        target = self.path / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), "utf-8")
        os.replace(tmp, target)
        return target

    def append_metrics(self, record: dict[str, Any]) -> None:
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def read_metrics(self) -> list[dict[str, Any]]:
        path = self.path / "metrics.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]

    def write_model_ref(
        self, ref: ModelRef, weights_sha256: str, config: ExperimentConfig, **extra: Any
    ) -> Path:
        return self.write_json(
            "model_ref.json",
            {
                **model_ref_to_dict(ref),
                "weights_sha256": weights_sha256,
                "config_hash": config.config_hash,
                "data_kind": config.data_kind,
                **extra,
            },
        )

    def read_model_ref(self) -> ModelRef:
        return model_ref_from_dict(json.loads((self.path / "model_ref.json").read_text("utf-8")))
