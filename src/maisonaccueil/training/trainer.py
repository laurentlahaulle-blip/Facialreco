"""Boucle d'apprentissage (PyTorch importé paresseusement).

``train`` :
1. crée ``runs/<id>/`` (configuration + effectifs) ;
2. réserve ``ResourceLock("gpu-0")`` si la configuration utilise CUDA ;
3. fixe les graines (random, numpy, torch) et le déterminisme ;
4. initialise ALÉATOIREMENT le réseau et la tête de perte ;
5. entraîne sur le seul découpage ``train`` ; journalise chaque pas ;
6. enregistre les poids du réseau (sans la tête), leur empreinte et le ``ModelRef``.

Aucune mesure de qualité n'est produite ici : l'évaluation appartient au lot G, sur des
séances ``tune``/``test`` indépendantes.
"""

from __future__ import annotations

import importlib.util
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from maisonaccueil.contracts import ModelRef

from .config import ExperimentConfig
from .data import AlignedTrainingSet
from .runs import RunDir, build_model_ref, device_guard, file_sha256

if TYPE_CHECKING:
    from .net import EmbeddingNet


class TorchUnavailableError(RuntimeError):
    pass


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise TorchUnavailableError(
            "PyTorch absent : installer l'extra 'training' dans le .venv du worktree"
        ) from exc
    return torch


@dataclass(frozen=True, slots=True)
class TrainResult:
    run: RunDir
    model_ref: ModelRef
    steps: int
    last_loss: float
    weights_sha256: str


def seed_everything(seed: int, uses_cuda: bool) -> None:
    torch = require_torch()
    if uses_cuda:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def train(
    config: ExperimentConfig,
    training_set: AlignedTrainingSet,
    runs_root: Path | None = None,
    lock_dir: Path | None = None,
    manifest_counts: dict[str, Any] | None = None,
) -> TrainResult:
    torch = require_torch()
    from .net import EmbeddingNet, MarginHead, count_parameters, state_dict_sha256

    if config.init_weights is not None:
        raise NotImplementedError(
            "Initialisation depuis des poids tiers non prise en charge dans ce lot "
            "(le modèle serait 'finetuned', pas 'scratch')"
        )
    if len(training_set) < 2:
        raise ValueError("Au moins 2 images d'apprentissage sont nécessaires")

    run = RunDir.create(config, runs_root)
    if manifest_counts is not None:
        run.write_json("manifest_counts.json", manifest_counts)

    t0 = time.monotonic_ns()
    with device_guard(config, lock_dir):
        seed_everything(config.seed, config.uses_cuda)
        device = torch.device(config.device)
        net = EmbeddingNet(config).to(device)
        head = MarginHead(
            config.embedding_dim,
            training_set.num_classes,
            config.loss,
            config.scale,
            config.margin,
        ).to(device)
        params = list(net.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

        n = len(training_set)
        drop_last = n > config.batch_size  # évite un dernier lot de taille 1 (BatchNorm)
        steps_per_epoch = n // config.batch_size if drop_last else 1
        total_steps = config.epochs * steps_per_epoch
        if config.max_steps is not None:
            total_steps = min(total_steps, config.max_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps)
        )
        loader_gen = torch.Generator().manual_seed(config.seed)
        flip_gen = torch.Generator().manual_seed(config.seed + 1)
        loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=config.num_workers,
            generator=loader_gen,
        )

        net.train()
        head.train()
        step, last_loss = 0, float("nan")
        for epoch in range(config.epochs):
            for images, labels in loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                if config.hflip:
                    mask = (torch.rand(images.shape[0], generator=flip_gen) < 0.5).to(device)
                    images = torch.where(mask[:, None, None, None], images.flip(3), images)
                loss = head(net(images), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Perte non finie au pas {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                last_loss = float(loss.detach().cpu())
                run.append_metrics(
                    {
                        "step": step,
                        "epoch": epoch,
                        "loss": last_loss,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_s": (time.monotonic_ns() - t0) / 1e9,
                    }
                )
                if step >= total_steps:
                    break
            if step >= total_steps:
                break

        net.eval()
        net.to("cpu")
        weights_sha = state_dict_sha256(net)
        torch.save(net.state_dict(), run.weights_path)

    ref = build_model_ref(config, weights_sha)
    run.write_model_ref(
        ref,
        weights_sha,
        config,
        weights_file_sha256=file_sha256(run.weights_path),
        classes=len(training_set.classes),
        parameters=count_parameters(net),
    )
    run.write_json(
        "summary.json",
        {
            "steps": step,
            "last_loss": last_loss,
            "duration_s": (time.monotonic_ns() - t0) / 1e9,
            "device": config.device,
            "torch_version": str(torch.__version__),
            "train_samples": n,
            "data_kind": config.data_kind,
            "trained_from_scratch": config.origin == "scratch",
            "quality_note": (
                "Aucune mesure de qualité : évaluation par le lot G sur séances tune/test."
                + (
                    " Données SYNTHÉTIQUES : ce modèle ne vaut rien pour reconnaître des "
                    "visages ; test logiciel uniquement."
                    if config.data_kind == "synthetic"
                    else ""
                )
            ),
        },
    )
    return TrainResult(run, ref, step, last_loss, weights_sha)


def load_embedding_net(run: RunDir) -> tuple[EmbeddingNet, ExperimentConfig, ModelRef]:
    """Recharge le réseau d'une expérience et vérifie l'empreinte des poids."""
    torch = require_torch()
    import json

    from .net import EmbeddingNet, state_dict_sha256

    config = ExperimentConfig.load(run.path / "config.json")
    meta = json.loads((run.path / "model_ref.json").read_text("utf-8"))
    ref = run.read_model_ref()
    net = EmbeddingNet(config)
    state = torch.load(run.weights_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    if state_dict_sha256(net) != meta["weights_sha256"]:
        raise ValueError(f"{run.weights_path} : empreinte des poids différente de model_ref.json")
    return net, config, ref
