"""Réseau de représentation et pertes à marge angulaire (PyTorch).

Ce module importe ``torch`` au chargement : ne l'importer que paresseusement (via
``trainer``/``export``/``embedder``), jamais depuis ``maisonaccueil.training.__init__``.

Tous les paramètres sont initialisés ALÉATOIREMENT (Kaiming / Xavier) à partir de la
graine de l'expérience ; aucun poids tiers n'est chargé ici.

Architecture « MaisonNet » (petit ResNet à blocs résiduels pré-activés, type IR) :

    entrée (N, 3, 112, 112) normalisée [-1, 1]
    souche : conv 3×3 (C0) → BN → PReLU                         112×112
    étage i : bloc IR stride 2 puis (blocs_i − 1) blocs stride 1  ÷2 par étage
    sortie : BN2d → Dropout → aplatissement → Linear → BN1d     (N, embedding_dim)

La représentation exportée est normalisée L2 (``NormalizedEmbedding``) ; la comparaison
se fait par similarité cosinus (``Metric.COSINE``).
"""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .config import ExperimentConfig


class IRBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.PReLU(cout),
            nn.Conv2d(cout, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.shortcut: nn.Module
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.shortcut(x)


class EmbeddingNet(nn.Module):
    """Réseau de représentation entraîné de zéro (sortie non normalisée)."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        channels = list(config.stage_channels)
        c0 = channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, 1, 1, bias=False), nn.BatchNorm2d(c0), nn.PReLU(c0)
        )
        blocks: list[nn.Module] = []
        cin = c0
        for cout, n in zip(channels, config.stage_blocks, strict=True):
            blocks.append(IRBlock(cin, cout, 2))
            blocks.extend(IRBlock(cout, cout, 1) for _ in range(n - 1))
            cin = cout
        self.stages = nn.Sequential(*blocks)
        spatial = config.input_size // (2 ** len(channels))
        self.output = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Dropout(config.dropout),
            nn.Flatten(),
            nn.Linear(cin * spatial * spatial, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialisation aléatoire explicite (dépend uniquement de la graine torch)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d | nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.output(self.stages(self.stem(x)))
        return out


class NormalizedEmbedding(nn.Module):
    """Enveloppe d'inférence/export : représentation normalisée L2."""

    def __init__(self, net: EmbeddingNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2.0, dim=1)


class MarginHead(nn.Module):
    """Tête de classification à marge (utilisée UNIQUEMENT pendant l'apprentissage).

    - CosFace (LMCL) : logit cible = s · (cos θ − m)
    - ArcFace        : logit cible = s · cos(θ + m), avec la correction usuelle quand
      θ + m > π (cos θ − m·sin m) pour garder une fonction décroissante.
    """

    def __init__(self, dim: int, num_classes: int, loss: str, scale: float, margin: float):
        super().__init__()
        if loss not in ("cosface", "arcface"):
            raise ValueError(f"Perte inconnue {loss!r}")
        self.loss, self.scale, self.margin = loss, scale, margin
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)
        self._cos_m, self._sin_m = math.cos(margin), math.sin(margin)
        self._threshold = math.cos(math.pi - margin)
        self._mm = math.sin(math.pi - margin) * margin

    def logits(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        target_mask = F.one_hot(labels, self.num_classes).bool()
        if self.loss == "cosface":
            target = cos - self.margin
        else:
            sin = torch.sqrt((1.0 - cos * cos).clamp_min(0.0))
            phi = cos * self._cos_m - sin * self._sin_m
            target = torch.where(cos > self._threshold, phi, cos - self._mm)
        return self.scale * torch.where(target_mask, target, cos)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(embeddings, labels), labels)


def state_dict_sha256(module: nn.Module) -> str:
    """Empreinte déterministe des poids (indépendante du format de sérialisation)."""
    h = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(repr(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
