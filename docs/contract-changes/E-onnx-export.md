# E — Dépendances d'export ONNX et installation de PyTorch CPU

**Statut** : différé (coordinateur, 2026-10-25) — nouvelles dépendances et index PyTorch CPU soumis à l'accord de l'utilisateur ; l'ouverture de download.pytorch.org relève de la configuration réseau de l'environnement (hors du ressort des sessions). À installer sur le PC cible. (lot E)

## Besoin

1. L'export (`training/export.py`) produit `model.onnx` ; vérifier qu'il reproduit le
   réseau PyTorch, et l'utiliser via `ScratchFaceEmbedder` (`OnnxBackend`), demande
   `onnxruntime` (présent seulement dans l'extra `recognition`) et, selon la version de
   torch, le paquet `onnx`. Aujourd'hui le test `test_onnx_export_matches_torch` est ignoré.
2. `torch` (extra `training`) n'a pas pu être installé dans le conteneur :
   `download.pytorch.org` refusé par le proxy (403) ; la roue PyPI Linux tire ≈ 3,0 Go
   compressés de bibliothèques CUDA (inutiles sans GPU) ; `--no-deps` ne s'importe pas.

## Changement proposé

```toml
[project.optional-dependencies]
training = ["torch>=2.2", "torchvision>=0.17", "onnx>=1.16", "onnxruntime>=1.17"]
```

`torchvision` n'est pas utilisé par le lot E (pourrait être retiré).

Pour les postes sans GPU (et la CI), source CPU de torch via uv :

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu", marker = "sys_platform == 'linux'" }
```

(à adapter au poste Windows cible, qui aura éventuellement un GPU : roue CUDA). Et
autoriser `download.pytorch.org` dans le proxy de l'environnement cloud.

## Compatibilité

Extras optionnels uniquement ; aucun impact sur le cœur. Licences : torch (BSD-3),
onnx (Apache-2.0), onnxruntime (MIT) — code ; aucun poids téléchargé.
