# Lot E — NOTRE IA : modèle de représentation faciale entraîné de zéro

Branche `lot/e-training` · code `src/maisonaccueil/training/` · tests `tests/training/`.

> **Aucun entraînement significatif n'a été lancé.** Il n'y a ni données réelles ni GPU
> réservé. Ce lot livre l'infrastructure ; aucune qualité n'est revendiquée. Les seuls
> essais prévus sont des tests de fumée de quelques pas sur données **synthétiques**
> (bruit, aucun visage), qui ne démontrent rien sur la reconnaissance.

## 1. Ce qui est entraîné de zéro, ce qui ne l'est pas

| Élément | Origine | Détail |
|---|---|---|
| Réseau de représentation `EmbeddingNet` | **de zéro** | Tous les poids initialisés aléatoirement (Kaiming normal pour les convolutions, Xavier pour la couche linéaire, BN à 1/0) à partir de la graine de l'expérience. Aucun poids tiers n'est chargé. |
| Tête de perte `MarginHead` (CosFace/ArcFace) | **de zéro** | Xavier uniforme ; utilisée seulement à l'apprentissage, non exportée. |
| Alignement 5 points (`preprocess.align_face`) | sans apprentissage | Similitude géométrique vers un gabarit fixe : aucun paramètre appris. |
| Détecteur de visages + 5 points | **préentraîné, hors lot E** | Fourni par le lot D (ex. YuNet). Indispensable en amont pour produire `Landmarks5`. |

Règle appliquée dans le code : `ExperimentConfig.origin` vaut `"scratch"` uniquement si
`init_weights is None` ; sinon `"finetuned"` (et `train` refuse ce cas dans ce lot).
`ModelRef.origin` est dérivé de cette propriété, jamais saisi à la main.

## 2. Entrée du réseau (prétraitement)

`ModelRef.preprocess = "align112-arcface5-v1"`, **le même identifiant que le prétraitement
de référence du lot D** (`maisonaccueil.vision.alignment.PREPROCESS_ID`, contrats 1.2.0) :

- `Landmarks5` en **coordonnées image** : œil à gauche dans l'image, œil à droite, nez,
  coin de bouche gauche-image, coin de bouche droite-image (ordre YuNet/ArcFace) ;
- similitude (`cv2.estimateAffinePartial2D`, LMEDS) vers le gabarit ArcFace 5 points 112×112 ;
- `warpAffine` 112×112, bilinéaire, bord noir → BGR uint8.

La règle « importer uniquement via les contrats » interdit d'importer `vision` depuis
`training` : la géométrie est recopiée dans `training/preprocess.py`. Vérifications :
`test_same_geometry_as_reference_alignment` (ignoré tant que `vision` n'est pas sur la
branche) et une comparaison ponctuelle avec `vision/alignment.py` de la branche
d'intégration (lecture seule) : sorties **identiques octet pour octet** sur 20 cas
aléatoires (rotation, échelle, bruit sur les points). Proposition pour supprimer la copie :
`docs/contract-changes/E-shared-alignment.md`.

Normalisation des pixels, **interne au modèle** (fixée par la configuration et les poids,
donc couverte par `ModelRef.version`) : BGR → RGB, float32, `(x − 127,5) / 127,5`,
disposition `(3, 112, 112)`.

Filtre de qualité de l'adaptateur (pas un prétraitement) : visage refusé (`None`) sans
points, si la distance inter-oculaire < 16 px, ou si la sortie n'est pas finie / nulle.

## 3. Architecture et perte

**MaisonNet** : petit ResNet à blocs résiduels pré-activés de type « IR ».

```
entrée (N, 3, 112, 112)
souche   conv3×3 C0 → BN → PReLU
étage i  bloc IR stride 2 (BN-conv3×3-BN-PReLU-conv3×3-BN + raccourci 1×1 si besoin)
         puis (blocs_i − 1) blocs IR stride 1
sortie   BN2d → Dropout → aplatissement → Linear → BN1d → (N, embedding_dim)
inférence/export : normalisation L2 ; comparaison cosinus (Metric.COSINE)
```

Configuration par défaut : canaux (32, 64, 128, 256), blocs (1, 2, 2, 1), dimension 128,
dropout 0,2 — environ 3,2 M paramètres (estimation à la main, non mesurée faute de
torch ici ; `model_ref.json` enregistre le compte exact). Choix : modèle assez petit pour
un jeu familial de quelques personnes et pour l'inférence CPU ; la taille réelle
dépendra du volume de données collectées.

**Perte** : CosFace (LMCL) par défaut, `s = 30`, `m = 0,35` ; ArcFace disponible
(`loss="arcface"`, correction usuelle quand θ + m > π). CosFace est choisie par défaut car
réputée plus stable au départ d'un apprentissage de zéro sur peu de classes. Ces
hyperparamètres sont des points de départ, non réglés.

Optimisation : AdamW (lr 1e-3, weight decay 5e-4), décroissance cosinus par pas, miroir
horizontal aléatoire (`hflip`), lots sans dernier lot partiel (BatchNorm).

## 4. Données : manifeste par séance

`manifest.py` — l'unité de découpage est la **séance** (`SessionId`, table
`collection_sessions`), jamais l'image.

- `manifest_from_database(db)` lit la base en **lecture seule** (`mode=ro`) : séances,
  découpage, images de référence. Séance sans découpage → refus (ou ignorée sur option).
  Les images de référence ne sont pas alignées : `preprocess = "raw"`.
- Contrôles (`Manifest.validate`, appelé à la construction) : séance déclarée deux fois,
  image rattachée à une séance inconnue, **même chemin ou même empreinte sha256 dans deux
  séances** → `SplitLeakError` ; séance sans profil (inconnus) en `train` → refus.
- Option stricte `check_identity_disjoint("train", "test")` : aucune personne commune.
- `training_samples()` ne renvoie que `train` ; `tune`, `test` et `enroll` ne sont jamais
  lus par l'apprentissage.
- `assign_splits(ids, fractions, seed)` propose un découpage **par séance**,
  déterministe ; il n'écrit rien en base (le lot C/B enregistre le choix).
- L'apprentissage exige un manifeste `preprocess = "align112-arcface5-v1"` (images déjà
  alignées 112×112) : un manifeste `raw` est refusé.

Étape manquante : produire les images alignées à partir des références `raw` demande les
5 points du détecteur du lot D (à brancher à l'intégration : détecteur via
`contracts.FaceDetector`, puis `align_face`, sortie dans `var/e/aligned/`).

## 5. Expériences : `runs/<id>/`

`id = <UTC>-<nom>-<hash config[:8]>`. Dossier ignoré par Git (`runs/`, `var/`, `*.pt`,
`*.onnx`). Contenu : `config.json` (configuration complète, schéma
`maisonaccueil-e-config/1`), `manifest_counts.json` (effectifs seulement), `metrics.jsonl`
(perte, lr, temps par pas, `time.monotonic_ns`), `weights.pt` (réseau sans la tête),
`model_ref.json` (ModelRef + `weights_sha256` + `config_hash` + `data_kind`),
`summary.json`, et après export `model.onnx` + `onnx.json`.

Graine : `random`, `numpy`, `torch` + `torch.use_deterministic_algorithms(True,
warn_only=True)`, générateurs dédiés pour le mélange et l'augmentation.

`ModelRef` produit :

| Champ | Valeur |
|---|---|
| `name` | `maisonaccueil-scratch` (`maisonaccueil-scratch-smoke` si `data_kind="synthetic"`, pour qu'un modèle de fumée ne soit jamais confondu avec un vrai) |
| `version` | `<nom d'expérience>-<sha256 des poids[:12]>` (empreinte des tenseurs, indépendante du format de fichier) |
| `embedding_dim` | dimension de la configuration |
| `metric` | `cosine` |
| `preprocess` | `align112-arcface5-v1` |
| `origin` | `scratch` (dérivé de `init_weights is None`) |

## 6. GPU

`device_guard(config, lock_dir)` réserve `ResourceLock("gpu-0", owner="lot-E")` pendant
**tout** l'entraînement dès que `device` commence par `cuda` ; GPU déjà réservé →
`ResourceBusyError`, pas d'entraînement concurrent. Sur CPU, rien n'est réservé. Test
matériel `@pytest.mark.gpu` (non exécuté ici).

## 7. Export et adaptateur `FaceEmbedder`

- `export.export_onnx(run)` : `NormalizedEmbedding` → ONNX (opset 17, axe lot dynamique,
  entrée `input` (N,3,112,112), sortie `embedding` normalisée L2), exporteur TorchScript
  (`dynamo=False`) pour ne pas dépendre d'`onnxscript`.
- `ScratchFaceEmbedder(model_ref, backend)` implémente `contracts.FaceEmbedder` :
  points → alignement → normalisation → backend → `Embedding` float32 normalisé portant
  le `ModelRef` de l'expérience. Backends : `OnnxBackend` (onnxruntime CPU, paresseux),
  `TorchBackend` (paresseux), ou un `Fake*` en test. `ScratchFaceEmbedder.from_run(path)`.

## 8. Ligne de commande

```
python -m maisonaccueil.training manifest --db var/e/base.sqlite --out var/e/manifest.json
python -m maisonaccueil.training check var/e/manifest.json [--disjoint train test]
python -m maisonaccueil.training train --config c.json --manifest m.json --data-root var/e/aligned
python -m maisonaccueil.training export runs/<id>
python -m maisonaccueil.training smoke --runs-root var/e/runs   # synthétique, 3 pas, CPU
```

## 9. Environnement et tests (état réel de cette session)

- **PyTorch non installé.** L'index CPU `download.pytorch.org` est refusé par le proxy
  (403). La roue PyPI Linux tire les bibliothèques CUDA : ≈ 3,0 Go **compressés**
  (torch 2.14.1 : 528 Mo + cuDNN, cuBLAS, NCCL…), au-delà de la limite fixée (~3 Go).
  Essai `--no-deps` (torch seul) : `import torch` échoue (`libcudart.so.13` absente). Il a
  été désinstallé ; `.venv` remis à l'état `uv sync --extra gui`.
- `onnx` / `onnxruntime` absents → test d'export ignoré.
- Le paquet `maisonaccueil.training` s'importe sans torch (test dédié en sous-processus).
- `uv run pytest tests/contracts tests/training` : **45 réussis, 2 ignorés**
  (`test_torch_smoke.py` entier : torch absent ; équivalence avec `vision.alignment` :
  module absent de cette branche). Suite complète : 188 réussis, 4 ignorés.
- **Non exécuté** : tout le code PyTorch (`net.py`, `trainer.py`, `export.py`,
  `TorchBackend`). Il compile, mais n'a jamais tourné. Premier geste quand torch sera
  disponible : `uv run pytest tests/training` (forme du réseau, pertes finies et
  différentiables, run de 3 pas, déterminisme à graine égale, adaptateur torch, export
  ONNX équivalent si onnxruntime présent).

## 10. Limites

- Aucune donnée réelle, aucun entraînement, aucune mesure : rien n'est affirmé sur la
  qualité. L'évaluation relève du lot G sur des séances `tune`/`test` indépendantes.
- Un réseau entraîné de zéro sur les quelques personnes d'une famille risque fortement le
  sur-apprentissage et une mauvaise généralisation aux inconnus ; il n'est pas certain
  qu'il puisse rivaliser avec le moteur de référence préentraîné. À mesurer, pas à supposer.
- Aucune validation pendant l'apprentissage (pas d'arrêt précoce) : volontaire pour ne pas
  toucher `tune`/`test` ; on pourrait réserver des séances `train` pour cela.
- Alignement recopié de D (voir proposition) ; convention des points dépendante de D.
- Pas de reprise d'entraînement interrompu, pas d'apprentissage multi-GPU.

## 11. Propositions de contrat

- `docs/contract-changes/E-onnx-export.md` : dépendances `onnx`/`onnxruntime` pour
  l'extra `training`, et installation de torch CPU.
- `docs/contract-changes/E-shared-alignment.md` : alignement partagé au lieu d'une copie.
- (Constat) `FaceEmbedder` n'est pas réexporté par `maisonaccueil.contracts` ; les tests
  l'importent depuis `maisonaccueil.contracts.interfaces` (rejoint la proposition du lot A).
