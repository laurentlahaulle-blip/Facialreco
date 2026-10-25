# Lot D — Reconnaissance de référence

Branche `lot/d-recognition`. Chaîne détection → alignement → suivi → qualité →
représentation → galerie, derrière les interfaces `FaceDetector`, `FaceTracker`,
`FaceEmbedder`, `Gallery`, `Recognizer` (`maisonaccueil.contracts.interfaces`).
Licences et empreintes des poids : `docs/LICENSES-D.md`.

## Modules

| Module | Contenu |
|---|---|
| `vision/model_files.py` | `YUNET_2023MAR`, `SFACE_2021DEC` (nom, sha256, taille, licence, source) ; `resolve_verified()` (taille + sha256), `locate()`, `default_models_dir()` (`MAISONACCUEIL_MODELS_DIR` sinon `models/d`) ; `ModelFileMissingError`, `ModelIntegrityError` |
| `vision/detection.py` | `YuNetDetector` (`cv2.FaceDetectorYN`, CPU) → `Detection` avec `Landmarks5`, boîte rognée à l'image |
| `vision/alignment.py` | `PREPROCESS_ID = "align112-arcface5-v1"`, `align_face()` (similitude LMEDS vers gabarit ArcFace 112×112), `landmarks_from_bbox()` (repli), `AlignmentError` |
| `vision/tracking.py` | `IouTracker` : association gloutonne IoU, pertes tolérées (`max_missed_frames`, `max_missed_ms`), `lost_tracks()` renvoyé une seule fois, `reset()` |
| `vision/quality.py` | `assess_quality()` → `frozenset[QualityFlag]` (FACE_TOO_SMALL, BLURRY, LOW_LIGHT, NOT_FRONTAL), `QualityConfig` |
| `vision/fakes.py` | `FakeBlobDetector`, `FakeScriptedDetector`, `synthetic_blob_image()` |
| `recognition/embedders.py` | `SFaceEmbedder` (`cv2.FaceRecognizerSF`, 128 d, normalisé L2), `SFACE_MODEL_REF` (`origin="pretrained"`), `embed_aligned()` |
| `recognition/gallery.py` | `InMemoryGallery(model, store)` : chargement filtré par `ModelRef`, `add(persist=)`, `remove_profile`, `reload`, `rank()` top-k par profil, `search()` |
| `recognition/recognizer.py` | `ReferenceRecognizer`, `RecognizerConfig`, `build_reference_recognizer(store, models_dir)` |
| `recognition/fakes.py` | `FakeColorEmbedder`, `FakeRandomEmbedder`, `FakeEmbeddingStore`, `FakeLeakyEmbeddingStore` |

## Comportements à connaître

- `ModelRef` SFace : `opencv-zoo-sface@2021dec-0ba9fbfa#align112-arcface5-v1/cosine/128`.
  Toute modification de l'alignement change `PREPROCESS_ID` → représentations à recalculer.
- Ordre des points : coordonnées image (voir `docs/contract-changes/D-landmarks-order.md`).
  Notre alignement est vérifié équivalent à `FaceRecognizerSF.alignCrop` (écart moyen < 3
  niveaux de gris sur image synthétique).
- La galerie lève `IncompatibleEmbeddingError` pour une requête, un ajout, ou une ligne de
  stockage d'un autre modèle (protection contre un stockage qui filtrerait mal).
- Score d'un profil = meilleur score parmi ses références ; `references_compared` = nombre de
  références comparées. Cosinus : plus haut = plus proche ; L2 : plus bas = plus proche.
  **Ce sont des scores techniques, pas des probabilités.**
- Le `Recognizer` émet un résultat par visage suivi, même sans représentation (candidats
  vides + indicateurs) ; `FACE_TOO_SMALL` court-circuite le calcul de représentation.
- Les seuils de qualité et de suivi sont des valeurs de départ **non mesurées** ; à régler
  par le lot G sur `tune`. `OCCLUDED` n'est pas évalué.
- Pas de caméra ni de GPU utilisés : aucun `ResourceLock` nécessaire. Exécution CPU.
  Les objets OpenCV DNN ne sont pas thread-safe : une instance par thread/processus.
- ONNX Runtime n'est pas nécessaire (OpenCV DNN suffit pour YuNet et SFace).

## Ordres de grandeur de temps (information technique, pas une mesure de qualité)

Conteneur Linux 4 cœurs, CPU, entrée de bruit synthétique : SFace ≈ 13 ms par visage aligné,
YuNet ≈ 10 ms par image 640×480 (moyenne de 20 appels, après préchauffage).

## Tests

```
uv run pytest tests/contracts tests/vision tests/recognition -rs
```

- Sans poids : `Fake*` détecteur/embedder, suivi sur boîtes synthétiques, galerie sur
  vecteurs aléatoires, chaîne complète sur rectangles colorés synthétiques.
- Avec poids (ignorés si absents de `models/d`) : fumée YuNet sur bruit (zéro visage
  accepté), SFace (forme, norme, déterminisme), équivalence d'alignement, chaîne réelle.
- Aucune photo de personne, aucune mesure de précision.

## Limites

- Non testé sur caméra réelle ni sur visages réels (aucune donnée autorisée ici).
- Suivi purement géométrique (pas d'apparence) : deux personnes qui se croisent peuvent
  échanger leurs `TrackId`.
- `InMemoryGallery.search` remplit `frame_id`/`track_id` neutres (voir
  `docs/contract-changes/D-gallery-search.md`) ; préférer `rank()`.
- Réserve de licence sur les données d'entraînement de SFace (voir `docs/LICENSES-D.md`).
