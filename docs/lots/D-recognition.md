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
| `recognition/enrollment.py` | `ReferenceEnroller.enroll(refs, store, already_represented=None)` → `EnrollmentReport` ; `embed_image()`, `select_face()`, `EnrollmentConfig`, `SkipReason`, `read_image()`, `build_reference_enroller(models_dir)` |
| `recognition/fakes.py` | `FakeColorEmbedder`, `FakeRandomEmbedder`, `FakeEmbeddingStore`, `FakeIndexedEmbeddingStore`, `FakeLeakyEmbeddingStore` |

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

## Enrôlement (calcul des représentations de référence)

Entrée : itérable de `ReferenceItem(profile_id, reference_id, image_path)` (ou tuples) fourni
par l'appelant (ex. `ReferenceRepository.list_for_profile` du lot C, split `enroll`) et un
`EmbeddingStore`. Le module ne lit jamais SQLite.

1. Lecture (`np.fromfile` + `cv2.imdecode` : chemins non ASCII sous Windows) ; image
   > 4096 px de côté refusée.
2. Détection ; visage principal = le plus grand. Si un second visage a une aire ≥ 50 % du
   plus grand → `ambiguous_faces` (risque d'enrôler la mauvaise personne). Pas de repli sur
   un visage plus petit si le principal est de mauvaise qualité.
3. Qualité : rejet si `FACE_TOO_SMALL`, `BLURRY`, `LOW_LIGHT`, `NOT_FRONTAL` (seuils non
   mesurés, configurables via `EnrollmentConfig`).
4. Représentation par le `FaceEmbedder` fourni (SFace, ou le moteur du lot E), puis
   `store.add_embedding(profile_id, reference_id, emb)` ; le `ModelRef` exact est porté par
   l'`Embedding`.
5. Bilan `EnrollmentReport` : `processed`, `added` (avec `embedding_id`), `skipped` (raison
   + détail), `summary()` en français, sans score.

Raisons d'exclusion : `already_represented`, `duplicate_in_batch`, `file_missing`,
`unreadable`, `no_face`, `ambiguous_faces`, `low_quality`, `embedding_failed`,
`store_error` (exceptions `KeyError`/`ValueError`/`RuntimeError` du stockage, ex. référence
d'un autre profil ; les autres exceptions, ex. base inaccessible, remontent).

Idempotence : voir `docs/contract-changes/D-represented-references.md` (méthode
`represented_reference_ids(model)` proposée ; utilisée si présente, sinon
`already_represented` fourni par l'appelant, sinon mémoire de l'instance).

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
- Enrôlement : sans `represented_reference_ids` côté C ni `already_represented`, une
  instance neuve recalcule (et duplique) les références déjà représentées.
- Réserve de licence sur les données d'entraînement de SFace (voir `docs/LICENSES-D.md`).
