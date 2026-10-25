# Session Lot D — RECONNAISSANCE (référence)

Branche : `lot/d-recognition` · Chemins : `src/maisonaccueil/vision/`, `src/maisonaccueil/recognition/`, `tests/vision/`, `tests/recognition/`, `docs/lots/D-recognition.md`, `docs/LICENSES-D.md`

## Objectif
Chaîne de reconnaissance de référence : détection, alignement, suivi, représentation,
galerie, derrière les interfaces `FaceDetector`, `FaceTracker`, `FaceEmbedder`, `Gallery`,
`Recognizer`.

## Travail attendu
1. **Licences d'abord** (`docs/LICENSES-D.md`) : pour chaque candidat (ex. détecteur YuNet
   et reconnaisseur SFace d'OpenCV Zoo, modèles InsightFace, MediaPipe, dlib, etc.),
   documenter SÉPARÉMENT la licence du code et celle des POIDS, la source exacte,
   la compatibilité avec un usage privé familial, et le verdict. **Aucun téléchargement de
   poids** avant ce document ; ne télécharger que des poids à licence claire et compatible,
   vers `models/` (ignoré par Git), avec hash sha256 vérifié et consigné.
2. Détecteur : adaptateur (ex. `cv2.FaceDetectorYN`) → `Detection` avec landmarks.
3. Alignement 5 points → visage normalisé (identifiant `preprocess` versionné).
4. Suivi : association IoU + gestion des pertes courtes → `TrackId` stable, `lost_tracks`.
5. Représentation de référence → `Embedding` avec `ModelRef(origin="pretrained")`.
6. Galerie en mémoire alimentée par un `EmbeddingStore` (interface) ; recherche top-k par
   profil ; `IncompatibleEmbeddingError` si modèles différents.
7. Indicateurs de qualité (`QualityFlag` : visage trop petit, flou…).
- Tests sans poids réels : `Fake*` détecteur/embedder, suivi sur boîtes synthétiques,
  galerie sur vecteurs aléatoires. Tests avec poids marqués et ignorés si absents.
- Aucun chiffre de précision : uniquement des tests de fonctionnement.
- Le moteur d'inférence ONNX Runtime est en extra `recognition` : vérifier la
  compatibilité avant de proposer son usage.

## Règles communes (obligatoires)

Tu es un travailleur du projet MaisonAccueil (reconnaissance faciale locale, usage
familial, accueil vocal). Le coordinateur possède les contrats et l'intégration.

1. Lis d'abord `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/OWNERSHIP.md`, `docs/DATA_POLICY.md`
   et TOUT `src/maisonaccueil/contracts/`.
2. Travaille UNIQUEMENT dans ton worktree et sur ta branche de lot. N'écris que dans les
   chemins de ton lot (liste ci-dessous) et, si nécessaire, dans
   `docs/contract-changes/<lot>-<sujet>.md`. N'écris jamais dans le dossier d'une autre session.
3. Ne modifie pas `contracts/`, `pipeline/`, `pyproject.toml`, `uv.lock`. Besoin d'un
   changement ou d'une dépendance → proposition écrite, puis contourne avec ce qui existe.
4. Ne fusionne pas, ne rebase pas la branche d'intégration, ne pousse rien vers un dépôt
   distant. Committe sur ta branche, en petits commits préfixés `lot-<X>: `.
5. Environnement : `uv sync --extra gui` dans ton worktree (crée `.venv` local). Aucune
   installation globale, aucun pilote.
6. Données de session dans `var/<lot>/` (ignoré par Git). Aucune photo réelle, aucune
   personne inventée ; données synthétiques uniquement pour tester le logiciel.
7. Caméra / GPU : `maisonaccueil.pipeline.ResourceLock` obligatoire ; tests matériels marqués
   `@pytest.mark.camera` / `@pytest.mark.gpu` et non exécutés par défaut.
8. Un score n'est pas une probabilité ; n'annonce aucune précision. Doubles de test nommés
   `Fake*`.
9. Avant chaque commit : `uv run ruff check src tests` et
   `uv run pytest tests/contracts tests/<tes modules>` doivent passer.
10. Termine par un rapport : fichiers créés, commits (hash), tests exécutés et résultats
    réels, limites, propositions de contrat, prochaine étape. Ne prétends pas avoir testé ce
    qui ne l'a pas été (caméra, GPU, son, GUI absents dans certains environnements).
