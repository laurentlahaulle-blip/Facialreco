# Session Lot E — NOTRE IA (modèle entraîné de zéro)

Branche : `lot/e-training` · Chemins : `src/maisonaccueil/training/`, `tests/training/`, `docs/lots/E-training.md`

## Objectif
Infrastructure d'un modèle de représentation faciale **entraîné de zéro** par le projet,
interchangeable via `FaceEmbedder`.

## Travail attendu
- Document `docs/lots/E-training.md` : architecture (ex. petit CNN type ResNet/MobileNet
  initialisé aléatoirement), perte (ex. ArcFace/CosFace), entrées alignées (même
  `preprocess` que D ou le sien, versionné). Indiquer **exactement** ce qui est entraîné
  de zéro et ce qui serait préentraîné (le détecteur/alignement de D l'est, par exemple).
  Un modèle adapté depuis des poids tiers s'appelle `origin="finetuned"`, jamais "scratch".
- Préparation des données : manifeste par **séance** (`SessionId`), découpage
  train/tune/test par séance, jamais par image ; refus de tout chevauchement.
- Scripts d'entraînement PyTorch (extra `training`, à installer localement dans `.venv`
  seulement si nécessaire), configuration d'expérience versionnée, graine, journal
  `runs/<id>/` (ignoré par Git) : config, métriques, hash des poids, `ModelRef`.
- Export ONNX + adaptateur `FaceEmbedder` produisant `ModelRef(origin="scratch")`.
- Réservation `ResourceLock("gpu-0")` avant tout entraînement.
- **Ne lance aucun entraînement significatif** : pas de données réelles, pas de GPU
  réservé. Seul un test de fumée de quelques itérations sur données synthétiques (CPU)
  est permis pour vérifier le code ; il ne démontre rien sur la qualité.
- Si PyTorch n'est pas installé, les tests correspondants sont ignorés proprement.

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
