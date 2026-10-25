# Session Lot B — INTERFACE

Branche : `lot/b-interface` · Chemins : `src/maisonaccueil/app/`, `tests/app/`, `docs/lots/B-interface.md`

## Objectif
Interface PySide6 de l'application, dans un processus distinct du traitement vidéo.

## Travail attendu
- Fenêtre principale : aperçu du flux, état système (caméra, moteur, `ModelRef`, images
  abandonnées), journal des arrivées.
- Gestion des profils : création (avec consentement obligatoire), édition du message
  d'accueil, suppression confirmée (suppression complète déléguée au stockage).
- Réglages : source caméra (`SourceConfig`), moteur, volume.
- Écran de collecte guidée (parcours de `docs/DATA_POLICY.md`) : affichage, consignes,
  progression ; la logique de stockage passe par des interfaces.
- Code dépendant uniquement des contrats ; dépôts et sources injectés (adaptateurs
  `Fake*` pour développer). **Ne jamais afficher de résultat de reconnaissance inventé** :
  en l'absence de moteur, afficher « moteur non disponible ».
- Scores affichés comme « score technique (non calibré) », jamais en %.
- Séparer modèles de vue (testables sans Qt) et widgets. Tests Qt marqués `gui`, exécutés
  avec `QT_QPA_PLATFORM=offscreen`, ignorés proprement si Qt ne se charge pas
  (ex. `libEGL` absent).
- Remplacer `app/main.py` (point d'entrée `maisonaccueil`).

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
