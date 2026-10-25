# Session Lot A — CAPTURE

Branche : `lot/a-capture` · Chemins : `src/maisonaccueil/capture/`, `tests/capture/`, `docs/lots/A-capture.md`

## Objectif
Implémenter des sources d'images conformes à `contracts.interfaces.FrameSource`.

## Travail attendu
- `OpenCVSource` : webcam USB (index) et flux réseau local (RTSP/HTTP) via `cv2.VideoCapture`
  (backend MSMF/DSHOW sous Windows configurable), fichier vidéo.
- `SyntheticSource` : générateur d'images (motifs, compteur) pour les tests — aucun visage réel.
- Thread de capture dédié ; `start` non bloquant ; `on_frame` appelé avec des `Frame`
  (`FrameId` croissant, `monotonic_ns`, UTC) ; le consommateur dépose dans `pipeline.LatestQueue`.
- États `CaptureStatus` publiés via `on_state` ; reconnexion automatique avec attente
  exponentielle bornée après coupure ; `FAILED` après N tentatives (paramétrable).
- Arrêt propre et idempotent (`stop(timeout_s)`), libération de la caméra, pas de thread orphelin.
- Réservation `ResourceLock("camera-<index>")` pour les sources USB.
- Configuration via `SourceConfig` ; fonction d'énumération des index USB disponibles
  (sans ouvrir de capture exclusive en parallèle d'une autre session).
- Aucun enregistrement sur disque.
- Tests : source synthétique, cycle d'états, reconnexion simulée (faux `VideoCapture`
  injectable), arrêt sous charge, abandon d'images via `LatestQueue`. Tests caméra réelle
  marqués `camera`.

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
