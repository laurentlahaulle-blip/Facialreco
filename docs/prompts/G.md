# Session Lot G — ÉVALUATION

Branche : `lot/g-evaluation` · Chemins : `src/maisonaccueil/evaluation/`, `tests/evaluation/`, `docs/lots/G-evaluation.md`

## Objectif
Mesurer de façon vérifiable, sur des passages réels futurs, la chaîne complète et chaque
moteur — sans jamais annoncer de chiffre sans données.

## Travail attendu
- Format d'annotation de **passages** (séance, intervalle temporel, personne attendue ou
  « inconnu non enregistré ») et format de journal d'exécution (résultats, décisions,
  arrivées, accueils horodatés) — JSON Lines documentés.
- Découpage par séances `train`/`tune`/`test` avec contrôle d'absence de chevauchement ;
  traçabilité « ce test a-t-il déjà servi au réglage ? » (test brûlé → nouveau test exigé).
- Métriques par passage ET par personne : bonne identification, mauvaise identité, connu
  non reconnu, inconnu accepté à tort, message répété, délai jusqu'à l'accueil ;
  effectifs toujours affichés, intervalles de confiance (ex. Wilson) quand pertinent.
- Comparaison de moteurs sur les **mêmes passages** ; balayage de seuils sur `tune`
  uniquement ; courbes score/erreurs. Calibration éventuelle documentée et marquée.
- Rejeu hors ligne : exécuter `Recognizer` + `DecisionEngine` (via interfaces) sur une
  vidéo de séance (fichier local, jamais committé).
- Rapport Markdown généré. Tests avec données synthétiques et `Fake*` uniquement pour
  vérifier les calculs ; mentionner explicitement que ces chiffres ne mesurent rien de
  biométrique.

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
