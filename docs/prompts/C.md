# Session Lot C — STOCKAGE

Branche : `lot/c-storage` · Chemins : `src/maisonaccueil/storage/`, `tests/storage/`, `docs/lots/C-storage.md`

## Objectif
Accès SQLite au schéma partagé `contracts/schema.sql` (propriété du coordinateur).

## Travail attendu
- Ouverture de base (`Database(path)`) : application du schéma, `PRAGMA foreign_keys=ON`,
  WAL, vérification/écriture de `schema_meta.schema_version` = `contracts.SCHEMA_VERSION`,
  refus d'une base de version supérieure.
- `ProfileRepository` : créer (consentement requis), lire, lister, modifier, désactiver,
  **supprimer complètement** (profil, références, représentations ET fichiers images locaux).
- `ReferenceRepository` : enregistrer une image de référence (fichier local, sha256),
  séances de collecte (`collection_sessions`, split).
- `SqliteEmbeddingStore` implémentant `contracts.interfaces.EmbeddingStore` : vecteurs
  float32 en BLOB, filtrage strict par `ModelRef.key`, refus d'un vecteur de dimension
  incohérente, purge par modèle.
- `SettingsRepository` (JSON) et journal `arrival_log` avec purge.
- Thread-safety : une connexion par thread ou verrou documenté.
- Tests sur base temporaire (`tmp_path`), aucune base partagée. Toute évolution du schéma
  → `docs/contract-changes/C-<sujet>.md`.

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
