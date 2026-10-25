# CLAUDE.md — Règles de collaboration MaisonAccueil

Ce fichier s'applique à **toute** session (coordinateur ou travailleur, humain ou agent).

## Projet

Application de bureau locale (Python, PySide6, OpenCV, SQLite) de reconnaissance faciale
familiale avec accueil vocal. Voir `docs/ARCHITECTURE.md`.

## Rôles

- **Coordinateur** (branche d'intégration `claude/maisonaccueil-coordinator-whqt6k`) :
  propriétaire de `src/maisonaccueil/contracts/`, `src/maisonaccueil/pipeline/`,
  `pyproject.toml`, `uv.lock`, `CLAUDE.md`, `docs/` (hors notes de lot), `tests/integration/`.
  Seul autorisé à fusionner dans la branche d'intégration.
- **Travailleurs** (lots A à G) : une session = une branche `lot/<lettre>-<nom>` = un worktree.
  Ils ne modifient QUE les chemins de leur lot (voir `docs/OWNERSHIP.md`).

## Règles impératives

1. N'écrire que dans **son propre worktree** et dans les chemins de **son lot**.
   Interdiction d'écrire dans le dossier d'une autre session (les worktrees ne sont pas une
   barrière de sécurité).
2. Ne jamais modifier `contracts/`, `pipeline/`, `pyproject.toml`, `uv.lock` ou le schéma SQL.
   Pour un besoin : écrire une proposition dans `docs/contract-changes/<lot>-<sujet>.md`
   (seul fichier hors périmètre autorisé) et la signaler dans le rapport final.
3. Ne jamais fusionner, rebaser ou pousser vers la branche d'intégration. Ne pas pousser
   vers un dépôt distant. Committer uniquement sur sa branche de lot.
4. Importer les autres modules **uniquement via `maisonaccueil.contracts`** (interfaces),
   jamais l'implémentation d'un autre lot.
5. Pas de dépendance globale, pas de modification de pilotes, pas d'outil supplémentaire.
   Environnement Python : `.venv` du worktree (`uv sync`).
6. Données : aucune photo ni donnée issue d'un autre projet ; aucune personne réelle
   inventée ; aucune donnée biométrique committée (`var/`, `data/`, `models/` sont ignorés).
   Les données synthétiques servent uniquement aux tests logiciels.
7. Chaque session utilise sa base SQLite et son dossier de sortie : `var/<lot>/`.
8. Caméra physique et GPU : réservation obligatoire via
   `maisonaccueil.pipeline.ResourceLock("camera-0" | "gpu-0", owner=...)`. Pas de capture
   exclusive ni d'entraînement concurrents. Tests matériels marqués `@pytest.mark.camera` /
   `@pytest.mark.gpu`.
9. Aucun enregistrement vidéo permanent par défaut. Pas de service réseau exposé ; flux
   réseau caméra en local uniquement.
10. **Scores ≠ probabilités** : un score de similarité ne s'affiche jamais en « % de
    fiabilité ». Aucune précision annoncée sans mesure du lot G sur un test indépendant.
11. Les représentations faciales portent leur `ModelRef` ; ne jamais comparer des
    représentations de modèles différents (`ensure_compatible`).
12. Licences : documenter séparément la licence du **code** et celle des **poids** avant
    tout téléchargement ou utilisation (`docs/LICENSES.md`, via proposition si besoin).
13. Aucun nettoyage destructif (`git clean -fdx`, `reset --hard`, suppression de branches)
    sur un travail non intégré.

## Conventions

- Python ≥ 3.11 (cible 3.12), `ruff` (ligne 100), typage complet ; `mypy` strict sur contrats.
- Tests : `tests/<module>/`. Lancer `uv run pytest tests/<module>` avant chaque commit.
- Doubles de test nommés `Fake*`, jamais présentés comme résultats réels.
- Durées : `time.monotonic_ns()` ; horodatages affichés : UTC timezone-aware.
- Langue : code et identifiants en anglais, docs et messages utilisateur en français.
- Commits petits, préfixés par le lot : `lot-A: ...`, `coord: ...`.

## Vérifications avant commit

```
uv run ruff check src tests
uv run pytest tests/<votre_module> tests/contracts
```
