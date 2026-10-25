# MaisonAccueil

Application de bureau **locale** de reconnaissance faciale pour un usage familial :
une caméra dans la salle à manger identifie les proches volontaires enregistrés et
diffuse un message vocal personnalisé à leur arrivée.

- Aucun service cloud requis, aucun enregistrement vidéo permanent par défaut.
- Aucune précision n'est annoncée avant d'avoir été mesurée (voir `docs/EVALUATION.md`).

## Démarrage (développement)

```powershell
# Windows (PowerShell), depuis le dossier du projet
uv venv --python 3.12 .venv
uv sync --extra gui
.venv\Scripts\python -m pytest
.venv\Scripts\maisonaccueil
```

```bash
# Linux / macOS
uv venv --python 3.12 .venv && uv sync --extra gui
.venv/bin/python -m pytest
```

Extras optionnels : `recognition` (ONNX Runtime), `training` (PyTorch — volumineux).

## Utilisation

```powershell
.venv\Scripts\maisonaccueil --check      # diagnostic (Qt, moteur)
.venv\Scripts\maisonaccueil --demo       # source synthétique, données en mémoire
.venv\Scripts\maisonaccueil              # base var\b\maisonaccueil.sqlite, moteur de référence
```

Le moteur de référence (YuNet + SFace, OpenCV Zoo) exige les poids dans `models\d\`
(sources et empreintes sha256 : `docs/LICENSES-D.md`). Sans eux, l'application affiche
« moteur non disponible ». Les images collectées dans des séances « enroll » sont
transformées automatiquement en représentations par le processus vision.

## Documentation

- `CLAUDE.md` — règles de collaboration (humains et agents).
- `docs/ARCHITECTURE.md` — architecture, processus, flux de données.
- `docs/OWNERSHIP.md` — répartition des fichiers par lot.
- `docs/TASKS.md` — tableau de suivi et dépendances.
- `docs/prompts/` — prompts autonomes des sessions de travail.
- `docs/PARALLEL_SESSIONS.md` — ouverture des sessions parallèles (worktrees, VS Code).
