# Répartition des fichiers et responsabilités

| Lot | Branche | Chemins autorisés en écriture | Dépend de |
|---|---|---|---|
| Coordinateur | `claude/maisonaccueil-coordinator-whqt6k` (intégration) | `src/maisonaccueil/contracts/`, `src/maisonaccueil/pipeline/`, `src/maisonaccueil/runtime/` (racine de composition du processus vision), `pyproject.toml`, `uv.lock`, `CLAUDE.md`, `README.md`, `docs/*.md`, `docs/prompts/`, `tests/contracts/`, `tests/pipeline/`, `tests/runtime/`, `tests/integration/`, `config/` | — |
| A — Capture | `lot/a-capture` | `src/maisonaccueil/capture/`, `tests/capture/`, `docs/lots/A-capture.md` | contrats |
| B — Interface | `lot/b-interface` | `src/maisonaccueil/app/`, `tests/app/`, `docs/lots/B-interface.md` | contrats (+ A, C via interfaces) |
| C — Stockage | `lot/c-storage` | `src/maisonaccueil/storage/`, `tests/storage/`, `docs/lots/C-storage.md` | contrats, schéma |
| D — Reconnaissance | `lot/d-recognition` | `src/maisonaccueil/vision/`, `src/maisonaccueil/recognition/`, `tests/vision/`, `tests/recognition/`, `docs/lots/D-recognition.md`, `docs/LICENSES-D.md` | contrats |
| E — Notre IA | `lot/e-training` | `src/maisonaccueil/training/`, `tests/training/`, `docs/lots/E-training.md` | contrats |
| F — Décision & accueil | `lot/f-decision` | `src/maisonaccueil/decision/`, `src/maisonaccueil/welcome/`, `tests/decision/`, `tests/welcome/`, `docs/lots/F-decision.md` | contrats |
| G — Évaluation | `lot/g-evaluation` | `src/maisonaccueil/evaluation/`, `tests/evaluation/`, `docs/lots/G-evaluation.md` | contrats |

Tous les lots peuvent créer `docs/contract-changes/<lot>-<sujet>.md` pour proposer un
changement de contrat, de schéma ou de dépendance. Le coordinateur décide et applique.

Ressources partagées : `camera-0`, `gpu-0` → `pipeline.ResourceLock` obligatoire.
Données de session : `var/<lot>/` dans le worktree (ignoré par Git).
