# Audit initial (2026-10-25)

## Emplacement

- La session s'exécute dans un **conteneur Linux cloud éphémère** (Claude Code distant),
  pas sur le PC Windows. `C:\DEV\MaisonAccueil` n'est donc pas accessible d'ici.
- Dossier du projet : `/home/user/Facialreco` — clone du dépôt dédié
  `laurentlahaulle-blip/Facialreco`, **vide** au départ (aucun commit, aucun fichier) :
  pas de réutilisation d'un dossier non vide, aucun autre projet présent dans le conteneur.
- Branche d'intégration : `claude/maisonaccueil-coordinator-whqt6k`.
- Sur le PC Windows, emplacement recommandé : `C:\DEV\MaisonAccueil` (hors OneDrive), voir
  `docs/PARALLEL_SESSIONS.md`.

## Système et outils

| Élément | Valeur |
|---|---|
| OS | Linux 6.18 (Ubuntu), conteneur |
| Git | disponible (worktrees fonctionnels) |
| Python | 3.11, 3.12, 3.13 système ; projet en **3.12** via `uv` 0.11 (`.venv` local) |
| Claude Code | 2.1.310 |
| VS Code | absent du conteneur |
| FFmpeg | 6.1.1 |
| CPU | Intel Xeon 2,1 GHz, 4 vCPU |
| Mémoire | 15 Gio, sans swap |
| GPU | **aucun** (pas de `nvidia-smi`) |
| Caméra | **aucune** (`/dev/video*` absent) |
| Disque | ~31 Go disponibles |
| Réseau | PyPI accessible via proxy |
| Bibliothèques système Qt | `libEGL.so.1` absent → PySide6 installé mais Qt ne se charge pas |

## Parallélisme

- Outil `Agent` avec isolation `worktree` disponible : sous-agents en arrière-plan, chacun
  dans son propre worktree Git. Utilisé pour les lots (voir rapport).
- Pas d'outil d'« équipe d'agents » expérimental activé ; aucun activé volontairement.
- 4 vCPU / 15 Gio : 4 travailleurs simultanés raisonnables pour du code et des tests
  légers ; pas d'entraînement significatif possible ici (ni GPU, ni données).

## Conséquences

- Les tests caméra, GPU et GUI s'exécutent sur le PC cible.
- Le conteneur est éphémère : le travail doit être poussé sur la branche distante dédiée
  pour être conservé : seule la branche d'intégration est poussée (accord de l'utilisateur
  obtenu le 2026-10-25) ; les branches de lot restent locales.
