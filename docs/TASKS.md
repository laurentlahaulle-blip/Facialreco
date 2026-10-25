# Tableau de suivi

Légende : ☐ à faire · ◐ en cours · ☑ intégré · ⛔ bloqué

## Socle (coordinateur)

| Tâche | Statut | Notes |
|---|---|---|
| Audit environnement | ☑ | `docs/AUDIT.md` |
| Squelette, `pyproject.toml`, `uv.lock`, `.venv` | ☑ | |
| Contrats typés v1.0.0 + schéma SQLite v1 | ☑ | `src/maisonaccueil/contracts/` |
| `LatestQueue`, `ResourceLock` | ☑ | `src/maisonaccueil/pipeline/` |
| CLAUDE.md, ownership, prompts | ☑ | |
| Orchestration processus vision ↔ interface | ☐ | après A + D (ou faux) + F |
| Tests d'intégration bout en bout (doubles de test) | ☐ | après A, C, F |

## Lots

| Lot | Tâches clés | Dépendances | Vague | Statut |
|---|---|---|---|---|
| A Capture | USB/réseau/fichier/synthétique, états, reconnexion, arrêt propre | contrats | 1 | ☐ |
| C Stockage | dépôt profils, références, représentations, réglages, suppression complète | contrats, schéma | 1 | ☐ |
| D Reconnaissance | étude licences, détection, alignement, suivi, adaptateur de référence, galerie | contrats | 1 | ☐ |
| F Décision & accueil | confirmation temporelle, inconnus, présence/répétitions, file audio | contrats | 1 | ☐ |
| B Interface | fenêtre, profils, réglages, aperçu, état, collecte guidée | A, C (interfaces) | 2 | ☐ |
| G Évaluation | protocole séances, métriques par passage/personne, rapport | contrats (F, D pour exécution réelle) | 2 | ☐ |
| E Notre IA | préparation données, modèle de zéro, scripts, suivi d'expériences | contrats ; données réelles + GPU pour entraîner | 2 | ☐ |

Maximum 4 travailleurs simultanés. Vague 2 lancée dès qu'une place se libère.

## Blocages connus

- Pas de caméra ni de GPU dans l'environnement actuel (conteneur cloud) → tests matériels
  à exécuter sur le PC cible.
- `libEGL` absent du conteneur : Qt ne se charge pas ici → tests GUI ignorés dans ce
  conteneur, à exécuter sur le PC cible.
- Aucune donnée familiale fournie → aucune mesure biométrique possible.
