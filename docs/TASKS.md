# Tableau de suivi

Légende : ☐ à faire · ◐ en cours · ☑ intégré · ⛔ bloqué

## Socle (coordinateur)

| Tâche | Statut | Notes |
|---|---|---|
| Audit environnement | ☑ | `docs/AUDIT.md` |
| Squelette, `pyproject.toml`, `uv.lock`, `.venv` | ☑ | |
| Contrats typés (v1.3.0) + schéma SQLite v1 | ☑ | `src/maisonaccueil/contracts/` |
| `LatestQueue`, `ResourceLock` | ☑ | `src/maisonaccueil/pipeline/` |
| CLAUDE.md, ownership, prompts | ☑ | |
| Boucle vision (`VisionLoop`, interfaces seules) | ☑ | testée avec doubles |
| Racine `runtime.engine.run_engine_worker` (A+C+D+F → publication B) | ☑ | thread testé ; repli sans moteur |
| Sélection du moteur dans le point d'entrée (`--engine`) | ☑ | lot B |
| Enrôlement : références `enroll` → représentations (idempotent) | ☑ | lots D + C, branché dans `runtime` |
| Tests d'intégration avec doubles (`tests/integration`) | ☑ | 4 tests |
| Tests d'intégration avec implémentations réelles A+C+D+F | ☑ | images synthétiques, aucune mesure |

## Lots

| Lot | Tâches clés | Dépendances | Vague | Statut |
|---|---|---|---|---|
| A Capture | USB/réseau/fichier/synthétique, états, reconnexion, arrêt propre | contrats | 1 | ☑ (tests caméra réelle à faire sur PC) |
| C Stockage | dépôt profils, références, représentations, réglages, suppression complète | contrats, schéma | 1 | ☑ (42 tests suite complète) |
| D Reconnaissance | étude licences, détection, alignement, suivi, adaptateur de référence, galerie | contrats | 1 | ☑ (YuNet+SFace ; réserve licence SFace) |
| F Décision & accueil | confirmation temporelle, inconnus, présence/répétitions, file audio | contrats | 1 | ☑ (son réel à tester sous Windows) |
| B Interface | fenêtre, profils, réglages, aperçu, état, collecte guidée | A, C (interfaces) | 2 | ☑ (fenêtre non affichée ici : libEGL) |
| G Évaluation | protocole séances, métriques par passage/personne, rapport | contrats (F, D pour exécution réelle) | 2 | ☑ (aucune mesure : pas de données) |
| E Notre IA | préparation données, modèle de zéro, scripts, suivi d'expériences | contrats ; données réelles + GPU pour entraîner | 2 | ☑ infrastructure (torch non exécuté ici) |

Maximum 4 travailleurs simultanés. Vague 2 lancée dès qu'une place se libère.

## Prochaines tâches

| Tâche | Responsable | Dépend de |
|---|---|---|
| Essais sur le PC Windows : fenêtre, webcam, son (`winsound`) | utilisateur + coordinateur | clone sur le PC |
| Décision licence SFace (`docs/LICENSES.md`) | utilisateur | — |
| Dépendances `pyttsx3`, torch CPU, onnx/onnxruntime (propositions F, E) | utilisateur puis coordinateur | accord |
| Journal d'exécution `maisonaccueil.runlog` écrit en fonctionnement normal (option) | coordinateur + G | — |
| Cadres des visages sur l'aperçu (`VisionOutput.faces`, proposition B) | coordinateur + B | — |
| Lecture de `audio.volume` par le lecteur | F | — |
| Factoriser l'alignement partagé D/E | coordinateur | avant 1er entraînement |
| Première collecte réelle (3 personnes, séances séparées, inconnus volontaires) | utilisateur | consentements |
| Réglage des seuils sur `tune`, puis test indépendant | G | collecte |
| Premier entraînement de notre modèle | E | données réelles + GPU réservé |

## Blocages connus

- Pas de caméra ni de GPU dans l'environnement actuel (conteneur cloud) → tests matériels
  à exécuter sur le PC cible.
- `libEGL` absent du conteneur : Qt ne se charge pas ici → tests GUI ignorés dans ce
  conteneur, à exécuter sur le PC cible.
- Aucune donnée familiale fournie → aucune mesure biométrique possible.
