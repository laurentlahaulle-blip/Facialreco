# Données, collecte et validation

## Principes

- Uniquement des proches **volontaires** ayant donné leur consentement (`profiles.consent_at`).
- Aucune donnée d'un autre projet ; aucune personne réelle inventée.
- Images de référence stockées localement (`var/` ou dossier configuré), jamais committées,
  supprimées avec le profil (suppression en cascade + fichiers).
- Pas d'enregistrement vidéo permanent par défaut. L'enregistrement de séances d'évaluation
  est une action explicite, limitée dans le temps, avec consentement.
- Données synthétiques : tests logiciels uniquement, **jamais** pour juger la qualité
  biométrique.

## Parcours de collecte guidée (par personne)

1. Création du profil + consentement.
2. Séance d'enrôlement (`split='enroll'`) : ~20 s face caméra à 1–3 m, légers mouvements
   de tête, 2 éclairages si possible. Sélection automatique d'images nettes et variées.
3. Séances d'évaluation distinctes, **d'autres jours** ou conditions : passages naturels
   (entrée dans la pièce), enregistrés uniquement pour l'évaluation.

## Découpage

- Unité de découpage = **séance complète** (`collection_sessions`), jamais l'image : des
  images voisines d'une même vidéo ne peuvent être à la fois en apprentissage et en test.
- `train` (lot E), `tune` (réglage des seuils), `test` (mesure finale).
- Des personnes **non enregistrées** (volontaires) figurent dans `tune` et `test` pour
  mesurer le rejet des inconnus.
- Après tout réglage fondé sur un résultat de test, constituer un **nouveau test
  indépendant** avant toute conclusion.
- Tous les moteurs sont comparés sur les **mêmes passages** de test.

## Mesures (lot G), par passage et par personne

bonne identification · mauvaise identité · connu non reconnu · inconnu accepté à tort ·
message répété · délai jusqu'à l'accueil. Effectifs toujours indiqués ; pas de pourcentage
sans le nombre de passages.
