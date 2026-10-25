# E — Alignement 5 points partagé (au lieu d'une copie dans `training`)

**Statut** : proposé (lot E)

## Besoin

Le lot E produit des représentations avec `preprocess = "align112-arcface5-v1"`, le même
identifiant que `maisonaccueil.vision.alignment` (lot D). La règle « importer uniquement
via les contrats » l'empêche d'importer `vision` : la géométrie (gabarit, LMEDS,
`warpAffine` bilinéaire, bord noir) est donc recopiée dans `training/preprocess.py`.
Constat ponctuel : sorties identiques octet pour octet sur 20 cas aléatoires ; un test de
cohérence (`tests/training/test_preprocess.py::test_same_geometry_as_reference_alignment`)
s'exécutera dès que `vision` sera sur la même branche. Mais une copie peut diverger
silencieusement tout en gardant le même identifiant.

## Changement proposé (au choix du coordinateur)

- **A.** Déplacer `PREPROCESS_ID`, `ARCFACE_TEMPLATE_112` et `align_face` dans un module
  partagé possédé par le coordinateur (ex. `maisonaccueil.contracts.alignment`, ou
  `maisonaccueil.pipeline.alignment`), importé par D et E.
- **B.** Autoriser explicitement `training` à importer `maisonaccueil.vision.alignment`
  (fonction pure, sans état).

Dans les deux cas, E supprimera sa copie et gardera seulement la normalisation des pixels,
interne au modèle.

## Compatibilité

Aucun changement de comportement (géométrie identique) ; aucune représentation à recalculer.
