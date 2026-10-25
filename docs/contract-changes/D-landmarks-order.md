# D — Ordre des points de `Landmarks5`

- **Besoin** : `Landmarks5` dit « œil gauche, œil droit, nez, coin gauche, coin droit de la
  bouche » sans préciser le repère (gauche de la personne ou gauche de l'image). L'alignement
  sur un gabarit dépend de cet ordre ; une inversion entre lots (A/D/E) produirait des visages
  miroirs et des représentations incomparables sans erreur visible.
- **Changement proposé** (docstring seulement, aucun changement de type) :

  ```python
  class Landmarks5:
      """Cinq points en COORDONNÉES IMAGE : œil situé à gauche dans l'image, œil situé à
      droite dans l'image, nez, coin de bouche à gauche dans l'image, coin à droite.
      (Ordre natif de YuNet et du gabarit ArcFace 112×112.)"""
  ```
- **Compatibilité** : c'est la convention déjà appliquée par le lot D
  (`maisonaccueil.vision.alignment`). Le lot E doit l'utiliser pour son prétraitement.
- **Statut** : accepté (coordinateur, 2026-10-25) — contrats 1.2.0.
