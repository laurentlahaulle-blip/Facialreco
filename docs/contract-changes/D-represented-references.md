# D — Savoir quelles références sont déjà représentées pour un modèle

- **Besoin** : l'enrôlement (`maisonaccueil.recognition.enrollment.ReferenceEnroller`) doit
  être idempotent : ne pas recalculer ni dupliquer la représentation d'une référence déjà
  traitée pour le même `ModelRef`. `EmbeddingStore.load_embeddings(model)` ne renvoie que
  `(ProfileId, Embedding)`, sans `reference_id` : impossible de savoir ce qui est déjà fait.
  Exemple : relancer l'enrôlement après l'ajout de 3 images à un profil recalcule et
  duplique les 20 références existantes.
- **Changement proposé** (additif) dans `contracts/interfaces.py` :

  ```python
  class EmbeddingStore(Protocol):
      ...
      def represented_reference_ids(self, model: ModelRef) -> set[int]:
          """reference_id ayant au moins une représentation de ``model`` (même key),
          profils actifs ET inactifs."""
          ...
  ```

  Implémentation lot C (une requête) :
  `SELECT DISTINCT reference_id FROM embeddings WHERE model_key = ? AND reference_id IS NOT NULL`.
- **Contournement actuel (lot D)** : `ReferenceEnroller.enroll(refs, store, already_represented=None)`
  1. utilise `store.represented_reference_ids(model)` si la méthode existe (détection par
     `getattr`, testée avec `FakeIndexedEmbeddingStore`) ;
  2. sinon l'appelant peut fournir `already_represented` ;
  3. sinon seule la mémoire de l'instance protège contre les doublons (une instance
     neuve sur une base existante recalculerait tout).
- **Compatibilité** : additif ; seul `SqliteEmbeddingStore` (lot C) et les `Fake*` doivent
  l'implémenter. Pas de changement de schéma.
- **Statut** : accepté (coordinateur, 2026-10-25) — implémenté par le lot C, Protocol `EmbeddingStore` étendu (contrats 1.3.0).
