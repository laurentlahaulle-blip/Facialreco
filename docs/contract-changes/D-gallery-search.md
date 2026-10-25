# D — `Gallery.search` renvoie un `RecognitionResult` incomplet

- **Besoin** : `Gallery.search(query, top_k) -> RecognitionResult | None` oblige la galerie à
  fabriquer un `RecognitionResult` alors qu'elle ne connaît ni `frame_id`, ni `track_id`, ni
  `monotonic_ns`. L'implémentation du lot D (`InMemoryGallery.search`) remplit donc des
  valeurs neutres (`FrameId(0)`, `TrackId("")`, `0`), ce qui est trompeur si le résultat
  circule tel quel.
- **Changement proposé** :

  ```python
  class Gallery(Protocol):
      @property
      def model(self) -> ModelRef: ...
      def rank(self, query: Embedding, top_k: int = 3) -> tuple[Candidate, ...]: ...
  ```

  (`search` pourrait être conservée un temps, dépréciée.) Le `Recognizer` construit le
  `RecognitionResult` complet. `InMemoryGallery.rank` existe déjà avec cette signature.
- **Compatibilité** : seul le lot D implémente `Gallery` ; B/F/G consomment des
  `RecognitionResult` produits par le `Recognizer`, inchangés.
- **Statut** : proposé.
