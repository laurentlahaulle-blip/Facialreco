# D — Réexporter les Protocols depuis `maisonaccueil.contracts`

- **Besoin** : `CLAUDE.md` demande d'importer « uniquement via `maisonaccueil.contracts` »,
  mais `FaceDetector`, `FaceTracker`, `FaceEmbedder`, `Gallery`, `Recognizer`,
  `EmbeddingStore`, etc. ne sont accessibles que via `maisonaccueil.contracts.interfaces`.
  Le lot D importe donc depuis ce sous-module (toujours dans le paquet contrats).
- **Changement proposé** : dans `contracts/__init__.py`,
  `from .interfaces import (ArrivalTracker, AudioPlayer, DecisionEngine, EmbeddingStore,
  FaceDetector, FaceEmbedder, FaceTracker, FrameSource, Gallery, Recognizer, WelcomeComposer)`
  et ajout à `__all__`.
- **Compatibilité** : purement additif.
- **Statut** : refusé (coordinateur, 2026-10-25) — les Protocols restent dans `maisonaccueil.contracts.interfaces` (import explicite voulu), même décision que pour la proposition A.
