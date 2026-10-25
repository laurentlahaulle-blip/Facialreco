# Architecture

## Vue d'ensemble

```
 ┌──────────────── Processus INTERFACE (PySide6, lot B) ─────────────────┐
 │  Fenêtre, profils, réglages, collecte guidée, aperçu, état système    │
 └───────▲──────────────────────────────────────────────┬────────────────┘
         │ aperçu réduit + états + événements            │ commandes
         │ (multiprocessing.Queue bornées)               ▼
 ┌──────────────── Processus VISION (orchestration : coordinateur) ──────┐
 │ Capture (A) ─► LatestQueue(maxsize=2, abandon des anciennes images)   │
 │   ─► Détection/Suivi (D) ─► Représentation (D réf. | E notre modèle)  │
 │   ─► Galerie (D, adossée au stockage C, filtrée par ModelRef)         │
 │   ─► Décision temporelle (F) ─► Arrivées (F) ─► Accueil audio (F)     │
 └───────────────────────────────────────────────────────────────────────┘
 Stockage SQLite (C) : profils, références, représentations, réglages.
 Évaluation (G) et Entraînement (E) : outils hors ligne, mêmes contrats.
```

- **Séparation des processus** : l'interface ne fait aucun calcul vidéo ; un gel de la
  vision ne bloque pas l'interface et inversement.
- **Latence bornée** : `pipeline.LatestQueue` ne bloque jamais le producteur et abandonne
  les images anciennes ; le compteur `frames_dropped` est exposé dans `CaptureState`.
- **Moteur interchangeable** : tout moteur implémente `FaceEmbedder` et publie un
  `ModelRef`. La galerie ne charge que les représentations du même `ModelRef.key`.
- **Ressources matérielles** : `pipeline.ResourceLock` (caméra, GPU).

## Concepts distincts (contrats)

| Concept | Type | Nature |
|---|---|---|
| Identité durable | `ProfileId` | stockée en base |
| Passage devant la caméra | `TrackId` | éphémère, suivi visuel |
| Score du détecteur | `Detection.detector_score` | « est-ce un visage » |
| Score du modèle | `RawScore` (cosinus / L2) | technique, **non calibré** |
| Décision finale | `Decision` (confirmé / inconnu / incertain) | politique versionnée |
| Arrivée | `ArrivalEvent` | une fois par présence |
| Accueil | `AudioRequest` | fichier local ou synthèse locale |

## Flux d'un passage

1. A produit des `Frame` (horodatage monotone + UTC).
2. D détecte, suit (`TrackedFace` avec `TrackId`), extrait une `Embedding`, cherche dans la
   galerie → `RecognitionResult` (candidats + `RawScore`).
3. F accumule par `TrackId` → `Decision` quand la politique est satisfaite (ex. N
   observations concordantes au-dessus du seuil en ≤ T ms, marge avec le 2ᵉ candidat).
4. F transforme une décision en `ArrivalEvent` sauf si la personne est déjà « présente »
   (présence maintenue pendant les pertes de suivi courtes et jusqu'à un délai d'absence).
5. F compose un `AudioRequest` ; le lecteur sérialise les accueils simultanés.

Les seuils ne sont fixés qu'à partir des mesures du lot G (ensemble de réglage), puis
vérifiés sur un ensemble de test indépendant.

## Choix techniques

- Python 3.12, PySide6 (LGPL), `opencv-python-headless` (évite le conflit de plugins Qt),
  SQLite (stdlib), ONNX Runtime pour l'inférence après vérification, PyTorch pour E.
- Audio local : à préciser par F (fichiers WAV + `winsound` sous Windows ; synthèse locale
  SAPI éventuelle). Aucun service cloud.
