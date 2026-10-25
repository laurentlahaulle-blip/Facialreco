# Registre des licences (coordinateur)

Le code et les poids ont des licences distinctes. Détail des recherches :
`docs/LICENSES-D.md` (lot D).

## Dépendances logicielles

| Composant | Licence | Usage |
|---|---|---|
| numpy | BSD-3-Clause | cœur |
| opencv-python-headless (dont `FaceDetectorYN`, `FaceRecognizerSF`) | Apache-2.0 (code) | capture, vision |
| PySide6 / Qt | LGPL-3.0 (liaison dynamique) | interface |
| onnxruntime | MIT | extra `recognition`, non utilisé à ce jour |
| torch / torchvision | BSD-3-Clause | extra `training` (lot E) |

## Poids de modèles (jamais committés, dossier `models/`)

| Poids | Licence des poids | Source | sha256 | Statut |
|---|---|---|---|---|
| YuNet `face_detection_yunet_2023mar.onnx` | MIT | OpenCV Zoo | `8f2383e4…52fa4` | retenu |
| SFace `face_recognition_sface_2021dec.onnx` | Apache-2.0 | OpenCV Zoo | `0ba9fbfa…34e79` | **retenu provisoirement — décision de l'utilisateur requise** |
| InsightFace (buffalo_l, etc.) | non commerciale / sur demande | — | — | écarté, non téléchargé |

**Réserve SFace** : les poids sont publiés sous Apache-2.0, mais OpenCV Zoo ne précise
pas le jeu d'entraînement des poids publiés ; l'article SFace cite CASIA-WebFace,
VGGFace2 et MS-Celeb-1M, dont les conditions sont restrictives ou qui ont été retirés.
Pour un usage privé familial non commercial, le risque est faible mais réel ; SFace sert
de **moteur de référence pour le développement et la comparaison** jusqu'à décision de
l'utilisateur. Le moteur entraîné de zéro (lot E) ne dépend d'aucun de ces poids.
