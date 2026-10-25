# Licences — Lot D (reconnaissance de référence)

Vérification faite le 2026-10-25 à partir des sources en ligne citées. Le **code** et les
**poids** sont traités séparément : la licence d'un dépôt de code ne couvre pas
automatiquement des poids, et des poids peuvent hériter de restrictions de leurs données
d'entraînement. Usage visé : application **privée, familiale, non commerciale**, exécutée
en local. Ce document n'est pas un avis juridique.

## Synthèse

| Candidat | Code | Poids | Verdict lot D |
|---|---|---|---|
| YuNet (OpenCV Zoo) — détecteur | `cv2.FaceDetectorYN` : Apache-2.0 (OpenCV) | MIT (fichier LICENSE du dossier) | **Retenu**, poids téléchargés |
| SFace (OpenCV Zoo) — représentation | `cv2.FaceRecognizerSF` : Apache-2.0 (OpenCV) | Apache-2.0 (fichier LICENSE du dossier) ; données d'entraînement non précisées | **Retenu avec réserve**, poids téléchargés |
| InsightFace (buffalo_l, antelopev2…) | MIT | « non-commercial research purposes only » ; buffalo_l : licence sur demande | **Écarté** (poids non téléchargés) |
| dlib (`dlib_face_recognition_resnet_model_v1`, `shape_predictor_5`, `mmod`) | Boost Software License 1.0 | Domaine public / CC0-1.0 (sauf `shape_predictor_68` : non commercial, 300-W) | Compatible, **non retenu** (dépendance `dlib` à compiler, absente de `pyproject.toml`) |
| MediaPipe Face Detector (BlazeFace) | Apache-2.0 | Non vérifié (documentation des modèles inaccessible depuis cet environnement) | **Non retenu** (licence des poids non confirmée ; pas de reconnaissance) |
| ONNX Runtime (moteur) | MIT | — | Compatible ; **pas nécessaire** pour la chaîne de référence (voir plus bas) |

## Détail par candidat

### YuNet — détection de visages + 5 points

- Code d'inférence : `cv2.FaceDetectorYN`, inclus dans `opencv-python-headless` (OpenCV ≥ 4.5,
  Apache-2.0), déjà dépendance du projet. Code de démonstration OpenCV Zoo : MIT.
- Poids : `face_detection_yunet_2023mar.onnx`.
  Source : <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
  — « All files in this directory are licensed under MIT License ».
  Texte : `LICENSE` du dossier, « MIT License, Copyright (c) 2020 Shiqi Yu ».
  Amont : libfacedetection / libfacedetection.train (Shiqi Yu). Évaluation publiée sur
  WIDER Face (jeu de validation) ; aucune donnée d'entraînement à licence restrictive n'est
  signalée par la source.
- Compatibilité usage familial privé : oui (MIT, sans restriction d'usage).
- **Verdict : retenu.**

### SFace — représentation faciale 128 dimensions

- Code d'inférence : `cv2.FaceRecognizerSF` (OpenCV, Apache-2.0).
- Poids : `face_recognition_sface_2021dec.onnx` (MobileFaceNet entraîné avec la perte SFace).
  Source : <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface>
  — « All files in this directory are licensed under Apache 2.0 License ».
  Texte : `LICENSE` du dossier (Apache License 2.0).
- Réserve : la source ne précise pas le jeu d'entraînement des poids publiés. L'article
  SFace (<https://github.com/zhongyy/SFace>) rapporte des expériences sur CASIA-WebFace,
  VGGFace2 et MS-Celeb-1M ; certains de ces jeux ont des conditions restrictives ou ont été
  retirés par leurs auteurs. Le dépôt amont n'affiche pas de licence. La licence déclarée des
  poids par OpenCV Zoo est Apache-2.0, sans restriction d'usage ; pour un usage privé non
  commercial, le risque résiduel est jugé faible mais il est signalé au coordinateur.
- Compatibilité usage familial privé : oui, sous la réserve ci-dessus.
- **Verdict : retenu avec réserve** (à confirmer par le coordinateur dans `docs/LICENSES.md`).

### InsightFace

- Code : MIT (<https://github.com/deepinsight/insightface>, section License).
- Poids : « The training data containing the annotation (and the models trained with these
  data) are available for non-commercial research purposes only. » Les modèles téléchargés
  automatiquement suivent cette politique ; pour buffalo_l, la page renvoie vers un contact
  de licence (recognition-oss-pack@insightface.ai).
- Un usage familial privé n'est pas de la « recherche » : clause ambiguë.
- **Verdict : écarté**, aucun poids téléchargé.

### dlib

- Code : Boost Software License 1.0.
- Poids : <https://github.com/davisking/dlib-models> — « released them into the public
  domain » (dépôt CC0-1.0), sauf `shape_predictor_68_face_landmarks` (données iBUG 300-W,
  usage commercial exclu).
- Compatible, mais exige le paquet `dlib` (compilation C++/CMake sous Windows) qui n'est pas
  dans les dépendances : **non retenu** pour la référence, alternative possible plus tard.

### MediaPipe

- Code : Apache-2.0 (<https://github.com/google-ai-edge/mediapipe>).
- Poids (BlazeFace) : licence indiquée dans les « model cards », non consultables depuis cet
  environnement (DNS de ai.google.dev indisponible). Ne fournit qu'une détection, pas de
  représentation. **Non retenu** ; à revérifier si besoin.

### ONNX Runtime (extra `recognition`)

- Licence : MIT (<https://github.com/microsoft/onnxruntime>). Roues CPU publiées pour
  Windows/Linux/macOS et CPython 3.11–3.13 (à revérifier à la version figée dans `uv.lock`).
- Les deux modèles retenus s'exécutent avec `cv2.dnn` (via `FaceDetectorYN` /
  `FaceRecognizerSF`) : **ONNX Runtime n'est pas nécessaire** pour la chaîne de référence et
  n'est pas utilisé par le lot D. Il reste utile au lot E (export de notre modèle).

## Poids téléchargés (jamais committés)

Dossier : `models/d/` (ignoré par Git, ainsi que `*.onnx`). Téléchargement depuis la branche
`main` d'OpenCV Zoo (stockage Git LFS, `media.githubusercontent.com`), le 2026-10-25.
Les sha256 calculés localement **sont identiques aux oid sha256 des pointeurs Git LFS**
publiés dans le dépôt (vérification croisée).

| Fichier | Taille (octets) | sha256 |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 232 589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | 38 696 353 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| `LICENSE-yunet` (copie du LICENSE MIT) | 1 085 | `c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958` |
| `LICENSE-sface` (copie du LICENSE Apache-2.0) | 11 358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

Ces empreintes sont codées dans `maisonaccueil.recognition.models` : un fichier dont
l'empreinte diffère est refusé (`ModelIntegrityError`).

Pour reproduire sur une autre machine : télécharger les deux `.onnx` depuis les URL des
dossiers ci-dessus, les placer dans `models/d/` (ou `MAISONACCUEIL_MODELS_DIR`), puis
vérifier avec `sha256sum` (ou `Get-FileHash` sous Windows).
