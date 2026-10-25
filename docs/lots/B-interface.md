# Lot B — Interface (PySide6)

Paquet : `maisonaccueil.app`. Point d'entrée : `maisonaccueil` (`app/main.py`).

## Architecture

```
Processus VISION (app.vision_worker, sans Qt)        Processus INTERFACE (Qt)
 FrameSource ─► PreviewTap ─► make_preview (réduit) ─► file "preview" (2)  ─┐
            └─► VisionLoop (intégration) ─► VisionPublisher ─► file "events" (256) ─► UiInbox
 états CaptureState, EngineStatus, EnrollmentSample ───────────┘                     │
                                                         AppController.poll() (50 ms) ┘
                                                         ─► modèles de vue ─► widgets
```

- L'interface ne fait **aucun calcul de vision** : elle lit des `VisionOutput`,
  `CaptureState`, `EngineStatus`, `EnrollmentSample` et un `PreviewImage` déjà réduit.
- `VisionProcess` lance un travailleur dans un processus `spawn`. Le travailleur
  (`worker`, défaut `app.vision_worker:run_preview_worker`) et la fabrique de source sont
  transmis par nom (`module:fonction`), avec des `worker_kwargs` vérifiés sérialisables
  dès le parent. Le fils résout la cible (`run_named_worker`) ; si elle est introuvable,
  il se replie sur l'aperçu seul et publie « moteur non disponible » avec la raison.
  `InProcessVision` accepte les mêmes `worker` et arguments (thread, pour les tests).
- Publication non bloquante (pertes comptées) ; lecture non bloquante côté interface.

## Modules

| Module | Rôle |
|---|---|
| `ports` | Protocols des dépôts injectés ; `ProfileRepository`, `ReferenceRepository`, `SettingsRepository`, `ArrivalLogRepository` (lot C) les satisfont (vérifié par mypy dans `wiring`) |
| `messages` | `PreviewImage`, `EngineStatus`, `EnrollmentSample` (proposition `docs/contract-changes/B-ui-messages.md`) |
| `ipc` | `VisionPublisher` (côté vision ; `on_output`/`on_state` pour `VisionLoop`), `UiInbox` |
| `vision_worker` | `make_preview`, `PreviewThrottle`, `PreviewTap`, `run_preview_worker`, `VisionProcess`, `InProcessVision` |
| `viewmodels` | logique des écrans, **sans Qt** |
| `controller` | `AppServices` (injection), `AppController` (distribution des messages) |
| `widgets.main_window` | `MainWindow` + onglets ; seul code important `QtWidgets` |
| `fakes` | `FakeFrameSource` (synthétique, sans visage), dépôts `Fake*` en mémoire |
| `wiring` | seul module qui importe `maisonaccueil.storage` ; résout par nom la capture du lot A (`capture.factory:create_source`, `capture.discovery:list_usb_cameras`) si présente |

## Écrans

- **Accueil** : aperçu, légende (image, résolution source/aperçu, « flux interrompu »),
  état caméra, moteur, `ModelRef` (nom, version des poids, origine, dimension, métrique,
  prétraitement), images capturées / abandonnées (capture et vision), temps de traitement,
  visages en cours, journal des arrivées (historique + direct, UTC).
- **Profils** : création avec case de consentement obligatoire (horodatage UTC), nom
  prononcé, message d'accueil (≤ 300 car.), suppression complète en deux temps (nom à
  retaper) déléguée à `ProfileRepository.delete` avec bilan affiché.
- **Collecte guidée** (DATA_POLICY, étape 2) : choix du profil (actif, consenti), consignes
  de préparation, 4 étapes guidées (~20 s), séance `split='enroll'`, progression
  (images retenues / visées), arrêt, annulation (supprime les images de la séance), durée
  maximale 60 s. Refus si caméra non disponible.
- **Réglages** : `SourceConfig` (USB / réseau / fichier / synthétique, résolution, i/s),
  recherche des webcams USB (injectée), moteur (« moteur non disponible » si aucun),
  volume 0–100. Caméra réseau : adresses privées, loopback, link-local, `localhost`,
  `*.local` ou nom court uniquement.

## Règles appliquées

- Sans `EngineStatus(available=True)`, aucun résultat n'est affiché : « moteur non
  disponible ». Un résultat ou une décision d'un autre `ModelRef` que le moteur déclaré est
  ignoré (compteur `ignored_results`).
- Scores : « score technique (non calibré) : 0.612 (cosinus, plus haut = plus proche) » ;
  jamais de %. Le meilleur candidat est libellé « pas une décision ».
- Un visage inconnu n'est jamais nommé.
- `run_preview_worker(lock_camera=False)` par défaut : la source du lot A prend elle-même
  `ResourceLock("camera-N")` ; l'option sert aux sources qui ne le font pas.

## Moteur (`--engine`)

- `--engine reference` : le processus vision exécute
  `maisonaccueil.runtime.engine:run_engine_worker` (coordinateur) avec
  `db_path` = le MÊME fichier que l'interface (chemin absolu) et `models_dir` (`--models-dir`,
  optionnel). Défaut si `maisonaccueil.runtime.engine` est importable, `none` sinon.
- `--engine none` : aperçu seul. `--demo` impose `none` (`--demo --engine reference` est refusé).
- Le choix est calculé par `main.plan_vision` (sans Qt, testé).
- L'interface n'annonce un moteur (onglet Réglages, état) que sur
  `EngineStatus(available=True)` reçu du processus ; sinon « moteur non disponible (détail
  reçu) », p. ex. « moteur de référence : Poids absents : … ».

## Lancer

```
uv sync --extra gui
uv run maisonaccueil --check        # diagnostic sans fenêtre
uv run maisonaccueil --demo         # source synthétique + données en mémoire, sans moteur
uv run maisonaccueil [--db var/b/maisonaccueil.sqlite] [--engine reference|none] \
    [--models-dir models/d] [--source-factory module:fonction]
```

## Tests

`QT_QPA_PLATFORM=offscreen uv run pytest tests/app tests/contracts -rs`

Les tests Qt (`tests/app/test_widgets_gui.py`, marqués `gui`) sont ignorés si Qt Widgets
ne se charge pas (cas du conteneur de développement : `libEGL.so.1` absent). La fenêtre
n'a **pas** été affichée ni testée dans cet environnement ; seuls les modèles de vue, l'IPC,
le processus vision (vrai processus `spawn` avec source synthétique), le raccordement à
`VisionLoop` (doubles `Fake*`) et le typage des widgets (mypy sur les stubs PySide6) le sont.

## Limites

- Pas de cadres de visages sur l'aperçu (boîtes absentes de `VisionOutput`, voir proposition).
- Pas de sélection automatique d'images : elle doit venir de la vision (`EnrollmentSample`).
- Changement de source appliqué au prochain démarrage (pas de redémarrage à chaud).
- Volume enregistré seulement (`audio.volume`) ; non lu par le lecteur (hors périmètre actuel).
- Le moteur réel (poids YuNet/SFace présents) n'a pas été exécuté ici : seul le repli
  « poids absents » l'a été, y compris dans un vrai processus `spawn`.
- La recherche de webcams s'exécute dans le processus interface (bloquant quelques secondes).
- Pas de saisie du message d'accueil audio préenregistré (`welcome_audio`).
