# Lot A — Capture

Branche `lot/a-capture` · Code : `src/maisonaccueil/capture/` · Tests : `tests/capture/`

## Contenu

| Module | Rôle |
|---|---|
| `base.py` | `ThreadedFrameSource` : thread dédié, états `CaptureStatus`, reconnexion (`ReconnectPolicy`), `FrameId` croissants, horodatage monotone + UTC, conversion BGR uint8 |
| `opencv_source.py` | `OpenCVSource` : USB (index), réseau local (RTSP/HTTP), fichier ; `VideoCapture` injectable |
| `synthetic.py` | `SyntheticSource` : motifs `bars` / `checker` / `gradient` + compteur (aucun visage) |
| `discovery.py` | `list_usb_cameras` : énumération des index USB sous réservation |
| `factory.py` | `create_source(SourceConfig, **options)` |

## Utilisation

```python
from maisonaccueil.capture import create_source
from maisonaccueil.contracts import CameraId, Frame, SourceConfig, SourceKind
from maisonaccueil.pipeline import LatestQueue

queue: LatestQueue[Frame] = LatestQueue(maxsize=2)
cfg = SourceConfig(CameraId("entree"), SourceKind.USB, "0", width=1280, height=720)
source = create_source(cfg, backend="msmf", drop_counter=lambda: queue.dropped,
                       lock_owner="vision")
source.start(queue.put, on_state)   # non bloquant ; ResourceBusyError si caméra réservée
...
source.stop(timeout_s=5.0)          # idempotent ; libère caméra et réservation
```

## Comportement

- **États** : `STOPPED → CONNECTING → RUNNING → STOPPED`. Coupure ou échec d'ouverture :
  `RECONNECTING` avec attente `initial_delay_s * factor^(n-1)` bornée par `max_delay_s`
  (défaut 0,5 s → 10 s), puis `FAILED` après `max_attempts` (défaut 8) tentatives
  consécutives sans image. Le compteur repart à zéro dès qu'une image est lue.
  Une coupure = `max_consecutive_read_failures` (défaut 5) lectures ratées d'affilée.
- Pendant `RUNNING`, un état est republié toutes les `stats_interval_s` (défaut 1 s) avec
  `frames_captured` / `frames_dropped`.
- `frames_dropped` vient du `drop_counter` fourni (typiquement `lambda: queue.dropped`) :
  la source ne connaît pas la file du consommateur (voir proposition A-frames-dropped).
- `FrameId` : croissant et continu, y compris à travers reconnexions et redémarrages.
- Fin de fichier ou `max_frames` atteint (synthétique) : `STOPPED`, détail « fin du flux ».
  `loop_file=True` relit le fichier. `realtime=True` cadence la lecture à la fréquence
  du fichier.
- `on_frame` / `on_state` sont appelés depuis le thread de capture ; une exception dans
  `on_frame` est journalisée et comptée (`callback_errors`), la capture continue.
  `stop()` appelé depuis un rappel demande l'arrêt sans attendre (pas d'interblocage).
- Si une lecture OpenCV reste bloquée au-delà de `timeout_s`, `stop` journalise un
  avertissement et rend la main ; le thread (démon) s'arrête à la lecture suivante. Pour
  les flux réseau, `CAP_PROP_OPEN_TIMEOUT_MSEC` / `CAP_PROP_READ_TIMEOUT_MSEC` (5 s) sont
  passés à l'ouverture pour limiter ce cas.
- **USB** : backend par défaut MSMF sous Windows (`backend="dshow"` possible), V4L2 sous
  Linux. Réservation `ResourceLock("camera-<index>")` prise dans `start`, relâchée à la fin
  du thread (y compris après `FAILED`). `CAP_PROP_BUFFERSIZE=1` pour limiter la latence.
- **Réseau** : seules les adresses locales sont acceptées (privées, boucle locale, lien
  local, `*.local`, `*.lan`, `*.home.arpa`, nom sans domaine) ; sinon `ValueError`.
  Identifiants et paramètres de l'URL masqués dans les états (`redact_url`).
- **Énumération** : `list_usb_cameras(max_index)` réserve chaque index sans attendre ; un
  index déjà réservé est signalé `BUSY` et n'est pas ouvert. Capture d'essai relâchée
  immédiatement.
- Aucune image écrite sur disque.

## Tests

`uv run pytest tests/contracts tests/capture` : source synthétique, cycle d'états,
reconnexion simulée (`FakeVideoCapture` / `FakeCaptureFactory`), `FAILED` et libération du
verrou, arrêt pendant l'attente de reconnexion, arrêt sous charge, abandon d'images via
`LatestQueue`, exclusivité de la réservation, URL locales, fichier vidéo synthétique
généré dans un dossier temporaire.

Tests matériels (`@pytest.mark.camera`, ignorés par défaut) :
`MAISONACCUEIL_CAMERA_TESTS=1 uv run pytest -m camera tests/capture`
(index via `MAISONACCUEIL_CAMERA_INDEX`). **Non exécutés à ce jour** : l'environnement de
développement (conteneur Linux) n'a pas de caméra.

## Limites

- Aucun test sur caméra physique ni flux RTSP réel ; backends MSMF/DSHOW non vérifiés.
- Une lecture OpenCV bloquante ne peut pas être interrompue de l'extérieur (limite
  d'OpenCV) ; la caméra reste réservée tant que le thread n'est pas sorti.
- La source tourne dans un thread du processus vision ; un processus de capture séparé
  n'est pas implémenté (non requis par le contrat).
