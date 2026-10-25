# G — `VisionOutput.lost_tracks`

- **Besoin** : le journal d'exécution du lot G (`maisonaccueil.runlog`) consigne les pertes
  de suivi (`track_lost`) pour pouvoir rejouer fidèlement la décision (`rerun_decisions`,
  balayage de seuils sur `tune`). `VisionOutput` ne contient pas les `TrackId` perdus
  renvoyés par `Recognizer.lost_tracks()` pendant `process_frame`. Le rejeu hors ligne du
  lot G contourne en enveloppant le `Recognizer` ; un journal **en direct** (écrit depuis
  `on_output` dans l'application) ne peut pas le faire sans dupliquer cette enveloppe dans
  l'orchestration.
- **Changement proposé** (`pipeline/vision_loop.py`, propriété du coordinateur) :

  ```python
  @dataclass(frozen=True, slots=True)
  class VisionOutput:
      ...
      lost_tracks: tuple[TrackId, ...] = ()
  ```

  rempli dans `process_frame` avec la liste déjà parcourue (`for track_id in
  self._recognizer.lost_tracks(): ...`).
- **Compatibilité** : champ avec valeur par défaut, aucun impact sur les lots existants.
  Le lot G utiliserait le champ s'il existe et garderait l'enveloppe sinon.
- **Statut** : proposé.
