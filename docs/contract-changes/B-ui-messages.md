# Proposition B — messages vision → interface et données d'affichage

**Statut** : proposé (lot B).

## Besoin

1. **État du moteur.** L'interface doit afficher « moteur non disponible » ou le `ModelRef`
   actif, y compris quand aucun visage n'est présent (donc sans `RecognitionResult`).
   `VisionOutput` ne porte pas cette information.
2. **Aperçu réduit.** L'architecture prévoit un aperçu transmis séparément de
   `VisionOutput` (sans pixels) ; aucun type commun n'existe.
3. **Collecte guidée.** La sélection automatique d'images nettes (côté vision, lot D) doit
   remettre les images retenues à l'interface, qui les enregistre (lot C).
4. **Cadres des visages sur l'aperçu.** `RecognitionResult` ne contient pas la boîte du
   visage : l'interface ne peut pas dessiner de cadre par passage (`TrackId`) ; elle
   n'affiche donc que du texte.

## Changement proposé

Reprendre dans `contracts/` (ou `pipeline/`) les types aujourd'hui définis par le lot B
dans `src/maisonaccueil/app/messages.py`, sans modification :

```python
@dataclass(frozen=True, slots=True)
class PreviewImage:      # BGR888 contigu, réduit côté vision
    frame_id: int; camera_id: str; monotonic_ns: int
    width: int; height: int; source_width: int; source_height: int; data: bytes

@dataclass(frozen=True, slots=True)
class EngineStatus:      # available=True ⇔ model renseigné
    available: bool; model: ModelRef | None = None; detail: str = ""

@dataclass(frozen=True, slots=True)
class EnrollmentSample:  # image encodée (PNG/JPEG) retenue par la vision
    monotonic_ns: int; data: bytes; extension: str = ".png"
    width: int | None = None; height: int | None = None; quality_note: str | None = None
```

Et, pour le point 4, ajouter à `VisionOutput` un champ optionnel
`faces: tuple[TrackedFace, ...] = ()` (boîtes dans le repère de l'image source ;
l'interface les met à l'échelle de l'aperçu grâce à `source_width/height`).

Files : `app.ipc` (canal `events` borné à 256, canal `preview` borné à 2 ; publication non
bloquante ; un `VisionOutput` portant des arrivées attend au plus 0,2 s avant abandon).
`VisionPublisher.on_output` / `on_state` se branchent directement sur `VisionLoop`, et
`PreviewTap(publisher, downstream=...)` s'intercale sur `on_frame`.

## Compatibilité

Ajouts uniquement ; aucun lot existant n'est cassé. Le champ `faces` a une valeur par
défaut. Si le coordinateur déplace les types, le lot B remplacera ses imports
(`app.messages` deviendrait une simple réexportation).
