# F — Signes de vie des passages pour `ArrivalTracker`

## Besoin

`ArrivalTracker` ne reçoit que des `Decision`. Or le moteur de décision n'émet pas une
décision à chaque image (une CONFIRMED par passage). Sans autre signal, le suivi des
présences ne sait pas qu'une personne est **toujours là** : une personne restée 60 s
puis réapparue sous un nouveau `TrackId` après une coupure serait accueillie une
seconde fois si l'absence se mesurait depuis la dernière décision.

Exemple : CONFIRMED(A, t1) à 0 s ; A reste devant la caméra jusqu'à 50 s (aucune
décision) ; perte de suivi, nouveau passage t2 confirmé A à 52 s. Avec un délai
d'absence de 30 s mesuré depuis la décision, A serait réaccueilli à tort.

## Changement proposé

Ajouter au Protocol `ArrivalTracker` (déjà implémentées par
`PresenceArrivalTracker`) :

```python
def note_track_activity(self, track_id: TrackId, monotonic_ns: int) -> None:
    """Appelé pour chaque RecognitionResult (même sans décision)."""

def track_lost(self, track_id: TrackId, monotonic_ns: int) -> None:
    """Appelé pour chaque TrackId renvoyé par Recognizer.lost_tracks()."""
```

Côté pipeline (coordinateur), pour chaque image :

```python
for r in recognizer.process(frame):
    arrivals.note_track_activity(r.track_id, r.monotonic_ns)
    d = engine.observe(r)
    if d and (ev := arrivals.on_decision(d)) and (req := composer.compose(ev)):
        player.submit(req)
for tid in recognizer.lost_tracks():
    engine.track_lost(tid, now); arrivals.track_lost(tid, now)
arrivals.tick(now)
```

## Compatibilité

Ajout de méthodes : l'implémentation du lot F les fournit déjà. Sans ces appels,
`PresenceArrivalTracker` reste correct mais moins précis : un passage est considéré
actif `track_stale_ms` après sa dernière décision, puis le délai d'absence court.

## Statut

proposé
