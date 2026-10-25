# Lot F — Décision et accueil

Chaîne : `RecognitionResult` → `TemporalDecisionEngine` → `PresenceArrivalTracker` →
`TemplateWelcomeComposer` → `QueuedAudioPlayer`.

## API publique

`maisonaccueil.decision`

| Nom | Rôle |
|---|---|
| `DecisionPolicy` | paramètres versionnés (`policy_id="temporal-v1"`) |
| `TemporalDecisionEngine(policy)` | `contracts.DecisionEngine` : `observe`, `track_lost`, `active_tracks` |
| `ArrivalPolicy` | délai d'absence, péremption d'un passage, intervalle minimal entre deux UNKNOWN |
| `PresenceArrivalTracker(policy, wall_clock, event_id_factory)` | `contracts.ArrivalTracker` : `on_decision`, `tick` ; en plus `note_track_activity`, `track_lost`, `present_profiles`, `presence` |

`maisonaccueil.welcome`

| Nom | Rôle |
|---|---|
| `ProfileGreeting(display_name, audio_path, text)` | données d'accueil d'un profil |
| `ProfileGreetingLookup` | `Callable[[ProfileId], ProfileGreeting \| None]` injecté (ex. adossé au lot C) |
| `WelcomeConfig`, `UnknownPolicy` | modèles, politique inconnus (`SILENT` par défaut, `TEXT`, `AUDIO`) |
| `TemplateWelcomeComposer(lookup, config)` | `contracts.WelcomeComposer` |
| `QueuedAudioPlayer(backend, clock, max_queue, on_played)` | `contracts.AudioPlayer` : `submit`, `close`, `stats`, `pending`, `wait_idle` |
| `AudioBackend` (Protocol : `play` bloquant, `stop`) | |
| `WinsoundBackend` | WAV via `winsound` (Windows uniquement) |
| `Pyttsx3Backend` | synthèse locale optionnelle (dépendance proposée, non installée) |
| `RoutingAudioBackend(file, text)` | aiguillage fichier / texte |
| `LoggingAudioBackend` | aucun son, journalisation (poste de développement) |
| `create_default_backend()` | Windows : winsound + pyttsx3 si présent ; ailleurs : journalisation |
| `FakeAudioBackend` | double de test (aucun son) |

## Politique de décision `temporal-v1`

Pour chaque `TrackId`, fenêtre glissante des observations exploitables (les
observations portant un drapeau de `blocking_flags` sont ignorées). Une observation est
*acceptée* pour le profil P si le meilleur candidat est P, son score passe le seuil
d'acceptation de sa métrique et la marge avec le 2ᵉ candidat est suffisante.

- **CONFIRMED(P)** : au moins `min_agreeing` observations acceptées pour P et au plus
  `max_conflicting` acceptées pour d'autres profils.
- **UNCERTAIN (conflit)** : plus de `max_conflicting` observations acceptées pour un
  autre profil que le majoritaire (oscillation entre deux identités).
- **UNKNOWN** : au moins `min_observations_unknown` observations, aucune acceptée.
- sinon UNCERTAIN (insuffisant), jamais émis.

Émission : CONFIRMED une fois par passage et par profil, puis « collante » (pas de
rétrogradation émise) ; seule une confirmation d'un autre profil (échange de suivi) est
émise. UNKNOWN une fois par passage tant qu'il n'est pas confirmé. UNCERTAIN uniquement
à l'entrée dans un conflit. Un changement de `ModelRef` sur un passage réinitialise son
état ; un passage sans observation depuis `stale_track_ms` est oublié.

### Valeurs par défaut PROVISOIRES (aucune mesure, à fixer avec le lot G)

| Paramètre | Défaut |
|---|---|
| `accept_cosine` | 0.45 (score ≥) |
| `accept_l2` | 1.10 (distance ≤) |
| `min_margin` | 0.05 |
| `window_ms` | 2000 |
| `min_agreeing` | 3 |
| `max_conflicting` | 1 |
| `min_observations_unknown` | 5 |
| `blocking_flags` | `FACE_TOO_SMALL`, `BLURRY`, `OCCLUDED` |
| `stale_track_ms` | 10000 |
| `ArrivalPolicy.absence_timeout_ms` | 30000 |
| `ArrivalPolicy.track_stale_ms` | 5000 |
| `ArrivalPolicy.unknown_cooldown_ms` | 5000 |
| `WelcomeConfig.expires_after_ms` | 10000 |

Ces seuils sont des scores techniques, pas des probabilités. Le lot G devra les régler
sur `tune` (bonne identification, inconnu accepté à tort, message répété, délai
jusqu'à l'accueil) puis les vérifier sur un test indépendant.

## Présences

Une présence par profil regroupe ses passages. Une seule `ArrivalEvent` KNOWN par
présence. Elle survit aux pertes de suivi et aux changements de `TrackId` ; elle
expire (dans `tick`) quand aucun passage n'est actif et que le dernier signe de vie
date de `absence_timeout_ms`. Pour une précision maximale, l'intégration appelle
`note_track_activity` à chaque résultat et `track_lost` (proposition
`docs/contract-changes/F-arrival-track-activity.md`). Inconnus : au plus un UNKNOWN par
passage, sans nom, et pas plus d'un toutes les `unknown_cooldown_ms` (limite : deux
inconnus distincts arrivant ensemble ne produisent qu'un événement).

## Arrivées simultanées : messages ENCHAÎNÉS

Le lecteur joue les messages l'un après l'autre (priorité, puis ordre d'arrivée), jamais
en parallèle. Choix retenu car les messages personnalisés WAV ne se fusionnent pas et
`AudioRequest` ne porte pas le nom à prononcer. Les demandes en double (`dedupe_key`
identique en file ou en lecture) sont ignorées ; une demande dont l'âge dépasse
`expires_after_ms` au moment d'être jouée est abandonnée (au-delà de quelques arrivées
simultanées, les derniers accueils peuvent donc être perdus). File bornée
(`max_queue=8`) : une demande plus prioritaire évince la moins prioritaire la plus
récente. Regroupement « Bonjour A et B » : proposition
`docs/contract-changes/F-audio-grouping.md`.

## Tests

`uv run pytest tests/decision tests/welcome` — horloge injectée (`FakeClock`),
`RecognitionResult` synthétiques, `FakeAudioBackend`. Scénarios couverts : pertes de
suivi, changement de `TrackId`, répétitions après absence, arrivées simultanées,
inconnu (silencieux / texte), oscillation entre deux identités, images floues.

## Limites

- Aucun son réel n'a été joué ni entendu : environnement de développement sans carte
  son (Linux). `WinsoundBackend` et `Pyttsx3Backend` ne sont pas testés (seul le refus
  hors Windows l'est).
- Seuils provisoires, non mesurés.
- `RecognitionResult` hors ordre temporel sur un même passage : toléré mais non optimisé.
