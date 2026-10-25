# Lot G — Évaluation

Branche `lot/g-evaluation` · Code : `src/maisonaccueil/evaluation/` · Tests : `tests/evaluation/`

> **Aucune mesure biométrique n'existe à ce jour.** Tous les chiffres produits par les tests
> viennent de scénarios synthétiques et de doubles `Fake*` : ils vérifient les calculs du
> logiciel d'évaluation et **ne mesurent rien de biométrique**. Tout rapport généré à partir
> de données `synthetic` (annotations ou journal) porte un avertissement en tête.

## Contenu

| Module | Rôle |
|---|---|
| `annotations.py` | Format `maisonaccueil.passages` v1 (séances + passages), lecture/écriture validées |
| `runlog.py` | Format `maisonaccueil.runlog` v1 (journal d'exécution), `RunLogWriter`, `load_runlog` |
| `splits.py` | Contrôle du découpage, lecture `collection_sessions` (SQLite, lecture seule), registre du test `TestLedger` |
| `matching.py` | Attribution des événements aux passages, issue de chaque passage |
| `metrics.py` | Wilson, proportions, délais, synthèse par passage et par personne |
| `compare.py` | Comparaison de configurations sur les mêmes passages (paires discordantes, McNemar exact) |
| `sweep.py` | Courbe score/erreurs, balayage de la politique de décision, choix de seuil, table de fiabilité — `tune` uniquement |
| `replay.py` | Rejeu hors ligne vidéo → `VisionLoop.process_frame` → journal ; rejeu de la décision seule depuis un journal |
| `report.py` | Rapport Markdown, CSV par passage, CSV de courbe |
| `protocol.py` | `evaluate()` : découpage + registre + attribution + métriques + comparaison |
| `cli.py` | `python -m maisonaccueil.evaluation check | report | curve | replay` |

Dépendances : contrats, `pipeline.VisionLoop`, `capture.create_source` (source fichier du lot A),
numpy/OpenCV déjà présents. Aucune nouvelle dépendance ; lots D, E et F uniquement via les
interfaces (`Recognizer`, `DecisionEngine`, `ArrivalTracker`, `WelcomeComposer`).

## Format d'annotation des passages (`maisonaccueil.passages` v1, JSON Lines)

Une ligne = un objet, champ `record`. Temps en **ms de séance** (depuis le début de la vidéo).

```jsonl
{"record":"header","format":"maisonaccueil.passages","version":1,"data_kind":"real","created_at":"2026-11-02T18:00:00Z","annotator":""}
{"record":"session","session_id":"ses-…","split":"test","recorded_at":"2026-11-02T17:30:00Z","camera_id":"entree","duration_ms":600000,"video_sha256":"…","conditions":"soir, lampe"}
{"record":"passage","passage_id":"p-0001","session_id":"ses-…","start_ms":12000,"end_ms":19500,"expected":"known","profile_id":"prf-…","welcome_expected":true,"note":""}
{"record":"passage","passage_id":"p-0002","session_id":"ses-…","start_ms":30000,"end_ms":36000,"expected":"unregistered","profile_id":null,"subject":"inconnu-A"}
```

- `header` obligatoire en première ligne ; `data_kind` = `real` | `synthetic` (mélange interdit).
- `split` ∈ `enroll`, `train`, `tune`, `test` (mêmes valeurs que `collection_sessions.split`).
- `expected` = `known` (avec `profile_id`) ou `unregistered` (sans `profile_id` ; `subject` =
  pseudonyme facultatif pour regrouper les passages d'un même inconnu).
- `welcome_expected=false` : personne déjà présente, aucun accueil ne doit être émis.
- Erreurs signalées avec `fichier:ligne` : JSON invalide, champ manquant, doublon, séance non
  déclarée, `end_ms < start_ms`, passage au-delà de `duration_ms`, horodatage sans fuseau.
- Aucun nom, aucune image, aucune représentation faciale dans le fichier.

## Format du journal d'exécution (`maisonaccueil.runlog` v1, JSON Lines)

Un fichier par (séance, configuration). `t_ms = (monotonic_ns − origin_monotonic_ns)/1e6`,
même repère que les annotations.

```jsonl
{"record":"run","format":"maisonaccueil.runlog","version":1,"run_id":"run-…","session_id":"ses-…","config_id":"sface@2021dec|temporal-v1|0.45","model":{"name":"…","version":"…","embedding_dim":128,"metric":"cosine","preprocess":"…","origin":"pretrained"},"policy":"temporal-v1","data_kind":"real","source":"replay:seance.mp4","origin_monotonic_ns":0,"messages_from":"welcome","started_at":"…Z","contracts_version":"1.0.0","note":""}
{"record":"frame","t_ms":0.0,"frame_id":1,"processing_ms":12.3}
{"record":"result","t_ms":500.0,"frame_id":6,"track_id":"trk-…","metric":"cosine","candidates":[{"profile_id":"prf-…","score":0.71,"refs":5}],"quality_flags":[]}
{"record":"decision","t_ms":500.0,"track_id":"trk-…","status":"confirmed","profile_id":"prf-…","best_score":0.71,"observations":3,"window_ms":800,"policy":"temporal-v1","reason":""}
{"record":"arrival","t_ms":500.0,"event_id":"evt-…","kind":"known","track_id":"trk-…","profile_id":"prf-…","wall_time":"…Z"}
{"record":"welcome","t_ms":500.0,"request_id":"evt-…","arrival_event_id":"evt-…","profile_id":"prf-…","dedupe_key":"…","mode":"text"}
{"record":"track_lost","t_ms":1600.0,"track_id":"trk-…"}
```

- `config_id` identifie moteur + politique + seuils : c'est l'unité comparée et l'unité suivie
  par le registre du test.
- `messages_from` = `welcome` (compositeur présent) ou `arrival` (rejeu sans compositeur).
- Le texte prononcé n'est pas journalisé. Pas d'image, pas de représentation.
- Le journal est trié par `t_ms` à la lecture (ordre d'écriture conservé à égalité).

## Découpage et traçabilité du test

`check_splits(ann, extra_sessions)` :

- **erreurs** : même `video_sha256` sous deux séances ; séance déclarée avec deux découpages
  (annotations vs base) ;
- **avertissements** : `tune`/`test` sans inconnu non enregistré ; passage `tune`/`test` le
  même jour (UTC) qu'une séance `enroll`/`train` de la même personne.

Registre `TestLedger` (`var/g/ledger.jsonl`, append-only) : enregistrements `evaluation`
(`split`, `set_id`, `sessions`, `config_ids`) et `tuning` (`set_id`, `informed_by`,
`config_before`, `config_after`). `set_id` = 16 premiers hex du SHA-256 des identifiants de
séances triés. Un test est **brûlé** si (1) il figure dans `informed_by` d'un réglage,
(2) un réglage a été consigné directement dessus, ou (3) il est évalué avec une
configuration absente de sa **première** évaluation. Reproduire exactement les mêmes
configurations (ou un sous-ensemble) ne le brûle pas. Évaluer un test brûlé lève
`BurnedTestSetError` (sauf `allow_burned=True` : le rapport affiche « TEST BRÛLÉ —
résultats non concluants »). `score_curve` / `sweep_decision` refusent tout journal hors
`tune` (`SplitViolationError`) et consignent un `tuning`.

Procédure : comparer tous les moteurs candidats **dans la même première évaluation** du test
(`evaluate` avec tous les journaux) ; tout changement ultérieur exige un nouveau test.

## Attribution et issues (définitions exactes)

- Fenêtre d'un passage : `[start_ms, end_ms + grace_ms]`, `grace_ms = 1000` par défaut.
- Un événement (arrivée, résultat, décision) est attribué à UN passage de sa séance dont la
  fenêtre le contient ; en cas de chevauchement : (1) passage `known` du même profil,
  (2) passage commencé le plus récemment (`start_ms` max ≤ t), (3) plus petit `passage_id`.
  Un `welcome` suit le passage de son arrivée (`arrival_event_id`). Sans passage : « hors
  passage » (compté à part).
- Issue d'un passage `known` de profil P : `wrong_identity` si une arrivée `known` ≠ P
  (prioritaire) ; sinon `correct` si une arrivée `known` = P (ou aucune arrivée `known` quand
  `welcome_expected=false`) ; sinon `missed`.
- Issue d'un passage `unregistered` : `false_accept` si une arrivée `known`, sinon
  `correct_reject`.
- Messages = `welcome` (ou `arrival` si `messages_from=arrival`) ; `repeated_messages =
  max(0, messages − (1 si welcome_expected sinon 0))`.
- Délai = premier message de profil P − `start_ms` (passages `known` `correct` avec accueil
  attendu).

## Métriques

| Mesure | Numérateur / dénominateur |
|---|---|
| Bonne identification | `correct` / passages `known` |
| Mauvaise identité | `wrong_identity` / passages `known` |
| Connu non reconnu | `missed` / passages `known` |
| Inconnu accepté à tort | `false_accept` / passages `unregistered` |
| Message répété | passages avec `repeated_messages > 0` / tous les passages (+ total des messages en trop) |
| Délai jusqu'à l'accueil | n, min, médiane, p90 (rang le plus proche), max |
| Arrivées hors passage | nombre d'arrivées `known` (et total) attribuées à aucun passage |

- Toujours `k/n` ; « non mesurable » si n = 0 ; jamais de pourcentage seul.
- Intervalle de **Wilson** 95 % (z = 1,959964) : centre `(p + z²/2n)/(1 + z²/n)`,
  demi-largeur `z·√(p(1−p)/n + z²/4n²)/(1 + z²/n)`, bornes exactes 0 / 1 si k = 0 / k = n.
  Suppose des passages indépendants → tableau **par personne** (mêmes colonnes, `k/n`) et
  nombre de personnes affichés.
- Comparaison A/B sur les **mêmes passages** (`PassageSetMismatchError` sinon) : réussite =
  `correct` ou `correct_reject` ; paires discordantes et p de McNemar exact bilatéral
  `min(1, 2·Σ_{i≤min(b,c)} C(b+c,i)/2^(b+c))`, indicatif.

## Réglage (`tune` uniquement)

- `score_curve` : règle de référence indépendante du lot F (meilleur score du 1er candidat
  parmi les résultats attribués au passage, accepté si `≥ t` en cosinus / `≤ t` en L2). Sert
  à comparer des moteurs à politique égale ; ce n'est pas la politique de l'appli.
- `sweep_decision(ann, logs, seuils, fabrique)` : rejoue la vraie politique sur les résultats
  journalisés (`rerun_decisions`) pour chaque seuil ; la fabrique renvoie un `DecisionSetup`
  neuf. Exemple (après intégration du lot F) :
  `lambda t: DecisionSetup(TemporalDecisionEngine(DecisionPolicy(accept_cosine=t)), PresenceArrivalTracker(), composer, "temporal-v1")`.
- `select_threshold` : maximise les bonnes identifications sous contraintes d'effectifs
  (`max_wrong_identity`, `max_false_accept`, 0 par défaut) ; à égalité le seuil le plus
  strict. C'est une **proposition** à confirmer sur un test indépendant.
- `reliability_table` : **calibration descriptive, non appliquée**, marquée
  `calibration_ref = calib-<hash(modèle, séances tune, bornes)>` et `status="descriptive"` ;
  par tranche de score : part des images dont le 1er candidat est la bonne personne (k/n,
  Wilson) et nombre de passages distincts ; tranche « exploitable » seulement si
  ≥ `min_passages_per_bin` passages (30 par défaut). Aucun `calibrated=True` n'est produit.

## Rejeu hors ligne

```python
from maisonaccueil.evaluation import replay_video
log, stats = replay_video("var/g/seance.mp4", session_id="ses-…", config_id="…",
                          model=embedder_model_ref, recognizer=rec, decision=engine,
                          arrivals=tracker, composer=composer,
                          out_path="var/g/runs/ses-….jsonl")
```

- Lecture via `capture.create_source` (source fichier du lot A, `realtime=False`), file
  bloquante : **aucune image abandonnée**. `Frame.monotonic_ns` réhorodaté sur le temps
  vidéo `(FrameId−1)/fps` → fenêtres de décision et `t_ms` dans le repère des annotations.
- Passage par `VisionLoop.process_frame` (même enchaînement que l'appli, y compris les
  appels `note_track_activity`/`track_lost` des contrats 1.2 une fois intégrés), sans son
  (`SilentPlayer`). Chaque résultat doit porter le `ModelRef` de l'en-tête
  (`ensure_compatible`).
- `rerun_decisions(log, …)` rejoue décision + arrivées (+ accueil) sur les résultats d'un
  journal, sans vidéo ni moteur de vision ; les `frame` journalisés servent d'instants de
  `tick` (sans eux, seuls les instants des résultats/pertes de suivi sont rejoués).
- La vidéo reste locale (`var/`, ignoré par Git) ; jamais committée.

## Ligne de commande

```
uv run python -m maisonaccueil.evaluation check var/g/annotations.jsonl [--db var/c/maison.sqlite]
uv run python -m maisonaccueil.evaluation curve --annotations … --runs var/g/runs/tune-*.jsonl --thresholds 0.30:0.70:0.05
uv run python -m maisonaccueil.evaluation report --annotations … --runs var/g/runs/test-*.jsonl --split test --csv var/g/passages.csv
uv run python -m maisonaccueil.evaluation replay --video var/g/seance.mp4 --session ses-… --config-id … --factory mon_module:fabrique --out var/g/runs/x.jsonl
```

`--factory module:fonction` : fonction sans argument renvoyant un objet portant `recognizer`,
`decision`, `arrivals`, `model` (et facultativement `composer`, `policy`). Sorties par défaut
sous `var/g/`.

## Tests

`uv run pytest tests/contracts tests/evaluation` → 69 réussis (60 pour `tests/evaluation`).
Couvrent : valeurs de référence de Wilson (dont 81/263 de Newcombe 1998), symétrie et bornes,
percentiles, McNemar exact ; formats et erreurs localisées ; contrôles du découpage ; règles
du registre (test brûlé) ; lecture SQLite du schéma partagé ; chaque issue, la priorité
« mauvaise identité », les chevauchements, la fenêtre de grâce, les messages répétés, les
accueils tardifs ; rejeu `replay_frames` avec `Fake*`, refus d'un `ModelRef` incompatible,
rejeu de la décision (identique puis seuil plus strict) ; balayage et courbe sur `tune`,
refus sur `test` ; comparaison et refus de passages différents ; rapport (avertissement
synthétique, test brûlé, `k/n`), CSV, CLI ; rejeu d'une **vidéo synthétique MJPG générée
dans un dossier temporaire** via la source fichier du lot A (réhorodatage, arrêt anticipé).

## Limites

- Aucune donnée réelle : aucune métrique n'a été mesurée ; seuils non réglés.
- Pas de rejeu exécuté avec les implémentations réelles D/F (non intégrées sur cette
  branche) ; seulement des `Fake*` conformes aux interfaces.
- Le rejeu mesure la logique, pas la latence temps réel (pas d'abandon d'images,
  `processing_ms` indicatif seulement, CPU du conteneur).
- Wilson et McNemar supposent des passages indépendants ; l'agrégat par personne est fourni
  mais aucun intervalle par grappes (bootstrap par personne) n'est implémenté.
- Attribution par fenêtres temporelles : des passages simultanés de plusieurs personnes
  sont départagés par règle ; une annotation par piste (`track_id`) n'est pas possible car
  les `TrackId` ne survivent pas au rejeu.
- Journal en direct (application) non branché : l'orchestration devra écrire un
  `RunLogWriter` depuis `VisionOutput` (voir proposition G-vision-output-lost-tracks).

## Propositions de contrat

- `docs/contract-changes/G-vision-output-lost-tracks.md` : ajouter `lost_tracks` à
  `VisionOutput` (contourné ici par une enveloppe du `Recognizer`).
