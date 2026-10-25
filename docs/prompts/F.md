# Session Lot F — DÉCISION ET ACCUEIL

Branche : `lot/f-decision` · Chemins : `src/maisonaccueil/decision/`, `src/maisonaccueil/welcome/`, `tests/decision/`, `tests/welcome/`, `docs/lots/F-decision.md`

## Objectif
Transformer les `RecognitionResult` en `Decision`, puis en `ArrivalEvent` et `AudioRequest`.

## Travail attendu
- `TemporalDecisionEngine` (`DecisionEngine`) : accumulation par `TrackId` dans une fenêtre
  glissante ; CONFIRMED si N observations concordantes (même meilleur profil) au-dessus
  d'un seuil d'acceptation avec marge suffisante sur le 2ᵉ candidat ; UNKNOWN si assez
  d'observations sans aucun profil accepté ; UNCERTAIN sinon / en cas de contradiction.
  Paramètres dans une `DecisionPolicy` versionnée (`policy="temporal-v1"`), valeurs par
  défaut **provisoires** à régler avec le lot G. Une décision CONFIRMED d'un passage n'est
  pas réémise à chaque image. Ignorer les observations à `QualityFlag` bloquants.
- `PresenceArrivalTracker` (`ArrivalTracker`) : présence par profil ; une seule arrivée
  par présence ; maintien pendant les pertes de suivi temporaires et changements de
  `TrackId` du même profil ; expiration après un délai d'absence configurable (`tick`).
  Inconnus : au plus un événement UNKNOWN par passage, sans nom.
- `WelcomeComposer` : message personnalisé (fichier audio du profil ou texte), modèle
  générique sinon ; politique pour les inconnus (silence par défaut, configurable).
  Le composer reçoit les données de profil via une interface/callable injecté.
- `QueuedAudioPlayer` (`AudioPlayer`) : thread dédié, file avec priorité, déduplication par
  `dedupe_key`, expiration (`expires_after_ms`) ; arrivées simultanées : messages
  enchaînés ou regroupés (« Bonjour A et B ») — documenter le choix. Backends : `winsound`
  (WAV, Windows), synthèse locale optionnelle (proposer la dépendance), et `FakeAudioBackend`
  pour les tests. Aucun service cloud.
- Horloge injectable pour des tests déterministes : pertes de suivi, répétitions, arrivées
  simultanées, inconnu, oscillation entre deux identités.

## Règles communes (obligatoires)

Tu es un travailleur du projet MaisonAccueil (reconnaissance faciale locale, usage
familial, accueil vocal). Le coordinateur possède les contrats et l'intégration.

1. Lis d'abord `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/OWNERSHIP.md`, `docs/DATA_POLICY.md`
   et TOUT `src/maisonaccueil/contracts/`.
2. Travaille UNIQUEMENT dans ton worktree et sur ta branche de lot. N'écris que dans les
   chemins de ton lot (liste ci-dessous) et, si nécessaire, dans
   `docs/contract-changes/<lot>-<sujet>.md`. N'écris jamais dans le dossier d'une autre session.
3. Ne modifie pas `contracts/`, `pipeline/`, `pyproject.toml`, `uv.lock`. Besoin d'un
   changement ou d'une dépendance → proposition écrite, puis contourne avec ce qui existe.
4. Ne fusionne pas, ne rebase pas la branche d'intégration, ne pousse rien vers un dépôt
   distant. Committe sur ta branche, en petits commits préfixés `lot-<X>: `.
5. Environnement : `uv sync --extra gui` dans ton worktree (crée `.venv` local). Aucune
   installation globale, aucun pilote.
6. Données de session dans `var/<lot>/` (ignoré par Git). Aucune photo réelle, aucune
   personne inventée ; données synthétiques uniquement pour tester le logiciel.
7. Caméra / GPU : `maisonaccueil.pipeline.ResourceLock` obligatoire ; tests matériels marqués
   `@pytest.mark.camera` / `@pytest.mark.gpu` et non exécutés par défaut.
8. Un score n'est pas une probabilité ; n'annonce aucune précision. Doubles de test nommés
   `Fake*`.
9. Avant chaque commit : `uv run ruff check src tests` et
   `uv run pytest tests/contracts tests/<tes modules>` doivent passer.
10. Termine par un rapport : fichiers créés, commits (hash), tests exécutés et résultats
    réels, limites, propositions de contrat, prochaine étape. Ne prétends pas avoir testé ce
    qui ne l'a pas été (caméra, GPU, son, GUI absents dans certains environnements).
