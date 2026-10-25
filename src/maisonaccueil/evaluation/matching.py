"""Attribution des événements journalisés aux passages annotés et issue de chaque passage.

Définitions exactes (voir aussi docs/lots/G-evaluation.md) :

**Fenêtre d'un passage** : ``[start_ms, end_ms + grace_ms]`` en temps de séance
(``grace_ms`` = 1000 par défaut : une décision peut tomber juste après la sortie du champ).

**Attribution** d'une arrivée (``arrival``) : parmi les passages de la même séance dont la
fenêtre contient ``t_ms`` ; s'il y en a plusieurs, on préfère (1) un passage ``known`` dont
le ``profile_id`` est celui de l'arrivée, puis (2) le passage commencé le plus récemment
(``start_ms`` maximal ≤ ``t_ms``), puis (3) le plus petit ``passage_id``. Sans passage
candidat, l'arrivée est **hors passage** (comptée à part : accueil intempestif potentiel).
Un accueil (``welcome``) suit le passage de son arrivée (``arrival_event_id``) ; à défaut, il
est attribué par le temps avec les mêmes règles. Les décisions sont attribuées par le temps
(information, non utilisées pour l'issue). Chaque ``result`` est attribué à UN seul
passage (même règle, profil = 1er candidat) : il sert au compte ``results``, à la courbe
de score et à la table de fiabilité.

**Messages** : les ``welcome`` du journal si ``messages_from = "welcome"``, sinon les
``arrival`` (exécution sans compositeur).

**Issue** (exclusive) d'un passage ``known`` de profil P :

- ``wrong_identity`` (mauvaise identité) : au moins une arrivée ``known`` attribuée avec un
  profil ≠ P (prioritaire : un mauvais prénom a été annoncé) ;
- sinon ``correct`` (bonne identification) : au moins une arrivée ``known`` de profil P, ou
  aucune arrivée ``known`` si ``welcome_expected = false`` (personne déjà présente) ;
- sinon ``missed`` (connu non reconnu) : aucune arrivée ``known`` (rien, ou une arrivée
  ``unknown``).

Issue d'un passage ``unregistered`` :

- ``false_accept`` (inconnu accepté à tort) : au moins une arrivée ``known`` attribuée ;
- sinon ``correct_reject``.

**Message répété** : ``repeated_messages = max(0, messages − autorisés)`` avec
autorisés = 1 si ``welcome_expected`` sinon 0 ; le passage est « répété » si > 0.

**Délai jusqu'à l'accueil** (passages ``known`` ``correct`` avec ``welcome_expected``) :
``t_ms`` du premier message de profil P attribué − ``start_ms`` du passage.
"""

from __future__ import annotations

import enum
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .annotations import AnnotationSet, Expected, Passage
from .runlog import RunLog


class Outcome(enum.StrEnum):
    CORRECT = "correct"
    WRONG_IDENTITY = "wrong_identity"
    MISSED = "missed"
    FALSE_ACCEPT = "false_accept"
    CORRECT_REJECT = "correct_reject"

    @property
    def ok(self) -> bool:
        return self in (Outcome.CORRECT, Outcome.CORRECT_REJECT)


OUTCOME_LABELS_FR = {
    Outcome.CORRECT: "bonne identification",
    Outcome.WRONG_IDENTITY: "mauvaise identité",
    Outcome.MISSED: "connu non reconnu",
    Outcome.FALSE_ACCEPT: "inconnu accepté à tort",
    Outcome.CORRECT_REJECT: "inconnu correctement rejeté",
}


@dataclass(frozen=True, slots=True)
class PassageResult:
    passage: Passage
    config_id: str
    outcome: Outcome
    known_arrivals: tuple[str, ...]  # profile_id des arrivées KNOWN attribuées
    unknown_arrivals: int
    messages: int
    repeated_messages: int
    delay_ms: float | None
    decisions: dict[str, int]  # nombre de décisions attribuées par statut
    results: int  # nombre de RecognitionResult dans la fenêtre

    @property
    def message_repeated(self) -> bool:
        return self.repeated_messages > 0


@dataclass(frozen=True, slots=True)
class UnattributedEvent:
    session_id: str
    config_id: str
    record: str
    t_ms: float
    kind: str
    profile_id: str | None


@dataclass(frozen=True, slots=True)
class MatchResult:
    config_id: str
    passages: tuple[PassageResult, ...]
    unattributed: tuple[UnattributedEvent, ...]
    data_kinds: frozenset[str]
    sessions: tuple[str, ...]

    def by_id(self) -> dict[str, PassageResult]:
        return {r.passage.passage_id: r for r in self.passages}


def _pick(cands: list[Passage], t_ms: float, profile: str | None) -> Passage:
    def key(p: Passage) -> tuple[int, float, str]:
        same = p.expected == Expected.KNOWN and profile is not None and p.profile_id == profile
        started = p.start_ms if p.start_ms <= t_ms else float("-inf")
        return (0 if same else 1, -started, p.passage_id)

    return min(cands, key=key)


def _window_candidates(passages: Sequence[Passage], t: float, grace_ms: float) -> list[Passage]:
    return [p for p in passages if p.start_ms <= t <= p.end_ms + grace_ms]


def attribute_results(
    passages: Sequence[Passage], log: RunLog, *, grace_ms: float = 1000.0
) -> dict[str, list[dict[str, Any]]]:
    """Résultats (``result``) du journal attribués à UN passage chacun, avec la même règle
    que les arrivées (profil = 1er candidat). Un résultat hors fenêtre n'est pas retenu."""
    sid = log.header.session_id
    ps = [p for p in passages if p.session_id == sid]
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in log.of("result"):
        t = float(e["t_ms"])
        cands = _window_candidates(ps, t, grace_ms)
        if not cands:
            continue
        top = e["candidates"][0]["profile_id"] if e["candidates"] else None
        out[_pick(cands, t, top).passage_id].append(e)
    return out


def match_session(
    passages: Sequence[Passage], log: RunLog, *, grace_ms: float = 1000.0
) -> tuple[list[PassageResult], list[UnattributedEvent]]:
    """Issues des passages d'UNE séance pour UN journal (même ``session_id``)."""
    sid = log.header.session_id
    cfg = log.header.config_id
    ps = [p for p in passages if p.session_id == sid]
    arrivals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    messages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    results = {k: len(v) for k, v in attribute_results(ps, log, grace_ms=grace_ms).items()}
    arrival_owner: dict[str, str] = {}
    unattributed: list[UnattributedEvent] = []
    use_welcome = log.header.messages_from == "welcome"

    for e in log.events:
        t = float(e["t_ms"])
        rec = e["record"]
        if rec in ("result", "track_lost", "frame"):
            continue
        profile = e.get("profile_id")
        owner: Passage | None = None
        if rec == "welcome" and e.get("arrival_event_id") in arrival_owner:
            pid = arrival_owner[e["arrival_event_id"]]
            owner = next(p for p in ps if p.passage_id == pid)
        else:
            cands = _window_candidates(ps, t, grace_ms)
            owner = _pick(cands, t, profile) if cands else None
        if owner is None:
            if rec in ("arrival", "welcome"):
                unattributed.append(
                    UnattributedEvent(sid, cfg, rec, t, str(e.get("kind", "")), profile)
                )
            continue
        if rec == "decision":
            decisions[owner.passage_id][str(e["status"])] += 1
        elif rec == "arrival":
            arrivals[owner.passage_id].append(e)
            arrival_owner[str(e["event_id"])] = owner.passage_id
            if not use_welcome:
                messages[owner.passage_id].append(e)
        elif rec == "welcome" and use_welcome:
            messages[owner.passage_id].append(e)

    out: list[PassageResult] = []
    for p in ps:
        arr = arrivals.get(p.passage_id, [])
        known = tuple(str(a["profile_id"]) for a in arr if a.get("kind") == "known")
        n_unknown = sum(1 for a in arr if a.get("kind") == "unknown")
        msgs = messages.get(p.passage_id, [])
        delay: float | None = None
        if p.expected == Expected.KNOWN:
            if any(k != p.profile_id for k in known):
                outcome = Outcome.WRONG_IDENTITY
            elif known or not p.welcome_expected:
                outcome = Outcome.CORRECT
            else:
                outcome = Outcome.MISSED
            if outcome == Outcome.CORRECT and p.welcome_expected:
                times = [float(m["t_ms"]) for m in msgs if m.get("profile_id") == p.profile_id]
                if times:
                    delay = max(0.0, min(times) - p.start_ms)
        else:
            outcome = Outcome.FALSE_ACCEPT if known else Outcome.CORRECT_REJECT
        allowed = 1 if p.welcome_expected else 0
        out.append(
            PassageResult(
                passage=p,
                config_id=cfg,
                outcome=outcome,
                known_arrivals=known,
                unknown_arrivals=n_unknown,
                messages=len(msgs),
                repeated_messages=max(0, len(msgs) - allowed),
                delay_ms=delay,
                decisions=dict(decisions.get(p.passage_id, {})),
                results=results.get(p.passage_id, 0),
            )
        )
    return out, unattributed


def match_runs(
    ann: AnnotationSet,
    logs: Sequence[RunLog],
    *,
    splits: Sequence[str] = ("test",),
    grace_ms: float = 1000.0,
    require_all_sessions: bool = True,
) -> dict[str, MatchResult]:
    """Regroupe les journaux par ``config_id`` et calcule l'issue de chaque passage des
    découpages ``splits``. Lève ``ValueError`` si un journal vise une séance hors de ces
    découpages, si une séance est couverte deux fois pour une même configuration, ou (par
    défaut) si une séance évaluée n'a pas de journal pour une configuration."""
    passages = ann.passages_in(splits)
    wanted_sessions = sorted({s for s, r in ann.sessions.items() if r.split in splits})
    by_cfg: dict[str, list[RunLog]] = defaultdict(list)
    for log in logs:
        sid = log.header.session_id
        if sid not in ann.sessions:
            raise ValueError(f"journal {log.source_path}: séance {sid} non annotée")
        if ann.sessions[sid].split not in splits:
            raise ValueError(
                f"journal {log.source_path}: séance {sid} du découpage "
                f"'{ann.sessions[sid].split}', hors de {list(splits)}"
            )
        by_cfg[log.header.config_id].append(log)
    out: dict[str, MatchResult] = {}
    for cfg, cfg_logs in sorted(by_cfg.items()):
        seen = [lg.header.session_id for lg in cfg_logs]
        dup = {s for s in seen if seen.count(s) > 1}
        if dup:
            raise ValueError(f"{cfg}: plusieurs journaux pour la séance {sorted(dup)}")
        missing = sorted(set(wanted_sessions) - set(seen))
        if missing and require_all_sessions:
            raise ValueError(f"{cfg}: séances sans journal : {', '.join(missing)}")
        results: list[PassageResult] = []
        unattr: list[UnattributedEvent] = []
        for lg in cfg_logs:
            r, u = match_session(passages, lg, grace_ms=grace_ms)
            results += r
            unattr += u
        results.sort(key=lambda r: (r.passage.session_id, r.passage.start_ms,
                                    r.passage.passage_id))
        out[cfg] = MatchResult(
            cfg,
            tuple(results),
            tuple(unattr),
            frozenset(str(lg.header.data_kind) for lg in cfg_logs),
            tuple(sorted(seen)),
        )
    return out
