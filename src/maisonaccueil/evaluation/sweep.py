"""Balayage de seuils (UNIQUEMENT sur ``tune``), courbes score/erreurs, calibration descriptive.

Deux niveaux :

1. ``score_curve`` — règle de référence indépendante du lot F, par passage : on retient,
   parmi les résultats de la fenêtre du passage, celui dont le score du 1er candidat est le
   meilleur ; il est « accepté » si ce score passe le seuil (``≥ t`` en cosinus, ``≤ t`` en
   L2). Issue : known P → accepté & P = correct, accepté & ≠ P = wrong_identity, sinon
   missed ; unregistered → accepté = false_accept, sinon correct_reject. Sert à comparer
   des moteurs à politique égale ; ce n'est PAS la politique de décision de l'appli.
2. ``sweep_decision`` — rejoue la vraie politique (``DecisionEngine`` + ``ArrivalTracker``
   construits par une fabrique ``seuil → DecisionSetup``) sur les résultats journalisés,
   puis applique les définitions de ``matching``.

Les deux refusent (``SplitViolationError``) tout journal d'une séance hors ``tune`` et
peuvent consigner le réglage dans le ``TestLedger``.

``reliability_table`` : table de fiabilité **descriptive** (par tranche de score du
1er candidat, part des résultats dont le 1er candidat est la bonne personne, k/n + Wilson).
Elle est marquée ``calibration_ref`` (empreinte : modèle + séances ``tune``) et n'est
JAMAIS appliquée automatiquement : un score reste un score, pas une probabilité. Les
résultats d'un même passage sont fortement corrélés ; la table n'est indicative que si
chaque tranche contient assez de passages distincts (``min_passages_per_bin``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from maisonaccueil.contracts import Metric
from maisonaccueil.contracts.interfaces import ArrivalTracker, DecisionEngine, WelcomeComposer

from .annotations import AnnotationSet, Expected, Passage
from .matching import Outcome, match_runs
from .metrics import Z95, Proportion, Summary, summarize
from .replay import rerun_decisions
from .runlog import RunLog
from .splits import TestLedger, require_split, set_id


def _passes(score: float, threshold: float, metric: Metric) -> bool:
    return score >= threshold if metric == Metric.COSINE else score <= threshold


def _better(a: float, b: float, metric: Metric) -> bool:
    return a > b if metric == Metric.COSINE else a < b


def _single_config(logs: Sequence[RunLog]) -> tuple[str, Metric]:
    cfgs = {lg.header.config_id for lg in logs}
    metrics = {lg.header.model.metric for lg in logs}
    keys = {lg.header.model.key for lg in logs}
    if len(cfgs) != 1 or len(keys) != 1:
        raise ValueError("un balayage porte sur UNE configuration et UN modèle à la fois")
    return cfgs.pop(), metrics.pop()


def _check_tune(ann: AnnotationSet, logs: Sequence[RunLog], split: str) -> list[str]:
    if split != "tune":
        raise ValueError("le réglage des seuils ne se fait que sur 'tune'")
    sessions = [lg.header.session_id for lg in logs]
    require_split(ann, sessions, "tune")
    return sorted(set(sessions))


@dataclass(frozen=True, slots=True)
class CurvePoint:
    threshold: float
    correct: Proportion
    wrong_identity: Proportion
    missed: Proportion
    false_accept: Proportion
    no_result: int  # passages sans aucun résultat dans la fenêtre


def _passage_best(
    p: Passage, log: RunLog, grace_ms: float, metric: Metric
) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for e in log.of("result"):
        t = float(e["t_ms"])
        if not p.start_ms <= t <= p.end_ms + grace_ms or not e["candidates"]:
            continue
        c = e["candidates"][0]
        s = float(c["score"])
        if best is None or _better(s, best[1], metric):
            best = (str(c["profile_id"]), s)
    return best


def score_curve(
    ann: AnnotationSet,
    logs: Sequence[RunLog],
    thresholds: Sequence[float],
    *,
    split: str = "tune",
    grace_ms: float = 1000.0,
    z: float = Z95,
    ledger: TestLedger | None = None,
    note: str = "",
) -> list[CurvePoint]:
    sessions = _check_tune(ann, logs, split)
    cfg, metric = _single_config(logs)
    by_session = {lg.header.session_id: lg for lg in logs}
    passages = [p for p in ann.passages_in(["tune"]) if p.session_id in by_session]
    bests = [(p, _passage_best(p, by_session[p.session_id], grace_ms, metric)) for p in passages]
    n_k = sum(1 for p in passages if p.expected == Expected.KNOWN)
    n_u = len(passages) - n_k
    out: list[CurvePoint] = []
    for t in thresholds:
        counts = {o: 0 for o in Outcome}
        for p, best in bests:
            accepted = best is not None and _passes(best[1], t, metric)
            if p.expected == Expected.KNOWN:
                if accepted and best is not None and best[0] == p.profile_id:
                    counts[Outcome.CORRECT] += 1
                elif accepted:
                    counts[Outcome.WRONG_IDENTITY] += 1
                else:
                    counts[Outcome.MISSED] += 1
            else:
                counts[Outcome.FALSE_ACCEPT if accepted else Outcome.CORRECT_REJECT] += 1
        out.append(
            CurvePoint(
                float(t),
                Proportion.of(counts[Outcome.CORRECT], n_k, z),
                Proportion.of(counts[Outcome.WRONG_IDENTITY], n_k, z),
                Proportion.of(counts[Outcome.MISSED], n_k, z),
                Proportion.of(counts[Outcome.FALSE_ACCEPT], n_u, z),
                sum(1 for _, b in bests if b is None),
            )
        )
    if ledger is not None:
        ledger.record_tuning(sessions, config_before=cfg, config_after="",
                             note=note or "score_curve")
    return out


@dataclass(frozen=True, slots=True)
class DecisionSetup:
    decision: DecisionEngine
    arrivals: ArrivalTracker
    composer: WelcomeComposer | None = None
    policy: str = ""


@dataclass(frozen=True, slots=True)
class SweepPoint:
    threshold: float
    config_id: str
    summary: Summary
    unattributed_known: int


def sweep_decision(
    ann: AnnotationSet,
    logs: Sequence[RunLog],
    thresholds: Sequence[float],
    factory: Callable[[float], DecisionSetup],
    *,
    split: str = "tune",
    grace_ms: float = 1000.0,
    z: float = Z95,
    ledger: TestLedger | None = None,
    note: str = "",
) -> list[SweepPoint]:
    """Rejoue la politique de décision pour chaque seuil. ``factory`` doit renvoyer des
    objets NEUFS à chaque appel (aucun état partagé entre seuils ni entre séances)."""
    sessions = _check_tune(ann, logs, split)
    base_cfg, _ = _single_config(logs)
    out: list[SweepPoint] = []
    for t in thresholds:
        cfg = f"{base_cfg}|seuil={t:g}"
        reruns: list[RunLog] = []
        for lg in logs:
            setup = factory(float(t))
            reruns.append(
                rerun_decisions(lg, config_id=cfg, decision=setup.decision,
                                arrivals=setup.arrivals, composer=setup.composer,
                                policy=setup.policy, note=f"balayage seuil {t:g}")
            )
        m = match_runs(ann, reruns, splits=("tune",), grace_ms=grace_ms,
                       require_all_sessions=False)[cfg]
        unk = sum(1 for u in m.unattributed if u.record == "arrival" and u.kind == "known")
        out.append(SweepPoint(float(t), cfg, summarize(m.passages, z), unk))
    if ledger is not None:
        ledger.record_tuning(sessions, config_before=base_cfg, config_after="",
                             note=note or "sweep_decision")
    return out


def select_threshold(
    points: Sequence[SweepPoint] | Sequence[CurvePoint],
    metric: Metric,
    *,
    max_wrong_identity: int = 0,
    max_false_accept: int = 0,
) -> float | None:
    """Seuil maximisant le nombre de bonnes identifications sous contraintes d'effectifs
    (mauvaises identités ≤ ``max_wrong_identity``, inconnus acceptés ≤ ``max_false_accept``).
    À égalité : le seuil le plus strict. ``None`` si aucun seuil ne satisfait les
    contraintes. Proposition à valider sur un test indépendant, pas une mesure."""
    best: tuple[int, float] | None = None
    for pt in points:
        s = pt.summary if isinstance(pt, SweepPoint) else pt
        if s.wrong_identity.k > max_wrong_identity or s.false_accept.k > max_false_accept:
            continue
        key = s.correct.k
        if (
            best is None
            or key > best[0]
            or (key == best[0] and _better(pt.threshold, best[1], metric))
        ):
            best = (key, pt.threshold)
    return None if best is None else best[1]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    low: float
    high: float
    top1_correct: Proportion  # résultats (images) dont le 1er candidat est la bonne personne
    passages: int  # passages distincts contribuant à la tranche
    usable: bool


@dataclass(frozen=True, slots=True)
class ReliabilityTable:
    calibration_ref: str
    model_key: str
    split: str
    sessions: tuple[str, ...]
    bins: tuple[ReliabilityBin, ...]
    status: str = "descriptive"  # jamais appliquée automatiquement


def reliability_table(
    ann: AnnotationSet,
    logs: Sequence[RunLog],
    edges: Sequence[float],
    *,
    split: str = "tune",
    grace_ms: float = 1000.0,
    min_passages_per_bin: int = 30,
    z: float = Z95,
) -> ReliabilityTable:
    """Tranches ``[edges[i], edges[i+1][`` (la dernière fermée). Seuls les résultats
    situés dans la fenêtre d'un passage annoté comptent ; un résultat d'un passage
    ``unregistered`` est toujours « incorrect » (aucun bon candidat n'existe)."""
    sessions = _check_tune(ann, logs, split)
    _single_config(logs)
    if len(edges) < 2 or list(edges) != sorted(edges):
        raise ValueError("edges : au moins deux bornes croissantes")
    nb = len(edges) - 1
    k = [0] * nb
    n = [0] * nb
    pas: list[set[str]] = [set() for _ in range(nb)]
    by_session = {lg.header.session_id: lg for lg in logs}
    for p in ann.passages_in(["tune"]):
        lg = by_session.get(p.session_id)
        if lg is None:
            continue
        for e in lg.of("result"):
            t = float(e["t_ms"])
            if not p.start_ms <= t <= p.end_ms + grace_ms or not e["candidates"]:
                continue
            c = e["candidates"][0]
            s = float(c["score"])
            idx = next((i for i in range(nb) if edges[i] <= s < edges[i + 1]), None)
            if idx is None and s == edges[-1]:
                idx = nb - 1
            if idx is None:
                continue
            n[idx] += 1
            pas[idx].add(p.passage_id)
            if p.expected == Expected.KNOWN and c["profile_id"] == p.profile_id:
                k[idx] += 1
    model_key = logs[0].header.model.key
    ref = hashlib.sha256(f"{model_key}|{set_id(sessions)}|{list(edges)}".encode()).hexdigest()
    bins = tuple(
        ReliabilityBin(float(edges[i]), float(edges[i + 1]), Proportion.of(k[i], n[i], z),
                       len(pas[i]), len(pas[i]) >= min_passages_per_bin)
        for i in range(nb)
    )
    return ReliabilityTable(f"calib-{ref[:12]}", model_key, split, tuple(sessions), bins)
