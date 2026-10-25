"""Enchaînement complet d'une évaluation : contrôle du découpage, registre du test,
attribution, métriques, comparaison."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .annotations import AnnotationSet, SessionRecord
from .compare import PairComparison, compare_all
from .matching import match_runs
from .metrics import Z95, EngineEvaluation, evaluate_match
from .runlog import RunLog
from .splits import SplitReport, TestLedger, TestSetStatus, check_splits


class SplitCheckError(ValueError):
    """Le découpage contient une erreur bloquante (chevauchement)."""


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    split: str
    evaluations: tuple[EngineEvaluation, ...]
    comparisons: tuple[PairComparison, ...]
    split_report: SplitReport
    test_status: TestSetStatus | None


def evaluate(
    ann: AnnotationSet,
    logs: Sequence[RunLog],
    *,
    split: str = "test",
    ledger: TestLedger | None = None,
    allow_burned: bool = False,
    grace_ms: float = 1000.0,
    z: float = Z95,
    extra_sessions: Sequence[SessionRecord] = (),
    note: str = "",
) -> EvaluationRun:
    """Évalue toutes les configurations présentes dans ``logs`` sur les passages de
    ``split``. Sur ``test`` avec un registre : l'évaluation est consignée et refusée si
    l'ensemble est brûlé (``BurnedTestSetError``) sauf ``allow_burned``."""
    if split not in ("tune", "test"):
        raise ValueError("seuls 'tune' et 'test' sont évaluables")
    rep = check_splits(ann, extra_sessions)
    if not rep.ok:
        raise SplitCheckError("découpage invalide : " + " ; ".join(rep.errors))
    matches = match_runs(ann, logs, splits=(split,), grace_ms=grace_ms)
    status: TestSetStatus | None = None
    sessions = list(ann.sessions_in(split))
    if ledger is not None:
        if split == "test":
            status = ledger.record_evaluation(sessions, sorted(matches),
                                              allow_burned=allow_burned, note=note)
        else:
            ledger.record_evaluation(sessions, sorted(matches), split=split, note=note)
    evaluations = tuple(evaluate_match(m, z) for _, m in sorted(matches.items()))
    comparisons = tuple(compare_all(matches)) if len(matches) > 1 else ()
    return EvaluationRun(split, evaluations, comparisons, rep, status)

