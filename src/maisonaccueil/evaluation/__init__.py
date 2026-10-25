"""Lot G — Évaluation hors ligne (voir docs/lots/G-evaluation.md).

Formats JSON Lines : annotations de passages (``annotations``), journal d'exécution
(``runlog``), registre du test (``splits.TestLedger``). Métriques par passage et par
personne avec effectifs et intervalles de Wilson (``metrics``), comparaison de moteurs
sur les mêmes passages (``compare``), balayage de seuils sur ``tune`` (``sweep``),
rejeu hors ligne (``replay``), rapport Markdown/CSV (``report``).

Aucun chiffre produit à partir de données synthétiques ne mesure quoi que ce soit de
biométrique ; le rapport l'indique explicitement.
"""

from .annotations import (
    AnnotationSet,
    DataKind,
    Expected,
    Passage,
    SessionRecord,
    load_annotations,
    save_annotations,
)
from .compare import PairComparison, PassageSetMismatchError, compare_all, compare_pair
from .jsonl import FormatError
from .matching import MatchResult, Outcome, PassageResult, match_runs, match_session
from .metrics import (
    DelayStats,
    EngineEvaluation,
    Proportion,
    Summary,
    evaluate_match,
    summarize,
    wilson_interval,
)
from .protocol import EvaluationRun, SplitCheckError, evaluate
from .replay import (
    ReplayStats,
    iter_video_frames,
    replay_frames,
    replay_video,
    rerun_decisions,
)
from .report import passages_csv, render_report
from .runlog import RunHeader, RunLog, RunLogWriter, load_runlog, load_runlogs
from .splits import (
    BurnedTestSetError,
    SplitViolationError,
    TestLedger,
    check_splits,
    set_id,
)
from .sweep import (
    DecisionSetup,
    reliability_table,
    score_curve,
    select_threshold,
    sweep_decision,
)

__all__ = [
    "AnnotationSet",
    "BurnedTestSetError",
    "DataKind",
    "DecisionSetup",
    "DelayStats",
    "EngineEvaluation",
    "EvaluationRun",
    "Expected",
    "FormatError",
    "MatchResult",
    "Outcome",
    "PairComparison",
    "Passage",
    "PassageResult",
    "PassageSetMismatchError",
    "Proportion",
    "ReplayStats",
    "RunHeader",
    "RunLog",
    "RunLogWriter",
    "SessionRecord",
    "SplitCheckError",
    "SplitViolationError",
    "Summary",
    "TestLedger",
    "check_splits",
    "compare_all",
    "compare_pair",
    "evaluate",
    "evaluate_match",
    "iter_video_frames",
    "load_annotations",
    "load_runlog",
    "load_runlogs",
    "match_runs",
    "match_session",
    "passages_csv",
    "reliability_table",
    "render_report",
    "replay_frames",
    "replay_video",
    "rerun_decisions",
    "save_annotations",
    "score_curve",
    "select_threshold",
    "set_id",
    "summarize",
    "sweep_decision",
    "wilson_interval",
]
