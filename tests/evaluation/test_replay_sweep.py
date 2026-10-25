"""Rejeu hors ligne, balayage de seuils et comparaison, avec doubles ``Fake*``.

Scénario scripté (10 images/s, 1 image = 100 ms) — AUCUNE valeur biométrique :
- p1 : connu prf-a, piste t1 (images 5–15), score 0.80 sur prf-a ;
- p2 : inconnu, piste t2 (images 30–40), 1er candidat prf-b à 0.55 ;
- p3 : connu prf-b, piste t3 (images 60–70), 0.65 sur prf-b ;
- p4 : connu prf-a, piste t4 (images 80–90), 1er candidat prf-c à 0.52.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from maisonaccueil.contracts import IncompatibleEmbeddingError, Metric
from maisonaccueil.evaluation import (
    DecisionSetup,
    Outcome,
    RunLogWriter,
    SplitViolationError,
    TestLedger,
    compare_all,
    evaluate,
    load_runlog,
    match_runs,
    reliability_table,
    replay_frames,
    replay_video,
    rerun_decisions,
    score_curve,
    select_threshold,
    sweep_decision,
)
from maisonaccueil.evaluation.replay import iter_video_frames

from .fakes import (
    FAKE_MODEL,
    FAKE_MODEL_B,
    FakeArrivalTracker,
    FakeDecisionEngine,
    FakeRecognizer,
    FakeWelcomeComposer,
    annset,
    frame,
    header,
    known,
    sess,
    unreg,
)

TRACKS = [("t1", "prf-a", 0.80, 5, 15), ("t2", "prf-b", 0.55, 30, 40),
          ("t3", "prf-b", 0.65, 60, 70), ("t4", "prf-c", 0.52, 80, 90)]


def script() -> tuple[dict[int, list[tuple[str, str, float]]], dict[int, list[str]]]:
    s: dict[int, list[tuple[str, str, float]]] = {}
    lost: dict[int, list[str]] = {}
    for track, prof, score, a, b in TRACKS:
        for i in range(a, b + 1):
            s.setdefault(i, []).append((track, prof, score))
        lost.setdefault(b + 1, []).append(track)
    return s, lost


def passages(session: str) -> list[object]:
    return [known(f"{session}-p1", session, 400, 1400, "prf-a"),
            unreg(f"{session}-p2", session, 2900, 3900, "inconnu-A"),
            known(f"{session}-p3", session, 5900, 6900, "prf-b"),
            known(f"{session}-p4", session, 7900, 8900, "prf-a")]


ANN = annset([sess("u", "tune"), sess("t", "test")], passages("u") + passages("t"))  # type: ignore[arg-type]


def run(session: str, threshold: float = 0.5, config: str = "cfg-A") -> object:
    s, lost = script()
    w = RunLogWriter(header(session, config))
    stats = replay_frames((frame(i) for i in range(1, 101)), recognizer=FakeRecognizer(s, lost),
                          decision=FakeDecisionEngine(threshold), arrivals=FakeArrivalTracker(),
                          composer=FakeWelcomeComposer(), writer=w)
    assert stats.frames == 100 and stats.results == sum(b - a + 1 for *_, a, b in TRACKS)
    return w.to_runlog()


def outcomes(log: object, split: str = "tune") -> dict[str, Outcome]:
    m = match_runs(ANN, [log], splits=(split,))  # type: ignore[list-item]
    (res,) = m.values()
    return {r.passage.passage_id.split("-")[1]: r.outcome for r in res.passages}


def test_replay_frames_logs_everything() -> None:
    log = run("u")
    kinds = [e["record"] for e in log.events]  # type: ignore[attr-defined]
    assert kinds.count("frame") == 100
    assert kinds.count("track_lost") == 4
    assert kinds.count("arrival") == 4 and kinds.count("welcome") == 4
    assert outcomes(log) == {"p1": Outcome.CORRECT, "p2": Outcome.FALSE_ACCEPT,
                             "p3": Outcome.CORRECT, "p4": Outcome.WRONG_IDENTITY}
    first_welcome = next(e for e in log.events if e["record"] == "welcome")  # type: ignore[attr-defined]
    assert first_welcome["t_ms"] == pytest.approx(500.0)  # 2e résultat de t1 (image 6)


def test_replay_refuses_incompatible_model() -> None:
    s, lost = script()
    w = RunLogWriter(header("u"))
    with pytest.raises(IncompatibleEmbeddingError):
        replay_frames((frame(i) for i in range(1, 20)),
                      recognizer=FakeRecognizer(s, lost, model=FAKE_MODEL_B),
                      decision=FakeDecisionEngine(), arrivals=FakeArrivalTracker(), writer=w)


def test_rerun_decisions_reproduces_and_changes_with_threshold(tmp_path: Path) -> None:
    original = run("u")
    # aller-retour fichier
    path = tmp_path / "run.jsonl"
    w = RunLogWriter(original.header, path)  # type: ignore[attr-defined]
    for rec in original.events:  # type: ignore[attr-defined]
        w._emit(dict(rec))
    w.close()
    loaded = load_runlog(path)
    assert loaded.recognition_results()[0].model == FAKE_MODEL
    same = rerun_decisions(loaded, config_id="cfg-A-rerun", decision=FakeDecisionEngine(0.5),
                           arrivals=FakeArrivalTracker(), composer=FakeWelcomeComposer())
    assert outcomes(same) == outcomes(original)
    stricter = rerun_decisions(loaded, config_id="cfg-0.6", decision=FakeDecisionEngine(0.6),
                               arrivals=FakeArrivalTracker())
    assert stricter.header.messages_from == "arrival"
    assert outcomes(stricter) == {"p1": Outcome.CORRECT, "p2": Outcome.CORRECT_REJECT,
                                  "p3": Outcome.CORRECT, "p4": Outcome.MISSED}


def _factory(t: float) -> DecisionSetup:
    return DecisionSetup(FakeDecisionEngine(t), FakeArrivalTracker(), FakeWelcomeComposer(),
                         "fake-v1")


def test_sweep_decision_on_tune_and_selection(tmp_path: Path) -> None:
    ledger = TestLedger(tmp_path / "ledger.jsonl")
    pts = sweep_decision(ANN, [run("u")], [0.5, 0.6, 0.7, 0.9], _factory, ledger=ledger)  # type: ignore[list-item]
    table = {p.threshold: (p.summary.correct.k, p.summary.wrong_identity.k,
                           p.summary.missed.k, p.summary.false_accept.k) for p in pts}
    assert table == {0.5: (2, 1, 0, 1), 0.6: (2, 0, 1, 0), 0.7: (1, 0, 2, 0),
                     0.9: (0, 0, 3, 0)}
    assert all(p.summary.correct.n == 3 and p.summary.false_accept.n == 1 for p in pts)
    assert pts[0].config_id == "cfg-A|seuil=0.5"
    assert select_threshold(pts, Metric.COSINE) == 0.6
    assert select_threshold(pts, Metric.COSINE, max_wrong_identity=1, max_false_accept=1) == 0.6
    assert select_threshold(pts[2:], Metric.COSINE) == 0.7
    assert ledger.records[-1]["record"] == "tuning"


def test_score_curve_matches_reference_rule() -> None:
    pts = score_curve(ANN, [run("u")], [0.5, 0.6, 0.85])  # type: ignore[list-item]
    rows = [(p.correct.k, p.wrong_identity.k, p.missed.k, p.false_accept.k, p.no_result)
            for p in pts]
    assert rows == [(2, 1, 0, 1, 0), (2, 0, 1, 0, 0), (0, 0, 3, 0, 0)]
    assert select_threshold(pts, Metric.COSINE) == 0.6


def test_tuning_on_test_is_refused() -> None:
    with pytest.raises(SplitViolationError):
        score_curve(ANN, [run("t")], [0.5])  # type: ignore[list-item]
    with pytest.raises(SplitViolationError):
        sweep_decision(ANN, [run("t")], [0.5], _factory)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="tune"):
        score_curve(ANN, [run("u")], [0.5], split="test")  # type: ignore[list-item]
    with pytest.raises(ValueError, match="UNE configuration"):
        score_curve(ANN, [run("u"), run("u", config="B")], [0.5])  # type: ignore[list-item]


def test_reliability_table_is_descriptive() -> None:
    tab = reliability_table(ANN, [run("u")], [0.0, 0.6, 1.0], min_passages_per_bin=2)  # type: ignore[list-item]
    low, high = tab.bins
    # [0, 0.6[ : t2 (11 images, inconnu) + t4 (11 images, prf-c ≠ prf-a) → 0/22
    assert (low.top1_correct.k, low.top1_correct.n, low.passages, low.usable) == (0, 22, 2, True)
    # [0.6, 1] : t1 (prf-a correct) + t3 (prf-b correct) → 22/22
    assert (high.top1_correct.k, high.top1_correct.n) == (22, 22)
    assert tab.status == "descriptive" and tab.calibration_ref.startswith("calib-")
    with pytest.raises(ValueError):
        reliability_table(ANN, [run("u")], [1.0, 0.0])  # type: ignore[list-item]


def test_evaluate_and_compare_on_same_test_passages(tmp_path: Path) -> None:
    ledger = TestLedger(tmp_path / "ledger.jsonl")
    logs = [run("t", 0.5, "A"), run("t", 0.6, "B")]
    ev = evaluate(ANN, logs, split="test", ledger=ledger)  # type: ignore[arg-type]
    assert ev.test_status is not None and not ev.test_status.burned
    by = {e.config_id: e for e in ev.evaluations}
    assert by["A"].overall.correct.k == 2 and by["B"].overall.false_accept.k == 0
    assert by["A"].overall.delay.n == 2 and by["A"].overall.delay.median_ms == pytest.approx(100)
    (cmp,) = ev.comparisons
    assert (cmp.config_a, cmp.config_b, cmp.n) == ("A", "B", 4)
    # A réussit p1, p3 ; B réussit p1, p2, p3 : B seul = p2 ; deux échoués = p4
    assert (cmp.both_ok, cmp.a_only, cmp.b_only, cmp.both_fail) == (2, 0, 1, 1)
    assert cmp.mcnemar_p == 1.0
    # Une nouvelle configuration sur le même test : refusée (test brûlé).
    from maisonaccueil.evaluation import BurnedTestSetError

    with pytest.raises(BurnedTestSetError):
        evaluate(ANN, [run("t", 0.55, "C")], split="test", ledger=ledger)  # type: ignore[list-item]
    # Comparaison sur des passages différents : refusée.
    from maisonaccueil.evaluation import PassageSetMismatchError

    m = match_runs(ANN, [run("t", 0.5, "A")], splits=("test",))  # type: ignore[list-item]
    m2 = match_runs(ANN, [run("u", 0.5, "B")], splits=("tune",))  # type: ignore[list-item]
    with pytest.raises(PassageSetMismatchError):
        compare_all({**m, **m2})


def _write_video(path: Path, n: int, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (32, 24))
    if not writer.isOpened():
        pytest.skip("Encodeur MJPG indisponible dans cet environnement")
    for i in range(n):
        writer.write(np.full((24, 32, 3), (i * 7) % 255, np.uint8))
    writer.release()


@pytest.fixture(autouse=True)
def isolated_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAISONACCUEIL_LOCK_DIR", str(tmp_path / "locks"))


def test_iter_video_frames_restamps_on_video_time(tmp_path: Path) -> None:
    path = tmp_path / "seance-synthetique.avi"  # fichier temporaire, jamais committé
    _write_video(path, 12)
    frames = list(iter_video_frames(path))
    assert len(frames) == 12
    assert [f.monotonic_ns for f in frames[:3]] == [0, 100_000_000, 200_000_000]
    assert [int(f.frame_id) for f in frames] == list(range(1, 13))
    # arrêt anticipé : le générateur libère la source
    assert len(list(iter_video_frames(path, max_frames=3))) == 3
    assert len(list(iter_video_frames(path, fps=5.0, max_frames=2))) == 2
    with pytest.raises(FileNotFoundError):
        list(iter_video_frames(tmp_path / "absent.avi"))


def test_replay_video_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "seance-synthetique.avi"
    _write_video(path, 100)
    s, lost = script()
    out = tmp_path / "var" / "run-u.jsonl"
    log, stats = replay_video(path, session_id="u", config_id="cfg-A", model=FAKE_MODEL,
                              recognizer=FakeRecognizer(s, lost),
                              decision=FakeDecisionEngine(0.5), arrivals=FakeArrivalTracker(),
                              composer=FakeWelcomeComposer(), out_path=out,
                              data_kind="synthetic", policy="fake-v1")  # type: ignore[arg-type]
    assert stats.frames == 100
    assert log.header.source == "replay:seance-synthetique.avi"
    assert outcomes(load_runlog(out)) == outcomes(run("u"))
