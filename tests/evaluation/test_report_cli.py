"""Rapport Markdown, CSV et ligne de commande (données synthétiques)."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from maisonaccueil.evaluation import (
    DataKind,
    RunLogWriter,
    TestLedger,
    evaluate,
    passages_csv,
    render_report,
    save_annotations,
)
from maisonaccueil.evaluation.cli import _thresholds, main
from maisonaccueil.evaluation.report import SYNTHETIC_WARNING, curve_csv
from maisonaccueil.evaluation.splits import set_id
from maisonaccueil.evaluation.sweep import score_curve

from .fakes import FAKE_MODEL, annset, known, sess, unreg
from .test_replay_sweep import ANN, run


def test_report_contains_warning_counts_and_ci(tmp_path: Path) -> None:
    ledger = TestLedger(tmp_path / "ledger.jsonl")
    ev = evaluate(ANN, [run("t", 0.5, "A"), run("t", 0.6, "B")], split="test",  # type: ignore[list-item]
                  ledger=ledger)
    curve = score_curve(ANN, [run("u")], [0.5, 0.6])  # type: ignore[list-item]
    text = render_report(ANN, ev.evaluations, split="test", split_report=ev.split_report,
                         test_status=ev.test_status, comparisons=ev.comparisons,
                         curves={"cfg-A": curve}, generated_at="2026-10-25T00:00:00Z")
    assert text.startswith("# Rapport d'évaluation MaisonAccueil\n\n" + SYNTHETIC_WARNING)
    assert "ne mesurent rien de biométrique" in text
    assert "Bonne identification (connus) | 2/3 (66.7 % ; IC95 Wilson" in text
    assert "Inconnu accepté à tort (inconnus) | 1/1 (100.0 %" in text  # config A
    assert "| `prf-a` | 2 | 1/2 | 1/2 | 0/2 | 0/0 | 0/2 |" in text
    assert "Comparaison sur les mêmes passages" in text
    assert "non brûlé" in text and "TEST BRÛLÉ" not in text
    assert "pas des probabilités" in text
    assert "%" not in text.split("## Courbe")[1].split("## Définitions")[0]  # que des k/n


def test_report_marks_burned_test_and_tune_split(tmp_path: Path) -> None:
    ledger = TestLedger(tmp_path / "ledger.jsonl")
    ledger.record_evaluation(["t"], ["A"])
    ledger.record_tuning(["u"], informed_by=[set_id(["t"])])
    ev = evaluate(ANN, [run("t", 0.5, "A")], split="test", ledger=ledger,  # type: ignore[list-item]
                  allow_burned=True)
    text = render_report(ANN, ev.evaluations, split="test", test_status=ev.test_status)
    assert "TEST BRÛLÉ" in text
    ev2 = evaluate(ANN, [run("u", 0.5, "A")], split="tune")  # type: ignore[list-item]
    text2 = render_report(ANN, ev2.evaluations, split="tune")
    assert "ne constituent pas une mesure finale" in text2


def test_real_report_has_no_synthetic_warning_and_handles_empty() -> None:
    ann = annset([sess("t", "test")], [known("p", "t", 0, 1000, "prf-a")], kind=DataKind.REAL)
    text = render_report(ann, [], split="test")
    assert SYNTHETIC_WARNING not in text and "Aucun journal" in text
    # mais un journal synthétique suffit à déclencher l'avertissement
    ev = evaluate(ann, [run_on(ann_session="t")], split="test")
    assert SYNTHETIC_WARNING in render_report(ann, ev.evaluations, split="test")
    assert "0/0 (non mesurable)" in render_report(ann, ev.evaluations, split="test")


def run_on(ann_session: str) -> object:
    return RunLogWriter(_header(ann_session)).to_runlog()


def _header(session: str) -> object:
    from .fakes import header

    return header(session, "A")


def test_csv_export() -> None:
    from maisonaccueil.evaluation import match_runs

    m = match_runs(ANN, [run("t", 0.5, "A")], splits=("test",))  # type: ignore[list-item]
    rows = list(csv.DictReader(io.StringIO(passages_csv(m))))
    assert len(rows) == 4
    assert {r["outcome"] for r in rows} == {"correct", "false_accept", "wrong_identity"}
    assert all(r["data_kinds"] == "synthetic" for r in rows)
    p1 = next(r for r in rows if r["passage_id"] == "t-p1")
    assert p1["delay_ms"] == "100.0" and p1["messages"] == "1"
    curve = score_curve(ANN, [run("u")], [0.5])  # type: ignore[list-item]
    assert curve_csv(curve, FAKE_MODEL.metric).splitlines()[1].startswith("cosine,0.5,2,3,1,0,1")


def test_thresholds_parser() -> None:
    assert _thresholds("0.3:0.5:0.1") == [0.3, 0.4, 0.5]
    assert _thresholds("0.2,0.4") == [0.2, 0.4]


def _write_log(path: Path, log: object) -> Path:
    w = RunLogWriter(log.header, path)  # type: ignore[attr-defined]
    for rec in log.events:  # type: ignore[attr-defined]
        w._emit(dict(rec))
    w.close()
    return path


def test_cli_check_report_curve(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ann_path = tmp_path / "ann.jsonl"
    save_annotations(ann_path, DataKind.SYNTHETIC, list(ANN.sessions.values()),
                     list(ANN.passages))
    assert main(["check", str(ann_path)]) == 0
    out = capsys.readouterr().out
    assert "AVERTISSEMENT" in out and "test: sessions=1, passages=4" in out

    la = _write_log(tmp_path / "a.jsonl", run("t", 0.5, "A"))
    lb = _write_log(tmp_path / "b.jsonl", run("t", 0.6, "B"))
    report, table = tmp_path / "out" / "report.md", tmp_path / "out" / "passages.csv"
    ledger = tmp_path / "out" / "ledger.jsonl"
    assert main(["report", "--annotations", str(ann_path), "--runs", str(la), str(lb),
                 "--ledger", str(ledger), "--out", str(report), "--csv", str(table)]) == 0
    assert SYNTHETIC_WARNING in report.read_text(encoding="utf-8")
    assert len(table.read_text(encoding="utf-8").splitlines()) == 9

    lu = _write_log(tmp_path / "u.jsonl", run("u"))
    curve = tmp_path / "out" / "curve.csv"
    assert main(["curve", "--annotations", str(ann_path), "--runs", str(lu),
                 "--thresholds", "0.5:0.7:0.1", "--ledger", str(ledger),
                 "--out", str(curve)]) == 0
    assert len(curve.read_text(encoding="utf-8").splitlines()) == 4


def test_cli_check_fails_on_overlap(tmp_path: Path) -> None:
    ann_path = tmp_path / "ann.jsonl"
    save_annotations(ann_path, DataKind.SYNTHETIC,
                     [sess("a", "tune", video_sha256="ff"), sess("b", "test", video_sha256="ff")],
                     [unreg("p", "a", 0, 1)])
    assert main(["check", str(ann_path)]) == 1
