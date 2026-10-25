"""Formats d'annotation, découpage et registre du test (données synthétiques)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from maisonaccueil.contracts import schema_sql
from maisonaccueil.evaluation import (
    BurnedTestSetError,
    DataKind,
    FormatError,
    SplitViolationError,
    TestLedger,
    check_splits,
    load_annotations,
    save_annotations,
    set_id,
)
from maisonaccueil.evaluation.splits import require_split, sessions_from_sqlite

from .fakes import annset, known, sess, unreg

D1 = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)
D2 = datetime(2026, 11, 5, 17, 0, tzinfo=UTC)


def _write(path: Path, lines: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


HEADER = {"record": "header", "format": "maisonaccueil.passages", "version": 1,
          "data_kind": "synthetic"}


def test_roundtrip(tmp_path: Path) -> None:
    sessions = [sess("ses-t", "test", recorded_at=D2, duration_ms=60_000, video_sha256="ab"),
                sess("ses-u", "tune", recorded_at=D1)]
    passages = [known("p1", "ses-t", 1000, 5000, "prf-a", welcome_expected=False),
                unreg("p2", "ses-t", 6000, 9000, "inconnu-A"),
                known("p3", "ses-u", 0, 1000, "prf-b")]
    path = tmp_path / "a.jsonl"
    save_annotations(path, DataKind.SYNTHETIC, sessions, passages, annotator="test")
    ann = load_annotations(path)
    assert ann.is_synthetic
    assert ann.sessions["ses-t"].recorded_at == D2  # type: ignore[index]
    assert {p.passage_id for p in ann.passages_in(["test"])} == {"p1", "p2"}
    p1 = next(p for p in ann.passages if p.passage_id == "p1")
    assert p1.welcome_expected is False and p1.profile_id == "prf-a"
    assert next(p for p in ann.passages if p.passage_id == "p2").person_key == "inconnu:inconnu-A"
    assert ann.sessions_in("tune") == ("ses-u",)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([{"record": "session", "session_id": "s", "split": "test"}], "en-tête"),
        ([HEADER, {"record": "session", "session_id": "s", "split": "bogus"}], "split"),
        ([HEADER, {"record": "session", "session_id": "s", "split": "test"},
          {"record": "session", "session_id": "s", "split": "tune"}], "deux fois"),
        ([HEADER, {"record": "passage", "passage_id": "p", "session_id": "nope",
                   "start_ms": 0, "end_ms": 1, "expected": "unregistered"}], "non déclarée"),
        ([HEADER, {"record": "session", "session_id": "s", "split": "test"},
          {"record": "passage", "passage_id": "p", "session_id": "s", "start_ms": 0,
           "end_ms": 1, "expected": "known"}], "profile_id"),
        ([HEADER, {"record": "session", "session_id": "s", "split": "test"},
          {"record": "passage", "passage_id": "p", "session_id": "s", "start_ms": 0,
           "end_ms": 1, "expected": "unregistered", "profile_id": "prf-a"}], "profile_id"),
        ([HEADER, {"record": "session", "session_id": "s", "split": "test"},
          {"record": "passage", "passage_id": "p", "session_id": "s", "start_ms": 5,
           "end_ms": 1, "expected": "unregistered"}], "end_ms"),
        ([HEADER, {"record": "session", "session_id": "s", "split": "test", "duration_ms": 10},
          {"record": "passage", "passage_id": "p", "session_id": "s", "start_ms": 0,
           "end_ms": 50, "expected": "unregistered"}], "durée"),
        ([HEADER, {"record": "session", "session_id": "s", "split": "test",
                   "recorded_at": "2026-11-02T17:00:00"}], "fuseau"),
        ([HEADER, {"record": "what"}], "inconnu"),
    ],
)
def test_format_errors(tmp_path: Path, lines: list[dict[str, object]], message: str) -> None:
    with pytest.raises(FormatError, match=message):
        load_annotations(_write(tmp_path / "a.jsonl", lines))


def test_mixing_real_and_synthetic_refused(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.jsonl", [HEADER])
    b = _write(tmp_path / "b.jsonl", [{**HEADER, "data_kind": "real"}])
    with pytest.raises(FormatError, match="mélange"):
        load_annotations(a, b)


def test_invalid_json_reports_line(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps(HEADER) + "\n{oops\n", encoding="utf-8")
    with pytest.raises(FormatError, match=":2:"):
        load_annotations(p)


def test_split_checks() -> None:
    ann = annset(
        [sess("ses-e", "enroll", recorded_at=D1),
         sess("ses-t", "test", recorded_at=D1, video_sha256="cafe"),
         sess("ses-u", "tune", recorded_at=D2, video_sha256="cafe")],
        [known("e1", "ses-e", 0, 10, "prf-a"), known("t1", "ses-t", 0, 10, "prf-a"),
         known("u1", "ses-u", 0, 10, "prf-a"), unreg("u2", "ses-u", 20, 30)],
    )
    rep = check_splits(ann)
    assert not rep.ok
    assert any("même vidéo" in e for e in rep.errors)
    assert any("'test' ne contient aucun passage d'inconnu" in w for w in rep.warnings)
    assert any("t1" in w and "même jour" in w for w in rep.warnings)
    assert not any("u1" in w for w in rep.warnings)
    assert rep.counts["tune"] == {"sessions": 1, "passages": 2, "known": 1, "unregistered": 1,
                                  "persons": 2}


def test_extra_session_with_conflicting_split() -> None:
    ann = annset([sess("ses-t", "test")], [unreg("t1", "ses-t", 0, 10)])
    rep = check_splits(ann, [sess("ses-t", "train")])
    assert any("à la fois" in e for e in rep.errors)


def test_require_split() -> None:
    ann = annset([sess("a", "tune"), sess("b", "test")], [])
    require_split(ann, ["a"], "tune")
    with pytest.raises(SplitViolationError, match="b"):
        require_split(ann, ["a", "b"], "tune")
    with pytest.raises(SplitViolationError):
        require_split(ann, ["zzz"], "tune")


def test_sessions_from_sqlite(tmp_path: Path) -> None:
    db = tmp_path / "x.sqlite"
    con = sqlite3.connect(db)
    con.executescript(schema_sql())
    con.execute("INSERT INTO collection_sessions(session_id, profile_id, started_at, split) "
                "VALUES ('ses-1', NULL, '2026-11-02T17:00:00+00:00', 'train'), "
                "('ses-2', NULL, '2026-11-02T18:00:00+00:00', NULL)")
    con.commit()
    con.close()
    rows = sessions_from_sqlite(db)
    assert [(r.session_id, r.split) for r in rows] == [("ses-1", "train")]
    assert rows[0].recorded_at == D1


def test_set_id_is_order_independent() -> None:
    assert set_id(["b", "a"]) == set_id(["a", "b", "a"]) != set_id(["a"])


class TestLedgerRules:
    def test_fresh_then_reproduction_is_fine(self, tmp_path: Path) -> None:
        led = TestLedger(tmp_path / "ledger.jsonl")
        st = led.record_evaluation(["s1", "s2"], ["A", "B"])
        assert not st.burned and st.evaluations == 0
        st = led.record_evaluation(["s2", "s1"], ["A"])  # sous-ensemble : reproduction
        assert not st.burned and st.evaluations == 1
        again = TestLedger(tmp_path / "ledger.jsonl")  # persistance
        assert len(again.records) == 2
        assert not again.status(set_id(["s1", "s2"]), ["B"]).burned

    def test_new_config_after_test_burns(self, tmp_path: Path) -> None:
        led = TestLedger(tmp_path / "ledger.jsonl")
        led.record_evaluation(["s1"], ["A"])
        with pytest.raises(BurnedTestSetError, match="nouveau test"):
            led.record_evaluation(["s1"], ["A2"])
        st = led.record_evaluation(["s1"], ["A2"], allow_burned=True)
        assert st.burned
        assert led.status(set_id(["s1"])).burned  # reste brûlé ensuite

    def test_informed_tuning_burns(self, tmp_path: Path) -> None:
        led = TestLedger(tmp_path / "ledger.jsonl")
        led.record_evaluation(["s1"], ["A"])
        led.record_tuning(["u1"], informed_by=[set_id(["s1"])], config_before="A",
                          config_after="A2")
        st = led.status(set_id(["s1"]), ["A"])
        assert st.burned and any("réglage" in r for r in st.reasons)
        with pytest.raises(BurnedTestSetError):
            led.record_evaluation(["s1"], ["A"])
        assert not led.status(set_id(["s9"]), ["A2"]).burned  # nouveau test : frais

    def test_tuning_directly_on_set_burns(self, tmp_path: Path) -> None:
        led = TestLedger(tmp_path / "ledger.jsonl")
        led.record_tuning(["s1"], split="test")
        assert led.status(set_id(["s1"])).burned

    def test_tune_evaluations_never_raise(self, tmp_path: Path) -> None:
        led = TestLedger(tmp_path / "ledger.jsonl")
        led.record_evaluation(["u1"], ["A"], split="tune")
        led.record_evaluation(["u1"], ["B"], split="tune")
        led.record_tuning(["u1"])
        assert len(led.records) == 3

    def test_corrupted_ledger(self, tmp_path: Path) -> None:
        p = tmp_path / "ledger.jsonl"
        p.write_text('{"record": "nope"}\n', encoding="utf-8")
        with pytest.raises(FormatError):
            TestLedger(p)
