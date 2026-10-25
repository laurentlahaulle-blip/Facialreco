"""Attribution des événements aux passages et issues (journaux écrits à la main)."""

from __future__ import annotations

from typing import Any

import pytest

from maisonaccueil.evaluation import Outcome, match_runs, match_session, summarize
from maisonaccueil.evaluation.metrics import evaluate_match, per_person

from .fakes import LogBuilder, annset, known, sess, unreg


def _one(passages: list[Any], log: Any, **kw: Any) -> dict[str, Any]:
    res, unattributed = match_session(passages, log, **kw)
    out: dict[str, Any] = {r.passage.passage_id: r for r in res}
    out["_unattributed"] = unattributed
    return out


def test_basic_outcomes_and_delay() -> None:
    ps = [
        known("ok", "s", 1000, 5000, "prf-a"),
        known("wrong", "s", 10_000, 14_000, "prf-a"),
        known("missed", "s", 20_000, 24_000, "prf-b"),
        unreg("fa", "s", 30_000, 34_000),
        unreg("rej", "s", 40_000, 44_000),
    ]
    b = LogBuilder("s")
    b.result(1500, "t1", "prf-a", 0.9).greet(2500, "prf-a", "t1")
    b.greet(11_000, "prf-c", "t2")
    b.arrival(21_000, None, "t3")  # inconnu annoncé pour un connu : non reconnu
    b.greet(31_000, "prf-a", "t4")
    b.arrival(41_000, None, "t5")
    r = _one(ps, b.build())
    assert r["ok"].outcome == Outcome.CORRECT and r["ok"].delay_ms == 1500.0
    assert r["ok"].results == 1
    assert r["wrong"].outcome == Outcome.WRONG_IDENTITY and r["wrong"].delay_ms is None
    assert r["missed"].outcome == Outcome.MISSED and r["missed"].unknown_arrivals == 1
    assert r["fa"].outcome == Outcome.FALSE_ACCEPT
    assert r["rej"].outcome == Outcome.CORRECT_REJECT
    assert r["_unattributed"] == []


def test_wrong_identity_has_priority_over_correct() -> None:
    ps = [known("p", "s", 0, 5000, "prf-a")]
    b = LogBuilder("s").greet(1000, "prf-a", "t1").greet(2000, "prf-b", "t2")
    r = _one(ps, b.build())
    assert r["p"].outcome == Outcome.WRONG_IDENTITY
    assert r["p"].known_arrivals == ("prf-a", "prf-b")


def test_repeated_messages_and_welcome_not_expected() -> None:
    ps = [known("p1", "s", 0, 5000, "prf-a"),
          known("p2", "s", 10_000, 15_000, "prf-a", welcome_expected=False),
          known("p3", "s", 20_000, 25_000, "prf-a", welcome_expected=False)]
    b = LogBuilder("s").greet(1000, "prf-a", "t1").greet(3000, "prf-a", "t1b")
    b.greet(11_000, "prf-a", "t2")
    r = _one(ps, b.build())
    assert r["p1"].messages == 2 and r["p1"].repeated_messages == 1 and r["p1"].message_repeated
    assert r["p1"].delay_ms == 1000.0  # premier message
    assert r["p2"].outcome == Outcome.CORRECT and r["p2"].repeated_messages == 1
    assert r["p2"].delay_ms is None
    assert r["p3"].outcome == Outcome.CORRECT and r["p3"].messages == 0
    s = summarize([r["p1"], r["p2"], r["p3"]])
    assert (s.repeated.k, s.repeated.n, s.repeated_messages_total) == (2, 3, 2)


def test_grace_window_and_unattributed() -> None:
    ps = [known("p", "s", 0, 1000, "prf-a")]
    b = LogBuilder("s").greet(1800, "prf-a", "t1").greet(5000, "prf-b", "t9")
    r = _one(ps, b.build())
    assert r["p"].outcome == Outcome.CORRECT
    un = r["_unattributed"]
    assert {(u.record, u.kind, u.profile_id) for u in un} == {("arrival", "known", "prf-b"),
                                                             ("welcome", "", "prf-b")}
    r2 = _one(ps, LogBuilder("s").greet(1800, "prf-a").build(), grace_ms=500)
    assert r2["p"].outcome == Outcome.MISSED


def test_overlapping_passages_prefer_matching_profile_then_latest() -> None:
    ps = [known("a", "s", 0, 10_000, "prf-a"), known("b", "s", 2000, 10_000, "prf-b"),
          unreg("u", "s", 3000, 10_000)]
    b = LogBuilder("s").greet(4000, "prf-a", "t1").greet(5000, "prf-b", "t2")
    b.arrival(6000, None, "t3")
    r = _one(ps, b.build())
    assert r["a"].outcome == Outcome.CORRECT and r["a"].known_arrivals == ("prf-a",)
    assert r["b"].outcome == Outcome.CORRECT
    assert r["u"].outcome == Outcome.CORRECT_REJECT and r["u"].unknown_arrivals == 1
    # Un nom inconnu en chevauchement va au passage commencé le plus récemment (u).
    r2 = _one(ps, LogBuilder("s").greet(4000, "prf-z", "t1").build())
    assert r2["u"].outcome == Outcome.FALSE_ACCEPT
    assert r2["a"].outcome == Outcome.MISSED


def test_welcome_follows_its_arrival() -> None:
    ps = [known("p", "s", 0, 1000, "prf-a")]
    b = LogBuilder("s")
    eid = b.arrival(900, "prf-a")
    b.welcome(4000, eid, "prf-a")  # accueil tardif (file audio) mais rattaché à l'arrivée
    r = _one(ps, b.build())
    assert r["p"].messages == 1 and r["p"].delay_ms == 4000.0
    assert r["_unattributed"] == []


def test_messages_from_arrivals_without_composer() -> None:
    ps = [known("p", "s", 0, 5000, "prf-a")]
    b = LogBuilder("s", messages_from="arrival")
    b.arrival(700, "prf-a")
    b.arrival(900, "prf-a")
    r = _one(ps, b.build())
    assert r["p"].messages == 2 and r["p"].repeated_messages == 1 and r["p"].delay_ms == 700.0


def test_match_runs_groups_by_config_and_checks_sessions() -> None:
    ann = annset([sess("s1", "test"), sess("s2", "test"), sess("u", "tune")],
                 [known("p1", "s1", 0, 1000, "prf-a"), unreg("p2", "s2", 0, 1000, "x"),
                  known("pu", "u", 0, 1000, "prf-a")])
    logs = [LogBuilder("s1", "A").greet(500, "prf-a").build(),
            LogBuilder("s2", "A").build(),
            LogBuilder("s1", "B").build(),
            LogBuilder("s2", "B").greet(100, "prf-a").build()]
    m = match_runs(ann, logs)  # type: ignore[arg-type]
    assert sorted(m) == ["A", "B"]
    ea, eb = evaluate_match(m["A"]), evaluate_match(m["B"])
    assert (ea.overall.correct.k, ea.overall.false_accept.k) == (1, 0)
    assert (eb.overall.missed.k, eb.overall.false_accept.k) == (1, 1)
    assert set(per_person(m["A"].passages)) == {"prf-a", "inconnu:x"}
    assert m["A"].data_kinds == frozenset({"synthetic"})

    with pytest.raises(ValueError, match="sans journal"):
        match_runs(ann, logs[:1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="plusieurs journaux"):
        match_runs(ann, [logs[0], logs[0], logs[1]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hors de"):
        match_runs(ann, [LogBuilder("u", "A").build()])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non annotée"):
        match_runs(ann, [LogBuilder("zz", "A").build()])  # type: ignore[arg-type]
