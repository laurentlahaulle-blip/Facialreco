import pytest

from maisonaccueil.contracts import DecisionStatus, QualityFlag, RawScore, TrackId
from maisonaccueil.contracts.interfaces import DecisionEngine
from maisonaccueil.decision import DecisionPolicy, TemporalDecisionEngine

from ._helpers import MODEL, MODEL_L2, A, B, C, result


def feed(engine, results):
    return [d for d in (engine.observe(r) for r in results) if d is not None]


def test_implements_protocol():
    assert isinstance(TemporalDecisionEngine(), DecisionEngine)


def test_confirmed_after_n_agreeing_and_not_reemitted():
    eng = TemporalDecisionEngine(DecisionPolicy(min_agreeing=3))
    out = feed(eng, [result("t1", i * 100, (A, 0.7), (B, 0.3)) for i in range(10)])
    assert len(out) == 1
    d = out[0]
    assert d.status == DecisionStatus.CONFIRMED and d.profile_id == A
    assert d.observations == 3 and d.window_ms == 200
    assert d.policy == "temporal-v1" and d.model == MODEL
    assert d.best_score == RawScore(0.7, MODEL.metric)
    assert d.monotonic_ns == 200 * 1_000_000


def test_below_threshold_gives_unknown_once():
    eng = TemporalDecisionEngine(DecisionPolicy(min_observations_unknown=5))
    out = feed(eng, [result("t1", i * 100, (A, 0.2), (B, 0.1)) for i in range(12)])
    assert [d.status for d in out] == [DecisionStatus.UNKNOWN]
    assert out[0].profile_id is None and out[0].observations == 5


def test_empty_gallery_counts_as_unknown_evidence():
    eng = TemporalDecisionEngine(DecisionPolicy(min_observations_unknown=3))
    out = feed(eng, [result("t1", i * 100) for i in range(3)])
    assert out[0].status == DecisionStatus.UNKNOWN and out[0].best_score is None


def test_insufficient_margin_is_not_accepted():
    eng = TemporalDecisionEngine(DecisionPolicy(min_margin=0.05, min_observations_unknown=4))
    out = feed(eng, [result("t1", i * 100, (A, 0.70), (B, 0.68)) for i in range(4)])
    assert [d.status for d in out] == [DecisionStatus.UNKNOWN]


def test_l2_metric_threshold_and_margin():
    pol = DecisionPolicy(accept_l2=1.0, min_margin=0.1, min_agreeing=2)
    eng = TemporalDecisionEngine(pol)
    out = feed(eng, [result("t1", i * 50, (A, 0.6), (B, 1.2), model=MODEL_L2)
                     for i in range(3)])
    assert out[0].status == DecisionStatus.CONFIRMED and out[0].profile_id == A
    assert pol.is_accepted(RawScore(0.9, MODEL_L2.metric))
    assert not pol.has_margin(RawScore(0.9, MODEL_L2.metric), RawScore(0.95, MODEL_L2.metric))


def test_blocking_quality_flags_are_ignored():
    eng = TemporalDecisionEngine(DecisionPolicy(min_agreeing=3))
    blurry = frozenset({QualityFlag.BLURRY})
    out = feed(eng, [result("t1", i * 100, (A, 0.9), flags=blurry) for i in range(10)])
    assert out == []
    # NOT_FRONTAL n'est pas bloquant par défaut
    side = frozenset({QualityFlag.NOT_FRONTAL})
    out = feed(eng, [result("t1", 2000 + i * 100, (A, 0.9), flags=side) for i in range(3)])
    assert out[0].status == DecisionStatus.CONFIRMED


def test_window_slides_old_observations_out():
    eng = TemporalDecisionEngine(DecisionPolicy(window_ms=500, min_agreeing=3))
    out = feed(eng, [result("t1", t, (A, 0.9)) for t in (0, 600, 1200, 1800)])
    assert out == []  # jamais 3 observations dans 500 ms
    out = feed(eng, [result("t1", t, (A, 0.9)) for t in (1900, 2000)])
    assert out[0].status == DecisionStatus.CONFIRMED and out[0].observations == 3


def test_oscillation_between_two_identities_is_uncertain():
    eng = TemporalDecisionEngine(DecisionPolicy(min_agreeing=3, max_conflicting=1))
    seq = [result("t1", i * 100, (A, 0.8) if i % 2 == 0 else (B, 0.8)) for i in range(12)]
    out = feed(eng, seq)
    assert [(d.status, d.reason) for d in out] == [(DecisionStatus.UNCERTAIN, "conflict")]
    assert out[0].profile_id is None


def test_single_stray_observation_is_tolerated():
    eng = TemporalDecisionEngine(DecisionPolicy(min_agreeing=3, max_conflicting=1))
    seq = [result("t1", 0, (A, 0.8)), result("t1", 100, (C, 0.8)),
           result("t1", 200, (A, 0.8)), result("t1", 300, (A, 0.8))]
    out = feed(eng, seq)
    assert [(d.status, d.profile_id) for d in out] == [(DecisionStatus.CONFIRMED, A)]


def test_confirmed_is_sticky_but_identity_switch_is_reported():
    eng = TemporalDecisionEngine(DecisionPolicy(window_ms=1000, min_agreeing=3))
    out = feed(eng, [result("t1", i * 100, (A, 0.8)) for i in range(3)])
    assert out[0].profile_id == A
    # Bruit / visage tourné : aucune rétrogradation émise
    assert feed(eng, [result("t1", 400 + i * 100, (A, 0.1)) for i in range(20)]) == []
    # Échange de suivi : B confirmé sur le même TrackId
    out = feed(eng, [result("t1", 3000 + i * 100, (B, 0.8)) for i in range(5)])
    assert [(d.status, d.profile_id) for d in out] == [(DecisionStatus.CONFIRMED, B)]


def test_unknown_then_confirmed_on_same_track():
    eng = TemporalDecisionEngine(DecisionPolicy(min_observations_unknown=3, window_ms=500))
    out = feed(eng, [result("t1", i * 100, (A, 0.1)) for i in range(3)])
    out += feed(eng, [result("t1", 1000 + i * 100, (A, 0.9)) for i in range(3)])
    assert [d.status for d in out] == [DecisionStatus.UNKNOWN, DecisionStatus.CONFIRMED]


def test_track_lost_resets_state_and_tracks_are_independent():
    eng = TemporalDecisionEngine(DecisionPolicy(min_agreeing=3))
    feed(eng, [result("t1", i * 100, (A, 0.9)) for i in range(2)])
    feed(eng, [result("t2", i * 100, (B, 0.9)) for i in range(2)])
    eng.track_lost(TrackId("t1"), 250_000_000)
    assert eng.active_tracks == {TrackId("t2")}
    assert feed(eng, [result("t1", 300, (A, 0.9))]) == []  # recompte depuis zéro
    out = feed(eng, [result("t2", 300, (B, 0.9))])
    assert out[0].track_id == TrackId("t2") and out[0].profile_id == B


def test_stale_tracks_are_forgotten():
    eng = TemporalDecisionEngine(DecisionPolicy(stale_track_ms=1000))
    eng.observe(result("t1", 0, (A, 0.9)))
    eng.observe(result("t2", 5000, (A, 0.9)))
    assert eng.active_tracks == {TrackId("t2")}


def test_model_change_resets_track():
    eng = TemporalDecisionEngine(DecisionPolicy(min_agreeing=3))
    feed(eng, [result("t1", i * 100, (A, 0.9)) for i in range(2)])
    assert feed(eng, [result("t1", 200, (A, 0.5), model=MODEL_L2)]) == []


@pytest.mark.parametrize("kw", [{"window_ms": 0}, {"min_agreeing": 0}, {"max_conflicting": -1}])
def test_policy_validation(kw):
    with pytest.raises(ValueError):
        DecisionPolicy(**kw)
