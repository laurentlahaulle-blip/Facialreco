"""Calculs statistiques (valeurs de référence connues ; aucune donnée biométrique)."""

from __future__ import annotations

import math

import pytest

from maisonaccueil.evaluation.compare import mcnemar_exact_p
from maisonaccueil.evaluation.metrics import (
    DelayStats,
    Proportion,
    median,
    percentile_nearest_rank,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [
        (0, 10, 0.0, 0.277533),
        (5, 10, 0.236593, 0.763407),
        (10, 10, 0.722467, 1.0),
        (1, 1, 0.206543, 1.0),
        (81, 263, 0.2553, 0.3662),  # Newcombe (1998), méthode 3
    ],
)
def test_wilson_reference_values(k: int, n: int, lo: float, hi: float) -> None:
    ci = wilson_interval(k, n)
    assert ci is not None
    assert ci[0] == pytest.approx(lo, abs=1e-4)
    assert ci[1] == pytest.approx(hi, abs=1e-4)


def test_wilson_symmetry_and_bounds() -> None:
    for n in (1, 7, 50):
        for k in range(n + 1):
            lo, hi = wilson_interval(k, n)  # type: ignore[misc]
            lo2, hi2 = wilson_interval(n - k, n)  # type: ignore[misc]
            assert 0.0 <= lo <= k / n <= hi <= 1.0
            assert lo == pytest.approx(1 - hi2) and hi == pytest.approx(1 - lo2)


def test_wilson_degenerate_and_invalid() -> None:
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    with pytest.raises(ValueError):
        wilson_interval(-1, 2)


def test_proportion_format_always_shows_counts() -> None:
    assert Proportion.of(0, 0).format() == "0/0 (non mesurable)"
    assert Proportion.of(0, 0).rate is None
    text = Proportion.of(5, 10).format()
    assert text.startswith("5/10 (50.0 %") and "IC95 Wilson [23.7 ; 76.3] %" in text


def test_percentile_and_median() -> None:
    v = [300.0, 100.0, 200.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    assert percentile_nearest_rank(v, 90) == 900.0
    assert percentile_nearest_rank(v, 100) == 1000.0
    assert percentile_nearest_rank([5.0], 90) == 5.0
    assert median(v) == 550.0
    assert median([3.0, 1.0, 2.0]) == 2.0
    with pytest.raises(ValueError):
        percentile_nearest_rank([], 50)
    with pytest.raises(ValueError):
        percentile_nearest_rank([1.0], 0)


def test_delay_stats() -> None:
    assert DelayStats.of([]).format() == "n=0 (non mesurable)"
    s = DelayStats.of([1000.0, 2000.0, 3000.0])
    assert (s.n, s.min_ms, s.median_ms, s.p90_ms, s.max_ms) == (3, 1000.0, 2000.0, 3000.0,
                                                                3000.0)


def test_mcnemar_exact() -> None:
    assert mcnemar_exact_p(0, 0) is None
    assert mcnemar_exact_p(3, 3) == 1.0
    # b=0, c=5 : p = 2 * (1/32)
    assert mcnemar_exact_p(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact_p(1, 9) == pytest.approx(2 * (1 + 10) / 2**10)
    assert not math.isnan(mcnemar_exact_p(40, 60) or 0.0)
