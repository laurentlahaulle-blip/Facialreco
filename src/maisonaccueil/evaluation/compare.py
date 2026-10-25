"""Comparaison de moteurs/configurations sur les MÊMES passages.

Pour deux configurations A et B, chaque passage est « réussi » si son issue est
``correct`` (known) ou ``correct_reject`` (unregistered). On compte les paires
discordantes : ``a_only`` (A réussit, B échoue) et ``b_only``. Le test de McNemar exact
(bilatéral) donne p = min(1, 2·Σ_{i ≤ min(b, c)} C(b+c, i) / 2^(b+c)). Il est fourni à titre
indicatif ; il suppose des passages indépendants (même réserve que Wilson).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass

from .matching import MatchResult


class PassageSetMismatchError(ValueError):
    """Les configurations comparées ne portent pas sur les mêmes passages."""


@dataclass(frozen=True, slots=True)
class PairComparison:
    config_a: str
    config_b: str
    n: int
    both_ok: int
    a_only: int
    b_only: int
    both_fail: int
    mcnemar_p: float | None  # None si aucune paire discordante


def mcnemar_exact_p(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def ensure_same_passages(matches: Mapping[str, MatchResult]) -> tuple[str, ...]:
    ids: tuple[str, ...] | None = None
    ref = ""
    for cfg, m in matches.items():
        cur = tuple(sorted(r.passage.passage_id for r in m.passages))
        if ids is None:
            ids, ref = cur, cfg
        elif cur != ids:
            diff = sorted(set(cur) ^ set(ids))
            raise PassageSetMismatchError(
                f"{cfg} et {ref} ne portent pas sur les mêmes passages "
                f"({len(diff)} différences, ex. {diff[:5]})"
            )
    return ids or ()


def compare_pair(a: MatchResult, b: MatchResult) -> PairComparison:
    ra, rb = a.by_id(), b.by_id()
    if set(ra) != set(rb):
        raise PassageSetMismatchError(f"{a.config_id} / {b.config_id} : passages différents")
    both_ok = a_only = b_only = both_fail = 0
    for pid, x in ra.items():
        ok_a, ok_b = x.outcome.ok, rb[pid].outcome.ok
        if ok_a and ok_b:
            both_ok += 1
        elif ok_a:
            a_only += 1
        elif ok_b:
            b_only += 1
        else:
            both_fail += 1
    return PairComparison(a.config_id, b.config_id, len(ra), both_ok, a_only, b_only, both_fail,
                          mcnemar_exact_p(a_only, b_only))


def compare_all(matches: Mapping[str, MatchResult]) -> list[PairComparison]:
    """Toutes les paires (ordre alphabétique des ``config_id``) ; vérifie les passages."""
    ensure_same_passages(matches)
    keys = sorted(matches)
    return [compare_pair(matches[a], matches[b]) for a, b in itertools.combinations(keys, 2)]
