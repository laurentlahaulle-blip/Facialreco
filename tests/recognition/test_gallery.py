"""Galerie sur vecteurs ALÉATOIRES : tests de fonctionnement, aucune mesure de précision."""

import numpy as np
import pytest

from maisonaccueil.contracts import (
    Embedding,
    IncompatibleEmbeddingError,
    Metric,
    ModelRef,
    ProfileId,
)
from maisonaccueil.contracts.interfaces import Gallery
from maisonaccueil.recognition import (
    FakeEmbeddingStore,
    FakeLeakyEmbeddingStore,
    FakeRandomEmbedder,
    InMemoryGallery,
)

M = ModelRef("fake-rand", "v1", 32, Metric.COSINE, "none-v1", "test-double")
M_OTHER = ModelRef("fake-rand", "v2", 32, Metric.COSINE, "none-v1", "test-double")
M_L2 = ModelRef("fake-rand", "v1", 32, Metric.L2, "none-v1", "test-double")


def _emb(model: ModelRef, v: np.ndarray) -> Embedding:
    return Embedding(model, v.astype(np.float32))


@pytest.fixture
def populated() -> tuple[InMemoryGallery, dict[ProfileId, list[Embedding]]]:
    rnd = FakeRandomEmbedder(M, seed=1)
    g = InMemoryGallery(M)
    refs: dict[ProfileId, list[Embedding]] = {}
    for i in range(5):
        pid = ProfileId(f"prf-{i}")
        refs[pid] = [rnd.random_embedding() for _ in range(i + 1)]
        for e in refs[pid]:
            g.add(pid, e)
    return g, refs


def test_protocol(populated):
    g, _ = populated
    assert isinstance(g, Gallery)
    assert len(g) == 15 and len(g.profile_ids) == 5


def test_top_k_per_profile_and_reference_counts(populated):
    g, refs = populated
    target = ProfileId("prf-3")
    q = refs[target][2]
    noisy = _emb(M, q.vector + 0.01 * np.random.default_rng(2).standard_normal(32))
    res = g.search(noisy, top_k=3)
    assert res is not None and res.model == M
    assert len(res.candidates) == 3
    assert res.best is not None and res.best.profile_id == target
    assert res.best.references_compared == 4
    assert len({c.profile_id for c in res.candidates}) == 3  # un candidat par profil
    values = [c.score.value for c in res.candidates]
    assert values == sorted(values, reverse=True)
    assert res.best.score.metric == Metric.COSINE
    assert len(g.rank(noisy, top_k=10)) == 5


def test_incompatible_query_rejected(populated):
    g, _ = populated
    with pytest.raises(IncompatibleEmbeddingError):
        g.search(_emb(M_OTHER, np.ones(32)))
    with pytest.raises(IncompatibleEmbeddingError):
        g.add(ProfileId("prf-x"), _emb(M_OTHER, np.ones(32)))


def test_empty_gallery_returns_no_candidates():
    res = InMemoryGallery(M).search(_emb(M, np.ones(32)))
    assert res is not None and res.candidates == () and res.best is None


def test_loads_only_matching_model_from_store():
    store = FakeEmbeddingStore()
    store.add_embedding(ProfileId("prf-a"), 1, _emb(M, np.ones(32)))
    store.add_embedding(ProfileId("prf-b"), 2, _emb(M_OTHER, np.ones(32)))
    g = InMemoryGallery(M, store)
    assert g.profile_ids == {ProfileId("prf-a")}
    g.add(ProfileId("prf-c"), _emb(M, -np.ones(32)), persist=True, reference_id=3)
    assert InMemoryGallery(M, store).profile_ids == {ProfileId("prf-a"), ProfileId("prf-c")}
    store.add_embedding(ProfileId("prf-d"), 4, _emb(M, np.arange(32.0)))
    assert g.reload() == 3


def test_leaky_store_is_detected():
    store = FakeLeakyEmbeddingStore()
    store.add_embedding(ProfileId("prf-b"), None, _emb(M_OTHER, np.ones(32)))
    with pytest.raises(IncompatibleEmbeddingError):
        InMemoryGallery(M, store)


def test_persist_without_store_fails():
    with pytest.raises(RuntimeError):
        InMemoryGallery(M).add(ProfileId("prf-a"), _emb(M, np.ones(32)), persist=True)


def test_l2_metric_lower_is_better():
    g = InMemoryGallery(M_L2)
    g.add(ProfileId("prf-near"), _emb(M_L2, np.zeros(32)))
    g.add(ProfileId("prf-far"), _emb(M_L2, np.full(32, 5.0)))
    cands = g.rank(_emb(M_L2, np.full(32, 0.1)), top_k=2)
    assert [c.profile_id for c in cands] == ["prf-near", "prf-far"]
    assert cands[0].score.is_better_than(cands[1].score)


def test_remove_profile(populated):
    g, _ = populated
    assert g.remove_profile(ProfileId("prf-4")) == 5
    assert ProfileId("prf-4") not in g.profile_ids
    with pytest.raises(ValueError):
        g.rank(_emb(M, np.ones(32)), top_k=0)
