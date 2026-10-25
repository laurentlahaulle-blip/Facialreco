from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from maisonaccueil.contracts import Metric, ModelRef
from maisonaccueil.storage import Database, ProfileRepository, ReferenceRepository

CONSENT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
# Octets synthétiques (pas une vraie photo) : seul le stockage est testé.
FAKE_IMAGE = b"\x89PNG\r\n\x1a\nFAKE-SYNTHETIC-BYTES"


def make_model(**kw: object) -> ModelRef:
    base: dict[str, object] = dict(
        name="fake-embedder", version="w1", embedding_dim=4, metric=Metric.COSINE,
        preprocess="p1", origin="test-double",
    )
    base.update(kw)
    return ModelRef(**base)  # type: ignore[arg-type]


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    d = Database(tmp_path / "test.sqlite", media_root=tmp_path / "media")
    yield d
    d.close()


@pytest.fixture
def profiles(db: Database) -> ProfileRepository:
    return ProfileRepository(db)


@pytest.fixture
def refs(db: Database) -> ReferenceRepository:
    return ReferenceRepository(db)
