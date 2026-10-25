from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from maisonaccueil.capture import ReconnectPolicy


@pytest.fixture(autouse=True)
def isolated_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Les verrous de test ne touchent jamais le dossier commun de la machine."""
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("MAISONACCUEIL_LOCK_DIR", str(lock_dir))
    return lock_dir


@pytest.fixture
def fast_policy() -> ReconnectPolicy:
    return ReconnectPolicy(
        initial_delay_s=0.001, max_delay_s=0.004, factor=2.0, max_attempts=4,
        max_consecutive_read_failures=2,
    )


@pytest.fixture(autouse=True)
def no_orphan_capture_threads() -> Iterator[None]:
    yield
    leftovers = [t for t in threading.enumerate() if t.name.startswith("capture-")]
    for t in leftovers:
        t.join(2.0)
    alive = [t.name for t in leftovers if t.is_alive()]
    assert not alive, f"threads de capture orphelins : {alive}"
