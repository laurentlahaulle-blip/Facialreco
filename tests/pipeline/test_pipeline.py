import os
import threading

import pytest

from maisonaccueil.pipeline import LatestQueue, ResourceBusyError, ResourceLock


def test_latest_queue_drops_oldest():
    q: LatestQueue[int] = LatestQueue(maxsize=2)
    for i in range(5):
        q.put(i)
    assert q.dropped == 3
    assert [q.get(0), q.get(0)] == [3, 4]
    assert q.get(timeout=0.01) is None


def test_latest_queue_close_unblocks_consumer():
    q: LatestQueue[int] = LatestQueue()
    out: list[object] = []
    t = threading.Thread(target=lambda: out.append(q.get(timeout=5)))
    t.start()
    q.close()
    t.join(2)
    assert out == [None]


def test_resource_lock_exclusive_and_stale(tmp_path):
    a = ResourceLock("camera-0", "test-a", lock_dir=tmp_path)
    b = ResourceLock("camera-0", "test-b", lock_dir=tmp_path)
    with a:
        with pytest.raises(ResourceBusyError):
            b.acquire()
    b.acquire()
    b.release()
    # verrou laissé par un processus disparu : repris
    (tmp_path / "gpu-0.lock").write_text('{"pid": 999999999, "owner": "ghost"}')
    with ResourceLock("gpu-0", "test", lock_dir=tmp_path) as lk:
        assert lk.holder()["pid"] == os.getpid()
