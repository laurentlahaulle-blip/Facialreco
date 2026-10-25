"""File bornée qui abandonne les éléments les plus anciens quand elle est pleine.

Objectif : la vision traite toujours une image récente ; la latence ne s'accumule pas
si le traitement est plus lent que la caméra. Thread-safe.
"""

from __future__ import annotations

import collections
import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestQueue(Generic[T]):
    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError("maxsize doit être >= 1")
        self._items: collections.deque[T] = collections.deque()
        self._maxsize = maxsize
        self._cond = threading.Condition()
        self._dropped = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped

    def put(self, item: T) -> None:
        """Ne bloque jamais. Abandonne le plus ancien élément si la file est pleine."""
        with self._cond:
            if self._closed:
                return
            while len(self._items) >= self._maxsize:
                self._items.popleft()
                self._dropped += 1
            self._items.append(item)
            self._cond.notify()

    def get(self, timeout: float | None = None) -> T | None:
        """Retourne l'élément le plus ancien encore présent, ou None (délai/fermeture)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._items or self._closed, timeout):
                return None
            return self._items.popleft() if self._items else None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)
