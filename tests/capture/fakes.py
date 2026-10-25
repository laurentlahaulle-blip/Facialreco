"""Doubles de test pour la capture. Aucune image réelle : motifs synthétiques."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from maisonaccueil.contracts import CaptureState, CaptureStatus, Frame


class FakeVideoCapture:
    """Faux ``cv2.VideoCapture`` scénarisé.

    ``opened`` : résultat de ``isOpened``.
    ``script`` : séquence de lectures ; ``True`` = image, ``False`` = lecture ratée.
    Au-delà du script : ``tail`` (True = images infinies, False = échecs infinis).
    """

    def __init__(
        self,
        opened: bool = True,
        script: Sequence[bool] = (),
        tail: bool = True,
        shape: tuple[int, ...] = (24, 32, 3),
        fps: float = 0.0,
    ) -> None:
        self.opened = opened
        self.script = list(script)
        self.tail = tail
        self.shape = shape
        self.fps = fps
        self.released = False
        self.reads = 0
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened and not self.released

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        ok = self.script.pop(0) if self.script else self.tail
        if not ok or self.released:
            return False, None
        return True, np.full(self.shape, self.reads % 256, np.uint8)

    def release(self) -> None:
        self.released = True

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = value
        return True

    def get(self, prop_id: int) -> float:
        import cv2

        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        return self.props.get(prop_id, 0.0)


class FakeCaptureFactory:
    """Fabrique scénarisée : chaque appel renvoie la capture suivante de la liste
    (la dernière est réutilisée sous forme de copie quand la liste est épuisée)."""

    def __init__(self, captures: Sequence[FakeVideoCapture]) -> None:
        self.captures = list(captures)
        self.created: list[FakeVideoCapture] = []
        self.calls: list[tuple[Any, int, list[int]]] = []

    def __call__(self, target: Any, api: int, params: Sequence[int]) -> FakeVideoCapture:
        self.calls.append((target, api, list(params)))
        if len(self.created) < len(self.captures):
            cap = self.captures[len(self.created)]
        else:
            last = self.captures[-1]
            cap = FakeVideoCapture(last.opened, list(last.script), last.tail, last.shape)
        self.created.append(cap)
        return cap


class Recorder:
    """Collecte thread-safe des images et états reçus."""

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.states: list[CaptureState] = []
        self._cond = threading.Condition()

    def on_frame(self, frame: Frame) -> None:
        with self._cond:
            self.frames.append(frame)
            self._cond.notify_all()

    def on_state(self, state: CaptureState) -> None:
        with self._cond:
            self.states.append(state)
            self._cond.notify_all()

    def statuses(self) -> list[CaptureStatus]:
        with self._cond:
            return [s.status for s in self.states]

    def wait_frames(self, n: int, timeout: float = 5.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: len(self.frames) >= n, timeout)

    def wait_status(self, status: CaptureStatus, timeout: float = 5.0) -> bool:
        with self._cond:
            return self._cond.wait_for(
                lambda: any(s.status == status for s in self.states), timeout
            )

    def distinct_statuses(self) -> list[CaptureStatus]:
        """Statuts successifs sans répétitions consécutives."""
        out: list[CaptureStatus] = []
        for s in self.statuses():
            if not out or out[-1] != s:
                out.append(s)
        return out
