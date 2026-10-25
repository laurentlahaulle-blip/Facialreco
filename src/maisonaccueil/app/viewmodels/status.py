"""État système : caméra, moteur, ``ModelRef``, images abandonnées, visages en cours.

Règle : aucun résultat de reconnaissance n'est affiché sans moteur déclaré disponible
(``EngineStatus.available``) ; un résultat dont le ``ModelRef`` diffère de celui du moteur
déclaré est ignoré et compté. Les scores sont des scores techniques non calibrés.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from maisonaccueil.contracts import (
    CaptureState,
    CaptureStatus,
    Decision,
    DecisionStatus,
    ProfileId,
    RecognitionResult,
)
from maisonaccueil.pipeline import VisionOutput

from ..messages import EngineStatus
from .base import Observable
from .formatting import (
    CAMERA_UNAVAILABLE,
    CAPTURE_LABELS,
    DECISION_LABELS,
    ENGINE_UNAVAILABLE,
    format_model_ref,
    format_raw_score,
    short_id,
)

NameLookup = Callable[[ProfileId], str | None]


class SystemStatusViewModel(Observable):
    def __init__(
        self,
        name_lookup: NameLookup,
        *,
        stale_after_ms: int = 3_000,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__()
        self._name_lookup = name_lookup
        self._stale_ns = stale_after_ms * 1_000_000
        self._clock = clock
        self._capture: CaptureState | None = None
        self._engine: EngineStatus | None = None
        self._last_output: VisionOutput | None = None
        self._last_output_at: int | None = None
        self._face_lines: list[str] = []
        self.ignored_results = 0

    # -- entrées --------------------------------------------------------------------
    def apply_capture_state(self, state: CaptureState) -> None:
        self._capture = state
        self._notify()

    def apply_engine_status(self, status: EngineStatus) -> None:
        self._engine = status
        if not status.available:
            self._face_lines = []
        self._notify()

    def apply_vision_output(self, output: VisionOutput) -> None:
        self._last_output = output
        self._last_output_at = self._clock()
        self._face_lines = self._build_face_lines(output)
        self._notify()

    def tick(self) -> None:
        """Efface les visages affichés si aucune sortie récente (passage terminé)."""
        if (
            self._face_lines
            and self._last_output_at is not None
            and self._clock() - self._last_output_at > self._stale_ns
        ):
            self._face_lines = []
            self._notify()

    # -- sorties --------------------------------------------------------------------
    @property
    def engine_available(self) -> bool:
        return self._engine is not None and self._engine.available

    @property
    def camera_running(self) -> bool:
        return self._capture is not None and self._capture.status == CaptureStatus.RUNNING

    @property
    def camera_text(self) -> str:
        s = self._capture
        if s is None:
            return CAMERA_UNAVAILABLE
        text = f"{s.camera_id} : {CAPTURE_LABELS[s.status]}"
        return f"{text} — {s.detail}" if s.detail else text

    @property
    def engine_text(self) -> str:
        e = self._engine
        if e is None or not e.available:
            detail = f" ({e.detail})" if e is not None and e.detail else ""
            return ENGINE_UNAVAILABLE + detail
        return "moteur disponible" + (f" — {e.detail}" if e.detail else "")

    @property
    def model_text(self) -> str:
        return format_model_ref(self._engine.model if self.engine_available and self._engine
                                else None)

    @property
    def dropped_text(self) -> str:
        cap = self._capture.frames_dropped if self._capture else 0
        captured = self._capture.frames_captured if self._capture else 0
        vis = self._last_output.frames_dropped if self._last_output else 0
        return (
            f"capturées : {captured} · abandonnées (capture) : {cap} · "
            f"abandonnées (vision) : {vis}"
        )

    @property
    def processing_text(self) -> str:
        if self._last_output is None:
            return "traitement : —"
        return f"traitement : {self._last_output.processing_ms:.0f} ms par image"

    @property
    def face_lines(self) -> list[str]:
        if not self.engine_available:
            return [ENGINE_UNAVAILABLE]
        return list(self._face_lines)

    # -- interne --------------------------------------------------------------------
    def _accepted(self, result: RecognitionResult) -> bool:
        e = self._engine
        return e is not None and e.available and result.model == e.model

    def _name(self, profile_id: ProfileId) -> str:
        return self._name_lookup(profile_id) or f"profil {short_id(profile_id)} (introuvable)"

    def _build_face_lines(self, output: VisionOutput) -> list[str]:
        if not self.engine_available:
            self.ignored_results += len(output.results) + len(output.decisions)
            return []
        lines: list[str] = []
        decided = {d.track_id: d for d in output.decisions if d.model == self._engine.model}  # type: ignore[union-attr]
        self.ignored_results += len(output.decisions) - len(decided)
        for result in output.results:
            if not self._accepted(result):
                self.ignored_results += 1
                continue
            lines.append(self._result_line(result, decided.pop(result.track_id, None)))
        for decision in decided.values():
            lines.append(self._decision_text(decision))
        return lines

    def _result_line(self, result: RecognitionResult, decision: Decision | None) -> str:
        head = f"Passage {short_id(result.track_id)}"
        best = result.best
        if best is None:
            body = "aucune référence compatible"
        else:
            body = (
                f"meilleur candidat (pas une décision) : {self._name(best.profile_id)} — "
                f"{format_raw_score(best.score)}"
            )
        if result.quality_flags:
            body += " ; qualité : " + ", ".join(sorted(str(f) for f in result.quality_flags))
        line = f"{head} : {body}"
        if decision is not None:
            line += f" | {self._decision_text(decision)}"
        return line

    def _decision_text(self, d: Decision) -> str:
        text = f"Décision ({d.policy}) : {DECISION_LABELS[d.status]}"
        if d.status == DecisionStatus.CONFIRMED and d.profile_id is not None:
            text += f" — {self._name(d.profile_id)}"
        return text + f" [{d.observations} obs. / {d.window_ms} ms]"
