"""Suivi de visages par association IoU, tolérant aux pertes courtes.

Un ``TrackId`` désigne un passage (silhouette suivie), jamais une identité. Un passage
qui n'est plus associé reste « en attente » pendant au plus ``max_missed_frames`` images
ET ``max_missed_ms`` millisecondes ; au-delà il est déclaré perdu et renvoyé une seule
fois par ``lost_tracks()``. Non thread-safe : un suiveur par flux.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from maisonaccueil.contracts import BoundingBox, Detection, Frame, TrackedFace, TrackId
from maisonaccueil.contracts import new_track_id as _new_track_id


@dataclass(slots=True)
class _Track:
    track_id: TrackId
    bbox: BoundingBox
    first_frame_index: int
    last_seen_ns: int
    missed: int = 0


class IouTracker:
    """Association gloutonne par IoU décroissant (implémente ``FaceTracker``)."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 15,
        max_missed_ms: int = 1500,
        new_track_id: Callable[[], str] | None = None,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold doit être dans ]0, 1]")
        if max_missed_frames < 0 or max_missed_ms < 0:
            raise ValueError("Tolérances de perte négatives")
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.max_missed_ms = max_missed_ms
        self._make_id: Callable[[], str] = new_track_id or _new_track_id
        self._tracks: list[_Track] = []
        self._lost: list[TrackId] = []
        self._frame_index = 0

    @property
    def active_track_ids(self) -> tuple[TrackId, ...]:
        return tuple(t.track_id for t in self._tracks)

    def update(self, frame: Frame, detections: Sequence[Detection]) -> Sequence[TrackedFace]:
        self._frame_index += 1
        now = frame.monotonic_ns

        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                iou = track.bbox.iou(det.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

        matched_tracks: set[int] = set()
        assignment: dict[int, _Track] = {}
        for _iou, ti, di in pairs:
            if ti in matched_tracks or di in assignment:
                continue
            matched_tracks.add(ti)
            assignment[di] = self._tracks[ti]

        survivors: list[_Track] = []
        for ti, track in enumerate(self._tracks):
            if ti in matched_tracks:
                survivors.append(track)
                continue
            track.missed += 1
            gone_ms = (now - track.last_seen_ns) / 1_000_000
            if track.missed > self.max_missed_frames or gone_ms > self.max_missed_ms:
                self._lost.append(track.track_id)
            else:
                survivors.append(track)

        out: list[TrackedFace] = []
        for di, det in enumerate(detections):
            track = assignment.get(di)
            if track is None:
                track = _Track(
                    track_id=TrackId(self._make_id()),
                    bbox=det.bbox,
                    first_frame_index=self._frame_index,
                    last_seen_ns=now,
                )
                survivors.append(track)
            track.bbox = det.bbox
            track.last_seen_ns = now
            track.missed = 0
            out.append(
                TrackedFace(
                    track_id=track.track_id,
                    detection=det,
                    monotonic_ns=now,
                    track_age_frames=self._frame_index - track.first_frame_index,
                )
            )
        self._tracks = survivors
        return out

    def lost_tracks(self) -> Sequence[TrackId]:
        lost, self._lost = self._lost, []
        return lost

    def reset(self) -> None:
        """Termine tous les passages (ex. coupure caméra) ; ``lost_tracks()`` les renverra."""
        self._lost.extend(t.track_id for t in self._tracks)
        self._tracks = []
