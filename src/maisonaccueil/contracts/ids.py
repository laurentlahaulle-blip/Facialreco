"""Identifiants typés. Ils ne sont PAS interchangeables.

- ``ProfileId`` : identité durable d'une personne enregistrée (stockée en base).
- ``TrackId``   : identifiant d'un *passage* devant la caméra (suivi visuel). Éphémère,
                  jamais stocké comme identité, ne survit pas au redémarrage.
- ``FrameId``   : numéro séquentiel d'une image dans une source donnée.
- ``CameraId``  : nom de la source configurée.
- ``EventId``   : identifiant d'un événement (arrivée, demande audio).
"""

from __future__ import annotations

import uuid
from typing import NewType

ProfileId = NewType("ProfileId", str)
TrackId = NewType("TrackId", str)
FrameId = NewType("FrameId", int)
CameraId = NewType("CameraId", str)
EventId = NewType("EventId", str)
SessionId = NewType("SessionId", str)  # séance de collecte (sert au découpage train/test)


def new_profile_id() -> ProfileId:
    return ProfileId(f"prf-{uuid.uuid4().hex}")


def new_track_id() -> TrackId:
    return TrackId(f"trk-{uuid.uuid4().hex[:12]}")


def new_event_id() -> EventId:
    return EventId(f"evt-{uuid.uuid4().hex}")


def new_session_id() -> SessionId:
    return SessionId(f"ses-{uuid.uuid4().hex[:12]}")
