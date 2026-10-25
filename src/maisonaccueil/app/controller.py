"""Contrôleur de l'interface, sans Qt : assemble les modèles de vue et leur distribue les
messages lus dans les files du processus vision (``UiInbox``). La fenêtre Qt n'appelle
que ``poll()`` (minuterie) et lit les modèles de vue."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from maisonaccueil.contracts import CameraId, CaptureState
from maisonaccueil.pipeline import VisionOutput

from .ipc import InboxBatch, UiInbox
from .messages import EngineStatus, EnrollmentSample
from .ports import ArrivalHistory, ProfileStore, ReferenceSink, SettingsStore
from .viewmodels import (
    ArrivalLogViewModel,
    EnrollmentViewModel,
    PreviewViewModel,
    ProfilesViewModel,
    SettingsViewModel,
    SystemStatusViewModel,
)
from .viewmodels.settings import CameraDiscovery


@dataclass(slots=True)
class AppServices:
    """Dépendances injectées (stockage réel du lot C ou doubles ``Fake*``)."""

    profiles: ProfileStore
    references: ReferenceSink
    settings: SettingsStore
    arrivals: ArrivalHistory | None = None
    available_engines: Sequence[str] = ()
    camera_id: CameraId | None = None
    demo: bool = False  # bannière « DÉMO — données synthétiques »
    camera_discovery: CameraDiscovery | None = None


class AppController:
    def __init__(self, services: AppServices, inbox: UiInbox | None = None) -> None:
        self.services = services
        self.inbox = inbox
        self.profiles = ProfilesViewModel(services.profiles)
        self.status = SystemStatusViewModel(self.profiles.name_of)
        self.preview = PreviewViewModel()
        self.arrivals = ArrivalLogViewModel(self.profiles.name_of, services.arrivals)
        self.settings = SettingsViewModel(
            services.settings,
            services.available_engines,
            camera_discovery=services.camera_discovery,
        )
        self.enrollment = EnrollmentViewModel(
            services.profiles, services.references, camera_id=services.camera_id
        )
        self.unexpected_messages = 0

    def start(self) -> None:
        self.profiles.refresh()
        self.arrivals.load_history()
        self._sync_enrollment_availability()

    def poll(self) -> InboxBatch | None:
        """Lecture non bloquante des files + mise à jour des modèles de vue."""
        batch = self.inbox.drain() if self.inbox is not None else None
        if batch is not None:
            self.dispatch(batch)
        self.status.tick()
        self.preview.tick()
        self.enrollment.tick()
        return batch

    def dispatch(self, batch: InboxBatch) -> None:
        self.unexpected_messages += batch.unexpected
        for event in batch.events:
            if isinstance(event, VisionOutput):
                self.status.apply_vision_output(event)
                self.arrivals.apply_vision_output(event)
            elif isinstance(event, CaptureState):
                self.status.apply_capture_state(event)
            elif isinstance(event, EngineStatus):
                self.status.apply_engine_status(event)
            elif isinstance(event, EnrollmentSample):
                self.enrollment.add_sample(event)
        if batch.preview is not None:
            self.preview.apply_preview(batch.preview)
        self._sync_enrollment_availability()

    def _sync_enrollment_availability(self) -> None:
        self.enrollment.set_availability(
            camera=self.status.camera_running, engine=self.status.engine_available
        )
