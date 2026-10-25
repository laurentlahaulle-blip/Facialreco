"""Racine de composition : SEUL module de l'interface qui importe l'implémentation du
stockage (lot C). Le reste de l'interface ne voit que ``app.ports``.

L'import est différé (dans la fonction) pour que l'interface reste importable si le
stockage est absent ou remplacé.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maisonaccueil.contracts import CameraId

from .controller import AppServices
from .fakes import (
    FakeArrivalHistory,
    FakeProfileStore,
    FakeReferenceSink,
    FakeSettingsStore,
)

DEFAULT_DB = Path("var") / "b" / "maisonaccueil.sqlite"

# Points d'entrée du lot A (capture), résolus PAR NOM à l'exécution : l'interface ne
# dépend pas de leur implémentation et reste utilisable s'ils sont absents.
CAPTURE_FACTORY_SPEC = "maisonaccueil.capture.factory:create_source"
USB_DISCOVERY_SPEC = "maisonaccueil.capture.discovery:list_usb_cameras"


def optional_callable(spec: str) -> Callable[..., Any] | None:
    """``module:fonction`` si le module existe et expose la fonction, sinon None."""
    module_name, _, attr = spec.partition(":")
    try:
        if importlib.util.find_spec(module_name) is None:
            return None
        fn = getattr(importlib.import_module(module_name), attr, None)
    except ImportError:
        return None
    return fn if callable(fn) else None


def default_source_factory_spec() -> str | None:
    """Fabrique du lot A si elle est installée (transmise par nom au processus vision)."""
    return CAPTURE_FACTORY_SPEC if optional_callable(CAPTURE_FACTORY_SPEC) else None


def default_camera_discovery() -> Callable[[], Any] | None:
    return optional_callable(USB_DISCOVERY_SPEC)


@dataclass(slots=True)
class StorageHandle:
    services: AppServices
    database: Any  # maisonaccueil.storage.Database, à fermer en fin d'exécution

    def close(self) -> None:
        self.database.close()


def open_storage_services(
    db_path: str | Path = DEFAULT_DB,
    *,
    available_engines: tuple[str, ...] = (),
    camera_id: CameraId | None = None,
) -> StorageHandle:
    from maisonaccueil.storage import (
        ArrivalLogRepository,
        Database,
        ProfileRepository,
        ReferenceRepository,
        SettingsRepository,
    )

    db = Database(db_path)
    services = AppServices(
        profiles=ProfileRepository(db),
        references=ReferenceRepository(db),
        settings=SettingsRepository(db),
        arrivals=ArrivalLogRepository(db),
        available_engines=available_engines,
        camera_id=camera_id,
        camera_discovery=default_camera_discovery(),
    )
    return StorageHandle(services, db)


def demo_services() -> AppServices:
    """Dépôts en mémoire (``Fake*``) : rien n'est écrit sur disque."""
    profiles = FakeProfileStore()
    refs = FakeReferenceSink()
    profiles.on_delete = refs.purge_profile
    return AppServices(
        profiles=profiles,
        references=refs,
        settings=FakeSettingsStore(),
        arrivals=FakeArrivalHistory(),
        camera_id=CameraId("fake-synthetic"),
        demo=True,
    )
