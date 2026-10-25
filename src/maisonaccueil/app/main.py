"""Point d'entrée ``maisonaccueil`` : interface PySide6 + processus vision séparé.

Modes :

- ``maisonaccueil``                : base ``var/b/maisonaccueil.sqlite`` (stockage lot C),
  source lue dans les réglages, ouverte par la fabrique du lot A
  (``maisonaccueil.capture.factory:create_source``) si elle est installée, ou par
  ``--source-factory module:fonction``. Sans fabrique : « caméra non disponible ».
- ``maisonaccueil --demo``         : source SYNTHÉTIQUE (``FakeFrameSource``), dépôts en
  mémoire ; titre de fenêtre marqué « DÉMO ».
- ``maisonaccueil --check``        : diagnostic sans fenêtre (Qt chargeable ? base ?).

Dans tous les cas, aucun moteur n'est branché ici : l'interface affiche
« moteur non disponible ». La chaîne complète (``pipeline.VisionLoop``) se raccorde via
``app.ipc.VisionPublisher`` lors de l'intégration.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from maisonaccueil import __version__
from maisonaccueil.contracts import (
    CONTRACTS_VERSION,
    SCHEMA_VERSION,
    CameraId,
    SourceConfig,
    SourceKind,
)

from .controller import AppController, AppServices
from .ipc import UiInbox

log = logging.getLogger("maisonaccueil.app")

DEMO_SOURCE = SourceConfig(CameraId("fake-synthetic"), SourceKind.SYNTHETIC, "demo",
                           640, 480, 15.0)
DEMO_FACTORY = "maisonaccueil.app.fakes:fake_source_factory"


def qt_available() -> tuple[bool, str]:
    """Vérifie que Qt Widgets se charge (ex. échoue si ``libEGL.so.1`` est absent)."""
    try:
        import PySide6.QtWidgets  # noqa: F401
    except ImportError as exc:
        return False, str(exc)
    return True, ""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="maisonaccueil", description="MaisonAccueil — interface")
    p.add_argument("--demo", action="store_true",
                   help="source synthétique et données en mémoire (aucune écriture disque)")
    p.add_argument("--db", type=Path, default=None,
                   help="fichier SQLite (défaut : var/b/maisonaccueil.sqlite)")
    p.add_argument("--source-factory", default=None,
                   help="fabrique de FrameSource 'module:fonction' (prend un SourceConfig)")
    p.add_argument("--check", action="store_true", help="diagnostic sans fenêtre")
    p.add_argument("--version", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    banner = (f"MaisonAccueil {__version__} — contrats {CONTRACTS_VERSION}, "
              f"schéma v{SCHEMA_VERSION}")
    if args.version:
        print(banner)
        return 0
    ok, reason = qt_available()
    if args.check:
        print(banner)
        print(f"Qt Widgets : {'disponible' if ok else 'NON disponible (' + reason + ')'}")
        print("Moteur : moteur non disponible (aucun moteur intégré)")
        return 0 if ok else 2
    if not ok:
        print(f"{banner}\nInterface impossible : Qt Widgets ne se charge pas ({reason}).",
              file=sys.stderr)
        return 2
    return _run_gui(args)


def _run_gui(args: argparse.Namespace) -> int:  # pragma: no cover - nécessite un affichage
    from PySide6.QtWidgets import QApplication

    from .viewmodels import SettingsViewModel
    from .vision_worker import VisionProcess
    from .widgets.main_window import MainWindow
    from .wiring import (
        DEFAULT_DB,
        default_source_factory_spec,
        demo_services,
        open_storage_services,
    )

    storage = None
    services: AppServices
    factory: str | None
    if args.demo:
        services = demo_services()
        source, factory = DEMO_SOURCE, DEMO_FACTORY
    else:
        storage = open_storage_services(args.db or DEFAULT_DB)
        services = storage.services
        source = SettingsViewModel(services.settings).current_source()
        factory = args.source_factory or default_source_factory_spec()

    vision: VisionProcess | None = None
    inbox: UiInbox | None = None
    if factory:
        vision = VisionProcess(factory, source)
        inbox = UiInbox(vision.events, vision.preview)
    ctrl = AppController(services, inbox)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(ctrl)
    ctrl.start()
    if vision is not None:
        vision.start()
    else:
        log.info("Aucune fabrique de source : caméra non disponible.")
    window.show()
    try:
        return int(app.exec())
    finally:
        if vision is not None:
            vision.stop()
        if storage is not None:
            storage.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main(sys.argv[1:]))
