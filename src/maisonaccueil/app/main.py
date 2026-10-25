"""Point d'entrée ``maisonaccueil`` : interface PySide6 + processus vision séparé.

Modes :

- ``maisonaccueil``                : base ``var/b/maisonaccueil.sqlite`` (stockage lot C),
  source lue dans les réglages, ouverte par la fabrique du lot A
  (``maisonaccueil.capture.factory:create_source``) si elle est installée, ou par
  ``--source-factory module:fonction``. Sans fabrique : « caméra non disponible ».
- ``--engine reference`` (défaut si ``maisonaccueil.runtime.engine`` est importable) :
  le processus vision exécute ``runtime.engine:run_engine_worker`` (YuNet + SFace,
  décision, accueil) sur le MÊME fichier de base que l'interface. Si les poids manquent,
  ce travailleur se replie lui-même et publie « moteur non disponible » avec la raison.
  ``--engine none`` : aperçu seul.
- ``maisonaccueil --demo``         : source SYNTHÉTIQUE (``FakeFrameSource``), dépôts en
  mémoire, aucun moteur ; titre de fenêtre marqué « DÉMO ».
- ``maisonaccueil --check``        : diagnostic sans fenêtre.

L'interface n'annonce un moteur que lorsque le processus vision publie
``EngineStatus(available=True)``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from .vision_worker import PREVIEW_WORKER
from .wiring import DEFAULT_DB, optional_callable

log = logging.getLogger("maisonaccueil.app")

DEMO_SOURCE = SourceConfig(CameraId("fake-synthetic"), SourceKind.SYNTHETIC, "demo",
                           640, 480, 15.0)
DEMO_FACTORY = "maisonaccueil.app.fakes:fake_source_factory"

ENGINE_NONE = "none"
ENGINE_REFERENCE = "reference"
ENGINE_WORKER = "maisonaccueil.runtime.engine:run_engine_worker"


class UsageError(ValueError):
    """Combinaison d'options impossible (message destiné à l'utilisateur)."""


@dataclass(frozen=True, slots=True)
class VisionPlan:
    """Ce que lance le processus vision. ``factory`` None : aucune capture."""

    engine: str
    worker: str
    worker_kwargs: dict[str, Any] = field(default_factory=dict)
    factory: str | None = None
    source: SourceConfig | None = None

    @property
    def engine_name(self) -> str | None:
        return None if self.engine == ENGINE_NONE else self.engine


def engine_worker_available() -> bool:
    return optional_callable(ENGINE_WORKER) is not None


def default_engine(demo: bool, engine_importable: bool) -> str:
    return ENGINE_REFERENCE if engine_importable and not demo else ENGINE_NONE


def plan_vision(
    *,
    demo: bool,
    engine: str | None,
    source: SourceConfig | None,
    factory: str | None,
    db_path: Path,
    models_dir: Path | None,
    engine_importable: bool,
) -> VisionPlan:
    """Choix du travailleur vision et de ses arguments (sérialisables), sans Qt."""
    chosen = engine or default_engine(demo, engine_importable)
    if chosen not in (ENGINE_NONE, ENGINE_REFERENCE):
        raise UsageError(f"Moteur inconnu : {chosen!r}")
    if demo:
        if chosen != ENGINE_NONE:
            raise UsageError("--demo n'utilise aucun moteur (données en mémoire, sans base)")
        return VisionPlan(ENGINE_NONE, PREVIEW_WORKER, {}, DEMO_FACTORY, DEMO_SOURCE)
    if chosen == ENGINE_NONE:
        return VisionPlan(ENGINE_NONE, PREVIEW_WORKER, {}, factory, source)
    if not engine_importable:
        raise UsageError("Moteur de référence non installé (maisonaccueil.runtime.engine)")
    kwargs: dict[str, Any] = {
        "db_path": str(db_path.resolve()),
        "models_dir": str(models_dir.resolve()) if models_dir is not None else None,
    }
    return VisionPlan(ENGINE_REFERENCE, ENGINE_WORKER, kwargs, factory, source)


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
                   help="fichier SQLite partagé avec le processus vision "
                        "(défaut : var/b/maisonaccueil.sqlite)")
    p.add_argument("--engine", choices=(ENGINE_NONE, ENGINE_REFERENCE), default=None,
                   help="moteur du processus vision (défaut : reference s'il est installé, "
                        "none sinon ; toujours none avec --demo)")
    p.add_argument("--models-dir", type=Path, default=None,
                   help="dossier des poids du moteur de référence (optionnel)")
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
    importable = engine_worker_available()
    if args.check:
        print(banner)
        print(f"Qt Widgets : {'disponible' if ok else 'NON disponible (' + reason + ')'}")
        if importable:
            print("Moteur de référence : installé (choisi par défaut ; poids vérifiés au "
                  "démarrage du processus vision, sinon « moteur non disponible »)")
        else:
            print("Moteur : moteur non disponible (maisonaccueil.runtime.engine absent)")
        return 0 if ok else 2
    try:
        plan = plan_vision(
            demo=args.demo, engine=args.engine, source=None, factory=None,
            db_path=args.db or DEFAULT_DB, models_dir=args.models_dir,
            engine_importable=importable,
        )
    except UsageError as exc:
        print(f"{banner}\n{exc}", file=sys.stderr)
        return 2
    if not ok:
        print(f"{banner}\nInterface impossible : Qt Widgets ne se charge pas ({reason}).",
              file=sys.stderr)
        return 2
    return _run_gui(args, plan.engine)


def _run_gui(args: argparse.Namespace, engine: str) -> int:  # pragma: no cover - affichage
    from PySide6.QtWidgets import QApplication

    from .viewmodels import SettingsViewModel
    from .vision_worker import VisionProcess
    from .widgets.main_window import MainWindow
    from .wiring import default_source_factory_spec, demo_services, open_storage_services

    storage = None
    services: AppServices
    db_path = args.db or DEFAULT_DB
    if args.demo:
        services = demo_services()
        source = None
    else:
        storage = open_storage_services(db_path)
        services = storage.services
        source = SettingsViewModel(services.settings).current_source()
    plan = plan_vision(
        demo=args.demo, engine=engine, source=source,
        factory=args.source_factory or default_source_factory_spec(),
        db_path=db_path, models_dir=args.models_dir,
        engine_importable=engine_worker_available(),
    )
    services.engine_name = plan.engine_name

    vision: VisionProcess | None = None
    inbox: UiInbox | None = None
    if plan.factory and plan.source is not None:
        vision = VisionProcess(plan.factory, plan.source, worker=plan.worker,
                               worker_kwargs=plan.worker_kwargs)
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
