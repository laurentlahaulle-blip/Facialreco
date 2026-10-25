"""Lancement du moteur dans le processus vision : choix de la cible, arguments, repli.

Aucun poids n'est fourni (``models_dir`` vide) : le moteur de référence doit se replier
et l'interface afficher « moteur non disponible » avec la raison reçue."""

from __future__ import annotations

import pickle
import queue
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from maisonaccueil.app import main as app_main
from maisonaccueil.app.controller import AppController
from maisonaccueil.app.ipc import UiInbox
from maisonaccueil.app.main import (
    DEMO_FACTORY,
    ENGINE_WORKER,
    UsageError,
    plan_vision,
)
from maisonaccueil.app.messages import EngineStatus, PreviewImage
from maisonaccueil.app.viewmodels import ENGINE_UNAVAILABLE
from maisonaccueil.app.vision_worker import (
    PREVIEW_WORKER,
    InProcessVision,
    VisionProcess,
)
from maisonaccueil.app.wiring import demo_services
from maisonaccueil.contracts import (
    CameraId,
    CaptureState,
    CaptureStatus,
    ModelRef,
    SourceConfig,
    SourceKind,
)

SYNTH = SourceConfig(CameraId("fake-synthetic"), SourceKind.SYNTHETIC, "demo", 320, 240, 30.0)
FAKE_FACTORY = "maisonaccueil.app.fakes:fake_source_factory"


def _plan(**kw: object):  # type: ignore[no-untyped-def]
    base: dict[str, object] = {
        "demo": False, "engine": None, "source": SYNTH, "factory": FAKE_FACTORY,
        "db_path": Path("var/b/x.sqlite"), "models_dir": None, "engine_importable": True,
    }
    base.update(kw)
    return plan_vision(**base)  # type: ignore[arg-type]


# -- choix de la cible ------------------------------------------------------------------


def test_default_engine_is_reference_when_importable(tmp_path: Path) -> None:
    plan = _plan(db_path=tmp_path / "b.sqlite", models_dir=tmp_path / "poids")
    assert plan.engine == "reference" and plan.engine_name == "reference"
    assert plan.worker == ENGINE_WORKER
    assert plan.worker_kwargs == {"db_path": str((tmp_path / "b.sqlite").resolve()),
                                  "models_dir": str((tmp_path / "poids").resolve())}
    assert Path(plan.worker_kwargs["db_path"]).is_absolute()
    pickle.dumps((plan.worker, plan.factory, plan.source, plan.worker_kwargs))


def test_default_engine_none_when_not_importable() -> None:
    plan = _plan(engine_importable=False)
    assert plan.engine == "none" and plan.engine_name is None
    assert plan.worker == PREVIEW_WORKER and plan.worker_kwargs == {}
    with pytest.raises(UsageError, match="non installé"):
        _plan(engine="reference", engine_importable=False)


def test_explicit_none_and_models_dir_default() -> None:
    assert _plan(engine="none").worker == PREVIEW_WORKER
    assert _plan().worker_kwargs["models_dir"] is None


def test_demo_keeps_no_engine() -> None:
    plan = _plan(demo=True, source=None, factory=None)
    assert plan.engine == "none" and plan.factory == DEMO_FACTORY
    assert plan.source is not None and plan.source.kind == SourceKind.SYNTHETIC
    with pytest.raises(UsageError, match="--demo"):
        _plan(demo=True, engine="reference")


def test_main_rejects_demo_with_engine(capsys: pytest.CaptureFixture[str]) -> None:
    assert app_main.main(["--demo", "--engine", "reference"]) == 2
    assert "--demo" in capsys.readouterr().err


def test_main_check_reports_engine(capsys: pytest.CaptureFixture[str],
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_main, "engine_worker_available", lambda: False)
    app_main.main(["--check"])
    assert "moteur non disponible" in capsys.readouterr().out
    monkeypatch.setattr(app_main, "engine_worker_available", lambda: True)
    app_main.main(["--check"])
    assert "Moteur de référence : installé" in capsys.readouterr().out


def test_engine_worker_is_importable_after_integration() -> None:
    assert app_main.engine_worker_available()


# -- arguments du processus -------------------------------------------------------------


def test_vision_process_rejects_unpicklable_kwargs() -> None:
    with pytest.raises(ValueError, match="non sérialisable"):
        VisionProcess(FAKE_FACTORY, SYNTH, worker_kwargs={"rappel": lambda: None})


def test_vision_process_merges_kwargs() -> None:
    proc = VisionProcess(FAKE_FACTORY, SYNTH, worker=ENGINE_WORKER,
                         worker_kwargs={"db_path": "/tmp/x.sqlite"}, max_preview_fps=5)
    try:
        assert proc.worker == ENGINE_WORKER
        assert proc.worker_kwargs == {"max_preview_fps": 5, "db_path": "/tmp/x.sqlite"}
    finally:
        proc.stop()  # jamais démarré : ferme seulement les files


# -- exécution et repli -----------------------------------------------------------------


def _collect(events: queue.Queue[object], preview: queue.Queue[object],
             cond: Callable[[list[object], int], bool], timeout: float) -> tuple[list[object], int]:
    got: list[object] = []
    previews = 0
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond(got, previews):
        while not events.empty():
            got.append(events.get_nowait())
        while not preview.empty():
            previews += isinstance(preview.get_nowait(), PreviewImage)
        time.sleep(0.02)
    return got, previews


def _engine_statuses(got: list[object]) -> list[EngineStatus]:
    return [m for m in got if isinstance(m, EngineStatus)]


def test_unknown_worker_falls_back_to_preview() -> None:
    events: queue.Queue[object] = queue.Queue()
    preview: queue.Queue[object] = queue.Queue()
    w = InProcessVision(FAKE_FACTORY, SYNTH, events, preview,
                        worker="maisonaccueil.module_absent:run", max_preview_fps=50)
    w.start()
    try:
        got, previews = _collect(events, preview, lambda g, p: p > 0, 5)
    finally:
        w.stop()
    (status,) = _engine_statuses(got)
    assert not status.available and "travailleur indisponible" in status.detail
    assert previews > 0


def test_engine_worker_without_weights_falls_back(tmp_path: Path) -> None:
    events: queue.Queue[object] = queue.Queue()
    preview: queue.Queue[object] = queue.Queue()
    w = InProcessVision(FAKE_FACTORY, SYNTH, events, preview, worker=ENGINE_WORKER,
                        db_path=str(tmp_path / "b.sqlite"),
                        models_dir=str(tmp_path / "sans-poids"), max_preview_fps=50)
    w.start()
    try:
        got, previews = _collect(events, preview, lambda g, p: p > 0, 15)
    finally:
        w.stop()
    statuses = _engine_statuses(got)
    assert statuses and not any(s.available for s in statuses)
    assert "Poids absents" in statuses[0].detail
    assert previews > 0
    assert any(isinstance(m, CaptureState) and m.status == CaptureStatus.RUNNING for m in got)


@pytest.mark.slow
def test_spawn_engine_worker_reaches_ui_with_real_reason(tmp_path: Path) -> None:
    """Vrai processus spawn exécutant runtime.engine:run_engine_worker (source synthétique,
    sans poids) ; le contrôleur affiche la raison et ne propose aucun moteur."""
    services = demo_services()
    services.engine_name = "reference"
    proc = VisionProcess(FAKE_FACTORY, SYNTH, worker=ENGINE_WORKER,
                         worker_kwargs={"db_path": str(tmp_path / "b.sqlite"),
                                        "models_dir": str(tmp_path / "sans-poids")},
                         max_preview_fps=20)
    ctrl = AppController(services, UiInbox(proc.events, proc.preview))
    ctrl.start()
    proc.start()
    try:
        end = time.monotonic() + 60
        while time.monotonic() < end and not (ctrl.preview.image and ctrl.status.engine_text
                                              != ENGINE_UNAVAILABLE):
            ctrl.poll()
            time.sleep(0.05)
    finally:
        proc.stop()
    assert ctrl.preview.image is not None
    assert ctrl.status.engine_text.startswith(ENGINE_UNAVAILABLE + " (moteur de référence")
    (choice,) = ctrl.settings.engine_choices
    assert choice[0] == "" and "Poids absents" in choice[1]
    assert not proc.alive


# -- ce que l'interface annonce ---------------------------------------------------------


def test_settings_follow_reported_engine(model: ModelRef) -> None:
    services = demo_services()
    services.engine_name = "reference"
    events: queue.Queue[object] = queue.Queue()
    ctrl = AppController(services, UiInbox(events, queue.Queue()))
    ctrl.start()
    assert ctrl.settings.engine_choices == [("", ENGINE_UNAVAILABLE)]
    events.put(EngineStatus(False, detail="moteur de référence : poids absents"))
    ctrl.poll()
    assert ctrl.settings.engine_choices == [
        ("", f"{ENGINE_UNAVAILABLE} (moteur de référence : poids absents)")]
    events.put(EngineStatus(True, model, detail="référence (YuNet + SFace)"))
    ctrl.poll()
    assert ("reference", "reference") in ctrl.settings.engine_choices
    assert ctrl.status.engine_text == "moteur disponible — référence (YuNet + SFace)"
