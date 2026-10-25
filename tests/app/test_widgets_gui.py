"""Tests Qt (marqués ``gui``), exécutés avec ``QT_QPA_PLATFORM=offscreen``.

Ignorés proprement si Qt Widgets ne se charge pas (ex. ``libEGL.so.1`` absent) :
``pytest.importorskip`` ne suffit pas, l'erreur n'étant pas un ``ModuleNotFoundError``.
"""

from __future__ import annotations

import os
import queue
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # libEGL absent, PySide6 non installé…
    pytest.skip(f"Qt Widgets indisponible : {exc}", allow_module_level=True)

from maisonaccueil.app.controller import AppController  # noqa: E402
from maisonaccueil.app.ipc import UiInbox  # noqa: E402
from maisonaccueil.app.messages import ENGINE_STATUS_UNAVAILABLE, PreviewImage  # noqa: E402
from maisonaccueil.app.viewmodels import ENGINE_UNAVAILABLE, ProfileForm  # noqa: E402
from maisonaccueil.app.widgets.main_window import MainWindow, preview_to_qimage  # noqa: E402
from maisonaccueil.app.wiring import demo_services  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


@pytest.fixture
def window(qapp: QApplication) -> Iterator[tuple[MainWindow, AppController, queue.Queue[object],
                                                 queue.Queue[object]]]:
    events: queue.Queue[object] = queue.Queue()
    preview: queue.Queue[object] = queue.Queue()
    ctrl = AppController(demo_services(), UiInbox(events, preview))
    w = MainWindow(ctrl)
    ctrl.start()
    yield w, ctrl, events, preview
    w.timer.stop()
    w.close()


def test_window_builds_with_demo_banner(window) -> None:  # type: ignore[no-untyped-def]
    w, ctrl, events, preview = window
    assert "DÉMO" in w.windowTitle()
    assert w.tabs.count() == 4
    events.put(ENGINE_STATUS_UNAVAILABLE)
    preview.put(PreviewImage(1, "cam", 1, 4, 2, 4, 2, bytes(24)))
    ctrl.poll()
    assert w.home.engine.text().startswith(ENGINE_UNAVAILABLE)
    assert w.home.faces.item(0).text() == ENGINE_UNAVAILABLE
    assert not w.home.preview_label.pixmap().isNull()


def test_profiles_tab_lists_created_profile(window) -> None:  # type: ignore[no-untyped-def]
    w, ctrl, _, _ = window
    ctrl.profiles.create(ProfileForm("Personne Test", consent_given=True))
    assert w.profiles.table.rowCount() == 1
    assert w.enrollment.profile.count() == 1


def test_preview_conversion(qapp: QApplication) -> None:
    img = preview_to_qimage(PreviewImage(1, "c", 1, 3, 2, 3, 2, bytes(range(18))))
    assert (img.width(), img.height()) == (3, 2)
