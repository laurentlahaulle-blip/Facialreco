"""Fenêtre principale et onglets. Widgets MINCES : toute la logique est dans les modèles de
vue (``app.viewmodels``) ; ici, uniquement l'affichage et la transmission des saisies."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from maisonaccueil.contracts import ProfileId, SourceKind

from ..controller import AppController
from ..messages import PreviewImage
from ..viewmodels import (
    GUIDANCE_STEPS,
    SOURCE_KIND_LABELS,
    EnrollmentPhase,
    ProfileForm,
    SettingsForm,
    ValidationError,
)
from ..viewmodels.enrollment import PREPARATION

POLL_INTERVAL_MS = 50


def preview_to_qimage(image: PreviewImage) -> QImage:
    """Conversion sans calcul : les pixels BGR888 sont déjà réduits côté vision."""
    qimg = QImage(image.data, image.width, image.height, image.width * 3,
                  QImage.Format.Format_BGR888)
    return qimg.copy()  # détache des octets Python


def _error(parent: QWidget, exc: Exception) -> None:
    QMessageBox.warning(parent, "MaisonAccueil", str(exc))


class HomeTab(QWidget):
    def __init__(self, ctrl: AppController) -> None:
        super().__init__()
        self._ctrl = ctrl
        self.preview_label = QLabel("aucune image reçue")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.caption = QLabel()
        self.camera = QLabel()
        self.engine = QLabel()
        self.model = QLabel()
        self.model.setWordWrap(True)
        self.dropped = QLabel()
        self.processing = QLabel()
        self.faces = QListWidget()
        self.arrivals = QListWidget()

        left = QVBoxLayout()
        left.addWidget(self.preview_label, 1)
        left.addWidget(self.caption)
        status = QGroupBox("État du système")
        form = QFormLayout(status)
        form.addRow("Caméra :", self.camera)
        form.addRow("Moteur :", self.engine)
        form.addRow("Modèle :", self.model)
        form.addRow("Images :", self.dropped)
        form.addRow("", self.processing)
        left.addWidget(status)

        right = QVBoxLayout()
        right.addWidget(QLabel("Visages en cours (scores techniques non calibrés) :"))
        right.addWidget(self.faces, 1)
        right.addWidget(QLabel("Journal des arrivées :"))
        right.addWidget(self.arrivals, 2)

        layout = QHBoxLayout(self)
        layout.addLayout(left, 3)
        layout.addLayout(right, 2)

        ctrl.preview.subscribe(self.refresh_preview)
        ctrl.status.subscribe(self.refresh_status)
        ctrl.arrivals.subscribe(self.refresh_arrivals)
        self.refresh_preview()
        self.refresh_status()
        self.refresh_arrivals()

    def refresh_preview(self) -> None:
        vm = self._ctrl.preview
        img = vm.image
        if img is None:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(vm.caption)
        else:
            pix = QPixmap.fromImage(preview_to_qimage(img))
            self.preview_label.setPixmap(
                pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            )
        self.caption.setText(vm.caption)

    def refresh_status(self) -> None:
        vm = self._ctrl.status
        self.camera.setText(vm.camera_text)
        self.engine.setText(vm.engine_text)
        self.model.setText(vm.model_text)
        self.dropped.setText(vm.dropped_text)
        self.processing.setText(vm.processing_text)
        self.faces.clear()
        self.faces.addItems(vm.face_lines)

    def refresh_arrivals(self) -> None:
        vm = self._ctrl.arrivals
        self.arrivals.clear()
        if vm.load_error:
            self.arrivals.addItem(vm.load_error)
        self.arrivals.addItems([f"{r.time_text} — {r.label}" for r in vm.rows])


class ProfilesTab(QWidget):
    COLUMNS = ("Nom", "Nom prononcé", "Message d'accueil", "Consentement", "Actif")

    def __init__(self, ctrl: AppController) -> None:
        super().__init__()
        self._ctrl = ctrl
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_select)

        self.name = QLineEdit()
        self.spoken = QLineEdit()
        self.welcome = QPlainTextEdit()
        self.welcome.setMaximumHeight(70)
        self.consent = QCheckBox(
            "La personne a donné son consentement à l'enregistrement de son visage"
        )
        self.consent_note = QLineEdit()
        create = QPushButton("Créer le profil")
        save = QPushButton("Enregistrer le message d'accueil")
        delete = QPushButton("Supprimer le profil…")
        create.clicked.connect(self._create)
        save.clicked.connect(self._save_welcome)
        delete.clicked.connect(self._delete)

        form = QFormLayout()
        form.addRow("Nom affiché :", self.name)
        form.addRow("Nom prononcé :", self.spoken)
        form.addRow("Message d'accueil :", self.welcome)
        form.addRow("", self.consent)
        form.addRow("Note de consentement :", self.consent_note)
        buttons = QHBoxLayout()
        for b in (create, save, delete):
            buttons.addWidget(b)
        self.error = QLabel()
        layout = QVBoxLayout(self)
        layout.addWidget(self.table, 1)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.error)
        ctrl.profiles.subscribe(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        rows = self._ctrl.profiles.rows
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            values = (r.display_name, r.spoken_name, r.welcome_text, r.consent_text,
                      "oui" if r.active else "non")
            for j, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setData(Qt.ItemDataRole.UserRole, r.profile_id)
                self.table.setItem(i, j, item)
        self.error.setText(self._ctrl.profiles.error or "")

    def _selected(self) -> ProfileId | None:
        items = self.table.selectedItems()
        return ProfileId(items[0].data(Qt.ItemDataRole.UserRole)) if items else None

    def _on_select(self) -> None:
        pid = self._selected()
        row = self._ctrl.profiles.get(pid) if pid else None
        if row is not None:
            self.name.setText(row.display_name)
            self.spoken.setText(row.spoken_name)
            self.welcome.setPlainText(row.welcome_text)

    def _create(self) -> None:
        try:
            self._ctrl.profiles.create(ProfileForm(
                display_name=self.name.text(),
                consent_given=self.consent.isChecked(),
                consent_note=self.consent_note.text(),
                spoken_name=self.spoken.text(),
                welcome_text=self.welcome.toPlainText(),
            ))
            self.consent.setChecked(False)
        except (ValidationError, ValueError) as exc:
            _error(self, exc)

    def _save_welcome(self) -> None:
        pid = self._selected()
        if pid is None:
            return
        try:
            self._ctrl.profiles.update_welcome(
                pid, welcome_text=self.welcome.toPlainText(), spoken_name=self.spoken.text()
            )
        except (ValidationError, ValueError) as exc:
            _error(self, exc)

    def _delete(self) -> None:
        pid = self._selected()
        if pid is None:
            return
        try:
            req = self._ctrl.profiles.request_deletion(pid)
            typed, ok = QInputDialog.getText(self, "Suppression complète", req.confirmation_text)
            if not ok:
                return
            QMessageBox.information(self, "Suppression",
                                    self._ctrl.profiles.confirm_deletion(req, typed))
        except (ValidationError, ValueError, KeyError) as exc:
            _error(self, exc)


class EnrollmentTab(QWidget):
    def __init__(self, ctrl: AppController) -> None:
        super().__init__()
        self._ctrl = ctrl
        self.profile = QComboBox()
        self.conditions = QLineEdit()
        self.conditions.setPlaceholderText("ex. lumière du jour, salon")
        self.select_btn = QPushButton("Choisir ce profil")
        self.start_btn = QPushButton("Démarrer la collecte")
        self.finish_btn = QPushButton("Terminer")
        self.cancel_btn = QPushButton("Annuler (supprime les images de la séance)")
        self.select_btn.clicked.connect(self._select)
        self.start_btn.clicked.connect(self._start)
        self.finish_btn.clicked.connect(self._finish)
        self.cancel_btn.clicked.connect(self._cancel)
        self.steps = QLabel("\n".join(
            f"{i + 1}. {s.title} — {s.instruction}" for i, s in enumerate(GUIDANCE_STEPS)
        ))
        self.steps.setWordWrap(True)
        self.message = QLabel(PREPARATION)
        self.message.setWordWrap(True)
        self.progress = QProgressBar()
        self.availability = QLabel()
        self.availability.setWordWrap(True)

        top = QFormLayout()
        top.addRow("Profil :", self.profile)
        top.addRow("Conditions :", self.conditions)
        buttons = QHBoxLayout()
        for b in (self.select_btn, self.start_btn, self.finish_btn, self.cancel_btn):
            buttons.addWidget(b)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(QLabel("Consignes :"))
        layout.addWidget(self.steps)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.message)
        layout.addWidget(self.availability)
        layout.addStretch(1)
        ctrl.enrollment.subscribe(self.refresh)
        ctrl.profiles.subscribe(self._reload_profiles)
        self._reload_profiles()
        self.refresh()

    def _reload_profiles(self) -> None:
        self.profile.clear()
        for r in self._ctrl.profiles.rows:
            if r.active:
                self.profile.addItem(r.display_name, r.profile_id)

    def refresh(self) -> None:
        vm = self._ctrl.enrollment
        self.progress.setMaximum(vm.target)
        self.progress.setValue(vm.collected)
        self.progress.setFormat("%v / %m images retenues")
        self.message.setText(vm.message)
        self.availability.setText(vm.availability_text)
        capturing = vm.phase == EnrollmentPhase.CAPTURING
        self.select_btn.setEnabled(not capturing)
        self.start_btn.setEnabled(vm.phase == EnrollmentPhase.READY and vm.camera_available)
        self.finish_btn.setEnabled(capturing)
        self.cancel_btn.setEnabled(capturing or vm.phase == EnrollmentPhase.DONE)

    def _select(self) -> None:
        pid = self.profile.currentData()
        if pid:
            try:
                self._ctrl.enrollment.select_profile(ProfileId(pid))
            except ValidationError as exc:
                _error(self, exc)

    def _start(self) -> None:
        try:
            self._ctrl.enrollment.start(self.conditions.text())
        except (ValidationError, ValueError) as exc:
            _error(self, exc)

    def _finish(self) -> None:
        try:
            self._ctrl.enrollment.finish()
        except ValidationError as exc:
            _error(self, exc)

    def _cancel(self) -> None:
        self._ctrl.enrollment.cancel()


class SettingsTab(QWidget):
    def __init__(self, ctrl: AppController) -> None:
        super().__init__()
        self._ctrl = ctrl
        self.camera_id = QLineEdit()
        self.kind = QComboBox()
        for kind, label in SOURCE_KIND_LABELS.items():
            self.kind.addItem(label, str(kind))
        self.location = QLineEdit()
        self.usb_list = QComboBox()
        self.usb_list.activated.connect(self._pick_usb)
        self.discover_btn = QPushButton("Rechercher les webcams USB")
        self.discover_btn.setEnabled(ctrl.settings.discovery_available)
        self.discover_btn.clicked.connect(self._discover)
        self.discovery_label = QLabel(ctrl.settings.discovery_message)
        self.width_edit = QLineEdit()
        self.height_edit = QLineEdit()
        self.fps = QLineEdit()
        self.engine = QComboBox()
        self._reload_engines()
        ctrl.settings.subscribe(self._reload_engines)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume_label = QLabel()
        self.volume.valueChanged.connect(lambda v: self.volume_label.setText(f"{v} / 100"))
        save = QPushButton("Enregistrer")
        save.clicked.connect(self._save)
        self.info = QLabel(
            "Le changement de source s'applique au prochain démarrage du processus vision. "
            "Caméra réseau : réseau local uniquement."
        )
        self.info.setWordWrap(True)
        form = QFormLayout(self)
        form.addRow("Nom de la caméra :", self.camera_id)
        form.addRow("Type de source :", self.kind)
        form.addRow("Index / URL / fichier :", self.location)
        form.addRow(self.discover_btn, self.usb_list)
        form.addRow("", self.discovery_label)
        form.addRow("Largeur (option) :", self.width_edit)
        form.addRow("Hauteur (option) :", self.height_edit)
        form.addRow("Images/s (option) :", self.fps)
        form.addRow("Moteur :", self.engine)
        form.addRow("Volume :", self.volume)
        form.addRow("", self.volume_label)
        form.addRow(save)
        form.addRow(self.info)
        self.load()

    def load(self) -> None:
        f = self._ctrl.settings.load()
        self.camera_id.setText(f.camera_id)
        self.kind.setCurrentIndex(max(0, self.kind.findData(f.source_kind)))
        self.location.setText(f.location)
        self.width_edit.setText(f.width)
        self.height_edit.setText(f.height)
        self.fps.setText(f.fps)
        self.engine.setCurrentIndex(max(0, self.engine.findData(f.engine)))
        self.volume.setValue(f.volume)
        self.volume_label.setText(f"{f.volume} / 100")
        if self._ctrl.settings.warning:
            self.info.setText(self._ctrl.settings.warning)

    def _reload_engines(self) -> None:
        current = self.engine.currentData()
        self.engine.clear()
        for value, label in self._ctrl.settings.engine_choices:
            self.engine.addItem(label, value)
        self.engine.setCurrentIndex(max(0, self.engine.findData(current)))

    def _discover(self) -> None:
        cams = self._ctrl.settings.discover_usb_cameras()
        self.usb_list.clear()
        for location, label in cams:
            self.usb_list.addItem(label, location)
        self.discovery_label.setText(self._ctrl.settings.discovery_message)

    def _pick_usb(self, index: int) -> None:
        location = self.usb_list.itemData(index)
        if location is not None:
            self.kind.setCurrentIndex(max(0, self.kind.findData(str(SourceKind.USB))))
            self.location.setText(str(location))

    def _save(self) -> None:
        form = SettingsForm(
            camera_id=self.camera_id.text(),
            source_kind=str(self.kind.currentData() or SourceKind.USB),
            location=self.location.text(),
            width=self.width_edit.text(),
            height=self.height_edit.text(),
            fps=self.fps.text(),
            engine=str(self.engine.currentData() or ""),
            volume=int(self.volume.value()),
        )
        try:
            self._ctrl.settings.save(form)
            self.info.setText("Réglages enregistrés (source appliquée au prochain démarrage).")
        except ValidationError as exc:
            _error(self, exc)


class MainWindow(QMainWindow):
    def __init__(self, ctrl: AppController) -> None:
        super().__init__()
        self._ctrl = ctrl
        title = "MaisonAccueil"
        if ctrl.services.demo:
            title += " — DÉMO (source synthétique, données en mémoire)"
        self.setWindowTitle(title)
        self.tabs = QTabWidget()
        self.home = HomeTab(ctrl)
        self.profiles = ProfilesTab(ctrl)
        self.enrollment = EnrollmentTab(ctrl)
        self.settings = SettingsTab(ctrl)
        self.tabs.addTab(self.home, "Accueil")
        self.tabs.addTab(self.profiles, "Profils")
        self.tabs.addTab(self.enrollment, "Collecte guidée")
        self.tabs.addTab(self.settings, "Réglages")
        self.setCentralWidget(self.tabs)
        self.resize(1100, 700)
        self.timer = QTimer(self)
        self.timer.setInterval(POLL_INTERVAL_MS)
        self.timer.timeout.connect(ctrl.poll)
        self.timer.start()
