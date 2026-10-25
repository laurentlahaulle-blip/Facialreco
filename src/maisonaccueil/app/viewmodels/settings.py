"""Réglages : source caméra (``SourceConfig``), moteur, volume. Persistés via
``SettingsStore`` (table ``settings`` du lot C, valeurs JSON).

Flux réseau caméra : réseau LOCAL uniquement (CLAUDE.md, règle 9) — une adresse IP
publique ou un nom de domaine complet non ``.local`` est refusé.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from maisonaccueil.contracts import CameraId, SourceConfig, SourceKind

from ..messages import EngineStatus
from ..ports import SettingsStore
from .base import Observable, ValidationError
from .formatting import ENGINE_UNAVAILABLE

KEY_SOURCE = "camera.source"
KEY_ENGINE = "engine.name"
KEY_VOLUME = "audio.volume"

DEFAULT_SOURCE = SourceConfig(CameraId("camera-0"), SourceKind.USB, "0")
DEFAULT_VOLUME = 70

SOURCE_KIND_LABELS: dict[SourceKind, str] = {
    SourceKind.USB: "Webcam USB (index)",
    SourceKind.NETWORK: "Caméra réseau locale (URL RTSP/HTTP)",
    SourceKind.FILE: "Fichier vidéo de test",
    SourceKind.SYNTHETIC: "Source synthétique (développement)",
}


@dataclass(frozen=True, slots=True)
class SettingsForm:
    """Valeurs saisies (texte brut des champs)."""

    camera_id: str
    source_kind: str
    location: str
    width: str = ""
    height: str = ""
    fps: str = ""
    engine: str = ""  # "" = aucun moteur
    volume: int = DEFAULT_VOLUME


def source_to_json(cfg: SourceConfig) -> dict[str, Any]:
    return {
        "camera_id": str(cfg.camera_id),
        "kind": str(cfg.kind),
        "location": cfg.location,
        "width": cfg.width,
        "height": cfg.height,
        "fps": cfg.fps,
    }


def source_from_json(data: Any) -> SourceConfig:
    if not isinstance(data, dict):
        raise ValueError("réglage camera.source illisible")
    return SourceConfig(
        camera_id=CameraId(str(data["camera_id"])),
        kind=SourceKind(data["kind"]),
        location=str(data["location"]),
        width=int(data["width"]) if data.get("width") is not None else None,
        height=int(data["height"]) if data.get("height") is not None else None,
        fps=float(data["fps"]) if data.get("fps") is not None else None,
    )


def _is_local_host(host: str) -> bool:
    host = host.strip("[]").lower()
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or host.endswith(".local") or "." not in host
    return ip.is_private or ip.is_loopback or ip.is_link_local


def _opt_int(text: str, label: str, lo: int, hi: int) -> int | None:
    t = text.strip()
    if not t:
        return None
    try:
        v = int(t)
    except ValueError:
        raise ValidationError(f"{label} : nombre entier attendu") from None
    if not lo <= v <= hi:
        raise ValidationError(f"{label} : entre {lo} et {hi}")
    return v


def validate_source(form: SettingsForm) -> SourceConfig:
    camera_id = form.camera_id.strip()
    if not camera_id or not camera_id.replace("-", "").replace("_", "").isalnum():
        raise ValidationError("Nom de caméra : lettres, chiffres, « - » ou « _ »")
    try:
        kind = SourceKind(form.source_kind)
    except ValueError:
        raise ValidationError(f"Type de source inconnu : {form.source_kind!r}") from None
    loc = form.location.strip()
    if kind == SourceKind.USB:
        if not loc.isdigit():
            raise ValidationError("Webcam USB : indiquer l'index (0, 1, …)")
    elif kind == SourceKind.NETWORK:
        parts = urlsplit(loc)
        if parts.scheme not in {"rtsp", "http", "https"}:
            raise ValidationError("Caméra réseau : URL rtsp://, http:// ou https:// attendue")
        if not _is_local_host(parts.hostname or ""):
            raise ValidationError(
                "Caméra réseau : réseau local uniquement (adresse privée, localhost ou .local)"
            )
    elif kind == SourceKind.FILE:
        if not loc or not Path(loc).is_file():
            raise ValidationError("Fichier vidéo introuvable")
    width = _opt_int(form.width, "Largeur", 64, 7680)
    height = _opt_int(form.height, "Hauteur", 48, 4320)
    fps: float | None = None
    if form.fps.strip():
        try:
            fps = float(form.fps.replace(",", "."))
        except ValueError:
            raise ValidationError("Images/s : nombre attendu") from None
        if not 1 <= fps <= 120:
            raise ValidationError("Images/s : entre 1 et 120")
    return SourceConfig(CameraId(camera_id), kind, loc, width, height, fps)


CameraDiscovery = Callable[[], Sequence[Any]]
"""Liste des webcams USB : objets portant ``location`` (str), ``status`` et ``detail``
(ex. ``capture.discovery.list_usb_cameras`` du lot A, injecté par ``wiring``)."""


class SettingsViewModel(Observable):
    def __init__(
        self,
        store: SettingsStore,
        available_engines: Sequence[str] = (),
        *,
        camera_discovery: CameraDiscovery | None = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._engines = list(available_engines)
        self._discovery = camera_discovery
        self.warning: str | None = None
        self.engine_detail = ""
        self._engine_reported = bool(self._engines)
        self.usb_cameras: list[tuple[str, str]] = []  # (index, libellé)
        self.discovery_message = (
            "" if camera_discovery else "recherche de webcams non disponible (capture absente)"
        )

    @property
    def discovery_available(self) -> bool:
        return self._discovery is not None

    def discover_usb_cameras(self) -> list[tuple[str, str]]:
        """Sonde les webcams USB (bloquant quelques secondes ; ouverture brève de chaque
        index sous ``ResourceLock`` côté lot A). Une webcam réservée est listée « occupée »."""
        if self._discovery is None:
            self.usb_cameras = []
        else:
            try:
                found = list(self._discovery())
            except Exception as exc:
                self.usb_cameras = []
                self.discovery_message = f"recherche impossible : {exc}"
                self._notify()
                return []
            self.usb_cameras = []
            for cam in found:
                loc = str(cam.location)
                status = str(getattr(cam, "status", ""))
                detail = str(getattr(cam, "detail", "") or "")
                label = f"Webcam {loc}"
                if status == "busy":
                    label += " — occupée" + (f" ({detail})" if detail else "")
                self.usb_cameras.append((loc, label))
            self.discovery_message = (
                f"{len(self.usb_cameras)} webcam(s) trouvée(s)" if self.usb_cameras
                else "aucune webcam trouvée"
            )
        self._notify()
        return list(self.usb_cameras)

    @property
    def engine_choices(self) -> list[tuple[str, str]]:
        """(identifiant, libellé). Sans moteur installé : une seule entrée explicite."""
        if not self._engines:
            label = ENGINE_UNAVAILABLE + (f" ({self.engine_detail})" if self.engine_detail else "")
            return [("", label)]
        return [("", "aucun moteur")] + [(e, e) for e in self._engines]

    def apply_engine_status(self, status: EngineStatus, engine_name: str | None) -> None:
        """Le moteur n'est proposé que si le processus vision le déclare disponible ;
        sinon la raison reçue (``detail``) est affichée."""
        if status.available and status.model is not None:
            self._engines = [engine_name or status.model.name]
        else:
            self._engines = []
        self.engine_detail = status.detail
        self._engine_reported = True
        self._notify()

    def current_source(self) -> SourceConfig:
        try:
            raw = self._store.get(KEY_SOURCE)
            return DEFAULT_SOURCE if raw is None else source_from_json(raw)
        except (ValueError, KeyError, TypeError) as exc:
            self.warning = f"Réglage caméra illisible, valeur par défaut utilisée ({exc})"
            return DEFAULT_SOURCE

    def load(self) -> SettingsForm:
        src = self.current_source()
        engine = self._store.get(KEY_ENGINE, "") or ""
        if engine and engine not in self._engines:
            if self._engine_reported:
                self.warning = f"Moteur « {engine} » enregistré mais non disponible"
            engine = ""
        volume = self._store.get(KEY_VOLUME, DEFAULT_VOLUME)
        if not isinstance(volume, int) or not 0 <= volume <= 100:
            volume = DEFAULT_VOLUME
        return SettingsForm(
            camera_id=str(src.camera_id),
            source_kind=str(src.kind),
            location=src.location,
            width="" if src.width is None else str(src.width),
            height="" if src.height is None else str(src.height),
            fps="" if src.fps is None else f"{src.fps:g}",
            engine=engine,
            volume=volume,
        )

    def save(self, form: SettingsForm) -> SourceConfig:
        cfg = validate_source(form)
        if form.engine and form.engine not in self._engines:
            raise ValidationError(f"Moteur indisponible : {form.engine}")
        if not 0 <= form.volume <= 100:
            raise ValidationError("Volume : entre 0 et 100")
        self._store.set(KEY_SOURCE, source_to_json(cfg))
        self._store.set(KEY_ENGINE, form.engine or None)
        self._store.set(KEY_VOLUME, int(form.volume))
        self.warning = None
        self._notify()
        return cfg
