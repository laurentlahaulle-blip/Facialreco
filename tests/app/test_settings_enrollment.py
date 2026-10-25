from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from maisonaccueil.app.fakes import FakeProfileStore, FakeReferenceSink, FakeSettingsStore
from maisonaccueil.app.messages import EnrollmentSample
from maisonaccueil.app.viewmodels import (
    ENGINE_UNAVAILABLE,
    GUIDANCE_STEPS,
    EnrollmentPhase,
    EnrollmentViewModel,
    SettingsForm,
    SettingsViewModel,
    ValidationError,
)
from maisonaccueil.app.viewmodels.settings import KEY_SOURCE, KEY_VOLUME
from maisonaccueil.app.wiring import StorageHandle, open_storage_services
from maisonaccueil.contracts import SourceKind


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[StorageHandle]:
    h = open_storage_services(tmp_path / "b.sqlite")
    yield h
    h.close()


def _sform(**kw: object) -> SettingsForm:
    base: dict[str, object] = {"camera_id": "camera-0", "source_kind": "usb", "location": "0"}
    base.update(kw)
    return SettingsForm(**base)  # type: ignore[arg-type]


# -- réglages ---------------------------------------------------------------------------


def test_settings_defaults_without_engine() -> None:
    vm = SettingsViewModel(FakeSettingsStore())
    form = vm.load()
    assert (form.source_kind, form.location, form.volume) == ("usb", "0", 70)
    assert vm.engine_choices == [("", ENGINE_UNAVAILABLE)]


def test_settings_roundtrip_with_storage(storage: StorageHandle) -> None:
    store = storage.services.settings
    vm = SettingsViewModel(store, ["ref-engine"])
    cfg = vm.save(_sform(source_kind="network", location="rtsp://192.168.1.20:554/flux",
                         width="1280", height="720", fps="15,5", engine="ref-engine",
                         volume=40))
    assert cfg.kind == SourceKind.NETWORK and cfg.width == 1280 and cfg.fps == 15.5
    assert store.get(KEY_VOLUME) == 40
    assert store.get(KEY_SOURCE)["location"] == "rtsp://192.168.1.20:554/flux"
    again = SettingsViewModel(store, ["ref-engine"]).load()
    assert again.fps == "15.5" and again.engine == "ref-engine" and again.width == "1280"
    # moteur enregistré mais disparu : signalé, pas utilisé
    vm2 = SettingsViewModel(store, [])
    assert vm2.load().engine == "" and vm2.warning is not None


@pytest.mark.parametrize("location", [
    "rtsp://8.8.8.8/flux", "http://camera.example.com/video", "ftp://192.168.1.2/x", "rtsp://",
])
def test_network_camera_must_be_local(location: str) -> None:
    with pytest.raises(ValidationError):
        SettingsViewModel(FakeSettingsStore()).save(_sform(source_kind="network",
                                                           location=location))


@pytest.mark.parametrize("location", [
    "rtsp://127.0.0.1/x", "http://localhost:8080/v", "rtsp://cam-entree.local/s",
    "rtsp://cam/s", "http://[fe80::1]/v", "rtsp://10.0.0.5/s",
])
def test_network_camera_local_accepted(location: str) -> None:
    SettingsViewModel(FakeSettingsStore()).save(_sform(source_kind="network", location=location))


@pytest.mark.parametrize("kw", [
    {"location": "abc"}, {"camera_id": "cam 0"}, {"source_kind": "bluetooth"},
    {"width": "x"}, {"fps": "500"}, {"volume": 101}, {"engine": "absent"},
    {"source_kind": "file", "location": "/nexiste/pas.mp4"},
])
def test_settings_validation(kw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SettingsViewModel(FakeSettingsStore()).save(_sform(**kw))


def test_corrupt_source_setting_falls_back() -> None:
    vm = SettingsViewModel(FakeSettingsStore({KEY_SOURCE: {"kind": "?"}}))
    assert vm.load().source_kind == "usb"
    assert vm.warning is not None


def test_usb_discovery_injected() -> None:
    from dataclasses import dataclass

    @dataclass
    class FakeUsbInfo:
        location: str
        status: str
        detail: str = ""

    vm = SettingsViewModel(FakeSettingsStore(), camera_discovery=lambda: [
        FakeUsbInfo("0", "available"), FakeUsbInfo("1", "busy", "réservée par lot-A")])
    assert vm.discovery_available
    assert vm.discover_usb_cameras() == [
        ("0", "Webcam 0"), ("1", "Webcam 1 — occupée (réservée par lot-A)")]
    assert vm.discovery_message == "2 webcam(s) trouvée(s)"

    def broken() -> list[object]:
        raise OSError("pilote")

    vm2 = SettingsViewModel(FakeSettingsStore(), camera_discovery=broken)
    assert vm2.discover_usb_cameras() == [] and "pilote" in vm2.discovery_message
    vm3 = SettingsViewModel(FakeSettingsStore())
    assert not vm3.discovery_available and vm3.discover_usb_cameras() == []
    assert "non disponible" in vm3.discovery_message


# -- collecte guidée --------------------------------------------------------------------


def _sample(i: int = 0) -> EnrollmentSample:
    return EnrollmentSample(i, b"\x89PNG-synthetique-" + bytes([i]), ".png", 112, 112, "net")


def _enroll_vm(clock, profiles, sink, fixed_utc, **kw):  # type: ignore[no-untyped-def]
    return EnrollmentViewModel(profiles, sink, clock=clock, now_utc=lambda: fixed_utc, **kw)


def test_enrollment_flow_with_storage(storage: StorageHandle, clock, fixed_utc) -> None:  # type: ignore[no-untyped-def]
    services = storage.services
    pid = services.profiles.create("Personne Test", consent_at=fixed_utc).profile_id
    vm = _enroll_vm(clock, services.profiles, services.references, fixed_utc, target_images=3)
    assert vm.phase == EnrollmentPhase.IDLE
    assert not vm.add_sample(_sample())  # hors collecte : ignoré
    vm.select_profile(pid)
    assert vm.phase == EnrollmentPhase.READY and "consentement" in vm.message
    with pytest.raises(ValidationError, match="caméra non disponible"):
        vm.start()
    vm.set_availability(camera=True, engine=False)
    assert ENGINE_UNAVAILABLE in vm.availability_text
    sid = vm.start("lumière du jour")
    session = services.references.get_session(sid)
    assert session.split == "enroll" and session.conditions == "lumière du jour"
    assert vm.current_step == GUIDANCE_STEPS[0]
    clock.advance_ms(6_000)
    vm.tick()
    assert vm.current_step_index == 1 and "Étape 2/4" in vm.message
    for i in range(3):
        assert vm.add_sample(_sample(i))
    assert vm.phase == EnrollmentPhase.DONE and vm.progress == 1.0
    stored = services.references.list_for_session(sid)
    assert len(stored) == 3 and all(r.quality_note == "net" for r in stored)
    assert vm.rejected_samples == 1


def test_enrollment_cancel_removes_images(storage: StorageHandle, clock, fixed_utc) -> None:  # type: ignore[no-untyped-def]
    services = storage.services
    pid = services.profiles.create("Personne Test", consent_at=fixed_utc).profile_id
    vm = _enroll_vm(clock, services.profiles, services.references, fixed_utc)
    vm.set_availability(camera=True, engine=False)
    vm.select_profile(pid)
    vm.start()
    vm.add_sample(_sample(1))
    vm.add_sample(_sample(2))
    assert vm.cancel() == 2
    assert services.references.list_for_profile(pid) == []
    assert vm.phase == EnrollmentPhase.CANCELLED


def test_enrollment_timeout_and_guards(clock, fixed_utc) -> None:  # type: ignore[no-untyped-def]
    profiles, sink = FakeProfileStore(), FakeReferenceSink()
    p = profiles.create("Personne Test", consent_at=fixed_utc)
    vm = _enroll_vm(clock, profiles, sink, fixed_utc, max_duration_s=30)
    with pytest.raises(ValidationError):
        vm.start()
    with pytest.raises(ValidationError):
        vm.select_profile("prf-inconnu")  # type: ignore[arg-type]
    vm.set_availability(camera=True, engine=True)
    vm.select_profile(p.profile_id)
    vm.start()
    with pytest.raises(ValidationError):
        vm.select_profile(p.profile_id)
    vm.add_sample(_sample())
    clock.advance_ms(31_000)
    vm.tick()
    assert vm.phase == EnrollmentPhase.DONE and "Durée maximale" in vm.message
    with pytest.raises(ValidationError):
        vm.finish()


def test_enrollment_refuses_inactive_profile(clock, fixed_utc) -> None:  # type: ignore[no-untyped-def]
    profiles = FakeProfileStore()
    p = profiles.create("Personne Test", consent_at=fixed_utc)
    profiles.profiles[p.profile_id] = p.__class__(p.profile_id, p.display_name, p.consent_at,
                                                  active=False)
    vm = _enroll_vm(clock, profiles, FakeReferenceSink(), fixed_utc)
    with pytest.raises(ValidationError, match="désactivé"):
        vm.select_profile(p.profile_id)


def test_enrollment_manual_finish(clock, fixed_utc: datetime) -> None:  # type: ignore[no-untyped-def]
    profiles, sink = FakeProfileStore(), FakeReferenceSink()
    p = profiles.create("Personne Test", consent_at=fixed_utc)
    vm = _enroll_vm(clock, profiles, sink, fixed_utc)
    vm.set_availability(camera=True, engine=False)
    vm.select_profile(p.profile_id)
    vm.start()
    vm.add_sample(_sample())
    vm.finish()
    assert vm.phase == EnrollmentPhase.DONE and vm.collected == 1
    assert next(iter(sink.sessions.values())).split == "enroll"
