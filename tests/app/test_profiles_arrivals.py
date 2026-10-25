from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from maisonaccueil.app.fakes import (
    FakeArrivalEntry,
    FakeArrivalHistory,
    FakeProfileStore,
    FakeReferenceSink,
)
from maisonaccueil.app.viewmodels import (
    ArrivalLogViewModel,
    ProfileForm,
    ProfilesViewModel,
    ValidationError,
)
from maisonaccueil.app.wiring import StorageHandle, open_storage_services
from maisonaccueil.contracts import (
    ArrivalEvent,
    ArrivalKind,
    EventId,
    ProfileId,
    TrackId,
)
from maisonaccueil.pipeline import VisionOutput


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[StorageHandle]:
    h = open_storage_services(tmp_path / "b.sqlite")
    yield h
    h.close()


def _form(**kw: object) -> ProfileForm:
    base: dict[str, object] = {"display_name": "Personne Test", "consent_given": True,
                               "welcome_text": "Bonjour"}
    base.update(kw)
    return ProfileForm(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["fake", "storage"])
def test_create_requires_consent(backend: str, storage: StorageHandle) -> None:
    store = FakeProfileStore() if backend == "fake" else storage.services.profiles
    vm = ProfilesViewModel(store)
    with pytest.raises(ValidationError, match="Consentement obligatoire"):
        vm.create(_form(consent_given=False))
    with pytest.raises(ValidationError, match="nom est obligatoire"):
        vm.create(_form(display_name="   "))
    vm.refresh()
    assert vm.rows == []
    row = vm.create(_form(spoken_name="Personne"))
    assert vm.rows == [row]
    assert row.welcome_text == "Bonjour"
    assert row.consent_text.startswith("consentement le ") and row.consent_text.endswith("UTC")
    assert vm.name_of(row.profile_id) == "Personne Test"


def test_length_limits() -> None:
    vm = ProfilesViewModel(FakeProfileStore())
    with pytest.raises(ValidationError):
        vm.create(_form(welcome_text="x" * 301))


def test_update_welcome_and_rename(storage: StorageHandle) -> None:
    vm = ProfilesViewModel(storage.services.profiles)
    row = vm.create(_form())
    row2 = vm.update_welcome(row.profile_id, welcome_text="  Salut  ", spoken_name="")
    assert row2.welcome_text == "Salut" and row2.spoken_name == ""
    assert vm.rename(row.profile_id, "Autre Nom").display_name == "Autre Nom"
    assert vm.name_of(row.profile_id) == "Autre Nom"


def test_deletion_requires_typed_name_and_reports(storage: StorageHandle,
                                                  fixed_utc: datetime) -> None:
    services = storage.services
    vm = ProfilesViewModel(services.profiles)
    row = vm.create(_form())
    refs = services.references
    sess = refs.create_session(row.profile_id, started_at=fixed_utc, split="enroll")
    refs.add_image_bytes(row.profile_id, b"\x89PNG-synthetique", captured_at=fixed_utc,
                         session_id=sess.session_id)
    req = vm.request_deletion(row.profile_id)
    assert "irréversible" in req.confirmation_text and "Personne Test" in req.confirmation_text
    with pytest.raises(ValidationError, match="ne correspond pas"):
        vm.confirm_deletion(req, "personne test")
    assert vm.get(row.profile_id) is not None
    text = vm.confirm_deletion(req, " Personne Test ")
    assert "1 image(s) de référence" in text and "1 séance(s)" in text
    assert "1 fichier(s) effacé(s)" in text
    assert vm.rows == []
    assert services.profiles.find(row.profile_id) is None
    with pytest.raises(ValidationError):
        vm.request_deletion(row.profile_id)


def test_fake_deletion_cascades_to_fake_refs(fixed_utc: datetime) -> None:
    profiles, refs = FakeProfileStore(), FakeReferenceSink()
    profiles.on_delete = refs.purge_profile
    vm = ProfilesViewModel(profiles)
    row = vm.create(_form())
    refs.add_image_bytes(row.profile_id, b"x", captured_at=fixed_utc)
    vm.confirm_deletion(vm.request_deletion(row.profile_id), "Personne Test")
    assert refs.references == {}


def test_refresh_error_is_displayed() -> None:
    class Broken(FakeProfileStore):
        def list(self, *, include_inactive: bool = False):  # type: ignore[no-untyped-def,override]
            raise OSError("disque absent")

    vm = ProfilesViewModel(Broken())
    vm.refresh()
    assert vm.error is not None and "disque absent" in vm.error


# -- journal des arrivées --------------------------------------------------------------

NAMES = {ProfileId("prf-a"): "Personne A"}


def _event(i: int, kind: ArrivalKind, pid: str | None, when: datetime) -> ArrivalEvent:
    return ArrivalEvent(EventId(f"evt-{i}"), kind, TrackId("trk"), i, when,
                        ProfileId(pid) if pid else None)


def test_arrivals_live_and_history(fixed_utc: datetime) -> None:
    history = FakeArrivalHistory([
        FakeArrivalEntry(EventId("evt-old"), ArrivalKind.KNOWN, fixed_utc - timedelta(hours=1),
                         ProfileId("prf-a")),
        FakeArrivalEntry(EventId("evt-del"), ArrivalKind.KNOWN, fixed_utc - timedelta(hours=2),
                         None),
    ])
    vm = ArrivalLogViewModel(NAMES.get, history)
    vm.load_history()
    vm.apply_vision_output(VisionOutput(1, 1, 1.0, arrivals=(
        _event(1, ArrivalKind.UNKNOWN, None, fixed_utc),
        _event(2, ArrivalKind.KNOWN, "prf-a", fixed_utc),
    )))
    vm.apply_event(_event(2, ArrivalKind.KNOWN, "prf-a", fixed_utc))  # doublon ignoré
    rows = vm.rows
    assert [r.event_id for r in rows] == ["evt-2", "evt-1", "evt-old", "evt-del"]
    assert rows[0].label == "Arrivée : Personne A" and rows[0].live
    assert rows[1].label == "Visage inconnu"
    assert rows[2].time_text == "2026-10-25 11:00:00 UTC" and not rows[2].live
    assert rows[3].label == "Personne enregistrée (profil supprimé)"


def test_arrivals_bounded(fixed_utc: datetime) -> None:
    vm = ArrivalLogViewModel(NAMES.get, max_rows=3)
    for i in range(5):
        vm.apply_event(_event(i, ArrivalKind.UNKNOWN, None, fixed_utc))
    assert [r.event_id for r in vm.rows] == ["evt-4", "evt-3", "evt-2"]
    vm.apply_event(_event(0, ArrivalKind.UNKNOWN, None, fixed_utc))  # sorti de la fenêtre
    assert vm.rows[0].event_id == "evt-0"


def test_arrivals_with_real_log(storage: StorageHandle, fixed_utc: datetime) -> None:
    from maisonaccueil.storage import ArrivalLogRepository

    log = storage.services.arrivals
    assert isinstance(log, ArrivalLogRepository)
    log.record(_event(9, ArrivalKind.UNKNOWN, None, fixed_utc), model_key="m", policy="p")
    vm = ArrivalLogViewModel(NAMES.get, log)
    vm.load_history()
    assert [r.label for r in vm.rows] == ["Visage inconnu"]


def test_arrivals_history_error() -> None:
    class Broken:
        def list_recent(self, limit: int = 100) -> list[object]:
            raise OSError("verrou")

    vm = ArrivalLogViewModel(NAMES.get, Broken())
    vm.load_history()
    assert vm.load_error == "journal indisponible : verrou"
