"""Gestion des profils : création avec consentement OBLIGATOIRE, message d'accueil,
suppression complète en deux temps (demande puis confirmation en retapant le nom)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from maisonaccueil.contracts import ProfileId

from ..ports import DeletionSummary, ProfileRecord, ProfileStore
from .base import Observable, ValidationError
from .formatting import format_utc

MAX_NAME = 80
MAX_WELCOME = 300


@dataclass(frozen=True, slots=True)
class ProfileRow:
    profile_id: ProfileId
    display_name: str
    spoken_name: str
    welcome_text: str
    consent_text: str
    active: bool


@dataclass(frozen=True, slots=True)
class ProfileForm:
    display_name: str
    consent_given: bool  # case « la personne a donné son consentement » (obligatoire)
    consent_note: str = ""
    spoken_name: str = ""
    welcome_text: str = ""


@dataclass(frozen=True, slots=True)
class DeletionRequest:
    profile_id: ProfileId
    display_name: str

    @property
    def confirmation_text(self) -> str:
        return (
            f"Supprimer définitivement « {self.display_name} » ? Seront effacés : le profil, "
            "ses images de référence (fichiers compris), ses représentations faciales, ses "
            "séances de collecte et ses entrées du journal d'arrivées. Action irréversible. "
            "Retapez le nom pour confirmer."
        )


def _clean(value: str, limit: int, label: str) -> str | None:
    v = value.strip()
    if len(v) > limit:
        raise ValidationError(f"{label} : {limit} caractères au maximum")
    return v or None


class ProfilesViewModel(Observable):
    def __init__(
        self,
        store: ProfileStore,
        *,
        now_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        super().__init__()
        self._store = store
        self._now = now_utc
        self._rows: list[ProfileRow] = []
        self._names: dict[ProfileId, str] = {}
        self.error: str | None = None

    # -- lecture --------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            profiles: list[ProfileRecord] = list(self._store.list(include_inactive=True))
        except Exception as exc:
            self.error = f"profils indisponibles : {exc}"
            self._notify()
            return
        self.error = None
        self._rows = [self._row(p) for p in profiles]
        self._names = {r.profile_id: r.display_name for r in self._rows}
        self._notify()

    @property
    def rows(self) -> list[ProfileRow]:
        return list(self._rows)

    def name_of(self, profile_id: ProfileId) -> str | None:
        return self._names.get(profile_id)

    def get(self, profile_id: ProfileId) -> ProfileRow | None:
        return next((r for r in self._rows if r.profile_id == profile_id), None)

    # -- écriture -------------------------------------------------------------------
    def create(self, form: ProfileForm) -> ProfileRow:
        if not form.consent_given:
            raise ValidationError(
                "Consentement obligatoire : la personne doit avoir accepté l'enregistrement "
                "de son visage avant toute création de profil."
            )
        name = _clean(form.display_name, MAX_NAME, "Nom")
        if name is None:
            raise ValidationError("Le nom est obligatoire.")
        record = self._store.create(
            name,
            consent_at=self._now(),
            consent_note=_clean(form.consent_note, MAX_WELCOME, "Note de consentement"),
            spoken_name=_clean(form.spoken_name, MAX_NAME, "Nom prononcé"),
            welcome_text=_clean(form.welcome_text, MAX_WELCOME, "Message d'accueil"),
        )
        self.refresh()
        return self._row(record)

    def update_welcome(
        self, profile_id: ProfileId, *, welcome_text: str, spoken_name: str = ""
    ) -> ProfileRow:
        record = self._store.update(
            profile_id,
            welcome_text=_clean(welcome_text, MAX_WELCOME, "Message d'accueil"),
            spoken_name=_clean(spoken_name, MAX_NAME, "Nom prononcé"),
        )
        self.refresh()
        return self._row(record)

    def rename(self, profile_id: ProfileId, display_name: str) -> ProfileRow:
        name = _clean(display_name, MAX_NAME, "Nom")
        if name is None:
            raise ValidationError("Le nom est obligatoire.")
        record = self._store.update(profile_id, display_name=name)
        self.refresh()
        return self._row(record)

    def request_deletion(self, profile_id: ProfileId) -> DeletionRequest:
        row = self.get(profile_id)
        if row is None:
            raise ValidationError("Profil introuvable.")
        return DeletionRequest(profile_id, row.display_name)

    def confirm_deletion(self, request: DeletionRequest, typed_name: str) -> str:
        """Suppression complète déléguée au stockage. Retourne un bilan lisible."""
        if typed_name.strip() != request.display_name:
            raise ValidationError("Le nom retapé ne correspond pas : suppression annulée.")
        report: DeletionSummary = self._store.delete(request.profile_id, purge_arrivals=True)
        self.refresh()
        text = (
            f"« {request.display_name} » supprimé : {report.references_deleted} image(s) de "
            f"référence, {report.embeddings_deleted} représentation(s), "
            f"{report.sessions_deleted} séance(s), {report.arrivals_deleted} entrée(s) du "
            f"journal, {report.files_deleted} fichier(s) effacé(s)."
        )
        if report.files_skipped:
            text += (
                f" Attention : {len(report.files_skipped)} fichier(s) non effacé(s) "
                "(hors du dossier géré ou erreur) : " + ", ".join(report.files_skipped)
            )
        return text

    # -- interne --------------------------------------------------------------------
    @staticmethod
    def _row(p: ProfileRecord) -> ProfileRow:
        return ProfileRow(
            profile_id=p.profile_id,
            display_name=p.display_name,
            spoken_name=p.spoken_name or "",
            welcome_text=p.welcome_text or "",
            consent_text=f"consentement le {format_utc(p.consent_at)}",
            active=p.active,
        )
