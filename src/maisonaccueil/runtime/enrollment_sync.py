"""Synchronisation de l'enrôlement : images de référence des séances ``enroll``
(lot C) → représentations du moteur courant (lot D), puis rechargement de la galerie.

Seules les séances au découpage ``enroll`` alimentent la galerie : les séances
``train``/``tune``/``test`` restent réservées à l'entraînement et à l'évaluation.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def enroll_pending(db: Any, enroller: Any, store: Any) -> Any:
    """Calcule les représentations manquantes ; retourne le ``EnrollmentReport``.

    ``db`` : ``storage.Database`` ; ``enroller`` : ``recognition.ReferenceEnroller`` ;
    ``store`` : ``EmbeddingStore``. L'idempotence repose sur
    ``store.represented_reference_ids`` quand il existe, sinon sur la mémoire de
    l'enrôleur (une instance par processus)."""
    from maisonaccueil.recognition.enrollment import ReferenceItem
    from maisonaccueil.storage import ProfileRepository, ReferenceRepository

    refs = ReferenceRepository(db)
    active = {p.profile_id for p in ProfileRepository(db).list()}
    items = [
        ReferenceItem(r.profile_id, r.reference_id, r.file_path)
        for s in refs.list_sessions(split="enroll")
        if s.profile_id in active
        for r in refs.list_for_session(s.session_id)
    ]
    report = enroller.enroll(items, store)
    if report.added or report.skipped:
        log.info("Enrôlement : %s", report.summary())
    return report
