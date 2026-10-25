"""Gestion des fichiers images de référence sous ``Database.media_root``.

Seuls les fichiers situés SOUS ``media_root`` sont créés ou supprimés par ce module ;
un chemin extérieur n'est jamais effacé (protection contre une suppression hors périmètre).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def write_atomic(path: Path, data: bytes) -> None:
    """Écrit ``data`` dans un fichier temporaire puis renomme (pas de fichier partiel)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def remove_managed_files(media_root: Path, paths: list[str]) -> tuple[int, list[str]]:
    """Supprime les fichiers listés s'ils sont sous ``media_root``.

    Retourne (nombre supprimé, chemins ignorés car hors ``media_root`` ou en erreur).
    Les dossiers parents devenus vides (jusqu'à ``media_root`` exclu) sont retirés.
    """
    removed = 0
    skipped: list[str] = []
    root = media_root.resolve()
    for raw in paths:
        p = Path(raw)
        if not is_within(root, p):
            skipped.append(raw)
            continue
        try:
            p.unlink(missing_ok=True)
            removed += 1
        except OSError:
            skipped.append(raw)
            continue
        parent = p.resolve().parent
        while parent != root and is_within(root, parent):
            try:
                parent.rmdir()  # échoue si non vide : on s'arrête
            except OSError:
                break
            parent = parent.parent
    return removed, skipped
