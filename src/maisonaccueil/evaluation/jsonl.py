"""Lecture/écriture JSON Lines avec erreurs localisées (fichier:ligne)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class FormatError(ValueError):
    """Fichier d'annotation ou journal mal formé (message en français, avec la ligne)."""


def read_jsonl(path: Path | str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Itère sur ``(numéro de ligne, objet)``. Lignes vides et lignes ``#`` ignorées."""
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise FormatError(f"{p}:{lineno}: JSON invalide ({exc.msg})") from None
            if not isinstance(obj, dict):
                raise FormatError(f"{p}:{lineno}: un objet JSON est attendu")
            yield lineno, obj


def dumps(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def write_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(dumps(rec) + "\n")


def append_jsonl(path: Path | str, record: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(dumps(record) + "\n")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value: str, where: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        raise FormatError(f"{where}: horodatage ISO 8601 invalide {value!r}") from None
    if dt.tzinfo is None:
        raise FormatError(f"{where}: horodatage sans fuseau (UTC attendu) {value!r}")
    return dt.astimezone(UTC)


def require(obj: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if key not in obj:
        raise FormatError(f"{where}: champ obligatoire manquant {key!r}")
    value = obj[key]
    if isinstance(value, bool) and kind in (int, float, (int, float)):
        raise FormatError(f"{where}: {key!r} doit être numérique")
    if not isinstance(value, kind):
        raise FormatError(f"{where}: {key!r} a un type invalide ({type(value).__name__})")
    return value


def optional(
    obj: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str, default: Any
) -> Any:
    if obj.get(key) is None:
        return default
    return require(obj, key, kind, where)
