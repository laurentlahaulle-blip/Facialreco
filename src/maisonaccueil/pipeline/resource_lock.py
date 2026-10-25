"""Réservation exclusive de ressources matérielles (caméra physique, GPU).

Les worktrees ne partagent pas de dossier ``var/`` : le verrou vit donc dans un
répertoire commun à toutes les sessions de la machine (``MAISONACCUEIL_LOCK_DIR`` ou,
par défaut, ``%LOCALAPPDATA%/MaisonAccueil/locks`` sous Windows,
``~/.cache/maisonaccueil/locks`` ailleurs). Un verrou dont le processus n'existe plus
est considéré comme périmé et peut être repris.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import TracebackType


class ResourceBusyError(RuntimeError):
    pass


def default_lock_dir() -> Path:
    env = os.environ.get("MAISONACCUEIL_LOCK_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "MaisonAccueil" / "locks"
    return Path.home() / ".cache" / "maisonaccueil" / "locks"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ResourceLock:
    """Usage : ``with ResourceLock("camera-0", owner="lot-A"): ...``"""

    def __init__(self, resource: str, owner: str, lock_dir: Path | None = None) -> None:
        if not resource.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Nom de ressource invalide")
        self.resource = resource
        self.owner = owner
        self.path = (lock_dir or default_lock_dir()) / f"{resource}.lock"
        self._held = False

    def holder(self) -> dict[str, object] | None:
        try:
            data: dict[str, object] = json.loads(self.path.read_text(encoding="utf-8"))
            return data
        except (FileNotFoundError, ValueError):
            return None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                info = self.holder()
                pid = info.get("pid") if info else None
                if isinstance(pid, int) and not _pid_alive(pid):
                    self.path.unlink(missing_ok=True)  # verrou périmé
                    continue
                raise ResourceBusyError(f"{self.resource} déjà réservé par {info}") from None
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "owner": self.owner, "since": time.time()}, f)
            self._held = True
            return
        raise ResourceBusyError(f"Impossible de réserver {self.resource}")

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> ResourceLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
