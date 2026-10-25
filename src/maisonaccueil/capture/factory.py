"""Construction d'une ``FrameSource`` à partir d'une ``SourceConfig``."""

from __future__ import annotations

from typing import Any

from maisonaccueil.contracts import SourceConfig, SourceKind

from .base import ThreadedFrameSource
from .opencv_source import OpenCVSource
from .synthetic import PATTERNS, SyntheticSource


def create_source(config: SourceConfig, **options: Any) -> ThreadedFrameSource:
    """``SYNTHETIC`` → ``SyntheticSource`` (``location`` = motif, ``""`` = ``"bars"``) ;
    ``USB`` / ``NETWORK`` / ``FILE`` → ``OpenCVSource``. Les options sont transmises au
    constructeur (``policy``, ``drop_counter``, ``backend``, ``lock_dir``…).

    Le résultat respecte ``contracts.FrameSource``."""
    if config.kind == SourceKind.SYNTHETIC:
        pattern = config.location or "bars"
        if pattern not in PATTERNS:
            raise ValueError(f"Motif synthétique inconnu : {pattern!r}")
        kwargs: dict[str, Any] = {"pattern": pattern}
        if config.width:
            kwargs["width"] = config.width
        if config.height:
            kwargs["height"] = config.height
        if config.fps:
            kwargs["fps"] = config.fps
        kwargs.update(options)
        return SyntheticSource(config.camera_id, **kwargs)
    return OpenCVSource(config, **options)
