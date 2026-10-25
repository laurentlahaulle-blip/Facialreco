"""Point d'entrée. Provisoire (socle) : le lot B le remplace par l'interface PySide6."""

from __future__ import annotations

import sys

from maisonaccueil import __version__
from maisonaccueil.contracts import CONTRACTS_VERSION, SCHEMA_VERSION


def main(argv: list[str] | None = None) -> int:
    print(
        f"MaisonAccueil {__version__} — contrats {CONTRACTS_VERSION}, schéma v{SCHEMA_VERSION}. "
        "Interface non encore intégrée."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
