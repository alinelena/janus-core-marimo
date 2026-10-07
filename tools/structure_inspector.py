"""Structure and periodic boundary condition (PBC) validator CLI & module wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from janus_marimo.inspector import (  # noqa: E402
    StructureReport,
    inspect_structure,
    main,
    validate_neb_endpoints,
)

__all__ = [
    "StructureReport",
    "inspect_structure",
    "validate_neb_endpoints",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
