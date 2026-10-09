from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


class AccessWindowClosed(RuntimeError):
    """Raised between files when the production access window has closed."""


def check_access_deadline(deadline: datetime | None) -> None:
    if deadline is not None and datetime.now() >= deadline:
        raise AccessWindowClosed(f"Production access window closed at {deadline:%H:%M}; stopping before the next file.")


def assert_not_production_output(path: str | Path, config: dict) -> None:
    candidate = os.path.normcase(os.path.abspath(os.fspath(path)))
    for configured_root in config.get("read_only_roots", []):
        root = os.path.normcase(os.path.abspath(configured_root))
        try:
            inside = os.path.commonpath((candidate, root)) == root
        except ValueError:
            inside = False
        if inside:
            raise ValueError(
                f"Refusing to write inside read-only production location: {configured_root}"
            )
