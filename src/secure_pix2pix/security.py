from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


class PathSecurityError(ValueError):
    """Raised before an unsafe filesystem operation can be attempted."""


def _canonical(path: str | Path, *, strict: bool) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve(strict=strict)
    except OSError as exc:
        raise PathSecurityError(f"Cannot resolve path safely: {candidate}") from exc


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _contains(parent: Path, child: Path) -> bool:
    try:
        return os.path.commonpath((_norm(parent), _norm(child))) == _norm(parent)
    except ValueError:
        # Different Windows drives cannot contain one another.
        return False


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def validate_readonly_directory(path: str | Path, *, label: str) -> Path:
    """Resolve and validate a source directory without writing to it."""

    resolved = _canonical(path, strict=True)
    if not resolved.is_dir():
        raise PathSecurityError(f"{label} is not a directory: {resolved}")
    return resolved


def validate_source_file(path: Path, source_root: Path) -> Path:
    """Ensure a discovered input file cannot escape its declared source root."""

    unresolved = Path(path)
    if _is_link_or_junction(unresolved):
        raise PathSecurityError(f"Symbolic links/junctions are forbidden in input data: {path}")
    resolved = _canonical(unresolved, strict=True)
    if not _contains(source_root, resolved):
        raise PathSecurityError(f"Input escaped its read-only root: {path}")
    if not resolved.is_file():
        raise PathSecurityError(f"Input is not a regular file: {resolved}")
    return resolved


def validate_write_root(
    path: str | Path,
    read_only_roots: Iterable[str | Path],
    *,
    create: bool = True,
) -> Path:
    """Reject any output/staging directory overlapping a protected source.

    Both directions are checked: a write root may neither be inside production
    nor be a parent which contains production.  The latter prevents a broad
    output path from accidentally encompassing a protected tree.
    """

    candidate = _canonical(path, strict=False)
    for configured_root in read_only_roots:
        protected = _canonical(configured_root, strict=False)
        if _contains(protected, candidate) or _contains(candidate, protected):
            raise PathSecurityError(
                f"Refusing write-root overlap: {candidate} conflicts with "
                f"read-only source {protected}"
            )

    if candidate.exists() and not candidate.is_dir():
        raise PathSecurityError(f"Write root is not a directory: {candidate}")
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def ensure_child_path(parent: Path, child: Path) -> Path:
    """Validate a generated destination before any local write."""

    parent_resolved = _canonical(parent, strict=False)
    child_resolved = _canonical(child, strict=False)
    if not _contains(parent_resolved, child_resolved):
        raise PathSecurityError(f"Generated path escaped write root: {child}")
    return child_resolved
