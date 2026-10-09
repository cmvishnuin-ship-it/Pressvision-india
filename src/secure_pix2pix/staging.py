from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .checkpoints import atomic_json_save
from .data import PairRecord
from .security import ensure_child_path, validate_write_root

MAX_STAGING_BYTES = 300 * 1000**3


class StagingWindowClosed(RuntimeError):
    """Raised before copying when the configured maintenance window is closed."""


def parse_clock(value: str) -> time:
    try:
        parsed = datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ValueError(f"Invalid 24-hour time {value!r}; expected HH:MM") from exc
    return parsed


@dataclass(frozen=True)
class CopyWindow:
    start: time
    end: time

    @classmethod
    def from_strings(cls, start: str, end: str) -> "CopyWindow":
        return cls(parse_clock(start), parse_clock(end))

    def contains(self, moment: datetime) -> bool:
        clock = moment.timetz().replace(tzinfo=None)
        if self.start < self.end:
            return self.start <= clock < self.end
        if self.start > self.end:  # Explicit support for overnight maintenance windows.
            return clock >= self.start or clock < self.end
        return False


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_source_sha256(path: Path) -> str:
    before = path.stat()
    digest = _sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while it was being verified: {path}")
    return digest


def _cheap_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _copy_verified_readonly_source(source: Path, destination: Path, staging_root: Path) -> str:
    """Copy one source to a local temp file and verify it was stable throughout."""

    destination = ensure_child_path(staging_root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    temporary = ensure_child_path(staging_root, temporary)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            while chunk := incoming.read(4 * 1024 * 1024):
                digest.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed during staging; pair not accepted: {source}")
        if temporary.stat().st_size != before.st_size or _sha256(temporary) != digest.hexdigest():
            raise RuntimeError(f"Staging verification failed: {source}")
        os.replace(temporary, destination)
        return digest.hexdigest()
    finally:
        if temporary.exists():
            temporary.unlink()


def stage_pairs(
    pairs: Sequence[PairRecord],
    staging_directory: str | Path,
    read_only_roots: Iterable[str | Path],
    copy_window: CopyWindow,
    *,
    now: Callable[[], datetime] | None = None,
    max_staging_bytes: int = MAX_STAGING_BYTES,
) -> tuple[Path, Path]:
    """Create a verified local stage without any write to source locations.

    The gate is checked once before creating the stage and again before every
    pair.  If the window closes, the function stops before starting another
    transfer; completed local pairs and the production sources remain intact.
    """

    clock = now or (lambda: datetime.now().astimezone())
    if not copy_window.contains(clock()):
        raise StagingWindowClosed("Staging copy is allowed only inside the configured maintenance window")

    if max_staging_bytes <= 0 or max_staging_bytes > MAX_STAGING_BYTES:
        raise ValueError("Staging is hard-limited to a maximum of 300 GB")
    staging_root = validate_write_root(staging_directory, read_only_roots, create=False)
    original_stage = ensure_child_path(staging_root, staging_root / "original")
    corrected_stage = ensure_child_path(staging_root, staging_root / "corrected")

    # Account for orphaned/older local stage files as well as this run. Capacity
    # is checked before any directory or copied file is created.
    projected_bytes = _tree_bytes(staging_root)
    for pair in pairs:
        pair_id = hashlib.sha256(pair.stable_key.encode("utf-8", "surrogatepass")).hexdigest()[:24]
        for source, destination in (
            (pair.original, original_stage / pair_id / pair.original.name),
            (pair.corrected, corrected_stage / pair_id / pair.corrected.name),
        ):
            if destination.exists() and destination.is_file():
                projected_bytes -= destination.stat().st_size
            projected_bytes += source.stat().st_size
    if projected_bytes > max_staging_bytes:
        raise RuntimeError(
            f"Staging would use {projected_bytes / 1000**3:.2f} GB; "
            "the hard safety limit is 300 GB"
        )

    staging_root.mkdir(parents=True, exist_ok=True)
    original_stage.mkdir(parents=True, exist_ok=True)
    corrected_stage.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    # Cheap size+mtime signatures from the previous run avoid a full re-hash
    # of both the staged copy and the (potentially large) production source
    # on every run when nothing has actually changed. Only fall back to a
    # full verify when there is no prior record or the signature differs.
    previous_manifest: dict[str, dict] = {}
    manifest_path = staging_root / "staging_manifest.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in payload.get("pairs", []):
                if isinstance(entry, dict) and isinstance(entry.get("pair_key"), str):
                    previous_manifest[entry["pair_key"]] = entry
        except (OSError, json.JSONDecodeError):
            previous_manifest = {}

    for pair in pairs:
        if not copy_window.contains(clock()):
            raise StagingWindowClosed("Maintenance window closed before the next pair copy")
        pair_id = hashlib.sha256(pair.stable_key.encode("utf-8", "surrogatepass")).hexdigest()[:24]
        source_destination = original_stage / pair_id / pair.original.name
        target_destination = corrected_stage / pair_id / pair.corrected.name

        source_signature = list(_cheap_signature(pair.original))
        target_signature = list(_cheap_signature(pair.corrected))
        previous_entry = previous_manifest.get(pair.stable_key)
        unchanged = (
            previous_entry is not None
            and source_destination.is_file()
            and target_destination.is_file()
            and previous_entry.get("original_source_signature") == source_signature
            and previous_entry.get("corrected_source_signature") == target_signature
        )
        if unchanged:
            source_hash = previous_entry.get("original_sha256")
            target_hash = previous_entry.get("corrected_sha256")
        else:
            source_hash = _sha256(source_destination) if source_destination.exists() else None
            target_hash = _sha256(target_destination) if target_destination.exists() else None
            current_source_hash = (
                _stable_source_sha256(pair.original) if source_destination.exists() else None
            )
            current_target_hash = (
                _stable_source_sha256(pair.corrected) if target_destination.exists() else None
            )
            if current_source_hash is None or source_hash != current_source_hash:
                source_hash = _copy_verified_readonly_source(
                    pair.original, source_destination, staging_root
                )
            if current_target_hash is None or target_hash != current_target_hash:
                target_hash = _copy_verified_readonly_source(
                    pair.corrected, target_destination, staging_root
                )
        entries.append(
            {
                "pair_key": pair.stable_key,
                "filename": pair.original.name,
                "original_sha256": source_hash,
                "corrected_sha256": target_hash,
                "original_source_signature": source_signature,
                "corrected_source_signature": target_signature,
            }
        )

    atomic_json_save(
        {
            "created_at": clock().isoformat(),
            "read_only_sources": True,
            "maximum_stage_bytes": max_staging_bytes,
            "projected_stage_bytes": projected_bytes,
            "pair_count": len(entries),
            "pairs": entries,
        },
        staging_root / "staging_manifest.json",
        staging_root,
    )
    return original_stage, corrected_stage
