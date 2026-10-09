from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import torch

from .security import ensure_child_path


def atomic_torch_save(payload: dict[str, Any], destination: Path, checkpoint_root: Path) -> None:
    """Write a checkpoint locally, fsync it, then atomically replace its name."""

    destination = ensure_child_path(checkpoint_root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary = ensure_child_path(checkpoint_root, temporary)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_torch_load(checkpoint: Path, checkpoint_root: Path, map_location: str | torch.device) -> dict[str, Any]:
    """Load only tensor/primitive checkpoint data from the controlled output root."""

    checkpoint = checkpoint.resolve(strict=True)
    ensure_child_path(checkpoint_root, checkpoint)
    payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    return payload


def atomic_json_save(payload: dict[str, Any], destination: Path, output_root: Path) -> None:
    destination = ensure_child_path(output_root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary = ensure_child_path(output_root, temporary)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prune_periodic_checkpoints(checkpoint_root: Path, keep: int) -> list[Path]:
    """Bound local recovery archives; never matches best.pt or last.pt."""

    if keep < 1:
        raise ValueError("At least one periodic recovery checkpoint must be retained")
    pattern = re.compile(r"^epoch_[0-9]{4,}\.pt$")
    periodic = sorted(
        (
            path
            for path in checkpoint_root.iterdir()
            if path.is_file() and pattern.fullmatch(path.name)
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    removed: list[Path] = []
    for old in periodic[keep:]:
        safe_old = ensure_child_path(checkpoint_root, old)
        safe_old.unlink()
        removed.append(safe_old)
    return removed
