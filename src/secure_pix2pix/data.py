from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .security import validate_readonly_directory, validate_source_file

SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})


@dataclass(frozen=True)
class PairRecord:
    original: Path
    corrected: Path
    original_relative: Path
    corrected_relative: Path
    group_key: str = ""

    @property
    def stable_key(self) -> str:
        return f"{self.original_relative.as_posix()}|{self.corrected_relative.as_posix()}"

    @property
    def split_group(self) -> str:
        return self.group_key or self.stable_key


@dataclass(frozen=True)
class PairingReport:
    pairs: tuple[PairRecord, ...]
    original_images: int
    corrected_images: int
    unmatched_originals: int
    ambiguous_filenames: tuple[str, ...]


@dataclass(frozen=True)
class CachePairingReport:
    pairs: tuple[PairRecord, ...]
    cache_records: int
    accepted_records: int
    replay_samples: int
    rejected_match_method: int
    rejected_filename: int
    rejected_geometry: int
    missing_files: int


def _discover_images(root: Path) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    for unresolved in root.rglob("*"):
        if not unresolved.is_file() or unresolved.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        resolved = validate_source_file(unresolved, root)
        found.append((resolved, resolved.relative_to(root)))
    return sorted(found, key=lambda item: item[1].as_posix())


def build_identical_filename_pairs(
    original_root: str | Path,
    corrected_root: str | Path,
) -> PairingReport:
    """Pair images by identical filename, preferring an identical relative path.

    If the same filename occurs in multiple corrected subdirectories, a source
    is accepted only when its identical relative path exists.  Ambiguous matches
    are never guessed, which prevents silently training on the wrong target.
    """

    original_base = validate_readonly_directory(original_root, label="Original root")
    corrected_base = validate_readonly_directory(corrected_root, label="Corrected root")
    originals = _discover_images(original_base)
    corrected = _discover_images(corrected_base)

    by_name: dict[str, list[tuple[Path, Path]]] = {}
    by_relative: dict[str, tuple[Path, Path]] = {}
    for resolved, relative in corrected:
        by_name.setdefault(resolved.name, []).append((resolved, relative))
        by_relative[relative.as_posix()] = (resolved, relative)

    pairs: list[PairRecord] = []
    ambiguous: set[str] = set()
    unmatched = 0
    for source, source_relative in originals:
        # Matching on the full relative path already guarantees the filename matches.
        exact_relative = by_relative.get(source_relative.as_posix())
        if exact_relative is not None:
            target, target_relative = exact_relative
        else:
            candidates = by_name.get(source.name, [])
            if len(candidates) != 1:
                unmatched += 1
                if len(candidates) > 1:
                    ambiguous.add(source.name)
                continue
            target, target_relative = candidates[0]

        pairs.append(
            PairRecord(
                original=source,
                corrected=target,
                original_relative=source_relative,
                corrected_relative=target_relative,
            )
        )

    return PairingReport(
        pairs=tuple(pairs),
        original_images=len(originals),
        corrected_images=len(corrected),
        unmatched_originals=unmatched,
        ambiguous_filenames=tuple(sorted(ambiguous)),
    )


def deterministic_split(
    pairs: Sequence[PairRecord], validation_fraction: float, seed: int
) -> tuple[list[PairRecord], list[PairRecord]]:
    if len(pairs) < 2:
        raise ValueError("At least two secure image pairs are required for train/validation splitting")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    groups: dict[str, list[PairRecord]] = {}
    for pair in pairs:
        groups.setdefault(pair.split_group, []).append(pair)
    if len(groups) < 2:
        raise ValueError("At least two independent pair groups are required for splitting")

    def ranking(group_key: str) -> bytes:
        payload = f"{seed}:{group_key}".encode("utf-8", "surrogatepass")
        return hashlib.sha256(payload).digest()

    ordered_groups = sorted(groups, key=ranking)
    validation_group_count = max(
        1, min(len(ordered_groups) - 1, round(len(ordered_groups) * validation_fraction))
    )
    validation_keys = set(ordered_groups[:validation_group_count])
    training = [pair for pair in pairs if pair.split_group not in validation_keys]
    validation = [pair for pair in pairs if pair.split_group in validation_keys]
    return training, validation


def split_with_frozen_validation(
    pairs: Sequence[PairRecord],
    validation_fraction: float,
    seed: int,
    frozen_group_keys: Sequence[str] | None = None,
) -> tuple[list[PairRecord], list[PairRecord], list[str]]:
    """Create or reuse a pair-group validation benchmark.

    Once persisted by the caller, the validation groups stay fixed as new cache
    records arrive. This makes PSNR/SSIM comparisons across daily sessions valid.
    """

    available = {pair.split_group for pair in pairs}
    if frozen_group_keys is None:
        training, validation = deterministic_split(pairs, validation_fraction, seed)
        selected = sorted({pair.split_group for pair in validation})
        return training, validation, selected

    selected = sorted({str(key) for key in frozen_group_keys if str(key) in available})
    if not selected:
        raise ValueError("The frozen validation groups are absent from the current cache")
    selected_set = set(selected)
    training = [pair for pair in pairs if pair.split_group not in selected_set]
    validation = [pair for pair in pairs if pair.split_group in selected_set]
    if not training or not validation:
        raise ValueError("Frozen validation must leave at least one training and validation pair")
    return training, validation, selected


def _production_filename(value: object) -> str:
    # PurePath selection is platform-specific; this trainer is deployed on Windows.
    return Path(str(value)).name


def load_verified_cache_pairs(
    cache_index: str | Path,
    *,
    exact_filename_only: bool = True,
    max_aspect_change: float = 0.02,
) -> CachePairingReport:
    """Read verified pairs directly from the V2 local cache index.

    The cache and its images are treated as read-only inputs. No production path
    is opened: only paths relative to the local cache root are accepted.
    """

    index_path = Path(cache_index).resolve(strict=True)
    cache_root = index_path.parent.resolve(strict=True)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read cache index safely: {index_path}") from exc
    records = payload.get("pairs")
    if payload.get("version") != 1 or not isinstance(records, dict):
        raise ValueError("Unsupported cache index format")

    accepted: list[PairRecord] = []
    accepted_records = replay_samples = rejected_method = rejected_name = 0
    rejected_geometry = missing_files = 0
    for pair_id, record in sorted(records.items()):
        if not isinstance(pair_id, str) or not isinstance(record, dict):
            continue
        pair_metadata = record.get("pair", {})
        if not isinstance(pair_metadata, dict):
            continue
        if pair_metadata.get("match_method") != "filename_and_visual":
            rejected_method += 1
            continue
        if exact_filename_only:
            original_name = _production_filename(record.get("production_original", ""))
            corrected_name = _production_filename(record.get("production_corrected", ""))
            if not original_name or original_name.casefold() != corrected_name.casefold():
                rejected_name += 1
                continue
        try:
            aspect_change = float(pair_metadata.get("aspect_change", 1.0))
        except (TypeError, ValueError):
            aspect_change = 1.0
        if aspect_change > max_aspect_change:
            rejected_geometry += 1
            continue

        samples: list[tuple[str, str]] = []
        if record.get("storage", "hot") == "replay":
            for sample in record.get("replay_pairs", []):
                if isinstance(sample, dict):
                    samples.append(
                        (str(sample.get("cached_original", "")), str(sample.get("cached_corrected", "")))
                    )
        else:
            samples.append(
                (str(record.get("cached_original", "")), str(record.get("cached_corrected", "")))
            )
        record_accepted = False
        for original_relative_text, corrected_relative_text in samples:
            if not original_relative_text or not corrected_relative_text:
                missing_files += 1
                continue
            original_relative = Path(original_relative_text)
            corrected_relative = Path(corrected_relative_text)
            try:
                original = validate_source_file(cache_root / original_relative, cache_root)
                corrected = validate_source_file(cache_root / corrected_relative, cache_root)
            except (FileNotFoundError, ValueError):
                missing_files += 1
                continue
            accepted.append(
                PairRecord(
                    original=original,
                    corrected=corrected,
                    original_relative=original.relative_to(cache_root),
                    corrected_relative=corrected.relative_to(cache_root),
                    group_key=pair_id,
                )
            )
            record_accepted = True
            if record.get("storage", "hot") == "replay":
                replay_samples += 1
        if record_accepted:
            accepted_records += 1

    return CachePairingReport(
        pairs=tuple(accepted),
        cache_records=len(records),
        accepted_records=accepted_records,
        replay_samples=replay_samples,
        rejected_match_method=rejected_method,
        rejected_filename=rejected_name,
        rejected_geometry=rejected_geometry,
        missing_files=missing_files,
    )


def dataset_fingerprint(pairs: Sequence[PairRecord]) -> str:
    digest = hashlib.sha256()
    for pair in sorted(pairs, key=lambda item: item.stable_key):
        for file_path, relative in (
            (pair.original, pair.original_relative),
            (pair.corrected, pair.corrected_relative),
        ):
            stat = file_path.stat()
            digest.update(relative.as_posix().encode("utf-8", "surrogatepass"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _load_rgb_tensor(path: Path, image_size: int, max_pixels: int) -> torch.Tensor:
    # Treat decompression warnings as errors and check dimensions before decode.
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with path.open("rb") as handle:
            with Image.open(handle) as opened:
                width, height = opened.size
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise ValueError(
                        f"Unsafe image dimensions {width}x{height} for read-only input {path}"
                    )
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
                array = np.asarray(image, dtype=np.float32)
    # HWC [0,255] -> contiguous CHW [-1,1], compatible with generator Tanh.
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(array).mul_(1.0 / 127.5).sub_(1.0)


class PairedStyleDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Read-only paired dataset with deterministic, aligned transforms."""

    def __init__(
        self,
        pairs: Sequence[PairRecord],
        *,
        image_size: int = 256,
        max_image_megapixels: float = 120.0,
    ) -> None:
        if image_size != 256:
            raise ValueError("This pix2pix pipeline requires exactly 256x256 training images")
        if max_image_megapixels <= 0:
            raise ValueError("max_image_megapixels must be positive")
        self.pairs = tuple(pairs)
        self.image_size = image_size
        self.max_pixels = int(max_image_megapixels * 1_000_000)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.pairs[index]
        original = _load_rgb_tensor(pair.original, self.image_size, self.max_pixels)
        corrected = _load_rgb_tensor(pair.corrected, self.image_size, self.max_pixels)
        return original, corrected, pair.stable_key
