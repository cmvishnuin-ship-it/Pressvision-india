from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from correction_ml.metrics import read_rgb, resize_to
from correction_ml.pairs import ImagePair
from correction_ml.safety import check_access_deadline


INDEX_VERSION = 1


def _pair_identity(pair: ImagePair) -> str:
    """Identity key for one cached pair.

    KNOWN LIMITATION: identity is derived from (set_name, casefolded path, size,
    mtime), not file content. Renaming or moving a production file with
    identical bytes therefore creates a new cache record and permanently
    orphans the old one under this identity scheme (cache entries are never
    auto-evicted). A real fix requires migrating to content-hash-based
    identity plus an index migration; that is out of scope here.
    """
    source = Path(pair.original)
    target = Path(pair.corrected)
    source_stat = source.stat()
    target_stat = target.stat()
    identity = "|".join(
        [
            pair.set_name,
            str(source).casefold(),
            str(target).casefold(),
            str(source_stat.st_size),
            str(source_stat.st_mtime_ns),
            str(target_stat.st_size),
            str(target_stat.st_mtime_ns),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _load_index(cache_root: Path) -> dict:
    path = cache_root / "cache_index.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == INDEX_VERSION and isinstance(data.get("pairs"), dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"version": INDEX_VERSION, "pairs": {}}


def _save_index(cache_root: Path, index: dict) -> None:
    path = cache_root / "cache_index.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(index, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _record_pairs(cache_root: Path, record: dict, status: str) -> list[ImagePair]:
    if record.get("storage", "hot") == "replay":
        result = []
        for sample in record.get("replay_pairs", []):
            original = cache_root / sample["cached_original"]
            corrected = cache_root / sample["cached_corrected"]
            if not original.is_file() or not corrected.is_file():
                continue
            values = dict(sample["pair"])
            values.update(original=str(original), corrected=str(corrected), source_status=status)
            result.append(ImagePair(**values))
        return result
    original = cache_root / record["cached_original"]
    corrected = cache_root / record["cached_corrected"]
    if not original.is_file() or not corrected.is_file():
        return []
    values = dict(record["pair"])
    values["original"] = str(original)
    values["corrected"] = str(corrected)
    values["source_status"] = status
    return [ImagePair(**values)]


def _save_png_atomic(data: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    Image.fromarray(data).save(temporary, format="PNG", compress_level=3)
    with Image.open(temporary) as image:
        image.verify()
    os.replace(temporary, path)


def _fit_within(image: np.ndarray, maximum_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= maximum_side:
        return image
    scale = maximum_side / max(height, width)
    return resize_to(image, max(1, round(width * scale)), max(1, round(height * scale)))


def _replay_samples(source: np.ndarray, target: np.ndarray, patch_size: int, patch_count: int):
    height, width = target.shape[:2]
    source = resize_to(source, width, height)
    global_target = _fit_within(target, 512)
    global_source = resize_to(source, global_target.shape[1], global_target.shape[0])
    yield "global", global_source, global_target

    if min(height, width) < patch_size:
        scale = patch_size / min(height, width)
        width, height = max(patch_size, round(width * scale)), max(patch_size, round(height * scale))
        source = resize_to(source, width, height)
        target = resize_to(target, width, height)
    positions = [
        (0, 0),
        (0, width - patch_size),
        (height - patch_size, 0),
        (height - patch_size, width - patch_size),
        ((height - patch_size) // 2, (width - patch_size) // 2),
    ]
    for number, (top, left) in enumerate(positions[: max(0, patch_count)]):
        yield (
            f"patch_{number + 1}",
            source[top : top + patch_size, left : left + patch_size],
            target[top : top + patch_size, left : left + patch_size],
        )


def _compact_record(
    root: Path, pair_id: str, record: dict, patch_size: int, patch_count: int
) -> tuple[int, list[Path]] | None:
    hot_pairs = _record_pairs(root, record, "historical")
    if len(hot_pairs) != 1:
        return None
    hot_pair = hot_pairs[0]
    source_path, target_path = Path(hot_pair.original), Path(hot_pair.corrected)
    try:
        source = read_rgb(source_path, max_side=1536)
        target = read_rgb(target_path, max_side=1536)
        replay_dir = root / "replay" / pair_id
        samples = []
        for label, before, after in _replay_samples(source, target, patch_size, patch_count):
            cached_source = replay_dir / f"{label}_original.png"
            cached_target = replay_dir / f"{label}_corrected.png"
            _save_png_atomic(before, cached_source)
            _save_png_atomic(after, cached_target)
            replay_pair = replace(
                hot_pair,
                key=f"{hot_pair.key}#replay-{label}",
                original=str(cached_source),
                corrected=str(cached_target),
                original_width=before.shape[1],
                original_height=before.shape[0],
                corrected_width=after.shape[1],
                corrected_height=after.shape[0],
                aspect_change=0.0,
                source_status="historical",
            )
            samples.append(
                {
                    "pair": asdict(replay_pair),
                    "cached_original": str(cached_source.relative_to(root)),
                    "cached_corrected": str(cached_target.relative_to(root)),
                }
            )
        replay_bytes = sum(
            (root / sample[path_key]).stat().st_size
            for sample in samples
            for path_key in ("cached_original", "cached_corrected")
        )
        old_paths = [source_path, target_path]
        record["storage"] = "replay"
        record["parameter_alias"] = [str(source_path), str(target_path)]
        record["replay_pairs"] = samples
        record["bytes"] = replay_bytes
        record["compacted_at"] = datetime.now().isoformat(timespec="seconds")
        return replay_bytes, old_paths
    except Exception:
        replay_dir = root / "replay" / pair_id
        if replay_dir.exists():
            shutil.rmtree(replay_dir, ignore_errors=True)
        return None


def _compact_aging_hot_records(
    root: Path,
    records: dict[str, dict],
    protected_ids: set[str],
    hot_days: int,
    hot_max_bytes: int,
    replay_max_bytes: int,
    replay_patch_size: int,
    replay_patches_per_pair: int,
    hot_bytes: int,
    replay_bytes: int,
    now: str,
) -> tuple[int, int, int, int, int, int, list[Path]]:
    """Compact hot records aged past hot_days, or any hot record while the hot
    tier exceeds hot_max_bytes, into replay storage.

    Returns (hot_bytes, replay_bytes, used_bytes_delta, compacted,
    compaction_failures, replay_capacity_skips, obsolete_hot_files).
    """
    compacted = compaction_failures = replay_capacity_skips = 0
    used_bytes_delta = 0
    obsolete_hot_files: list[Path] = []
    cutoff = datetime.now() - timedelta(days=max(0, hot_days))
    candidates = sorted(
        (
            (pair_id, record)
            for pair_id, record in records.items()
            if record.get("storage", "hot") == "hot"
        ),
        key=lambda item: item[1].get("first_seen", ""),
    )
    for pair_id, record in candidates:
        # Preserve the exact local files used by the frozen pix2pix benchmark.
        # This is a local-cache retention rule and never touches production.
        if pair_id in protected_ids:
            continue
        try:
            first_seen = datetime.fromisoformat(record.get("first_seen", now))
        except ValueError:
            first_seen = datetime.now()
        if first_seen > cutoff and hot_bytes <= hot_max_bytes:
            continue
        old_record = dict(record)
        compacted_result = _compact_record(root, pair_id, record, replay_patch_size, replay_patches_per_pair)
        if compacted_result is None:
            compaction_failures += 1
            continue
        new_bytes, old_paths = compacted_result
        if replay_bytes + new_bytes > replay_max_bytes:
            replay_capacity_skips += 1
            replay_dir = root / "replay" / pair_id
            shutil.rmtree(replay_dir, ignore_errors=True)
            record.clear()
            record.update(old_record)
            continue
        old_bytes = int(old_record.get("bytes", 0))
        hot_bytes -= old_bytes
        replay_bytes += new_bytes
        used_bytes_delta += new_bytes - old_bytes
        obsolete_hot_files.extend(old_paths)
        compacted += 1
    return hot_bytes, replay_bytes, used_bytes_delta, compacted, compaction_failures, replay_capacity_skips, obsolete_hot_files


def update_training_cache(
    current_pairs: list[ImagePair],
    cache_root: str | Path,
    max_bytes: int,
    hot_days: int = 30,
    hot_max_bytes: int = 50 * 1024**3,
    replay_max_bytes: int = 150 * 1024**3,
    replay_patch_size: int = 192,
    replay_patches_per_pair: int = 4,
    protected_pair_ids: set[str] | None = None,
    deadline: datetime | None = None,
) -> tuple[list[ImagePair], dict, dict[tuple[str, str], tuple[str, str]]]:
    """Persist current pairs locally and return current plus historical cache pairs.

    Production paths are read only. Cache files are never automatically evicted;
    when the configured capacity is reached, new pairs are skipped and reported.
    With ``deadline`` it stops (AccessWindowClosed) before copying past it; the
    index is then left as it was, and the next ingest copies the rest.
    """
    root = Path(cache_root).resolve()
    protected_ids = set(protected_pair_ids or ())
    root.mkdir(parents=True, exist_ok=True)
    index = _load_index(root)
    records: dict[str, dict] = index["pairs"]
    used_bytes = sum(int(record.get("bytes", 0)) for record in records.values())
    now = datetime.now().isoformat(timespec="seconds")
    hot_bytes = replay_bytes = 0
    compacted = compaction_failures = replay_capacity_skips = 0
    obsolete_hot_files: list[Path] = []

    def _run_compaction_pass() -> None:
        nonlocal hot_bytes, replay_bytes, used_bytes, compacted, compaction_failures, replay_capacity_skips
        # Recompute fresh each call: the admission loop below adds new hot
        # records without maintaining hot_bytes/replay_bytes incrementally.
        current_hot_bytes = sum(int(record.get("bytes", 0)) for record in records.values() if record.get("storage", "hot") == "hot")
        current_replay_bytes = sum(int(record.get("bytes", 0)) for record in records.values() if record.get("storage") == "replay")
        (
            hot_bytes, replay_bytes, delta, pass_compacted, pass_failures, pass_skips, pass_obsolete,
        ) = _compact_aging_hot_records(
            root, records, protected_ids, hot_days, hot_max_bytes, replay_max_bytes,
            replay_patch_size, replay_patches_per_pair, current_hot_bytes, current_replay_bytes, now,
        )
        used_bytes += delta
        compacted += pass_compacted
        compaction_failures += pass_failures
        replay_capacity_skips += pass_skips
        obsolete_hot_files.extend(pass_obsolete)

    # First pass: compact already-aged/over-ceiling hot records *before*
    # admitting new current pairs below, so capacity freed here is available
    # to this run's admission pass (previously only visible on the next run).
    _run_compaction_pass()

    current_ids: set[str] = set()
    uncached_current: list[ImagePair] = []
    added = reused = skipped_capacity = copy_failures = 0

    for pair in current_pairs:
        check_access_deadline(deadline)
        try:
            pair_id = _pair_identity(pair)
            current_ids.add(pair_id)
            existing = records.get(pair_id)
            if existing is not None and _record_pairs(root, existing, "current"):
                existing["last_seen"] = now
                reused += 1
                continue

            source = Path(pair.original)
            target = Path(pair.corrected)
            required_bytes = source.stat().st_size + target.stat().st_size
            if used_bytes + required_bytes > max_bytes:
                skipped_capacity += 1
                uncached_current.append(replace(pair, source_status="current_uncached"))
                continue

            pair_dir = root / "pairs" / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            cached_source = pair_dir / f"original{source.suffix.casefold()}"
            cached_target = pair_dir / f"corrected{target.suffix.casefold()}"
            source_temp = cached_source.with_suffix(cached_source.suffix + ".partial")
            target_temp = cached_target.with_suffix(cached_target.suffix + ".partial")
            try:
                shutil.copy2(source, source_temp)
                shutil.copy2(target, target_temp)
                os.replace(source_temp, cached_source)
                os.replace(target_temp, cached_target)
            except Exception:
                for temporary in (source_temp, target_temp):
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                copy_failures += 1
                uncached_current.append(replace(pair, source_status="current_uncached"))
                continue

            cached_pair = replace(pair, source_status="current")
            records[pair_id] = {
                "pair": asdict(cached_pair),
                "production_original": pair.original,
                "production_corrected": pair.corrected,
                "cached_original": str(cached_source.relative_to(root)),
                "cached_corrected": str(cached_target.relative_to(root)),
                "storage": "hot",
                "bytes": required_bytes,
                "first_seen": now,
                "last_seen": now,
            }
            used_bytes += required_bytes
            added += 1
        except Exception:
            copy_failures += 1
            uncached_current.append(replace(pair, source_status="current_uncached"))

    # Second pass: compact any pair admitted just now that immediately pushes
    # the hot tier over its ceiling, matching the pre-existing same-run
    # compaction behavior for a freshly overflowing hot tier.
    _run_compaction_pass()

    index["updated_at"] = now
    index["max_bytes"] = max_bytes
    _save_index(root, index)
    for obsolete in obsolete_hot_files:
        try:
            obsolete.unlink(missing_ok=True)
        except OSError:
            pass

    cached_pairs = []
    parameter_aliases: dict[tuple[str, str], tuple[str, str]] = {}
    missing_cache_files = 0
    for pair_id, record in records.items():
        record_pairs = _record_pairs(root, record, "current" if pair_id in current_ids else "historical")
        if not record_pairs:
            missing_cache_files += 1
            continue
        cached_pairs.extend(record_pairs)
        alias = record.get("parameter_alias")
        if alias and len(alias) == 2:
            for cached in record_pairs:
                parameter_aliases[(cached.original, cached.corrected)] = (alias[0], alias[1])
    cached_pairs.extend(uncached_current)
    cached_pairs.sort(key=lambda pair: (pair.set_name.casefold(), pair.key.casefold(), pair.original.casefold()))
    report = {
        "cache_root": str(root),
        "maximum_bytes": max_bytes,
        "used_bytes": used_bytes,
        "used_gigabytes": used_bytes / (1024**3),
        "hot_bytes": hot_bytes,
        "hot_gigabytes": hot_bytes / (1024**3),
        "replay_bytes": replay_bytes,
        "replay_gigabytes": replay_bytes / (1024**3),
        "hot_pair_record_count": sum(record.get("storage", "hot") == "hot" for record in records.values()),
        "replay_pair_record_count": sum(record.get("storage") == "replay" for record in records.values()),
        "cached_pair_count": len(cached_pairs),
        "current_pair_count": sum(pair.source_status == "current" for pair in cached_pairs),
        "historical_pair_count": sum(pair.source_status == "historical" for pair in cached_pairs),
        "added_pair_count": added,
        "reused_pair_count": reused,
        "skipped_capacity_count": skipped_capacity,
        "copy_failure_count": copy_failures,
        "missing_cache_file_count": missing_cache_files,
        "current_uncached_count": len(uncached_current),
        "compacted_record_count": compacted,
        "compaction_failure_count": compaction_failures,
        "replay_capacity_skip_count": replay_capacity_skips,
        "hot_retention_days": hot_days,
        "hot_maximum_bytes": hot_max_bytes,
        "replay_maximum_bytes": replay_max_bytes,
        "replay_patch_size": replay_patch_size,
        "replay_patches_per_pair": replay_patches_per_pair,
        "automatic_full_image_deletion": "only after verified replay compaction",
        "protected_validation_record_count": sum(
            pair_id in protected_ids and record.get("storage", "hot") == "hot"
            for pair_id, record in records.items()
        ),
    }
    (root / "cache_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return cached_pairs, report, parameter_aliases
