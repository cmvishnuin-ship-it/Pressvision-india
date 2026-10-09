from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .safety import check_access_deadline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImagePair:
    set_name: str
    key: str
    original: str
    corrected: str
    original_width: int
    original_height: int
    corrected_width: int
    corrected_height: int
    aspect_change: float
    time_difference_hours: float
    split: str
    match_method: str = "filename_and_visual"
    visual_similarity: float = 1.0
    source_status: str = "current"


@dataclass(frozen=True)
class VisualSignature:
    width: int
    height: int
    normalized_luma: np.ndarray
    normalized_edges: np.ndarray
    perceptual_hash: np.ndarray
    color_histogram: np.ndarray
    descriptors: np.ndarray | None


def load_config(path: str | Path = "config.json") -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_stem(path: Path) -> str:
    value = path.stem.casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).casefold(),
    )


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def _normalized_vector(values: np.ndarray) -> np.ndarray:
    vector = values.astype(np.float32).reshape(-1)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-6)


def visual_signature(path: Path) -> VisualSignature:
    """Create a compact content signature robust to normal photo corrections."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luma = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    gradient_x = cv2.Sobel(luma.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luma.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)

    dct = cv2.dct(luma.astype(np.float32) / 255.0)[:8, :8]
    perceptual_hash = dct > np.median(dct[1:])
    histograms = []
    for channel in range(3):
        histogram = cv2.calcHist([rgb], [channel], None, [16], [0, 256]).reshape(-1)
        histogram /= max(1.0, float(histogram.sum()))
        histograms.append(histogram)

    orb = cv2.ORB_create(nfeatures=300, fastThreshold=12)
    _, descriptors = orb.detectAndCompute(gray, None)
    return VisualSignature(
        width=width,
        height=height,
        normalized_luma=_normalized_vector(luma),
        normalized_edges=_normalized_vector(edges),
        perceptual_hash=perceptual_hash,
        color_histogram=np.concatenate(histograms),
        descriptors=descriptors,
    )


def _orb_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or len(left) < 2 or len(right) < 2:
        return 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left, right, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    denominator = max(12.0, min(float(len(left)), float(len(right)), 60.0))
    return min(1.0, len(good) / denominator)


def visual_similarity(left: VisualSignature, right: VisualSignature, include_orb: bool = True) -> float:
    """Return 0..1 similarity based on layout, edges, pHash, colour, and ORB."""
    luma = float(np.clip(np.dot(left.normalized_luma, right.normalized_luma), -1, 1))
    edge = float(np.clip(np.dot(left.normalized_edges, right.normalized_edges), -1, 1))
    luma = 0.5 * (luma + 1.0)
    edge = 0.5 * (edge + 1.0)
    phash = 1.0 - float(np.mean(left.perceptual_hash != right.perceptual_hash))
    color = float(np.minimum(left.color_histogram, right.color_histogram).sum() / 3.0)
    base = 0.38 * luma + 0.24 * edge + 0.20 * phash + 0.08 * color
    if not include_orb:
        return float(np.clip(base / 0.90, 0, 1))
    orb = _orb_similarity(left.descriptors, right.descriptors)
    return float(np.clip(base + 0.10 * orb, 0, 1))


def _stable_split(key: str, validation_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    number = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if number < validation_fraction else "train"


def discover_pairs(config: dict, deadline: datetime | None = None) -> tuple[list[ImagePair], list[dict]]:
    """Match originals to corrected images. With ``deadline`` it stops (AccessWindowClosed) before reading past it."""
    max_seconds = float(config.get("max_pair_age_days", 3)) * 86400.0
    validation_fraction = float(config.get("validation_fraction", 0.15))
    seed = int(config.get("seed", 42))
    visual_enabled = bool(config.get("visual_matching", True))
    visual_threshold = float(config.get("visual_match_threshold", 0.76))
    filename_threshold = float(config.get("visual_filename_threshold", 0.58))
    visual_margin = float(config.get("visual_match_margin", 0.045))
    visual_max_aspect = float(config.get("visual_match_max_aspect_change", 0.10))
    visual_candidate_limit = max(1, int(config.get("visual_match_candidates_per_source", 6)))
    pairs: list[ImagePair] = []
    inventory: list[dict] = []

    for item in config["image_sets"]:
        check_access_deadline(deadline)
        source_root = Path(item["original"])
        target_root = Path(item["corrected"])
        sources = image_files(source_root)
        targets = image_files(target_root)
        signatures: dict[Path, VisualSignature] = {}
        unreadable_paths: set[Path] = set()

        def signature(path: Path) -> VisualSignature | None:
            if path in signatures:
                return signatures[path]
            if path in unreadable_paths:
                return None
            check_access_deadline(deadline)  # every production image read passes here
            try:
                signatures[path] = visual_signature(path)
                return signatures[path]
            except Exception:
                unreadable_paths.add(path)
                return None

        target_times = sorted((target.stat().st_mtime, target) for target in targets)
        time_values = [row[0] for row in target_times]
        preliminary: dict[Path, list[tuple[float, float, bool, Path, Path]]] = {}
        source_stems = {path: normalized_stem(path) for path in sources}
        target_stems = {path: normalized_stem(path) for path in targets}
        scores_by_source: dict[Path, list[float]] = {}
        scores_by_target: dict[Path, list[float]] = {}
        for source in sources:
            source_time = source.stat().st_mtime
            lower = bisect_left(time_values, source_time - max_seconds)
            upper = bisect_right(time_values, source_time + max_seconds)
            source_signature = signature(source)
            if source_signature is None:
                continue
            for target_time, target in target_times[lower:upper]:
                same_name = source_stems[source] == target_stems[target]
                if not visual_enabled and not same_name:
                    continue
                target_signature = signature(target)
                if target_signature is None:
                    continue
                source_aspect = source_signature.width / source_signature.height
                target_aspect = target_signature.width / target_signature.height
                aspect_change = abs(target_aspect / source_aspect - 1.0)
                if aspect_change > visual_max_aspect and not same_name:
                    continue
                similarity = visual_similarity(source_signature, target_signature, include_orb=False)
                required = filename_threshold if same_name else visual_threshold
                # Full scoring can add at most 0.10 from ORB features.
                if similarity < max(0.0, (required - 0.10) / 0.90):
                    continue
                difference = abs(target_time - source_time)
                preliminary.setdefault(source, []).append((similarity, difference, same_name, source, target))

        scored: list[tuple[float, float, bool, Path, Path]] = []
        for source, candidates in preliminary.items():
            same_name_candidates = [row for row in candidates if row[2]]
            visual_candidates = sorted((row for row in candidates if not row[2]), key=lambda row: (-row[0], row[1]))[
                :visual_candidate_limit
            ]
            for _, difference, same_name, _, target in same_name_candidates + visual_candidates:
                similarity = visual_similarity(signatures[source], signatures[target], include_orb=True)
                required = filename_threshold if same_name else visual_threshold
                if similarity < required:
                    continue
                # Filename is only a small tie-breaker; content recognition remains
                # mandatory even when names are identical.
                ranking_score = similarity + (0.025 if same_name else 0.0)
                scored.append((ranking_score, difference, same_name, source, target))
                scores_by_source.setdefault(source, []).append(ranking_score)
                scores_by_target.setdefault(target, []).append(ranking_score)

        best_for_source: dict[Path, float] = {path: max(values) for path, values in scores_by_source.items()}
        best_for_target: dict[Path, float] = {path: max(values) for path, values in scores_by_target.items()}
        second_for_source = {
            path: sorted(values, reverse=True)[1] if len(values) > 1 else -1.0
            for path, values in scores_by_source.items()
        }
        second_for_target = {
            path: sorted(values, reverse=True)[1] if len(values) > 1 else -1.0
            for path, values in scores_by_target.items()
        }

        # Mutual-best matching prevents one visually similar corrected image from
        # supervising several originals. A margin rejects ambiguous lookalikes.
        accepted = 0
        filename_matches = 0
        visual_matches = 0
        ambiguous = 0
        used_sources: set[Path] = set()
        used_targets: set[Path] = set()
        for ranking_score, difference, same_name, source, target in sorted(scored, key=lambda row: (-row[0], row[1])):
            if source in used_sources or target in used_targets:
                continue
            if ranking_score != best_for_source[source] or ranking_score != best_for_target[target]:
                continue
            source_margin = ranking_score - second_for_source[source]
            target_margin = ranking_score - second_for_target[target]
            if (not same_name) and min(source_margin, target_margin) < visual_margin:
                ambiguous += 1
                continue
            source_signature = signatures[source]
            target_signature = signatures[target]
            sw, sh = source_signature.width, source_signature.height
            tw, th = target_signature.width, target_signature.height
            source_aspect = sw / sh
            target_aspect = tw / th
            aspect_change = abs(target_aspect / source_aspect - 1.0)
            key = normalized_stem(source)
            similarity = ranking_score - (0.025 if same_name else 0.0)
            pairs.append(
                ImagePair(
                    set_name=item["name"],
                    key=key,
                    original=str(source),
                    corrected=str(target),
                    original_width=sw,
                    original_height=sh,
                    corrected_width=tw,
                    corrected_height=th,
                    aspect_change=aspect_change,
                    time_difference_hours=difference / 3600.0,
                    split=_stable_split(key, validation_fraction, seed),
                    match_method="filename_and_visual" if same_name else "visual_content",
                    visual_similarity=similarity,
                )
            )
            accepted += 1
            used_sources.add(source)
            used_targets.add(target)
            if same_name:
                filename_matches += 1
            else:
                visual_matches += 1

        inventory.append(
            {
                "set_name": item["name"],
                "original_root": str(source_root),
                "corrected_root": str(target_root),
                "original_count": len(sources),
                "corrected_count": len(targets),
                "matched_count": accepted,
                "filename_and_visual_count": filename_matches,
                "visual_content_count": visual_matches,
                "ambiguous_visual_count": ambiguous,
                "unreadable_count": len(unreadable_paths),
                "visual_matching_enabled": visual_enabled,
            }
        )
    return pairs, inventory


def write_manifest(pairs: Iterable[ImagePair], path: str | Path) -> None:
    rows = [asdict(pair) for pair in pairs]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a stop during the write never leaves a truncated manifest.
    temporary = output.with_name(output.name + ".partial")
    if not rows:
        temporary.write_text("", encoding="utf-8")
    else:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, output)


def read_manifest(path: str | Path) -> list[ImagePair]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                ImagePair(
                    set_name=row["set_name"],
                    key=row["key"],
                    original=row["original"],
                    corrected=row["corrected"],
                    original_width=int(row["original_width"]),
                    original_height=int(row["original_height"]),
                    corrected_width=int(row["corrected_width"]),
                    corrected_height=int(row["corrected_height"]),
                    aspect_change=float(row["aspect_change"]),
                    time_difference_hours=float(row["time_difference_hours"]),
                    split=row["split"],
                    match_method=row.get("match_method", "legacy_filename"),
                    visual_similarity=float(row.get("visual_similarity", 1.0)),
                    source_status=row.get("source_status", "current"),
                )
            )
        return rows
