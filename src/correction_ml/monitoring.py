"""Local-only quality gates and reproducible evaluation helpers.

Nothing in this module writes to an image source.  Reports, frozen benchmark
manifests and preview sheets are always written beneath the caller's artifacts
directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def pair_identifier(pair) -> str:
    text = "\x1f".join((str(pair.set_name), str(pair.key), str(pair.original), str(pair.corrected)))
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def verify_image(path: str | Path) -> str | None:
    """Return an error string for an unsafe/corrupt image, otherwise ``None``."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(path) as image:
                image.verify()
            # ``verify`` does not force every TIFF strip to be decoded.
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
    except Exception as exc:  # Pillow has several format-specific exception types.
        return f"{type(exc).__name__}: {exc}"
    return None


def image_category(row: dict) -> str:
    """Conservative, explainable grouping for reports and future model routing."""
    width, height = float(row.get("original_width", 0)), float(row.get("original_height", 0))
    brightness = float(row.get("original_brightness", 0.5))
    edge_density = float(row.get("original_edge_density", 0.0))
    entropy = float(row.get("original_entropy", 8.0))
    aspect = width / max(1.0, height)
    if brightness < 0.16:
        return "very_dark"
    if aspect > 2.2 or aspect < 0.45:
        return "unusual_aspect"
    # High edge density with limited photographic entropy is commonly a graphic,
    # layout, screenshot, or text-heavy image.  It is deliberately not called OCR.
    if edge_density >= 0.16 and entropy < 7.0:
        return "text_or_graphic_like"
    if width * height > 12_000_000:
        return "large_photo"
    return "photo_like"


def outlier_reasons(row: dict, *, max_aspect_change: float) -> list[str]:
    reasons: list[str] = []
    if float(row.get("aspect_change", 0.0)) > max_aspect_change:
        reasons.append("geometry_changed")
    if float(row.get("visual_similarity", 1.0)) < 0.70:
        reasons.append("low_visual_similarity")
    if float(row.get("mae", 0.0)) > 0.25:
        reasons.append("extreme_pixel_difference")
    if float(row.get("time_difference_hours", 0.0)) > 48.0:
        reasons.append("large_time_gap")
    return reasons


def ensure_frozen_benchmark(pairs: Iterable, artifacts: Path, count: int, seed: int) -> set[str]:
    """Persist a deterministic, immutable benchmark selection once.

    Existing benchmark IDs are never replaced automatically.  This prevents
    daily cache growth from silently changing the score used for promotion.
    """
    path = artifacts / "frozen_benchmark.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("pair_ids", [])
            if isinstance(values, list) and all(isinstance(item, str) for item in values):
                return set(values)
        except (OSError, json.JSONDecodeError):
            pass
    grouped: dict[str, list] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair.set_name)].append(pair)
    available = sum(len(items) for items in grouped.values())
    target = min(max(1, int(count)), available)
    selected: list[str] = []
    # Give every populated image set representation before filling proportionally.
    for set_name in sorted(grouped):
        ordered = sorted(grouped[set_name], key=lambda item: hashlib.sha256(
            f"{seed}:{pair_identifier(item)}".encode()).hexdigest())
        selected.append(pair_identifier(ordered[0]))
    if len(selected) > target:
        # More populated sets than the requested count: keep the deterministic
        # seeded order instead of silently persisting more than was requested.
        selected.sort(key=lambda pair_id: hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest())
        selected = selected[:target]
    remaining = max(0, target - len(selected))
    candidates = [pair for items in grouped.values() for pair in items if pair_identifier(pair) not in selected]
    candidates.sort(key=lambda item: hashlib.sha256(f"{seed}:{pair_identifier(item)}".encode()).hexdigest())
    selected.extend(pair_identifier(pair) for pair in candidates[:remaining])
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": seed,
        "requested_pair_count": int(count),
        "pair_ids": selected,
        "policy": "frozen local benchmark; pairs are excluded from training and working validation",
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return set(selected)


def write_quality_reports(rows: list[dict], rejected: list[dict], artifacts: Path, max_aspect_change: float) -> None:
    for row in rows:
        row["image_category"] = image_category(row)
        row["outlier_reasons"] = ";".join(outlier_reasons(row, max_aspect_change=max_aspect_change))
    fields = ["set_name", "key", "original", "corrected", "image_category", "outlier_reasons", "visual_similarity", "mae", "aspect_change", "time_difference_hours"]
    outliers = [{key: row.get(key, "") for key in fields} for row in rows if row["outlier_reasons"]]
    for name, records in (("pair_outliers.csv", outliers), ("rejected_images.csv", rejected)):
        output = artifacts / name
        if not records:
            output.write_text("", encoding="utf-8")
            continue
        with output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    categories = Counter(row["image_category"] for row in rows)
    (artifacts / "data_quality_summary.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(),
        "analyzed_pairs": len(rows), "rejected_images": len(rejected),
        "outlier_pairs": len(outliers), "categories": dict(sorted(categories.items())),
    }, indent=2), encoding="utf-8")


def write_pair_review_sheet(pairs: Iterable, destination: Path, limit: int = 12) -> None:
    """Write a compact original/target audit sheet without changing either input."""
    selected = list(pairs)[:limit]
    if not selected:
        return
    cell_w, cell_h, label_h = 220, 165, 22
    canvas = Image.new("RGB", (cell_w * 2, (cell_h + label_h) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, pair in enumerate(selected):
        top = index * (cell_h + label_h)
        for column, path in enumerate((pair.original, pair.corrected)):
            try:
                with Image.open(path) as image:
                    preview = ImageOps.exif_transpose(image).convert("RGB")
                    preview.thumbnail((cell_w - 4, cell_h - 4), Image.Resampling.LANCZOS)
                    x = column * cell_w + (cell_w - preview.width) // 2
                    y = top + (cell_h - preview.height) // 2
                    canvas.paste(preview, (x, y))
            except Exception:
                pass
        draw.text((3, top + cell_h + 3), f"{pair.set_name}: original", fill="black")
        draw.text((cell_w + 3, top + cell_h + 3), "historical corrected", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90)


def write_model_status(artifacts: Path) -> None:
    """Create one small, current source-of-truth status file from local artifacts."""
    def read_json(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    status = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "v2": read_json(artifacts / "latest_training_session.json"),
        "pix2pix": read_json(artifacts / "pix2pix" / "shadow_status.json"),
        "data_quality": read_json(artifacts / "data_quality_summary.json"),
    }
    benchmark = read_json(artifacts / "frozen_benchmark.json")
    status["benchmark"] = {"pair_count": len(benchmark.get("pair_ids", [])), "policy": benchmark.get("policy")}
    temporary = artifacts / "model_status.tmp"
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(artifacts / "model_status.json")
