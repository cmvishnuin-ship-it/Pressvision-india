"""Score whole validation images: how much of the editors' change each model reproduces.

Local-cache only. For every publication it answers two questions:

* How much of the editors' change is reproduced: error against the editors'
  result, as a share of the error of leaving the image unchanged.
* Did the model learn the right pattern: for each measured effect (brightness,
  contrast, saturation, colour temperature and tint, shadows, highlights,
  sharpening), the model's change next to the editors' change.

"unchanged" is the original image. The two "ceiling_*" rows are the best a single
per-image colour curve / 3D colour map could do, fitted to the editors' own
result, so they show what a whole-picture correction alone can reach.

Writes artifacts/whole_image_report.json and appends artifacts/whole_image_history.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from correction_ml.inference import correct_array, load_model
from correction_ml.metrics import image_parameters, read_rgb, resize_to
from correction_ml.pairs import read_manifest
from train import inside_folder

MODEL_CHECKPOINTS = {"v2": "best_v2.pt", "v3_global": "best_v3_global.pt", "v2_pretrained": "best_v2_pretrained.pt"}
EFFECTS = ("brightness", "contrast", "saturation", "temperature_proxy", "tint_proxy", "shadow_mean", "highlight_mean")
HISTORY_FIELDS = ["generated_at", "publication", "model", "images", "mae", "psnr", "delta_e", "reproduced"]


def arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score whole validation images against the editors' corrections.")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--cache-root", default=None, help="Local cache folder the images must be in; defaults to <artifacts>/training_cache.")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--max-images", type=int, default=300, help="Deterministic subset of the validation images; 0 uses all.")
    parser.add_argument("--max-aspect-change", type=float, default=0.02)
    parser.add_argument("--tile-size", type=int, default=512)
    return parser.parse_args(argv)


def validation_images(artifacts: Path, max_images: int, max_aspect_change: float, cache_root: Path | None = None) -> list:
    """Whole validation images inside the local cache (replay crops are not whole images)."""
    cache_root = str(cache_root or artifacts / "training_cache")
    pairs = [
        pair for pair in read_manifest(artifacts / "manifest.csv")
        if pair.split == "validation"
        and pair.aspect_change <= max_aspect_change
        and not Path(pair.original).name.startswith("patch_")
        and inside_folder(pair.original, cache_root)
        and inside_folder(pair.corrected, cache_root)
    ]
    pairs.sort(key=lambda pair: hashlib.sha256(pair.original.encode("utf-8")).hexdigest())
    return pairs[:max_images] if max_images > 0 else pairs


def ceiling_curves(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best per-channel tone curve for this image (mean target value for every source value)."""
    result = np.empty(source.shape, np.float32)
    for channel in range(3):
        values, wanted = source[..., channel].ravel(), target[..., channel].ravel().astype(np.float64)
        counts = np.bincount(values, minlength=256)
        known = np.flatnonzero(counts)
        lut = np.interp(np.arange(256), known, np.bincount(values, weights=wanted, minlength=256)[known] / counts[known])
        result[..., channel] = lut[source[..., channel]]
    return result


def ceiling_colour_map(source: np.ndarray, target: np.ndarray, bins: int = 16) -> np.ndarray:
    """Best per-image 3D colour map on a bins^3 grid (captures cross-channel colour changes)."""
    cell = (source // (256 // bins)).astype(np.int64)
    key = ((cell[..., 0] * bins + cell[..., 1]) * bins + cell[..., 2]).ravel()
    counts = np.maximum(np.bincount(key, minlength=bins**3), 1)
    result = np.empty(source.shape, np.float32)
    for channel in range(3):
        result[..., channel] = (np.bincount(key, weights=target[..., channel].ravel().astype(np.float64), minlength=bins**3) / counts)[key].reshape(source.shape[:2])
    return result


def colour_difference(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean CIE76 colour difference (Delta E) in Lab."""
    lab_prediction = cv2.cvtColor(np.clip(prediction, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    lab_target = cv2.cvtColor(target.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return float(np.linalg.norm(lab_prediction - lab_target, axis=2).mean())


def effect_changes(source_stats: dict, result: np.ndarray) -> dict[str, float]:
    stats = image_parameters(np.clip(result + 0.5, 0, 255).astype(np.uint8))
    changes = {name: stats[name] - source_stats[name] for name in EFFECTS}
    changes["sharpening_log2"] = float(np.log2(max(stats["sharpness_laplacian"], 1e-6) / max(source_stats["sharpness_laplacian"], 1e-6)))
    return changes


def evaluate(artifacts: Path, max_side: int, max_images: int, max_aspect_change: float, tile_size: int, cache_root: Path | None = None) -> dict:
    device = torch.device("cpu")
    models, checkpoints = {}, {}
    for name, filename in MODEL_CHECKPOINTS.items():
        path = artifacts / filename
        if path.is_file():
            models[name], data = load_model(path, device)
            checkpoints[name] = {"file": filename, "epoch": int(data.get("epoch", 0))}

    scores = defaultdict(lambda: defaultdict(list))  # group -> model -> per-image score dicts
    editors = defaultdict(list)  # group -> editors' effect changes per image
    pairs = validation_images(artifacts, max_images, max_aspect_change, cache_root)
    for pair in pairs:
        source = read_rgb(pair.original, max_side)
        target = read_rgb(pair.corrected, max_side)
        source = resize_to(source, target.shape[1], target.shape[0])
        target_float = target.astype(np.float32)
        source_stats = image_parameters(source)
        editors_change = effect_changes(source_stats, target_float)
        results = {"unchanged": source.astype(np.float32), "ceiling_curves": ceiling_curves(source, target), "ceiling_colour_map": ceiling_colour_map(source, target)}
        for name, model in models.items():
            results[name] = correct_array(model, source, device, tile_size=tile_size, overlap=48).astype(np.float32)
        for group in ("ALL", pair.set_name):
            editors[group].append(editors_change)
            for name, result in results.items():
                mse = float(np.mean(((result - target_float) / 255.0) ** 2))
                scores[group][name].append({
                    "mae": float(np.abs(result - target_float).mean() / 255.0),
                    "psnr": 10.0 * np.log10(1.0 / max(mse, 1e-10)),
                    "delta_e": colour_difference(result, target),
                    "changes": effect_changes(source_stats, result),
                })

    publications = {}
    for group, by_model in scores.items():
        baseline = np.mean([score["mae"] for score in by_model["unchanged"]])
        editors_mean = {effect: float(np.mean([change[effect] for change in editors[group]])) for effect in editors[group][0]}
        models_summary = {}
        for name, per_image in by_model.items():
            mae = float(np.mean([score["mae"] for score in per_image]))
            models_summary[name] = {
                "images": len(per_image),
                "mae": mae,
                "psnr": float(np.mean([score["psnr"] for score in per_image])),
                "delta_e": float(np.mean([score["delta_e"] for score in per_image])),
                "reproduced": float(1.0 - mae / baseline) if baseline > 0 else 0.0,
                "effects": {
                    effect: {
                        "change": float(np.mean([score["changes"][effect] for score in per_image])),
                        "error_vs_editors": float(np.mean([abs(score["changes"][effect] - editor[effect]) for score, editor in zip(per_image, editors[group])])),
                    }
                    for effect in editors_mean
                },
            }
        publications[group] = {"editors_change": editors_mean, "models": models_summary}
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "images": len(pairs),
        "max_side": max_side,
        "checkpoints": checkpoints,
        "publications": publications,
        "note": "reproduced = share of the editors' change reproduced (1 - error / unchanged error). ceiling_* rows are per-image fits to the editors' own result, not models.",
    }


def print_summary(report: dict) -> None:
    print(f"Whole-image check on {report['images']} validation images ({report['max_side']} px)")
    print(f"{'publication':16s} {'model':20s} {'images':>6s} {'error':>7s} {'reproduced':>10s} {'colour dE':>9s}")
    for group, summary in sorted(report["publications"].items()):
        for name, values in summary["models"].items():
            print(f"{group:16s} {name:20s} {values['images']:6d} {values['mae']:7.4f} {values['reproduced']:10.0%} {values['delta_e']:9.2f}")
    overall = report["publications"].get("ALL")
    if overall:
        names = [name for name in overall["models"] if not name.startswith("ceiling_")]
        print("\nPattern check (ALL): editors' average change and each model's average change")
        print(f"{'effect':18s} {'editors':>9s} " + " ".join(f"{name:>12s}" for name in names))
        for effect, editors_change in overall["editors_change"].items():
            print(f"{effect:18s} {editors_change:9.4f} " + " ".join(f"{overall['models'][name]['effects'][effect]['change']:12.4f}" for name in names))


def write_report(report: dict, artifacts: Path) -> None:
    temporary = artifacts / "whole_image_report.json.partial"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(artifacts / "whole_image_report.json")
    history = artifacts / "whole_image_history.csv"
    new_file = not history.exists()
    with history.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        for group, summary in sorted(report["publications"].items()):
            for name, values in summary["models"].items():
                writer.writerow({
                    "generated_at": report["generated_at"], "publication": group, "model": name, "images": values["images"],
                    "mae": round(values["mae"], 6), "psnr": round(values["psnr"], 3), "delta_e": round(values["delta_e"], 4),
                    "reproduced": round(values["reproduced"], 4),
                })


def main(argv=None) -> dict:
    args = arguments(argv)
    artifacts = Path(args.artifacts)
    report = evaluate(artifacts, args.max_side, args.max_images, args.max_aspect_change, args.tile_size, Path(args.cache_root) if args.cache_root else None)
    if report["images"]:
        write_report(report, artifacts)
    print_summary(report)
    return report


if __name__ == "__main__":
    main()
