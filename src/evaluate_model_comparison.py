"""Score the unchanged image, V2, V3-global and pix2pix side by side on a shared local benchmark.

Local-cache-only: every pair comes from artifacts/training_cache (via the
already-hardened secure_pix2pix.data.load_verified_cache_pairs reader), never
from the production shares, so this can run at any time of day, not only
inside the 04:00-10:00 access window.

Usage:
    python evaluate_model_comparison.py --artifacts artifacts --config config.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from correction_ml.inference import correct_array, load_model
from correction_ml.metrics import read_rgb, resize_to
from correction_ml.safety import assert_not_production_output
from secure_pix2pix.checkpoints import safe_torch_load
from secure_pix2pix.data import load_verified_cache_pairs
from secure_pix2pix.metrics import denormalize_image, l1_per_image, psnr_per_image, ssim_per_image
from secure_pix2pix.models import UNetGenerator

PIX2PIX_IMAGE_SIZE = 256
MINIMUM_SCORABLE_SIDE = 11  # ssim_per_image's Gaussian window requirement


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score V2 and pix2pix predictions on a shared local benchmark.")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--cache-index", default=None, help="Defaults to <artifacts>/training_cache/cache_index.json")
    parser.add_argument("--v2-checkpoint", default=None, help="Defaults to <artifacts>/best_v2.pt")
    parser.add_argument("--pretrained-v2-checkpoint", default=None, help="Defaults to <artifacts>/best_v2_pretrained.pt, if present")
    parser.add_argument("--global-checkpoint", default=None, help="Defaults to <artifacts>/best_v3_global.pt, if present")
    parser.add_argument("--pix2pix-checkpoint", default=None, help="Defaults to <artifacts>/pix2pix/checkpoints/best.pt")
    parser.add_argument("--benchmark-pairs", type=int, default=40)
    parser.add_argument("--max-aspect-change", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_model_comparison_benchmark(pairs, artifacts: Path, count: int, seed: int) -> set[str]:
    """Persist a deterministic, immutable benchmark selection once.

    This is a *separate* identity space from correction_ml's frozen_benchmark.json
    (keyed by a hash of production paths) and secure_pix2pix's own
    frozen_validation.json (keyed by cache group_key under its own train/validation
    split policy): it exists only to give this harness a stable, shared set of
    local cache pairs to score every model against. It is not guaranteed disjoint
    from either model's own training data -- unifying that would require
    reconciling both models' train/validation splits, which is out of scope here.
    """
    path = artifacts / "model_comparison_benchmark.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("group_keys", [])
            if isinstance(values, list) and all(isinstance(item, str) for item in values):
                return set(values)
        except (OSError, json.JSONDecodeError):
            pass
    available = sorted({pair.group_key for pair in pairs if pair.group_key})
    ordered = sorted(available, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    selected = ordered[: max(0, int(count))]
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": seed,
        "requested_pair_count": int(count),
        "group_keys": selected,
        "policy": "local-cache-only benchmark for cross-model comparison; not guaranteed disjoint from either model's training data",
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return set(selected)


def _to_unit_tensor(rgb: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1)) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _run_pix2pix(generator: torch.nn.Module, source_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    height, width = source_rgb.shape[:2]
    resized = cv2.resize(source_rgb, (PIX2PIX_IMAGE_SIZE, PIX2PIX_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(np.ascontiguousarray(resized.astype(np.float32).transpose(2, 0, 1)))
    tensor = tensor.mul(1.0 / 127.5).sub(1.0).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = denormalize_image(generator(tensor))
    prediction = (output[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    return resize_to(prediction, width, height)


def _score(prediction: np.ndarray, target: np.ndarray) -> dict[str, float] | None:
    if min(prediction.shape[0], prediction.shape[1]) < MINIMUM_SCORABLE_SIDE:
        return None
    prediction_tensor = _to_unit_tensor(prediction)
    target_tensor = _to_unit_tensor(target)
    return {
        "psnr_db": float(psnr_per_image(prediction_tensor, target_tensor)[0]),
        "ssim": float(ssim_per_image(prediction_tensor, target_tensor)[0]),
        "l1": float(l1_per_image(prediction_tensor, target_tensor)[0]),
    }


def _load_v2(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        return None
    model, _ = load_model(checkpoint_path, device)
    return model


def _load_pix2pix(checkpoint_path: Path, device: torch.device) -> torch.nn.Module | None:
    if not checkpoint_path.is_file():
        return None
    payload = safe_torch_load(checkpoint_path, checkpoint_path.parent, device)
    generator = UNetGenerator().to(device)
    generator.load_state_dict(payload["generator"])
    generator.eval()
    return generator


def write_comparison_sheet(rows: list[tuple], labels: list[str], destination: Path, limit: int = 12) -> None:
    rows = rows[:limit]
    if not rows or not labels:
        return
    width, height, label_h = 200, 150, 22
    canvas = Image.new("RGB", (width * len(labels), (height + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, images in enumerate(rows):
        top = row_index * (height + label_h)
        for column, array in enumerate(images):
            if array is None:
                continue
            thumbnail = Image.fromarray(array)
            thumbnail.thumbnail((width - 4, height - 4), Image.Resampling.LANCZOS)
            x = column * width + (width - thumbnail.width) // 2
            y = top + (height - thumbnail.height) // 2
            canvas.paste(thumbnail, (x, y))
            draw.text((column * width + 3, top + height + 3), labels[column], fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90)


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "mean_psnr_db": float(np.mean([r["psnr_db"] for r in records])),
        "mean_ssim": float(np.mean([r["ssim"] for r in records])),
        "mean_l1": float(np.mean([r["l1"] for r in records])),
    }


def main() -> None:
    args = arguments()
    artifacts = Path(args.artifacts)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert_not_production_output(artifacts, config)
    artifacts.mkdir(parents=True, exist_ok=True)

    cache_index = Path(args.cache_index) if args.cache_index else artifacts / "training_cache" / "cache_index.json"
    if not cache_index.is_file():
        raise SystemExit(f"No local cache index found at {cache_index}. Run analyze.py at least once first.")
    cache_report = load_verified_cache_pairs(cache_index, exact_filename_only=True, max_aspect_change=args.max_aspect_change)
    if not cache_report.pairs:
        raise SystemExit("No verified local cache pairs available to compare models on.")

    benchmark_keys = ensure_model_comparison_benchmark(cache_report.pairs, artifacts, args.benchmark_pairs, args.seed)
    by_key: dict[str, object] = {}
    for pair in cache_report.pairs:
        by_key.setdefault(pair.group_key, pair)
    benchmark = [by_key[key] for key in sorted(benchmark_keys) if key in by_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v2_checkpoint = Path(args.v2_checkpoint) if args.v2_checkpoint else artifacts / "best_v2.pt"
    pretrained_checkpoint = Path(args.pretrained_v2_checkpoint) if args.pretrained_v2_checkpoint else artifacts / "best_v2_pretrained.pt"
    pix2pix_checkpoint = Path(args.pix2pix_checkpoint) if args.pix2pix_checkpoint else artifacts / "pix2pix" / "checkpoints" / "best.pt"
    global_checkpoint = Path(args.global_checkpoint) if args.global_checkpoint else artifacts / "best_v3_global.pt"

    # "unchanged" scores the original image itself: every model must beat it
    # before its corrections are worth anything.
    models: dict[str, dict] = {"unchanged": {"kind": "identity", "model": None}}
    v2_model = _load_v2(v2_checkpoint, device)
    if v2_model is not None:
        models["v2"] = {"kind": "v2", "model": v2_model}
    pretrained_model = _load_v2(pretrained_checkpoint, device)
    if pretrained_model is not None:
        models["v2_pretrained"] = {"kind": "v2", "model": pretrained_model}
    global_model = _load_v2(global_checkpoint, device)
    if global_model is not None:
        models["v3_global"] = {"kind": "v2", "model": global_model}
    pix2pix_generator = _load_pix2pix(pix2pix_checkpoint, device)
    if pix2pix_generator is not None:
        models["pix2pix"] = {"kind": "pix2pix", "model": pix2pix_generator}

    per_model_metrics: dict[str, list[dict]] = {name: [] for name in models}
    skipped: list[dict] = []
    sheet_rows: list[tuple] = []
    sheet_labels = ["original", "historical target"] + list(models.keys())

    for pair in benchmark:
        try:
            source = read_rgb(pair.original, max_side=2048)
            target = read_rgb(pair.corrected, max_side=2048)
        except Exception as exc:
            skipped.append({"group_key": pair.group_key, "reason": f"read_error: {type(exc).__name__}: {exc}"})
            continue
        if source.shape != target.shape:
            skipped.append({"group_key": pair.group_key, "reason": "geometry_mismatch"})
            continue

        predictions: dict[str, np.ndarray] = {}
        for name, entry in models.items():
            try:
                if entry["kind"] == "identity":
                    predictions[name] = source
                elif entry["kind"] == "v2":
                    predictions[name] = correct_array(entry["model"], source, device, tile_size=512, overlap=48)
                else:
                    predictions[name] = _run_pix2pix(entry["model"], source, device)
            except Exception as exc:
                skipped.append({"group_key": pair.group_key, "model": name, "reason": f"inference_error: {type(exc).__name__}: {exc}"})

        for name, prediction in predictions.items():
            score = _score(prediction, target)
            if score is None:
                continue
            score["group_key"] = pair.group_key
            per_model_metrics[name].append(score)

        if len(sheet_rows) < 12:
            sheet_rows.append((source, target, *(predictions.get(name) for name in models)))

    if sheet_rows:
        write_comparison_sheet(sheet_rows, sheet_labels, artifacts / "model_comparison_review.jpg")

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "benchmark_pair_count": len(benchmark),
        "models_evaluated": sorted(models.keys()),
        "models_unavailable": sorted({"v2", "v2_pretrained", "v3_global", "pix2pix"} - set(models.keys())),
        "per_model": {name: summarize(records) for name, records in per_model_metrics.items()},
        "per_pair": {
            name: [
                {"group_key": record["group_key"], "psnr_db": record["psnr_db"], "ssim": record["ssim"], "l1": record["l1"]}
                for record in records
            ]
            for name, records in per_model_metrics.items()
        },
        "skipped": skipped,
        "note": "Local-cache-only comparison; not guaranteed disjoint from either model's own training data.",
    }
    (artifacts / "model_comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_pair"}, indent=2))


if __name__ == "__main__":
    main()
