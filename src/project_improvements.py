"""Run the V1 local quality gates without writing to production image folders.

Use after analysis/training, or schedule it after the shadow learner:
    python project_improvements.py --artifacts artifacts --config config.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from correction_ml.inference import correct_array, input_domain_report, load_model
from correction_ml.metrics import read_rgb
from correction_ml.monitoring import (
    ensure_frozen_benchmark,
    pair_identifier,
    verify_image,
    write_model_status,
    write_pair_review_sheet,
    write_quality_reports,
)
from correction_ml.pairs import read_manifest
from correction_ml.safety import assert_not_production_output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local-only data quality, benchmark and promotion reports.")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", default=None, help="Defaults to artifacts/best_v2.pt")
    parser.add_argument("--max-evaluation-pairs", type=int, default=24, help="0 evaluates every frozen benchmark pair.")
    parser.add_argument("--benchmark-pairs", type=int, default=200)
    return parser.parse_args()


def parameter_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def preview(rgb: np.ndarray, side: int = 512) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, side / max(height, width))
    if scale == 1.0:
        return rgb
    return cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def write_v2_sheet(rows: list[tuple], destination: Path) -> None:
    if not rows:
        return
    width, height, label = 210, 158, 22
    canvas = Image.new("RGB", (width * 3, (height + label) * len(rows)), "white")
    labels = ("original", "V2 prediction", "historical target")
    draw = ImageDraw.Draw(canvas)
    for row_index, images in enumerate(rows):
        top = row_index * (height + label)
        for column, array in enumerate(images):
            image = Image.fromarray(array)
            image.thumbnail((width - 4, height - 4), Image.Resampling.LANCZOS)
            x = column * width + (width - image.width) // 2
            y = top + (height - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((column * width + 3, top + height + 3), labels[column], fill="black")
    canvas.save(destination, quality=90)


def main() -> None:
    args = arguments()
    artifacts = Path(args.artifacts)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert_not_production_output(artifacts, config)
    pairs = read_manifest(artifacts / "manifest.csv")
    benchmark_ids = ensure_frozen_benchmark(pairs, artifacts, args.benchmark_pairs, int(config.get("seed", 42)))
    benchmark = [pair for pair in pairs if pair_identifier(pair) in benchmark_ids]
    write_pair_review_sheet(benchmark, artifacts / "benchmark_pairs.jpg")

    rejected = []
    valid = []
    for pair in pairs:
        error = verify_image(pair.original) or verify_image(pair.corrected)
        if error:
            rejected.append({"set_name": pair.set_name, "key": pair.key, "original": pair.original, "corrected": pair.corrected, "reason": error})
        else:
            valid.append(pair)
    rows = parameter_rows(artifacts / "correction_parameters.csv")
    write_quality_reports(rows, rejected, artifacts, float(config.get("max_aspect_change_for_training", 0.02)))

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else artifacts / "best_v2.pt"
    report: dict = {"benchmark_pair_count": len(benchmark), "evaluated_pair_count": 0, "status": "checkpoint_missing"}
    if checkpoint_path.is_file() and benchmark:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, checkpoint = load_model(checkpoint_path, device)
        chosen = benchmark if args.max_evaluation_pairs == 0 else benchmark[:max(1, args.max_evaluation_pairs)]
        metrics, grouped, sheet_rows, skipped = [], defaultdict(list), [], []
        for pair in chosen:
            domain = input_domain_report(checkpoint, Path(pair.original))
            # This is a hard safety gate for any future inference integration.
            if domain.get("status") == "outside_training_domain":
                skipped.append({"key": pair.key, "reason": "outside_training_domain", "domain": domain})
                continue
            try:
                source, target = preview(read_rgb(pair.original)), preview(read_rgb(pair.corrected))
                if source.shape != target.shape:
                    skipped.append({"key": pair.key, "reason": "geometry_changed"})
                    continue
                prediction = correct_array(model, source, device, tile_size=512, overlap=48)
                mae = float(np.mean(np.abs(prediction.astype(np.float32) - target.astype(np.float32))) / 255.0)
                psnr = float(cv2.PSNR(target, prediction))
                record = {"key": pair.key, "set_name": pair.set_name, "mae": mae, "psnr_db": psnr, "domain": domain["status"]}
                metrics.append(record)
                grouped[pair.set_name].append(record)
                if len(sheet_rows) < 12:
                    sheet_rows.append((source, prediction, target))
            except Exception as exc:
                skipped.append({"key": pair.key, "reason": f"evaluation_error: {type(exc).__name__}: {exc}"})
        if sheet_rows:
            write_v2_sheet(sheet_rows, artifacts / "benchmark_v2_review.jpg")
        report = {
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
            "checkpoint": str(checkpoint_path), "benchmark_pair_count": len(benchmark),
            "evaluated_pair_count": len(metrics), "skipped_pair_count": len(skipped),
            "mean_mae": float(np.mean([row["mae"] for row in metrics])) if metrics else None,
            "mean_psnr_db": float(np.mean([row["psnr_db"] for row in metrics])) if metrics else None,
            "per_set": {name: {"count": len(values), "mean_mae": float(np.mean([v["mae"] for v in values])), "mean_psnr_db": float(np.mean([v["psnr_db"] for v in values]))} for name, values in grouped.items()},
            "skipped": skipped,
            "promotion": "hold: learning-only mode; require a stable benchmark improvement across multiple runs before any assist-mode trial",
        }
    (artifacts / "benchmark_v2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_model_status(artifacts)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
