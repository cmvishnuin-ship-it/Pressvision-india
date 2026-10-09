"""Fast local status command for the learning-only service.

This does not train, infer, access production paths, or inspect image pixels.
It only summarizes already-written local artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from correction_ml.monitoring import write_model_status


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Show current local learning-model status without training.")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    write_model_status(artifacts)
    status = read_json(artifacts / "model_status.json")
    v2 = status.get("v2", {})
    shadow = status.get("pix2pix", {})
    quality = status.get("data_quality", {})
    print("V2")
    print(f"  pairs: train={v2.get('training_pairs', 'unknown')}, validation={v2.get('validation_pairs', 'unknown')}")
    print(f"  dataset: {v2.get('dataset_status', 'unknown')}")
    print("Pix2pix shadow")
    print(f"  status={shadow.get('status', 'not started')}, epoch={shadow.get('epoch', 'unknown')}, best PSNR={shadow.get('best_psnr_db', 'unknown')}")
    print("Data quality")
    print(f"  analyzed={quality.get('analyzed_pairs', 'unknown')}, rejected={quality.get('rejected_images', 'unknown')}, outliers={quality.get('outlier_pairs', 'unknown')}")
    print(f"Full JSON status: {(artifacts / 'model_status.json').resolve()}")


if __name__ == "__main__":
    main()
