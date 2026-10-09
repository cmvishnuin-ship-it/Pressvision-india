from __future__ import annotations

import argparse
from pathlib import Path

import torch

from correction_ml.inference import correct_file, load_model, write_parameter_report
from correction_ml.pairs import IMAGE_EXTENSIONS
from correction_ml.pairs import load_config
from correction_ml.safety import assert_not_production_output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a trained correction model to an image or folder.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="artifacts/best_v2.pt")
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--config", default="config.json")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    source = Path(args.input)
    destination = Path(args.output)
    config = load_config(args.config)
    if config.get("learning_only", True):
        raise SystemExit(
            "Inference is disabled: this project is configured for learning and analysis only."
        )
    assert_not_production_output(destination, config)
    if source.is_file():
        output_file = destination / f"{source.stem}.jpg" if destination.is_dir() else destination
        correct_file(model, source, output_file, device, args.tile_size, args.quality)
        report = write_parameter_report(model, checkpoint, source, output_file, device)
        print(output_file)
        if report:
            print(report)
        return
    files = [path for path in source.rglob("*") if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS]
    for number, path in enumerate(files, start=1):
        relative = path.relative_to(source).with_suffix(".jpg")
        output_file = destination / relative
        correct_file(model, path, output_file, device, args.tile_size, args.quality)
        write_parameter_report(model, checkpoint, path, output_file, device)
        print(f"[{number}/{len(files)}] {output_file}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")


if __name__ == "__main__":
    main()
