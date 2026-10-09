"""Explicit, guarded inference for future assist-mode trials.

Unlike ``infer.py`` this command is usable only when learning_only is false,
never writes to protected roots, and refuses out-of-domain inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from correction_ml.inference import correct_file, input_domain_report, load_model, write_parameter_report
from correction_ml.pairs import IMAGE_EXTENSIONS, load_config
from correction_ml.safety import assert_not_production_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create guarded, review-only V2 proposals.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="A separate proposal/review directory.")
    parser.add_argument("--checkpoint", default="artifacts/best_v2.pt")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("learning_only", True):
        raise SystemExit("Inference remains disabled while learning_only is true.")
    output = Path(args.output)
    assert_not_production_output(output, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    source = Path(args.input)
    files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS]
    rejected = []
    for path in files:
        domain = input_domain_report(checkpoint, path)
        if domain.get("status") != "typical_training_domain":
            rejected.append({"source": str(path), "reason": domain.get("status"), "domain": domain})
            continue
        target = output / (path.stem + ".jpg") if source.is_file() else output / path.relative_to(source).with_suffix(".jpg")
        correct_file(model, path, target, device, tile_size=768, quality=95)
        write_parameter_report(model, checkpoint, path, target, device)
        print(target)
    if rejected:
        import json
        (output / "rejected_out_of_domain.json").parent.mkdir(parents=True, exist_ok=True)
        (output / "rejected_out_of_domain.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
        print(f"Skipped {len(rejected)} non-typical inputs; no production image was changed.")


if __name__ == "__main__":
    main()
