from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image, ImageOps

from .dataset import context_view
from .metrics import image_parameters, read_rgb
from .model import CorrectionUNet, CorrectionUNetV2
from .model_global import CorrectionUNetV3Global
from .model_pretrained import CorrectionUNetV2PretrainedBackbone


def _context_tensor(model: torch.nn.Module, rgb: np.ndarray, device: torch.device) -> torch.Tensor | None:
    """Whole-image view for context-aware models, shared by every tile of one image."""
    size = getattr(model, "context_size", 0)
    if not size:
        return None
    view = context_view(rgb, size)
    return torch.from_numpy(np.ascontiguousarray(view.transpose(2, 0, 1))).float().unsqueeze(0).to(device) / 255.0


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture = data.get("architecture")
    if architecture == "correction_unet_v3_global":
        model = CorrectionUNetV3Global(
            base_channels=int(data.get("base_channels", 32)),
            parameter_count=len(data.get("parameter_names", [])) or 34,
            context_size=int(data.get("context_size") or 128),
        ).to(device)
    elif architecture == "correction_unet_v2_pretrained_resnet18":
        # pretrained_weights_path=None is intentional here: load_state_dict below
        # overwrites every parameter (including the resnet18 submodule's, which
        # is captured in the checkpoint after the first training session), so
        # the bundled weights file is not needed to load an existing checkpoint.
        model = CorrectionUNetV2PretrainedBackbone(
            base_channels=int(data.get("base_channels", 32)),
            parameter_count=len(data.get("parameter_names", [])) or 34,
            pretrained_weights_path=None,
        ).to(device)
    elif architecture == "correction_unet_v2":
        model = CorrectionUNetV2(
            base_channels=int(data.get("base_channels", 32)),
            parameter_count=len(data.get("parameter_names", [])) or 34,
        ).to(device)
    else:
        model = CorrectionUNet(base_channels=int(data.get("base_channels", 24))).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, data


@torch.inference_mode()
def correct_array(model: torch.nn.Module, rgb: np.ndarray, device: torch.device, tile_size: int = 768, overlap: int = 64) -> np.ndarray:
    height, width = rgb.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    step = max(4, tile_size - overlap)
    y_positions = list(range(0, max(1, height - tile_size + 1), step))
    x_positions = list(range(0, max(1, width - tile_size + 1), step))
    y_positions.append(max(0, height - tile_size))
    x_positions.append(max(0, width - tile_size))
    context = _context_tensor(model, rgb, device)

    for top in sorted(set(y_positions)):
        for left in sorted(set(x_positions)):
            tile = rgb[top : top + tile_size, left : left + tile_size]
            tile_height, tile_width = tile.shape[:2]
            # V2/V3 halve the image three times, so each side must be a multiple of 8.
            pad_h = (-tile_height) % 8
            pad_w = (-tile_width) % 8
            padded = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            corrected = model(tensor) if context is None else model(tensor, context)
            prediction = corrected[0, :, :tile_height, :tile_width].permute(1, 2, 0).cpu().numpy()
            output[top : top + tile_height, left : left + tile_width] += prediction
            weights[top : top + tile_height, left : left + tile_width] += 1.0
    return np.clip(output / np.maximum(weights, 1e-6) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def correct_file(model: torch.nn.Module, source: Path, destination: Path, device: torch.device, tile_size: int, quality: int) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(image)
    corrected = correct_array(model, rgb, device, tile_size=tile_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrected).save(destination, format="JPEG", quality=quality, subsampling=0, optimize=True)


@torch.inference_mode()
def predicted_parameters(model: torch.nn.Module, checkpoint: dict, source: Path, device: torch.device) -> dict[str, float]:
    names = checkpoint.get("parameter_names")
    center = checkpoint.get("parameter_center")
    scale = checkpoint.get("parameter_scale")
    if not names or center is None or scale is None or not hasattr(model, "forward_with_parameters"):
        return {}
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    pad_h, pad_w = (-height) % 8, (-width) % 8
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    context = _context_tensor(model, rgb, device)
    _, normalized = model.forward_with_parameters(tensor) if context is None else model.forward_with_parameters(tensor, context)
    values = normalized[0].cpu().numpy() * np.asarray(scale) + np.asarray(center)
    return {str(name): float(value) for name, value in zip(names, values)}


def write_parameter_report(model: torch.nn.Module, checkpoint: dict, source: Path, destination: Path, device: torch.device) -> Path | None:
    values = predicted_parameters(model, checkpoint, source, device)
    if not values:
        return None
    report = destination.with_suffix(".parameters.json")
    report.write_text(
        json.dumps(
            {
                "source": str(source),
                "corrected_output": str(destination),
                "note": "Predicted observable correction effects; these are not hidden editor slider values.",
                "input_domain": input_domain_report(checkpoint, source),
                "parameters": values,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def input_domain_report(checkpoint: dict, source: Path) -> dict:
    names = checkpoint.get("input_feature_names")
    center = checkpoint.get("input_feature_center")
    scale = checkpoint.get("input_feature_scale")
    if not names or center is None or scale is None:
        return {
            "status": "unavailable",
            "human_review_recommended": True,
            "reason": "Checkpoint predates input-domain monitoring; continue training once to add it.",
        }
    rgb = read_rgb(source, max_side=3072)
    measured = image_parameters(rgb)
    values = np.asarray([measured[name] for name in names], dtype=np.float32)
    z_scores = np.abs((values - np.asarray(center)) / np.asarray(scale))
    score = float(np.mean(np.clip(z_scores, 0.0, 10.0)))
    confidence = float(100.0 * np.exp(-score / 2.5))
    if score < 2.0:
        status = "typical_training_domain"
    elif score < 3.5:
        status = "unusual_review_advised"
    else:
        status = "outside_training_domain"
    largest = np.argsort(z_scores)[-3:][::-1]
    return {
        "status": status,
        "heuristic_confidence_percent": round(confidence, 2),
        "distance_score": round(score, 4),
        "human_review_recommended": score >= 2.0,
        "most_unusual_features": [
            {"name": names[int(index)], "standardized_distance": round(float(z_scores[index]), 3)}
            for index in largest
        ],
        "note": "Confidence measures similarity to training images; it is not a guarantee of correction quality.",
    }
