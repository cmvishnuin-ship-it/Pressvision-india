from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity


def read_rgb(path: str | Path, max_side: int = 3072) -> np.ndarray:
    with Image.open(path) as image:
        if max_side > 0 and max(image.size) > max_side:
            # JPEG draft decoding avoids materializing the full-resolution raster
            # when the decoder supports it.
            image.draft("RGB", (max_side, max_side))
        image = ImageOps.exif_transpose(image)
        if max_side > 0 and max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image.convert("RGB"))


def resize_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    interpolation = cv2.INTER_AREA if width < image.shape[1] or height < image.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _luma(rgb: np.ndarray) -> np.ndarray:
    data = rgb.astype(np.float32) / 255.0
    return 0.2126 * data[..., 0] + 0.7152 * data[..., 1] + 0.0722 * data[..., 2]


def image_parameters(rgb: np.ndarray) -> dict[str, float]:
    data = rgb.astype(np.float32) / 255.0
    luma = _luma(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray8 = np.clip(luma * 255.0, 0, 255).astype(np.uint8)
    laplacian = cv2.Laplacian(gray8, cv2.CV_32F)
    highpass = gray8.astype(np.float32) - cv2.GaussianBlur(gray8, (0, 0), 1.2)
    smooth_luma = cv2.GaussianBlur(luma, (0, 0), 5.0)
    local_detail = luma - smooth_luma
    gradient_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    edges = cv2.Canny(gray8, 80, 160)
    histogram = np.histogram(gray8, bins=256, range=(0, 256))[0].astype(np.float64)
    probability = histogram / max(1.0, histogram.sum())
    entropy = -np.sum(probability[probability > 0] * np.log2(probability[probability > 0]))
    red_green = data[..., 0] - data[..., 1]
    yellow_blue = 0.5 * (data[..., 0] + data[..., 1]) - data[..., 2]
    colorfulness = np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2) + 0.3 * np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    hue_radians = hsv[..., 0] * (2.0 * np.pi / 180.0)
    saturation = hsv[..., 1] / 255.0
    hue_weight = np.maximum(saturation, 1e-4)
    hue_sin = float(np.sum(np.sin(hue_radians) * hue_weight) / np.sum(hue_weight))
    hue_cos = float(np.sum(np.cos(hue_radians) * hue_weight) / np.sum(hue_weight))
    percentiles = np.percentile(luma, [10, 25, 50, 75, 90])
    shadow = luma[luma < 0.25]
    midtone = luma[(luma >= 0.25) & (luma <= 0.75)]
    highlight = luma[luma > 0.75]
    median = float(percentiles[2])
    gamma_proxy = float(np.log(np.clip(median, 1e-4, 0.9999)) / np.log(0.5))
    return {
        "brightness": float(luma.mean()),
        "contrast": float(luma.std()),
        "red_mean": float(data[..., 0].mean()),
        "green_mean": float(data[..., 1].mean()),
        "blue_mean": float(data[..., 2].mean()),
        "saturation": float((hsv[..., 1] / 255.0).mean()),
        "saturation_std": float(saturation.std()),
        "colorfulness": float(colorfulness),
        "temperature_proxy": float(data[..., 0].mean() - data[..., 2].mean()),
        "tint_proxy": float(data[..., 1].mean() - 0.5 * (data[..., 0].mean() + data[..., 2].mean())),
        "hue_sin": hue_sin,
        "hue_cos": hue_cos,
        "shadow_mean": float(shadow.mean()) if shadow.size else 0.0,
        "midtone_mean": float(midtone.mean()) if midtone.size else 0.5,
        "highlight_mean": float(highlight.mean()) if highlight.size else 1.0,
        "luma_p10": float(percentiles[0]),
        "luma_p25": float(percentiles[1]),
        "luma_p50": median,
        "luma_p75": float(percentiles[3]),
        "luma_p90": float(percentiles[4]),
        "dynamic_range": float(percentiles[4] - percentiles[0]),
        "gamma_proxy": gamma_proxy,
        "sharpness_laplacian": float(laplacian.var()),
        "high_frequency": float(np.abs(highpass).mean()),
        "local_contrast": float(np.abs(local_detail).mean()),
        "gradient_mean": float(gradient.mean()),
        "edge_density": float((edges > 0).mean()),
        "noise_estimate": float(np.median(np.abs(highpass)) * 1.4826 / 255.0),
        "entropy": float(entropy),
        "clipped_black_fraction": float((luma <= 2 / 255).mean()),
        "clipped_white_fraction": float((luma >= 253 / 255).mean()),
    }


def pair_parameters(original: np.ndarray, corrected: np.ndarray) -> dict[str, float | str]:
    height, width = corrected.shape[:2]
    aligned = resize_to(original, width, height)
    # Metrics do not need full press-resolution pixels and are more stable at a
    # common analysis size.
    max_side = 1024
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        width2, height2 = max(1, round(width * scale)), max(1, round(height * scale))
        aligned = resize_to(aligned, width2, height2)
        corrected = resize_to(corrected, width2, height2)

    before = image_parameters(aligned)
    after = image_parameters(corrected)
    result: dict[str, float | str] = {}
    for name in before:
        result[f"original_{name}"] = before[name]
        result[f"corrected_{name}"] = after[name]
        result[f"delta_{name}"] = after[name] - before[name]

    a = aligned.astype(np.float32) / 255.0
    b = corrected.astype(np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    result["mae"] = float(np.mean(np.abs(a - b)))
    result["psnr"] = float(99.0 if mse == 0 else 10.0 * math.log10(1.0 / mse))
    result["ssim"] = float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
    return result
