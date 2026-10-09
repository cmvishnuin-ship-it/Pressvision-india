from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# Observable correction effects predicted by the auxiliary model head. Values are
# learned jointly with corrected pixels, making the checkpoint explainable without
# pretending that hidden Photoshop slider values can be recovered uniquely.
PARAMETER_NAMES = [
    "scale_x",
    "scale_y",
    "aspect_change",
    "delta_brightness",
    "delta_contrast",
    "delta_red_mean",
    "delta_green_mean",
    "delta_blue_mean",
    "delta_saturation",
    "delta_saturation_std",
    "delta_colorfulness",
    "delta_temperature_proxy",
    "delta_tint_proxy",
    "delta_hue_sin",
    "delta_hue_cos",
    "delta_shadow_mean",
    "delta_midtone_mean",
    "delta_highlight_mean",
    "delta_luma_p10",
    "delta_luma_p25",
    "delta_luma_p50",
    "delta_luma_p75",
    "delta_luma_p90",
    "delta_dynamic_range",
    "delta_gamma_proxy",
    "delta_clipped_black_fraction",
    "delta_clipped_white_fraction",
    "delta_sharpness_laplacian",
    "delta_high_frequency",
    "delta_local_contrast",
    "delta_gradient_mean",
    "delta_edge_density",
    "delta_noise_estimate",
    "delta_entropy",
]


# Resolution-stable source characteristics used to detect images unlike the
# training distribution. Detail/noise metrics are excluded because they change
# materially with preview resolution.
INPUT_FEATURE_NAMES = [
    "brightness",
    "contrast",
    "red_mean",
    "green_mean",
    "blue_mean",
    "saturation",
    "saturation_std",
    "colorfulness",
    "temperature_proxy",
    "tint_proxy",
    "dynamic_range",
    "gamma_proxy",
    "clipped_black_fraction",
    "clipped_white_fraction",
    "entropy",
]


def load_parameter_targets(path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    targets: dict[tuple[str, str], np.ndarray] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                values = np.asarray([float(row[name]) for name in PARAMETER_NAMES], dtype=np.float32)
            except (KeyError, TypeError, ValueError):
                continue
            if np.all(np.isfinite(values)):
                targets[(row["original"], row["corrected"])] = values
    return targets


def load_input_features(path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    features: dict[tuple[str, str], np.ndarray] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                values = np.asarray([float(row[f"original_{name}"]) for name in INPUT_FEATURE_NAMES], dtype=np.float32)
            except (KeyError, TypeError, ValueError):
                continue
            if np.all(np.isfinite(values)):
                features[(row["original"], row["corrected"])] = values
    return features


def robust_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0).astype(np.float32)
    q25 = np.percentile(values, 25, axis=0)
    q75 = np.percentile(values, 75, axis=0)
    scale = ((q75 - q25) / 1.349).astype(np.float32)
    standard = values.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, np.where(standard > 1e-6, standard, 1.0))
    return center, scale
