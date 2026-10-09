from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def denormalize_image(value: torch.Tensor) -> torch.Tensor:
    """Convert a Tanh-compatible [-1,1] tensor to clamped [0,1]."""

    return value.add(1.0).mul(0.5).clamp(0.0, 1.0)


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio for [0,1] NCHW tensors, in decibels."""

    mse = (prediction - target).square().flatten(1).mean(dim=1)
    # 120 dB is the finite representation used for an exact match.
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1.0e-12))


def l1_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error for [0,1] NCHW tensors, per image."""

    return (prediction - target).abs().flatten(1).mean(dim=1)


def _gaussian_window(size: int, sigma: float, channels: int, reference: torch.Tensor) -> torch.Tensor:
    coordinates = torch.arange(size, dtype=reference.dtype, device=reference.device)
    coordinates = coordinates - (size - 1) / 2.0
    vector = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    vector = vector / vector.sum()
    kernel = torch.outer(vector, vector)
    return kernel.expand(channels, 1, size, size).contiguous()


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Structural similarity using the standard Gaussian-window formulation."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM expects prediction and target with equal NCHW shapes")
    if min(prediction.shape[-2:]) < window_size:
        raise ValueError(f"SSIM inputs must be at least {window_size}x{window_size}")

    channels = prediction.shape[1]
    window = _gaussian_window(window_size, sigma, channels, prediction)
    mu_prediction = F.conv2d(prediction, window, groups=channels)
    mu_target = F.conv2d(target, window, groups=channels)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target

    sigma_prediction = F.conv2d(prediction.square(), window, groups=channels) - mu_prediction_sq
    sigma_target = F.conv2d(target.square(), window, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(prediction * target, window, groups=channels) - mu_product

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    numerator = (2.0 * mu_product + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        sigma_prediction + sigma_target + c2
    )
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)).flatten(1).mean(dim=1)
