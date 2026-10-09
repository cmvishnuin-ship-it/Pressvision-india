from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .metrics import read_rgb, resize_to
from .pairs import ImagePair


def context_view(rgb: np.ndarray, size: int) -> np.ndarray:
    """Whole-image view used by the global-context model (aspect ratio is not kept)."""
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def replay_whole_image_paths(pair: ImagePair) -> tuple[Path, Path] | None:
    """Replay patches are crops; their whole images are the sibling global replay pair."""
    original = Path(pair.original)
    if original.name.startswith("patch_"):
        source, target = original.with_name("global_original.png"), original.with_name("global_corrected.png")
        if source.is_file() and target.is_file():
            return source, target
    return None


class PairedPatchDataset(Dataset):
    """Aligned original/corrected patches.

    Validation datasets (``augment=False``) crop at positions derived from the
    seed and pair path, so every epoch scores the model on identical pixels.
    The most recently decoded pair is kept, so a sampler that repeats an index
    (see train.RepeatedSampler) cuts several patches from one decode.

    With ``context_size`` each item also carries whole-image views of the
    original and of the editors' result: the first is the model's context, the
    second supervises its image-wide correction.
    """

    def __init__(self, pairs: list[ImagePair], patch_size: int = 192, patches_per_image: int = 8, augment: bool = True, parameter_targets=None, parameter_center=None, parameter_scale=None, context_size: int = 0, seed: int = 42):
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.parameter_targets = parameter_targets
        self.parameter_center = parameter_center
        self.parameter_scale = parameter_scale
        self.context_size = context_size
        self.seed = seed
        self._decoded: tuple[int, np.ndarray, np.ndarray, np.ndarray | None] | None = None

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def _decode(self, pair_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self._decoded is not None and self._decoded[0] == pair_index:
            return self._decoded[1:]
        pair = self.pairs[pair_index]
        source = read_rgb(pair.original)
        target = read_rgb(pair.corrected)
        height, width = target.shape[:2]
        source = resize_to(source, width, height)
        context = None
        if self.context_size > 0:
            siblings = replay_whole_image_paths(pair)
            whole_source, whole_target = (read_rgb(siblings[0]), read_rgb(siblings[1])) if siblings else (source, target)
            # Original and editors' views stacked as six channels, so flips and
            # rotations below always keep them aligned.
            context = np.concatenate((context_view(whole_source, self.context_size), context_view(whole_target, self.context_size)), axis=2)

        minimum = self.patch_size
        if min(height, width) < minimum:
            scale = minimum / min(height, width)
            new_width = max(minimum, round(width * scale))
            new_height = max(minimum, round(height * scale))
            source = resize_to(source, new_width, new_height)
            target = resize_to(target, new_width, new_height)
        self._decoded = (pair_index, source, target, context)
        return source, target, context

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pair_index = index // self.patches_per_image
        pair = self.pairs[pair_index]
        source, target, context = self._decode(pair_index)
        height, width = target.shape[:2]
        minimum = self.patch_size

        chooser = random if self.augment else random.Random(f"{self.seed}:{pair.original}:{index % self.patches_per_image}")
        top = chooser.randint(0, height - minimum)
        left = chooser.randint(0, width - minimum)
        source = source[top : top + minimum, left : left + minimum]
        target = target[top : top + minimum, left : left + minimum]

        if self.augment:
            if random.random() < 0.5:
                source, target = source[:, ::-1], target[:, ::-1]
                context = context[:, ::-1] if context is not None else None
            if random.random() < 0.5:
                source, target = source[::-1], target[::-1]
                context = context[::-1] if context is not None else None
            rotations = random.randint(0, 3)
            source, target = np.rot90(source, rotations), np.rot90(target, rotations)
            context = np.rot90(context, rotations) if context is not None else None

        source_tensor = torch.from_numpy(np.ascontiguousarray(source.transpose(2, 0, 1))).float() / 255.0
        target_tensor = torch.from_numpy(np.ascontiguousarray(target.transpose(2, 0, 1))).float() / 255.0
        if self.parameter_targets is None:
            return source_tensor, target_tensor
        raw = self.parameter_targets[(pair.original, pair.corrected)]
        normalized = np.clip((raw - self.parameter_center) / self.parameter_scale, -6.0, 6.0)
        item = (source_tensor, target_tensor, torch.from_numpy(normalized.astype(np.float32)))
        if context is not None:
            views = torch.from_numpy(np.ascontiguousarray(context.transpose(2, 0, 1))).float() / 255.0
            item += (views[:3], views[3:])
        return item


class MaterializedDataset(Dataset):
    """Decode a deterministic dataset once and keep its samples in memory.

    Image tensors are stored as uint8 (they originate from 8-bit pixels, so the
    round trip is exact); other tensors are kept unchanged.
    """

    def __init__(self, dataset: Dataset):
        self.items = []
        for index in range(len(dataset)):
            self.items.append(tuple(
                (value * 255.0).round().to(torch.uint8) if value.dim() == 3 else value
                for value in dataset[index]
            ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(value.float() / 255.0 if value.dtype == torch.uint8 else value for value in self.items[index])


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return torch.nn.functional.l1_loss(pred_x, target_x) + torch.nn.functional.l1_loss(pred_y, target_y)


def color_moment_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimensions = (2, 3)
    mean_loss = torch.nn.functional.l1_loss(prediction.mean(dimensions), target.mean(dimensions))
    std_loss = torch.nn.functional.l1_loss(prediction.std(dimensions), target.std(dimensions))
    return mean_loss + std_loss


def _gaussian_window(size: int, sigma: float, channels: int, reference: torch.Tensor) -> torch.Tensor:
    coordinates = torch.arange(size, dtype=reference.dtype, device=reference.device) - (size - 1) / 2.0
    vector = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    vector = vector / vector.sum()
    kernel = torch.outer(vector, vector)
    return kernel.expand(channels, 1, size, size).contiguous()


def gaussian_ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1 - mean SSIM using a proper Gaussian window.

    Padded (unlike secure_pix2pix's eval-only per-image ssim_per_image, which
    this mirrors) so it keeps operating over the full training patch instead
    of shrinking the spatial map by window_size-1 on each side.
    """
    channels = prediction.shape[1]
    window = _gaussian_window(window_size, sigma, channels, prediction)
    pad = window_size // 2
    mu_p = torch.nn.functional.conv2d(prediction, window, groups=channels, padding=pad)
    mu_t = torch.nn.functional.conv2d(target, window, groups=channels, padding=pad)
    mu_p_sq, mu_t_sq, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sigma_p = torch.nn.functional.conv2d(prediction * prediction, window, groups=channels, padding=pad) - mu_p_sq
    sigma_t = torch.nn.functional.conv2d(target * target, window, groups=channels, padding=pad) - mu_t_sq
    sigma_pt = torch.nn.functional.conv2d(prediction * target, window, groups=channels, padding=pad) - mu_pt
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / ((mu_p_sq + mu_t_sq + c1) * (sigma_p + sigma_t + c2) + 1e-8)
    return 1.0 - ssim.mean()
