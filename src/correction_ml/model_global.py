"""V2 correction model with a whole-image context branch (V3 "global").

Production edits are dominated by image-wide adjustments (levels/curves,
colour balance, saturation) that depend on the whole photograph. V2 trains on
64x64 crops and cannot tell whether the image around a crop is dark or bright,
so it can only learn an average local correction. This variant adds:

* a small encoder over a low-resolution view of the *whole* source image,
* per-image monotonic tone curves (one per RGB channel) and a 3x3 colour mix
  predicted from that context and applied to the full-resolution pixels,
* FiLM modulation of V2's bottleneck and last decoder stage by the context,
* a context term in the global-parameter head.

Every new output layer is zero-initialised, so a model warm-started from a V2
checkpoint (``load_v2_weights``) reproduces V2's output exactly before its first
update. Like model_pretrained.py this is a sibling of model.py: V2 and its
checkpoints are not modified.
"""
from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .model import CorrectionUNetV2

HISTOGRAM_BINS = 16
# Layers V3 adds to V2; every other parameter is warm-started from a V2 checkpoint.
NEW_LAYER_PREFIXES = ("context_encoder.", "middle_film.", "output_film.", "curve_head.", "color_head.", "parameter_context.")


class GlobalContextEncoder(nn.Module):
    """Summarise a low-resolution whole-image view into one context vector."""

    def __init__(self, output_features: int = 128):
        super().__init__()
        layers = []
        channels = 3
        for width in (16, 32, 64, 128):
            layers += [
                nn.Conv2d(channels, width, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, width), width),
                nn.SiLU(inplace=True),
            ]
            channels = width
        self.convolutions = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        # CNN features plus explicit colour statistics: per-channel mean and
        # standard deviation and a luma histogram, which tone decisions rely on.
        self.mixer = nn.Sequential(
            nn.Linear(channels + 6 + HISTOGRAM_BINS, output_features),
            nn.SiLU(inplace=True),
            nn.Linear(output_features, output_features),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def image_statistics(context: torch.Tensor) -> torch.Tensor:
        flat = context.flatten(2)
        luma = 0.2126 * flat[:, 0] + 0.7152 * flat[:, 1] + 0.0722 * flat[:, 2]
        bins = (luma * HISTOGRAM_BINS).long().clamp(0, HISTOGRAM_BINS - 1)
        histogram = functional.one_hot(bins, HISTOGRAM_BINS).float().mean(1)
        return torch.cat((flat.mean(2), flat.std(2), histogram), dim=1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        features = self.convolutions(context)
        return self.mixer(torch.cat((features, self.image_statistics(context)), dim=1))


class CorrectionUNetV3Global(CorrectionUNetV2):
    """CorrectionUNetV2 conditioned on a low-resolution view of the whole image."""

    def __init__(self, base_channels: int = 32, parameter_count: int = 34, context_size: int = 128,
                 curve_knots: int = 16, context_features: int = 128):
        super().__init__(base_channels, parameter_count)
        c = base_channels
        self.context_size = context_size
        self.curve_knots = curve_knots
        self.context_encoder = GlobalContextEncoder(context_features)
        self.middle_film = nn.Linear(context_features, 2 * c * 8)
        self.output_film = nn.Linear(context_features, 2 * c)
        self.curve_head = nn.Linear(context_features, 3 * curve_knots)
        self.color_head = nn.Linear(context_features, 12)
        self.parameter_context = nn.Linear(context_features, parameter_count)
        for layer in (self.middle_film, self.output_film, self.curve_head, self.color_head, self.parameter_context):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def load_v2_weights(self, state_dict: dict) -> None:
        """Warm-start from a CorrectionUNetV2 state dict; new layers keep their identity init."""
        result = self.load_state_dict(state_dict, strict=False)
        unexpected_missing = [key for key in result.missing_keys if not key.startswith(NEW_LAYER_PREFIXES)]
        if unexpected_missing or result.unexpected_keys:
            raise ValueError(
                f"Not a compatible V2 state dict: missing={unexpected_missing[:5]} unexpected={result.unexpected_keys[:5]}"
            )

    def freeze_v2_layers(self) -> None:
        """Train only the layers V3 adds, so the warm-started V2 layers cannot drift."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(NEW_LAYER_PREFIXES))

    def context_from(self, value: torch.Tensor) -> torch.Tensor:
        """Fallback context when no whole-image view is supplied: the input itself."""
        return functional.interpolate(value, size=(self.context_size, self.context_size), mode="area")

    def global_tone(self, value: torch.Tensor, context_vector: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        knots = self.curve_knots
        # Positive slopes keep every channel curve monotonic; exp(0) = 1 is identity.
        slopes = torch.exp(self.curve_head(context_vector).view(batch, 3, knots).clamp(-3.0, 3.0))
        segment = 1.0 / knots
        curved = torch.zeros_like(value)
        for knot in range(knots):
            curved = curved + slopes[:, :, knot, None, None] * (value - knot * segment).clamp(0.0, segment)
        mix = self.color_head(context_vector)
        matrix = mix[:, :9].view(batch, 3, 3)
        bias = mix[:, 9:]
        return curved + torch.einsum("bij,bjhw->bihw", matrix, curved) + bias[:, :, None, None]

    def whole_image_correction(self, context: torch.Tensor) -> torch.Tensor:
        """The image-wide part of the correction (tone curves and colour mix) applied to the whole-image view.

        Training compares this with the editors' whole image, so the curves learn
        from the entire picture and not only through one small crop.
        """
        return torch.clamp(self.global_tone(context, self.context_encoder(context)), 0.0, 1.0)

    def forward_with_parameters(self, value: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if context is None:
            context = self.context_from(value)
        context_vector = self.context_encoder(context)
        e1 = self.enc1(value)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        middle = self.middle(self.pool(e3))
        scale, shift = self.middle_film(context_vector).chunk(2, dim=1)
        middle = middle * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        parameters = self.parameter_head(middle) + self.parameter_context(context_vector)
        d3 = self.dec3(torch.cat((self.up3(middle), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        scale, shift = self.output_film(context_vector).chunk(2, dim=1)
        d1 = d1 * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        corrected = torch.clamp(self.global_tone(value, context_vector) + 0.35 * torch.tanh(self.residual(d1)), 0.0, 1.0)
        return corrected, parameters

    def forward(self, value: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        corrected, _ = self.forward_with_parameters(value, context)
        return corrected
