from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class CorrectionUNet(nn.Module):
    """Small residual U-Net which preserves image size and predicts corrections."""

    def __init__(self, base_channels: int = 24):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.bottleneck = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.residual = nn.Conv2d(c, 3, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(value)
        e2 = self.enc2(self.pool(e1))
        middle = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat((self.up2(middle), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        # Starting as identity is important: before training the network cannot
        # accidentally destroy production images.
        return torch.clamp(value + 0.25 * torch.tanh(self.residual(d1)), 0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.layers(value)


class AttentionBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.convolutions = ConvBlock(input_channels, output_channels)
        self.attention = ChannelAttention(output_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.attention(self.convolutions(value))


class CorrectionUNetV2(nn.Module):
    """Higher-capacity correction model with an interpretable parameter head."""

    def __init__(self, base_channels: int = 32, parameter_count: int = 34):
        super().__init__()
        c = base_channels
        self.enc1 = AttentionBlock(3, c)
        self.enc2 = AttentionBlock(c, c * 2)
        self.enc3 = AttentionBlock(c * 2, c * 4)
        self.middle = AttentionBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = AttentionBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = AttentionBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = AttentionBlock(c * 2, c)
        self.residual = nn.Conv2d(c, 3, 1)
        self.parameter_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c * 4, parameter_count),
        )
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward_with_parameters(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(value)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        middle = self.middle(self.pool(e3))
        parameters = self.parameter_head(middle)
        d3 = self.dec3(torch.cat((self.up3(middle), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        corrected = torch.clamp(value + 0.35 * torch.tanh(self.residual(d1)), 0.0, 1.0)
        return corrected, parameters

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        corrected, _ = self.forward_with_parameters(value)
        return corrected
