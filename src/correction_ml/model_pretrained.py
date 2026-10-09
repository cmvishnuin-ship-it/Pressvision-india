"""V2 correction model with an additional frozen ImageNet-pretrained feature branch.

This is a sibling to model.py, not a modification of it: CorrectionUNetV2 stays
byte-for-byte unchanged so existing best_v2.pt/last_v2.pt checkpoints are never
put at risk. This variant is experimental and runs only as a shadow worker
(see learning_service.py's pretrained_backbone_shadow) -- it never replaces the
primary V2 checkpoint automatically.

Bootstrap the bundled weights file once, on a machine with internet access:
    python -c "import torch; from torchvision.models import resnet18, ResNet18_Weights; \
        torch.save(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict(), \
        'artifacts/pretrained/resnet18_imagenet.pt')"
The packaged service never downloads weights at runtime; it only reads that
local file (see learning_service.bootstrap_pretrained_weights).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .model import AttentionBlock


class _FrozenResNet18Features(nn.Module):
    """ImageNet-pretrained stem+layer1+layer2, frozen, loaded from a local file only."""

    def __init__(self, weights_path: str | Path | None) -> None:
        super().__init__()
        from torchvision.models import resnet18  # local import: keep torchvision optional at module load

        backbone = resnet18(weights=None)  # never triggers a network download
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            backbone.load_state_dict(state)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()

    def train(self, mode: bool = True) -> "_FrozenResNet18Features":
        # Stay in eval mode unconditionally: BatchNorm running stats must not
        # drift even though the parent model's train(True) is called every
        # epoch. Freezing requires_grad alone does not stop that drift.
        return super().train(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = (value - self.mean) / self.std
        return self.layer2(self.layer1(self.stem(normalized)))


class CorrectionUNetV2PretrainedBackbone(nn.Module):
    """CorrectionUNetV2 with an additional frozen ImageNet-pretrained feature branch.

    The frozen branch runs directly on the raw input image (stride 8, 128
    channels at resnet18's layer2) and is fused with the pooled enc3 output,
    which sits at the same resolution and channel count -- the decoder,
    parameter head, and every other stage are identical to CorrectionUNetV2.
    """

    def __init__(self, base_channels: int = 32, parameter_count: int = 34,
                 pretrained_weights_path: str | Path | None = None) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = AttentionBlock(3, c)
        self.enc2 = AttentionBlock(c, c * 2)
        self.enc3 = AttentionBlock(c * 2, c * 4)
        self.resnet_features = _FrozenResNet18Features(pretrained_weights_path)
        self.fusion = nn.Conv2d(c * 4 + 128, c * 4, kernel_size=1)
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
        pooled_e3 = self.pool(e3)
        fused = self.fusion(torch.cat((pooled_e3, self.resnet_features(value)), dim=1))
        middle = self.middle(fused)
        parameters = self.parameter_head(middle)
        d3 = self.dec3(torch.cat((self.up3(middle), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        corrected = torch.clamp(value + 0.35 * torch.tanh(self.residual(d1)), 0.0, 1.0)
        return corrected, parameters

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        corrected, _ = self.forward_with_parameters(value)
        return corrected
