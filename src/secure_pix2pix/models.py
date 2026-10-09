from __future__ import annotations

import torch
from torch import nn


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            # The first decoder receives the encoder bottleneck directly. This
            # activation must not mutate that tensor before its encoder gradient
            # is computed; keeping every decoder ReLU out-of-place is safest.
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class UNetGenerator(nn.Module):
    """Eight-level 256x256 U-Net generator built only from native PyTorch layers."""

    def __init__(self, input_channels: int = 3, output_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        b = base_channels
        self.down1 = _DownBlock(input_channels, b, normalize=False)       # 256 -> 128
        self.down2 = _DownBlock(b, b * 2)                                # 128 -> 64
        self.down3 = _DownBlock(b * 2, b * 4)                            # 64 -> 32
        self.down4 = _DownBlock(b * 4, b * 8)                            # 32 -> 16
        self.down5 = _DownBlock(b * 8, b * 8)                            # 16 -> 8
        self.down6 = _DownBlock(b * 8, b * 8)                            # 8 -> 4
        self.down7 = _DownBlock(b * 8, b * 8)                            # 4 -> 2
        self.down8 = _DownBlock(b * 8, b * 8, normalize=False)           # 2 -> 1

        self.up1 = _UpBlock(b * 8, b * 8, dropout=True)                  # 1 -> 2
        self.up2 = _UpBlock(b * 16, b * 8, dropout=True)                 # 2 -> 4
        self.up3 = _UpBlock(b * 16, b * 8, dropout=True)                 # 4 -> 8
        self.up4 = _UpBlock(b * 16, b * 8)                               # 8 -> 16
        self.up5 = _UpBlock(b * 16, b * 4)                               # 16 -> 32
        self.up6 = _UpBlock(b * 8, b * 2)                                # 32 -> 64
        self.up7 = _UpBlock(b * 4, b)                                    # 64 -> 128
        self.final = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(b * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(value)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat((u1, d7), dim=1))
        u3 = self.up3(torch.cat((u2, d6), dim=1))
        u4 = self.up4(torch.cat((u3, d5), dim=1))
        u5 = self.up5(torch.cat((u4, d4), dim=1))
        u6 = self.up6(torch.cat((u5, d3), dim=1))
        u7 = self.up7(torch.cat((u6, d2), dim=1))
        return self.final(torch.cat((u7, d1), dim=1))


class PatchDiscriminator(nn.Module):
    """Conditional 70x70 PatchGAN discriminator for 256x256 pairs."""

    def __init__(self, input_channels: int = 3, target_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        b = base_channels

        def block(in_channels: int, out_channels: int, stride: int, normalize: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=not normalize,
                )
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.network = nn.Sequential(
            *block(input_channels + target_channels, b, 2, False),
            *block(b, b * 2, 2, True),
            *block(b * 2, b * 4, 2, True),
            *block(b * 4, b * 8, 1, True),
            nn.Conv2d(b * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((original, candidate), dim=1))


def initialize_pix2pix_weights(module: nn.Module) -> None:
    """Paper-compatible normal initialization for convolution and BN layers."""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
