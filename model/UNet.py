import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) twice"""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv. Uses ConvTranspose2d."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: 현재 디코더 특성, x2: 해당 인코더 특성 (스킵 연결)
        x1 = self.up(x1)
        
        # 일관성을 위해 넣은 패딩 코드
        diffy = x2.size(2) - x1.size(2)
        diffx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1x1 convolution to desired output channels"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    """
    2D U-Net implementation. (Replaced 3D with 2D ops)

    Args:
      in_channels: input channels (e.g., 3 for RGB)
      out_channels: number of output channels (e.g., 3 for predicted noise)
      base_channels: number of filters at first level (will double each down)
      depth: how many down/up levels (default 4)
      use_transpose: whether to use ConvTranspose2d for upsampling
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 64,
                 depth: int = 4):
        super().__init__()
        self.depth = depth

        # Encoder path
        self.inc = DoubleConv(in_channels, base_channels)
        downs: List[nn.Module] = []
        chs = base_channels
        for _ in range(depth - 1):
            downs.append(Down(chs, chs * 2))
            chs *= 2
        self.downs = nn.ModuleList(downs)

        # Decoder path
        ups: List[nn.Module] = []
        for _ in range(depth - 1):
            ups.append(Up(chs * 2, chs // 2))
            chs //= 2
        self.ups = nn.ModuleList(ups)

        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x_encs = []
        x1 = self.inc(x)
        x_encs.append(x1)
        xi = x1
        for down in self.downs:
            xi = down(xi)
            x_encs.append(xi)

        # Decoder
        xj = x_encs[-1]
        for i, up in enumerate(self.ups):
            # corresponding encoder feature (skip)
            skip = x_encs[-2 - i]
            xj = up(xj, skip)

        out = self.outc(xj)
        return out


# Example usage:
# LDM의 VAE 출력(Latent)이 보통 (B, 4, H, W)이므로, in/out_channels=4로 설정하는 것이 일반적입니다.
# model = UNet2D(in_channels=4, out_channels=4, base_channels=64, depth=4)
# input = torch.randn(1, 4, 64, 64) # (B, C, H, W)
# output = model(input)  # -> shape (1, 4, 64, 64)