import torch
import torch.nn as nn
from typing import List
import yaml

from blocks import DoubleConv, Down, Up, OutConv


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class UNet2D(nn.Module):
    """
    2D U-Net implementation
    """
    def __init__(self, config_path: str):
        super().__init__()
        
        self.config = load_config(config_path)['unet']
        
        in_channels = int(self.config.get('in_channels', 4))
        out_channels = int(self.config.get('out_channels', 4))
        base_channels = int(self.config.get('base_channels', 64))
        depth = int(self.config.get('depth', 4))

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