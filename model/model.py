import torch
import torch.nn as nn
from typing import List
import yaml

from condition import ConditionProjector
from blocks import DoubleConv, Down, Up, OutConv
from attention import ConditionalBlock


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class UNet2DConditional(nn.Module):
    """
    2D U-Net with Cross-Attention for Conditional Input (LDM style).
    """
    
    def __init__(self, config_path: str):
        super().__init__()
        
        self.config = load_config(config_path)['unet']
        
        in_channels = int(self.config.get('in_channels', 4))
        out_channels = int(self.config.get('out_channels', 4))
        base_channels = int(self.config.get('base_channels', 64))
        depth = int(self.config.get('depth', 4))
        context_dim = int(self.config.get('context_dim', 512))
        
        self.depth = depth
        self.context_dim = context_dim
        
        # Condition Projector
        # Assuming the sparse input (x, y, z, value) is batched as (B, N, 4)
        self.cond_proj = ConditionProjector('configs/config.yaml')
        
        # Encoder path
        self.inc = DoubleConv(in_channels, base_channels)
        downs: List[nn.Module] = []
        attns: List[nn.Module] = []
        chs = base_channels
        for _ in range(depth - 1):
            # Downscaling part
            downs.append(Down(chs, chs * 2))
            chs *= 2
            # Cross-Attention block after each Down/DoubleConv (except first and last)
            # Standard LDM often places attention in the residual blocks or bottlenecks.
            attns.append(ConditionalBlock(config_path))
        
        self.downs = nn.ModuleList(downs)
        self.attns = nn.ModuleList(attns)
        
        # Bottleneck (Optional: Add ConditionalBlock here too)
        self.bottleneck_attn = ConditionalBlock(config_path)
        
        # Decoder path
        ups: List[nn.Module] = []
        for _ in range(depth - 1):
            ups.append(Up(chs * 2, chs // 2))
            chs //= 2
        self.ups = nn.ModuleList(ups)
        
        self.outc = OutConv(base_channels, out_channels)
    
    def forward(self, x: torch.Tensor, conditional_input: torch.Tensor) -> torch.Tensor:
        # 1. Condition Projection
        context = self.cond_proj(conditional_input)  # (B, K, context_dim)
        
        # 2. Encoder
        x_encs = []
        x1 = self.inc(x)
        x_encs.append(x1)
        xi = x1
        
        for i, down in enumerate(self.downs):
            xi = down(xi)  # Down + DoubleConv (B, C*2, H/2, W/2)
            
            # Cross-Attention Integration
            xi = self.attns[i](xi, context)
            
            x_encs.append(xi)
        
        # 3. Bottleneck (Deepest level)
        xj = x_encs[-1]
        xj = self.bottleneck_attn(xj, context)
        
        # 4. Decoder
        # x_encs: [x1, x2, ..., x_last] (x1: 가장 큰 해상도, x_last: 가장 작은 해상도)
        for i, up in enumerate(self.ups):
            # skip: x_encs[-2 - i] (해당 디코더 레벨과 같은 해상도의 인코더 특성)
            skip = x_encs[-2 - i]
            xj = up(xj, skip)
        
        out = self.outc(xj)
        return out