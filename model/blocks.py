import torch
import torch.nn as nn
import torch.nn.functional as F


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