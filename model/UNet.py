
import torch
import torch.nn as nn


class Conv3dLayer(nn.Sequential):
def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
super().__init__(
nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
stride=stride, padding=padding, bias=bias),
nn.BatchNorm3d(out_channels),
nn.ReLU()
)


class CloudPointEncoder(nn.Module):
def __init__(self, input_channels=1, output_dim=768, base_channels=32):
super().__init__()
self.down = nn.Sequential(
# Input: (B, 1, 256, 107, 37)
Conv3dLayer(input_channels, base_channels, 3, 2, 1),
Conv3dLayer(base_channels, base_channels * 2, 3, 2, 1),
Conv3dLayer(base_channels * 2, base_channels * 4, 3, 2, 1),
nn.MaxPool3d(kernel_size=(4, 2, 2))
)
final_feature_channels = base_channels * 4
final_spatial_dim = 4 * 16 * 4
flatten_dim = final_feature_channels * final_spatial_dim
self.to_embedding = nn.Sequential(
nn.AdaptiveAvgPool3d((4, 4, 4)),
nn.Flatten(),
nn.Linear(final_feature_channels * 4 * 4 * 4, output_dim)
)
def forward(self, x):
# x shape: (B, 1, 256, 107, 37)
x = self.down(x)
embedding = self.to_embedding(x)
return embedding.unsqueeze(1)