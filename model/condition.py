import torch
import torch.nn as nn
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class ConditionProjector(nn.Module):
    """
    복셀희소행렬을 크로스 어텐션을 위한 임베딩벡터로 변환합니다.
    Input: (N, 4) sparse voxel matrix (x, y, z, value)
    Output: (1, K, context_dim) context embedding
    """
    def __init__(self, config_path: str):
        super().__init__()
        
        self.config = load_config(config_path)['condition']
        
        self.in_features = int(self.config.get('in_features', 4))
        self.context_dim = int(self.config.get('context_dim', 512))
        self.num_tokens = int(self.config.get('num_tokens', 1024))
        
        # Simple projection for each voxel (x, y, z, value)
        self.projection = nn.Sequential(
            nn.Linear(self.in_features, self.context_dim),
            nn.GELU(),
            nn.LayerNorm(self.context_dim)
        )
    
    def forward(self, condition_input: torch.Tensor) -> torch.Tensor:
        # condition_input: (B, N, 4) where N is the number of sparse voxels (can be variable)
        B, N, _ = condition_input.shape
        
        # 1. Project each voxel feature
        projected = self.projection(condition_input)  # (B, N, context_dim)
        
        # 2. Simplified Aggregation/Padding/Truncation to fixed K=num_tokens
        # We'll use a simple truncation/padding for fixed length context.
        if N > self.num_tokens:
            context = projected[:, :self.num_tokens, :]  # Truncate
        elif N < self.num_tokens:
            padding = torch.zeros(B, self.num_tokens - N, self.context_dim,
                                  dtype=projected.dtype, device=projected.device)
            context = torch.cat([projected, padding], dim=1)  # Pad
        else:
            context = projected
        
        return context  # (B, num_tokens, context_dim)