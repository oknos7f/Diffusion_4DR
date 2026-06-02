import torch
import torch.nn as nn
import torch.nn.functional as F

class PointPillarsEncoder(nn.Module):
    """
    PointPillars Encoder for 4D Radar.
    Voxelizes RPC into pillars, applies PFN, and scatters to a pseudo-image 
    before projecting to the LDM conditioning dimension.
    """
    def __init__(self, 
                 num_points: int = 512, 
                 grid_size: int = 32, 
                 feat_channels: int = 64, 
                 out_dim: int = 768, 
                 token_len: int = 77):
        super().__init__()
        self.grid_size = grid_size
        self.num_points = num_points
        self.token_len = token_len
        
        # Simple Pillar Feature Network (PFN)
        self.pfn = nn.Sequential(
            nn.Linear(5, feat_channels), # [x, y, z, p, d]
            nn.BatchNorm1d(feat_channels),
            nn.ReLU(),
            nn.Linear(feat_channels, feat_channels),
            nn.BatchNorm1d(feat_channels),
            nn.ReLU()
        )
        
        # Feature Mapping to sequence
        # grid_size^2 -> token_len, feat_channels -> out_dim
        self.project_feat = nn.Linear(feat_channels, out_dim)
        self.project_seq = nn.Linear(grid_size * grid_size, token_len)

    def forward(self, rpc: torch.Tensor) -> torch.Tensor:
        """
        rpc: [Batch, N, 5]
        returns: [Batch, 77, 768]
        """
        B, N, C = rpc.shape
        
        # 1. Simple Voxelization (Pillar-style)
        # For simplicity, we assume rpc is already within a normalized range [-1, 1]
        # and we map them to a grid_size x grid_size grid.
        coords = ((rpc[:, :, :2] + 1) / 2 * (self.grid_size - 1)).long()
        coords = torch.clamp(coords, 0, self.grid_size - 1)
        
        # 2. Apply PFN to each point
        flat_rpc = rpc.view(-1, C)
        feat = self.pfn(flat_rpc) # [B*N, feat_channels]
        feat = feat.view(B, N, -1)
        
        # 3. Scatter to Pseudo-Image (Vectorized)
        pseudo_image = torch.zeros(B, self.grid_size * self.grid_size, feat.shape[-1], device=rpc.device)
        indices = coords[:, :, 0] * self.grid_size + coords[:, :, 1] # [B, N]
        
        # Reshape for scatter_add
        # indices needs to be [B, N, C]
        indices = indices.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
        pseudo_image.scatter_add_(1, indices, feat)
            
        # 4. Final Projection to [Batch, 77, 768]
        out = self.project_feat(pseudo_image) # [B, 1024, 768]
        out = out.transpose(1, 2) # [B, 768, 1024]
        out = self.project_seq(out) # [B, 768, 77]
        return out.transpose(1, 2)
