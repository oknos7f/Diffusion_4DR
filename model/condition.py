import torch
import torch.nn as nn


class ConditionProjector(nn.Module):
    """
    Projects the Conditional Input (Sparse Voxel Matrix) into a context
    embedding suitable for Cross-Attention.
    Input: (N, 4) sparse voxel matrix (x, y, z, value)
    Output: (1, K, context_dim) context embedding
    """
    
    def __init__(self, in_features: int = 4, context_dim: int = 512, num_tokens: int = 77):
        super().__init__()
        self.context_dim = context_dim
        self.num_tokens = num_tokens  # Standard LDM context length
        
        # Simple projection for each voxel (x, y, z, value)
        self.projection = nn.Sequential(
            nn.Linear(in_features, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim)
        )
        
        # If the input is sparse/variable-length (N), we need to aggregate or pad/truncate.
        # For simplicity, we'll assume we can aggregate or pad to a fixed `num_tokens`.
        # Here we'll use a simple aggregation/linear layer approach for demonstration.
        # This part is highly dependent on the exact voxel encoding scheme.
        
        # Since the input is (N, 4), we'll simplify:
        # 1. Project N voxels to (N, context_dim).
        # 2. Pad/Truncate to (num_tokens, context_dim). (This assumes batch size B=1 for simplicity
        #    but we'll design it to handle B>1).
        
        # A more robust LDM approach uses a transformer encoder, but for minimal modification:
        # We will assume the input is already pre-processed/padded to (B, num_tokens, 4)
        # or we will pad it to (B, N_max, 4) and use a mask if needed, but given the
        # sparsity and (N, 4) format, let's assume we map N tokens to K tokens.
        
        # Simplified aggregation: Linear layer to map N features to K tokens.
        # A more rigorous LDM would use a Transformer/CNN for this.
        
        # Let's assume input is (B, N, 4) and we project it to (B, num_tokens, context_dim).
        # We'll use a placeholder/simplified aggregation that might need refinement.
    
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