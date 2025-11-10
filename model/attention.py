import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism.
    Q (Query): U-Net feature map (latent)
    K, V (Key, Value): Conditional Input (context)
    """
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear layers for Q, K, V projections
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)  # K/V dim is query_dim for easier concat
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, H*W, query_dim) (U-Net feature map)
        # context: (B, N, context_dim) (Conditional Input)
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Multi-head attention split
        # (B, L, D) -> (B, L, H, D/H) -> (B, H, L, D/H)
        q = q.view(q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention calculation: QK^T
        # (B, H, Lq, D/H) @ (B, H, D/H, Lk) -> (B, H, Lq, Lk)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum: Attn @ V
        # (B, H, Lq, Lk) @ (B, H, Lk, D/H) -> (B, H, Lq, D/H)
        out = torch.matmul(attn, v)
        
        # Concat heads
        # (B, H, Lq, D/H) -> (B, Lq, H, D/H) -> (B, Lq, D)
        out = out.transpose(1, 2).contiguous().view(out.size(0), -1, self.num_heads * self.head_dim)
        
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """
    Combines block with Residual connection and Cross-Attention (Diffusion style).
    Uses GroupNorm-Attention/FFN-Residual structure (Pre-Norm).
    """
    
    def __init__(self, config_path: str):
        super().__init__()
        
        # ... (config loading parts are the same) ...
        self.config = load_config(config_path)['attention']
        ch = int(self.config.get('ch', 256))
        context_dim = int(self.config.get('context_dim', 512))
        num_heads = int(self.config.get('num_heads', 8))
        dropout = float(self.config.get('dropout', 0.0))
        
        # GroupNorm은 (B, C, H, W)에 적용됩니다.
        # GroupNorm의 그룹 수는 보통 32가 사용됩니다.
        self.norm1 = nn.GroupNorm(32, ch)
        self.attention = CrossAttention(query_dim=ch,
                                        context_dim=context_dim,
                                        num_heads=num_heads,
                                        dropout=dropout)
        self.norm2 = nn.GroupNorm(32, ch)
        self.linear_feedforward = nn.Sequential(
            nn.Linear(ch, ch * 4),
            nn.GELU(),
            nn.Linear(ch * 4, ch)
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # 1. Cross-Attention with Residual connection (Pre-Norm style)
        residual_attn = x  # Residual 연결을 위한 원본 (B, C, H, W)
        
        # A. Norm
        normed_x = self.norm1(x)
        
        # B. Reshape for Attention: (B, C, H, W) -> (B, H*W, C)
        normed_x_flat = normed_x.view(B, C, -1).transpose(1, 2)  # (B, L, C)
        
        # C. Attention
        attn_output = self.attention(normed_x_flat, context)  # (B, L, C)
        
        # D. Add (Residual): (B, C, H, W) -> (B, L, C) 변환 후 더하기
        # x를 Flatten해서 더합니다.
        x_attn_flat = residual_attn.view(B, C, -1).transpose(1, 2) + attn_output  # (B, L, C)
        
        # ------------------------------------------------------------------
        
        # 2. Feed-Forward Network (FFN) with Residual connection (Pre-Norm style)
        residual_ffn = x_attn_flat  # Residual 연결을 위한 Attention 출력 (B, L, C)
        
        # E. Norm: (B, L, C) -> (B, C, L) -> (B, C, H, W) -> Norm -> (B, C, H, W)
        # FFN에 들어가기 전 다시 GroupNorm을 적용합니다.
        normed_x_attn = self.norm2(x_attn_flat.transpose(1, 2).view(B, C, H, W))
        
        # F. Reshape back for FFN: (B, C, H, W) -> (B, L, C)
        normed_x_attn_flat = normed_x_attn.view(B, C, -1).transpose(1, 2)  # (B, L, C)
        
        # G. FFN
        ff_output = self.linear_feedforward(normed_x_attn_flat)  # (B, L, C)
        
        # H. Add (Residual)
        x_out_flat = residual_ffn + ff_output  # (B, L, C)
        
        # I. Reshape back to feature map: (B, L, C) -> (B, C, H, W)
        x_out = x_out_flat.transpose(1, 2).view(B, C, H, W)
        
        return x_out