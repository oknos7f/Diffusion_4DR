import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism.
    Q (Query): U-Net feature map (latent)
    K, V (Key, Value): Conditional Input (context)
    """
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.0):
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
    Combines DoubleConv block with Residual connection and Cross-Attention.
    """
    
    def __init__(self, ch: int, context_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, ch)
        self.attention = CrossAttention(query_dim=ch, context_dim=context_dim)
        self.norm2 = nn.GroupNorm(32, ch)
        self.linear_feedforward = nn.Sequential(
            nn.Linear(ch, ch * 4),
            nn.GELU(),
            nn.Linear(ch * 4, ch)
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Reshape for Attention: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        # 1. Self-Attention (or Identity in this minimal example) is often here,
        #    but we focus on Cross-Attention integration first.
        
        # 2. Cross-Attention with Residual connection
        # (B, H*W, C) + CrossAttention(Norm(B, H*W, C), context)
        # Note: GroupNorm needs (B, C, H, W) input, so we apply it on the original feature map.
        
        # Cross-Attention path (Applying Norm before Q projection for attention input)
        # Applying Norm on the flattened input for simplicity (often it is applied before Q projection)
        normed_x_flat = self.norm1(x).view(B, C, -1).transpose(1, 2)
        
        attn_output = self.attention(normed_x_flat, context)
        
        # Residual connection
        x_attn = x_flat + attn_output
        
        # 3. Feed-Forward Network (FFN) with Residual connection
        # (B, H*W, C) + FFN(Norm(B, H*W, C))
        normed_x_attn = self.norm2(x_attn.transpose(1, 2).view(B, C, H, W)).view(B, C, -1).transpose(1, 2)
        ff_output = self.linear_feedforward(normed_x_attn)
        
        x_out = x_attn + ff_output
        
        # Reshape back to feature map: (B, H*W, C) -> (B, C, H, W)
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        
        return x_out