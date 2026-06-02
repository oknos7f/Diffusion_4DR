import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate Euclidean squared distance between two point sets.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Select points using indices.
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Sample npoint furthest points from xyz dataset.
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Group points within a ball radius.
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction layer for PointNet++, decoupling spatial geometry and features.
    """
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: list):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> tuple:
        """
        Forward pass. xyz is [B, N, 3], points is [B, N, C].
        Returns grouped new_xyz and new_points.
        """
        xyz = xyz.contiguous()
        B, N, C = xyz.shape
        
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, 3)
        
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
            
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+3, nsample, npoint]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        new_points = torch.max(new_points, 2)[0] # [B, C_out, npoint]
        return new_xyz, new_points.permute(0, 2, 1)

class PhysicsAwareRadarEncoder(nn.Module):
    """
    Custom Hierarchical PointNet++ taking [x, y, z, power_tilde, doppler].
    Decouples geometry [x,y,z] from attributes. Final features mapped to [Batch, 77, 768].
    """
    def __init__(self):
        super().__init__()
        # Physics-Aware Embedding Layer for [power, doppler]
        self.physics_embed = nn.Linear(2, 16)
        
        # SA1: 512 points -> 128 points
        # in_channel for SA1 is 16 (physics embed) + 3 (xyz) = 19 -> wait, SetAbstraction adds +3 internally.
        # So we pass 16 to SA, and it concatenates internal grouped_xyz (3), total 19.
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=16 + 3, mlp=[64, 64, 128])
        
        # SA2: 128 points -> 128 points (Wait, prompt says output map of dimension [Batch, 128, 512])
        # If output is 128 points with 512 features, npoint=128, mlp ends in 512.
        # Let's say we do one more SA step, or just use SA1 and SA2.
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[256, 256, 512])

        # MLP Alignment: Project raw feature map [B, 128, 512] -> [B, 77, 768]
        # We can treat the 128 points as sequence length and project both sequence length and feature dim.
        # First project features: 512 -> 768
        self.align_features = nn.Linear(512, 768)
        # Then project sequence length: 128 -> 77
        self.align_sequence = nn.Linear(128, 77)

    def forward(self, rpc: torch.Tensor) -> torch.Tensor:
        """
        rpc shape: [Batch, N, 5].
        Returns: [Batch, 77, 768].
        """
        xyz = rpc[:, :, :3]
        attributes = rpc[:, :, 3:]
        
        # Physics-aware embedding
        phys_feat = F.relu(self.physics_embed(attributes)) # [B, N, 16]
        
        # Hierarchical grouping
        l1_xyz, l1_points = self.sa1(xyz, phys_feat) # l1_xyz: [B, 128, 3], l1_points: [B, 128, 128]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # l2_xyz: [B, 128, 3], l2_points: [B, 128, 512]
        
        # l2_points is [Batch, 128, 512]. This is our raw feature map.
        # Map to [Batch, 77, 768]
        feat_proj = self.align_features(l2_points) # [B, 128, 768]
        # Transpose to project sequence dimension
        feat_proj = feat_proj.transpose(1, 2) # [B, 768, 128]
        aligned_seq = self.align_sequence(feat_proj) # [B, 768, 77]
        out = aligned_seq.transpose(1, 2) # [B, 77, 768]
        
        return out
