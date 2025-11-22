import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    두 점 집합 간의 유클리드 거리 제곱 계산 (Broadcast 활용)
    src: (B, N, C), dst: (B, M, C) -> dist: (B, N, M)
    """
    return torch.sum((src[:, :, None] - dst[:, None, :]) ** 2, dim=-1)


def index_points(points, idx):
    """
    인덱스에 해당하는 포인트를 추출
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
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
        self.group_all = group_all
    
    def forward(self, xyz, points):
        """
        Input:
            xyz: (B, N, 3) 좌표
            points: (B, N, C) 특징 (intensity 등)
        Return:
            new_xyz: (B, npoint, 3)
            new_points: (B, npoint, mlp[-1])
        """
        # Sampling & Grouping
        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3, device=xyz.device)  # Dummy
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            grouped_points = points.view(points.shape[0], points.shape[1], 1, -1) if points is not None else None
            if points is not None:
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            # 1. FPS (Sampling)
            idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, idx)
            
            # 2. Ball Query (Grouping)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.view(new_xyz.shape[0], self.npoint, 1, 3)  # Centering
            
            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+3, nsample, npoint]
        
        # 3. PointNet (Feature Extraction)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # 4. Pooling (Max over neighbors)
        new_points = torch.max(new_points, 2)[0]  # [B, MLP[-1], npoint]
        new_xyz = new_xyz
        
        return new_xyz, new_points.permute(0, 2, 1)


class RadarPointNetPlusPlus(nn.Module):
    def __init__(self, sd_hidden_dim=768):
        super().__init__()
        
        # Set Abstraction Layers
        # npoint: 샘플링할 점 개수 / radius: 반경 / nsample: 반경 내 이웃 개수 / mlp: 채널 변화
        
        # SA1: {yaml:max_points} -> 512 points
        # in_channel = 3 (xyz) + 1 (intensity) = 4
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 1, mlp=[64, 64, 128],
                                          group_all=False)
        
        # SA2: 512 -> 128 points (여기서 남은 128개를 토큰으로 씁니다!)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        
        # Final Projection to SD Dimension
        self.final_proj = nn.Linear(256, sd_hidden_dim)
        self.norm = nn.LayerNorm(sd_hidden_dim)
    
    def forward(self, x):
        # x: (B, N, 4) - [x, y, z, intensity]
        
        # PointNet++은 좌표(xyz)와 특징(features)을 분리해서 받음
        xyz = x[..., :3]  # (B, N, 3)
        features = x[..., 3:]  # (B, N, 1) - Intensity
        
        # Layer 1
        l1_xyz, l1_points = self.sa1(xyz, features)  # l1_points: (B, 512, 128)
        
        # Layer 2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # l2_points: (B, 128, 256)
        
        # Projection
        # (B, 128, 256) -> (B, 128, 768)
        out = self.final_proj(l2_points)
        out = self.norm(out)
        
        return out  # 이 결과가 Cross Attention의 encoder_hidden_states로 들어갑니다.