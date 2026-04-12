import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# Point Transformer Components
# ==========================================

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return torch.clamp(dist, min=0)

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
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

def query_ball_point(radius, nsample, xyz, new_xyz):
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
    
    # 防止索引越界
    group_idx[group_idx >= N] = 0
    return group_idx

class PointTransformerBlock(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = planes // 1
        self.out_planes = planes
        self.share_planes = share_planes
        self.nsample = nsample
        
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, planes)
        
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, mid_planes)
        )
        
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.BatchNorm1d(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes // share_planes, mid_planes // share_planes)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, px, xyz):
        # px: [B, N, C] (features)
        # xyz: [B, N, 3] (coords)
        B, N, C = px.shape
        
        # Simple KNN instead of ball query for attention usually works better/faster
        # But we reuse ball query utils for consistency
        # For full implementation, we usually use KNN k=16
        
        # Here we use a simplified version: assume neighbors are pre-calculated or calculate on fly
        # Calculating KNN on fly
        # Memory Optimization: For large N, calculating global distance matrix is O(N^2)
        # We assume N is moderate (e.g. < 5000) or we accept the cost.
        # If N is huge, this line is the bottleneck.
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.nsample]  # [B, N, K]
        
        knn_xyz = index_points(xyz, knn_idx) # [B, N, K, 3]
        
        q = self.linear_q(px).view(B, N, 1, self.mid_planes)
        k = index_points(self.linear_k(px), knn_idx).view(B, N, self.nsample, self.mid_planes)
        v = index_points(self.linear_v(px), knn_idx).view(B, N, self.nsample, self.out_planes)
        
        x_k = knn_xyz - xyz.view(B, N, 1, 3) # Relative coords [B, N, K, 3]
        
        # Position encoding
        # Processing x_k: [B, N, K, 3] -> [B*N*K, 3]
        pos_enc = self.linear_p(x_k.view(-1, 3)).view(B, N, self.nsample, self.mid_planes)
        
        w = q - k + pos_enc
        w = self.linear_w(w.view(-1, self.mid_planes)).view(B, N, self.nsample, self.mid_planes // self.share_planes)
        w = self.softmax(w) # Attention weights
        
        # Aggregation
        # v + pos_enc (broadcast needed? usually V is enough or V+pos)
        # Standard PT: y = sum(softmax(gamma(q-k+pos)) * (v+pos))
        # Here we simplify slightly to: sum(w * (v + pos_enc))
        
        v = v + pos_enc # Add position info to value
        
        # w: [B, N, K, mid//share]
        # v: [B, N, K, mid] -> reshape to [B, N, K, share, mid//share]
        
        v = v.view(B, N, self.nsample, self.share_planes, self.mid_planes // self.share_planes)
        w = w.view(B, N, self.nsample, 1, self.mid_planes // self.share_planes)
        
        res = torch.sum(w * v, dim=2) # [B, N, share, mid//share]
        res = res.view(B, N, self.out_planes)
        
        return res + px # Residual connection

class PointTransformerStage(nn.Module):
    """
    Helper class to stack multiple PointTransformerBlocks
    """
    def __init__(self, planes, num_blocks, share_planes=8, nsample=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            PointTransformerBlock(planes, planes, share_planes, nsample)
            for _ in range(num_blocks)
        ])
        
    def forward(self, px, xyz):
        for block in self.blocks:
            px = block(px, xyz)
        return px

class TransitionDown(nn.Module):
    def __init__(self, npoint, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride = stride
        self.npoint = npoint
        self.nsample = nsample
        
        self.mlp = nn.Sequential(
            nn.Linear(3+in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, px, xyz):
        # px: [B, N, C]
        # xyz: [B, N, 3]
        B, N, C = px.shape
        
        # FPS sampling
        # Handle case where current N < target npoint (though unlikely in downsampling design)
        if N > self.npoint:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
        else:
            new_xyz = xyz
            self.npoint = N # Adjust dynamically
        
        # Grouping (KNN)
        dists = square_distance(new_xyz, xyz)
        idx = dists.argsort()[:, :, :self.nsample]
        
        grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, 3]
        grouped_px = index_points(px, idx)   # [B, npoint, nsample, C]
        
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, 3)
        
        grouped_points = torch.cat([grouped_px, grouped_xyz_norm], dim=-1) # [B, npoint, nsample, C+3]
        
        # MLP
        new_px = self.mlp(grouped_points.view(-1, C+3))
        new_px = torch.max(new_px.view(B, self.npoint, self.nsample, -1), dim=2)[0]
        
        return new_px, new_xyz

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes, skip_planes=None):
        super().__init__()
        if skip_planes is None:
            skip_planes = out_planes
            
        self.linear1 = nn.Sequential(
            nn.Linear(skip_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, px1, xyz1, px2, xyz2):
        # px1: features from down layer [B, N1, C1]
        # xyz1: coords from down layer [B, N1, 3]
        # px2: features from skip connection [B, N2, C2]
        # xyz2: coords from skip connection [B, N2, 3]
        
        # Interpolate px1 to N2 points
        dist, idx = square_distance(xyz2, xyz1).sort(dim=-1)
        dist, idx = dist[:, :, :3], idx[:, :, :3]  # 3 NN
        
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        
        interpolated_px1 = torch.sum(index_points(px1, idx) * weight.view(xyz2.shape[0], xyz2.shape[1], 3, 1), dim=2)
        
        # Sum with skip connection
        # linear1 transforms px2 (skip) to out_planes
        # linear2 transforms px1 (down) to out_planes
        
        feat_skip = self.linear1(px2.view(-1, px2.shape[-1])).view(px2.shape[0], px2.shape[1], -1)
        feat_down = self.linear2(interpolated_px1.view(-1, interpolated_px1.shape[-1])).view(px2.shape[0], px2.shape[1], -1)
        
        x = feat_skip + feat_down
            
        return x

class PointTransformerSeg(nn.Module):
    def __init__(self, num_classes=2, in_channels=14):
        super().__init__()
        # Input: 14 channels (XYZ + RGB + Theta + Var_RGB + Normal + Curvature)
        self.in_channels = in_channels
        self.in_planes = 32
        
        # Initial Embedding
        self.linear_in = nn.Sequential(
            nn.Conv1d(in_channels, self.in_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.in_planes),
            nn.ReLU(inplace=True)
        )
        
        # Encoder Configuration
        # Original: 2048 -> 512 -> 128 -> 32 (Too aggressive)
        # New:      1024 -> 512 -> 256 -> 128 (Conservative)
        
        # Block depths: [2, 3, 4, 3] (Standard configuration)
        block_depths = [2, 3, 4, 3]
        
        # Stage 1: Full Resolution (N points)
        self.tf1 = PointTransformerStage(self.in_planes, num_blocks=block_depths[0])
        self.td1 = TransitionDown(npoint=1024, in_planes=self.in_planes, out_planes=64, nsample=16)
        
        # Stage 2: 1024 points
        self.tf2 = PointTransformerStage(64, num_blocks=block_depths[1])
        self.td2 = TransitionDown(npoint=512, in_planes=64, out_planes=128, nsample=16)
        
        # Stage 3: 512 points
        self.tf3 = PointTransformerStage(128, num_blocks=block_depths[2])
        self.td3 = TransitionDown(npoint=256, in_planes=128, out_planes=256, nsample=16)
        
        # Stage 4: 256 points
        self.tf4 = PointTransformerStage(256, num_blocks=block_depths[3])
        self.td4 = TransitionDown(npoint=128, in_planes=256, out_planes=512, nsample=16)
        
        # Stage 5: 128 points
        self.tf5 = PointTransformerStage(512, num_blocks=2) # Deepest layer
        
        # Decoder
        # tu1: in(out4)=512, skip(out3)=256 -> out=256
        self.tu1 = TransitionUp(512, 256, skip_planes=256)
        self.tf6 = PointTransformerStage(256, num_blocks=2)
        
        # tu2: in(up1)=256, skip(out2)=128 -> out=128
        self.tu2 = TransitionUp(256, 128, skip_planes=128)
        self.tf7 = PointTransformerStage(128, num_blocks=2)
        
        # tu3: in(up2)=128, skip(out1)=64 -> out=64
        self.tu3 = TransitionUp(128, 64, skip_planes=64)
        self.tf8 = PointTransformerStage(64, num_blocks=2)
        
        # tu4: in(up3)=64, skip(out0)=32 -> out=32
        self.tu4 = TransitionUp(64, 32, skip_planes=32)
        self.tf9 = PointTransformerStage(32, num_blocks=2)
        
        self.linear_out = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # x: [B, C, N] or [B, N, C]
        if x.shape[1] > 100: # Heuristic: if C is very large, it's likely N
            x = x.permute(0, 2, 1)
            
        # x: [B, C, N]
        B, C, N = x.shape
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous() # [B, N, 3]
        
        # 1. Embedding [B, C, N] -> [B, 32, N]
        out0 = self.linear_in(x) 
        
        # 2. Transformer blocks expect [B, N, C]
        out0 = out0.permute(0, 2, 1).contiguous() # [B, N, 32]
        out0 = self.tf1(out0, xyz)
        
        # Down
        out1, xyz1 = self.td1(out0, xyz) # 1024
        out1 = self.tf2(out1, xyz1)
        
        out2, xyz2 = self.td2(out1, xyz1) # 512
        out2 = self.tf3(out2, xyz2)
        
        out3, xyz3 = self.td3(out2, xyz2) # 256
        out3 = self.tf4(out3, xyz3)
        
        out4, xyz4 = self.td4(out3, xyz3) # 128
        out4 = self.tf5(out4, xyz4)
        
        # Up
        up1 = self.tu1(out4, xyz4, out3, xyz3)
        up1 = self.tf6(up1, xyz3)
        
        up2 = self.tu2(up1, xyz3, out2, xyz2)
        up2 = self.tf7(up2, xyz2)
        
        up3 = self.tu3(up2, xyz2, out1, xyz1)
        up3 = self.tf8(up3, xyz1)
        
        up4 = self.tu4(up3, xyz1, out0, xyz)
        up4 = self.tf9(up4, xyz)
        
        # 4. Final Head
        # up4: [B, N, 32] -> needs [B, 32, N] for linear_out (Conv1d)
        feat_final = up4.permute(0, 2, 1).contiguous() # [B, 32, N]
        logits = self.linear_out(feat_final) # [B, num_classes, N]
        
        # Return logits [B, NumClasses, N] for CrossEntropyLoss
        return logits, None
