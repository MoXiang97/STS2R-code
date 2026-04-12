import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# Utility Functions
# ============================================

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src^2, dim=-1) + sum(dst^2, dim=-1) - 2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
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

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
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

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
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

# ============================================
# PointNeXt Components
# ============================================

class InvResMLP(nn.Module):
    """
    Inverted Residual MLP Block (PointNeXt)
    Structure: Conv1x1 (Exp) -> GELU -> Depth-wise (Group+Conv+Pool) -> GELU -> Conv1x1 (Red)
    """
    def __init__(self, in_channels, radius, nsample, expansion=4):
        super().__init__()
        self.in_channels = in_channels
        self.radius = radius
        self.nsample = nsample
        mid_channels = in_channels * expansion
        
        # 1. Expansion
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1)
        # Replaced BatchNorm1d with GroupNorm
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=mid_channels)
        self.act1 = nn.GELU()
        
        # 2. Depth-wise Feature Aggregation
        # We simulate depth-wise conv by: Grouping -> Conv2d(groups=C) -> MaxPool
        self.dw_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, groups=mid_channels, bias=False)
        # Replaced BatchNorm2d with GroupNorm
        self.dw_bn = nn.GroupNorm(num_groups=4, num_channels=mid_channels)
        self.act2 = nn.GELU()
        
        # 3. Reduction
        self.conv2 = nn.Conv1d(mid_channels, in_channels, 1)
        # Replaced BatchNorm1d with GroupNorm
        self.bn2 = nn.GroupNorm(num_groups=4, num_channels=in_channels)
        
        # Residual Connection is implicit in forward if shapes match
        
    def forward(self, x, pos):
        """
        x: [B, C, N] - Features
        pos: [B, 3, N] - Coordinates (for neighbor query)
        """
        identity = x
        
        # 1. Expand
        x = self.act1(self.bn1(self.conv1(x))) # [B, mid, N]
        
        # 2. Depth-wise (Local Aggregation)
        # Query Neighbors
        pos_t = pos.permute(0, 2, 1) # [B, N, 3]
        idx = query_ball_point(self.radius, self.nsample, pos_t, pos_t) # [B, N, K]
        
        # Group Features
        # x: [B, C, N] -> [B, C, N, K]
        grouped_x = index_points(x.permute(0, 2, 1), idx).permute(0, 3, 1, 2) # [B, C, N, K]
        
        # Depth-wise Conv (Spatial mixing/weighting of neighbors)
        grouped_x = self.dw_bn(self.dw_conv(grouped_x)) # [B, C, N, K]
        
        # Max Pooling (Aggregation)
        x = grouped_x.max(dim=-1)[0] # [B, C, N]
        
        x = self.act2(x)
        
        # 3. Reduce
        x = self.bn2(self.conv2(x))
        
        # Residual
        x += identity
        
        return x

class DownsampleLayer(nn.Module):
    """
    Downsampling Stage: Subsample -> Group -> Feature Embedding
    """
    def __init__(self, in_channels, out_channels, npoint, radius, nsample):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            # Replaced BatchNorm1d with GroupNorm
            nn.GroupNorm(num_groups=4, num_channels=out_channels),
            nn.GELU()
        )

    def forward(self, x, pos):
        # x: [B, C, N] - Features (can be None)
        # pos: [B, 3, N] - Coordinates
        
        pos_t = pos.permute(0, 2, 1) # [B, N, 3]
        
        # 1. FPS Downsampling
        if self.npoint < pos.shape[2]:
            fps_idx = farthest_point_sample(pos_t, self.npoint)
            new_pos = index_points(pos_t, fps_idx) # [B, S, 3]
        else:
            new_pos = pos_t
            
        # 2. Grouping
        idx = query_ball_point(self.radius, self.nsample, pos_t, new_pos) # [B, S, K]
        
        # 3. Group Coords (Relative)
        grouped_pos = index_points(pos_t, idx) # [B, S, K, 3]
        new_pos_expand = new_pos.view(pos.shape[0], new_pos.shape[1], 1, 3)
        grouped_pos_norm = grouped_pos - new_pos_expand # [B, S, K, 3]
        grouped_pos_norm = grouped_pos_norm.permute(0, 3, 1, 2) # [B, 3, S, K]
        
        # 4. Group Features
        if x is not None:
            grouped_x = index_points(x.permute(0, 2, 1), idx).permute(0, 3, 1, 2) # [B, C, S, K]
            grouped_feat = torch.cat([grouped_x, grouped_pos_norm], dim=1) # [B, C+3, S, K]
        else:
            grouped_feat = grouped_pos_norm # [B, 3, S, K]
        
        # 5. Feature Reduction/Embedding
        B, C_feat, S, K = grouped_feat.shape
        grouped_feat = grouped_feat.reshape(B, C_feat, S*K)
        
        x_out = self.mlp(grouped_feat) # [B, Out, S*K]
        x_out = x_out.view(B, -1, S, K)
        x_out = x_out.max(dim=-1)[0] # [B, Out, S]
        
        new_pos = new_pos.permute(0, 2, 1) # [B, 3, S]
        
        return x_out, new_pos

class Stage(nn.Module):
    """
    PointNeXt Stage: Downsample -> Stack of InvResMLP Blocks
    """
    def __init__(self, in_channels, out_channels, npoint, radius, nsample, depth=1):
        super().__init__()
        
        # Downsampling Layer
        # Input to MLP is in_channels + 3 (Relative Pos)
        self.downsample = DownsampleLayer(in_channels + 3, out_channels, npoint, radius, nsample)
        
        # Stack of InvResMLP Blocks
        self.blocks = nn.ModuleList([
            InvResMLP(out_channels, radius, nsample) for _ in range(depth)
        ])
        
    def forward(self, x, pos):
        # Downsample
        x, pos = self.downsample(x, pos)
        
        # Process
        for block in self.blocks:
            x = block(x, pos)
            
        return x, pos

class UpSampleLayer(nn.Module):
    """
    Upsampling + Feature Fusion + InvResMLP
    """
    def __init__(self, in_channels_skip, in_channels_up, out_channels, radius, nsample):
        super().__init__()
        
        # Fusion MLP (Conv1x1)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(in_channels_skip + in_channels_up, out_channels, 1),
            # Replaced BatchNorm1d with GroupNorm
            nn.GroupNorm(num_groups=4, num_channels=out_channels),
            nn.GELU()
        )
        
        # Processing Block
        self.block = InvResMLP(out_channels, radius, nsample)
        
    def forward(self, x_up, pos_up, x_skip, pos_skip):
        # x_up: [B, C_up, S] (Source - coarser)
        # pos_up: [B, 3, S]
        # x_skip: [B, C_skip, N] (Target - finer)
        # pos_skip: [B, 3, N]
        
        # 1. Interpolate x_up to pos_skip
        pos_skip_t = pos_skip.permute(0, 2, 1)
        pos_up_t = pos_up.permute(0, 2, 1)
        
        dist, idx = square_distance(pos_skip_t, pos_up_t).sort(dim=-1)
        dist, idx = dist[:, :, :3], idx[:, :, :3] # 3 NN
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        
        interpolated = torch.sum(index_points(x_up.permute(0, 2, 1), idx) * weight.view(pos_skip.shape[0], pos_skip.shape[2], 3, 1), dim=2)
        interpolated = interpolated.permute(0, 2, 1) # [B, C_up, N]
        
        # 2. Concatenate
        if x_skip is not None:
            x_fused = torch.cat([x_skip, interpolated], dim=1)
        else:
            x_fused = interpolated
            
        # 3. Fuse and Process
        x = self.fusion_mlp(x_fused)
        x = self.block(x, pos_skip)
        
        return x

class PointNeXt(nn.Module):
    def __init__(self, num_classes=2, input_channels=6, width=32):
        super().__init__()
        
        # Configuration (PointNeXt-S style, adjusted for segmentation)
        # Depths: [1, 1, 3, 1]
        # Widths: [32, 64, 128, 256] -> Decoder reverses
        # Npoints: [1024, 512, 256, 128] (Less aggressive)
        
        # Input Handling
        self.in_channels = input_channels
        start_dim = input_channels - 3 # Features only
        if start_dim < 0: start_dim = 0
        
        # Encoder
        # Stage 1
        self.stage1 = Stage(start_dim, width, npoint=1024, radius=0.1, nsample=32, depth=1)
        # Stage 2
        self.stage2 = Stage(width, width*2, npoint=512, radius=0.15, nsample=32, depth=1)
        # Stage 3
        self.stage3 = Stage(width*2, width*4, npoint=256, radius=0.2, nsample=32, depth=3)
        # Stage 4
        self.stage4 = Stage(width*4, width*8, npoint=128, radius=0.3, nsample=32, depth=1)
        
        # Decoder
        # FP4: Up from Stage 4 to Stage 3
        self.fp4 = UpSampleLayer(width*4, width*8, width*4, radius=0.3, nsample=32)
        
        # FP3: Up from FP4 to Stage 2
        self.fp3 = UpSampleLayer(width*2, width*4, width*2, radius=0.2, nsample=32)
        
        # FP2: Up from FP3 to Stage 1
        self.fp2 = UpSampleLayer(width, width*2, width, radius=0.15, nsample=32)
        
        # FP1: Up from FP2 to Original
        self.fp1 = UpSampleLayer(start_dim, width, width, radius=0.1, nsample=32)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(width, width, 1),
            # Replaced BatchNorm1d with GroupNorm
            nn.GroupNorm(num_groups=4, num_channels=width),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Conv1d(width, num_classes, 1)
        )

    def forward(self, x):
        # x: [B, C, N] or [B, N, C]
        if x.shape[1] > 100: # Heuristic: if C is very large, it's likely N
            x = x.permute(0, 2, 1)
            
        # x: [B, C, N]
        # Assume first 3 channels are XYZ
        pos = x[:, :3, :]
        if x.shape[1] > 3:
            features = x[:, 3:, :]
        else:
            features = None 
            
        # Encoder
        # l0: Original
        l0_pos = pos
        l0_x = features
        
        # l1: Stage 1
        l1_x, l1_pos = self.stage1(l0_x, l0_pos)
        
        # l2: Stage 2
        l2_x, l2_pos = self.stage2(l1_x, l1_pos)
        
        # l3: Stage 3
        l3_x, l3_pos = self.stage3(l2_x, l2_pos)
        
        # l4: Stage 4
        l4_x, l4_pos = self.stage4(l3_x, l3_pos)
        
        # Decoder
        # FP4 (l4 -> l3)
        l3_up = self.fp4(l4_x, l4_pos, l3_x, l3_pos)
        
        # FP3 (l3_up -> l2)
        l2_up = self.fp3(l3_up, l3_pos, l2_x, l2_pos)
        
        # FP2 (l2_up -> l1)
        l1_up = self.fp2(l2_up, l2_pos, l1_x, l1_pos)
        
        # FP1 (l1_up -> l0)
        l0_up = self.fp1(l1_up, l1_pos, l0_x, l0_pos)
        
        # Classifier
        out = self.classifier(l0_up)
        
        # Return logits [B, NumClasses, N] for CrossEntropyLoss
        return out, None
