import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src: source points, [B, N, C]
    dst: target points, [B, M, C]
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
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C] or [B, S, K, C]
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

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class GeometricAffine(nn.Module):
    def __init__(self, dim, nsample=24):
        super(GeometricAffine, self).__init__()
        self.dim = dim
        self.nsample = nsample
        
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x, xyz, new_xyz=None):
        # x: [B, C, N]
        # xyz: [B, 3, N]
        
        if new_xyz is None:
            new_xyz = xyz
            
        B, C, N = x.shape
        xyz_trans = xyz.permute(0, 2, 1) # [B, N, 3]
        new_xyz_trans = new_xyz.permute(0, 2, 1) # [B, N, 3]
        
        idx = knn_point(self.nsample, xyz_trans, new_xyz_trans) # [B, N, k]
        
        # Group features
        grouped_x = index_points(x.permute(0, 2, 1), idx).permute(0, 3, 1, 2) # [B, C, N, k]
        
        # Calculate std and mean
        std = torch.std(grouped_x, dim=-1, keepdim=True) # [B, C, N, 1]
        mean = torch.mean(grouped_x, dim=-1, keepdim=True) # [B, C, N, 1]
        
        # Affine transformation
        x_transformed = (x.unsqueeze(-1) - mean) / (std + 1e-5) * self.alpha + self.beta
        return x_transformed.squeeze(-1)

class PointMLPBlock(nn.Module):
    def __init__(self, dim, nsample=24):
        super(PointMLPBlock, self).__init__()
        self.gam = GeometricAffine(dim, nsample)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.ReLU()
        
    def forward(self, x, xyz):
        # Pre-activation style residual block
        shortcut = x
        x = self.gam(x, xyz)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x + shortcut

class PointMLP(nn.Module):
    def __init__(self, num_classes=2, input_channels=6, points=4096, embed_dim=64, groups=1):
        super(PointMLP, self).__init__()
        
        self.points = points
        
        # Initial embedding
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # Encoder Stages (Simplified for Segmentation)
        # Stage 1
        self.block1 = PointMLPBlock(embed_dim)
        
        # Stage 2
        self.conv2 = nn.Sequential(nn.Conv1d(embed_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.block2 = PointMLPBlock(128)
        
        # Stage 3
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.block3 = PointMLPBlock(256)
        
        # Stage 4
        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU())
        self.block4 = PointMLPBlock(512)
        
        # Decoder (Feature Propagation / U-Net style)
        self.decode4 = nn.Sequential(nn.Conv1d(512 + 256, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.decode3 = nn.Sequential(nn.Conv1d(256 + 128, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.decode2 = nn.Sequential(nn.Conv1d(128 + embed_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        
        # Final prediction
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, x):
        # x: [B, C_in, N]
        xyz = x[:, :3, :] # Use XYZ for geometry
        
        # Embedding
        x0 = self.embedding(x) # [B, 64, N]
        
        # Encoder
        x1 = self.block1(x0, xyz) # [B, 64, N]
        
        x2_in = self.conv2(x1) # [B, 128, N]
        x2 = self.block2(x2_in, xyz) # [B, 128, N]
        
        x3_in = self.conv3(x2) # [B, 256, N]
        x3 = self.block3(x3_in, xyz) # [B, 256, N]
        
        x4_in = self.conv4(x3) # [B, 512, N]
        x4 = self.block4(x4_in, xyz) # [B, 512, N]
        
        # Decoder (Simple skip connections since point count doesn't change in this simplified version)
        # In full PointMLP, there is downsampling. Here we keep N constant for simplicity and speed in this segmentation task.
        # This acts like a deep ResNet-MLP on points with local geometric affine normalization.
        
        d4 = torch.cat([x4, x3], dim=1)
        d4 = self.decode4(d4)
        
        d3 = torch.cat([d4, x2], dim=1)
        d3 = self.decode3(d3)
        
        d2 = torch.cat([d3, x1], dim=1)
        d2 = self.decode2(d2)
        
        out = self.classifier(d2)
        # Return raw logits [B, NumClasses, N] for CrossEntropyLoss
        
        return out, None
