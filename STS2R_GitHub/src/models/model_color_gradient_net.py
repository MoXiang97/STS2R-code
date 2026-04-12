import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim_coor=3):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim_coor == 3:
            idx = knn(x[:, :3, :], k=k)  # [Local_XYZ] for KNN
        else:
            idx = knn(x, k=k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # Feature engineering: [x_i, x_i - x_j] or specific [RGB_i, RGB_i - RGB_j]
    # For ColorGradientNet, we focus on RGB difference.
    # The input x is [Local_XYZ(3), RGB(3), Other(4)]
    
    # Extract RGB (assume index 3:6)
    rgb_i = x[:, :, :, 3:6]
    rgb_j = feature[:, :, :, 3:6]
    delta_rgb = rgb_i - rgb_j
    
    # Concatenate [x_i, delta_rgb]
    # user asked: (Local_XYZ, RGB, Delta_RGB) 
    local_xyz_i = x[:, :, :, 0:3]
    rgb_i = x[:, :, :, 3:6]
    
    # Combine core features
    combined_feature = torch.cat((local_xyz_i, rgb_i, delta_rgb), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return combined_feature

class ColorGradientNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=10, k=20):
        super(ColorGradientNet, self).__init__()
        self.k = k
        
        # Input features: [Local_XYZ(3), RGB(3), Delta_RGB(3)] -> 9 channels
        self.conv1 = nn.Sequential(nn.Conv2d(9, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024 + 256 + 128 + 64, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # Stage 1: EdgeConv focusing on Color Gradient
        x1 = get_graph_feature(x, k=self.k)      # (batch_size, 9, num_points, k)
        x1 = self.conv1(x1)                       # (batch_size, 64, num_points, k)
        x1 = self.conv2(x1)                       # (batch_size, 64, num_points, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points)

        # Stage 2: Feature refinement
        # For simplicity and speed, we use 1x1 convs to aggregate features
        x2 = self.conv3(torch.cat((x1, x1), dim=1).unsqueeze(-1)) # (batch_size, 128, num_points, 1)
        x2 = x2.squeeze(-1)
        
        x3 = self.conv4(x2.unsqueeze(-1))
        x3 = x3.squeeze(-1)
        
        x4 = x3.max(dim=-1, keepdim=True)[0]     # (batch_size, 256, 1)
        
        # Global feature
        x_global = self.conv5(x4.repeat(1, 1, num_points)) # (batch_size, 1024, num_points)
        x_global_max = x_global.max(dim=-1, keepdim=True)[0] # (batch_size, 1024, 1)
        
        # Concatenate features for segmentation
        # [Global_Max, x3, x2, x1]
        x_feat = torch.cat((x_global_max.repeat(1, 1, num_points), x3, x2, x1), dim=1) # (batch_size, 1024+256+128+64, num_points)
        
        x_feat = x_feat.permute(0, 2, 1).reshape(-1, 1024 + 256 + 128 + 64)
        
        x_feat = F.leaky_relu(self.bn6(self.linear1(x_feat)), negative_slope=0.2)
        x_feat = self.dp1(x_feat)
        x_feat = F.leaky_relu(self.bn7(self.linear2(x_feat)), negative_slope=0.2)
        x_feat = self.dp2(x_feat)
        x_feat = self.linear3(x_feat)
        
        return x_feat.view(batch_size, num_points, -1).transpose(1, 2)
