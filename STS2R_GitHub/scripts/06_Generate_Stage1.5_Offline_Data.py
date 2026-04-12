import os
import sys
import glob
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial import cKDTree

# Try to import open3d for normal estimation
try:
    import open3d as o3d
except ImportError:
    print("Warning: open3d not found. Normal estimation will fail.")

# ============================================
# 1. Config 管理路径与超参数
# ============================================
class Config:
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1 共享前端模型 Checkpoint
    STAGE1_CHECKPOINT = r"D:\Data\PaperAI\SIM2REAL\experiments\stage1_results\checkpoints\best_adaptive_s1.pth"

    # 输入目录 (仿真数据 - 经过 05 ROI 过滤后的 NPY)
    SIM_INPUT_ROOT = r"outputs\Ablation_ROI_NPY"
    SIM_MODES = ["01_Base", "02_Geo", "03_Phys", "04_STS2R"]

    # 输入目录 (真实数据 - 原始 TXT)
    REAL_INPUT_PATHS = {
        "shoe_a": r"assets\Real_Data\Real_ShoeA",
        "shoe_b": r"assets\Real_Data\Real_ShoeB",
        "shoe_c": r"assets\Real_Data\Real_ShoeC"
    }

    # 输出根目录 (保存 Stage 1.5 离线特征数据)
    OUTPUT_ROOT = r"outputs\stage1.5_offline_data"

    # 推理参数
    K_MICRO = 32
    K_MACRO = 64
    M_POINTS = 8192
    INPUT_CHANNELS = 10
    NUM_CLASSES = 2

# ============================================
# 2. 模型定义 (CGA-MSG-Net)
# ============================================
class MSGAttention(nn.Module):
    def __init__(self, in_channels):
        super(MSGAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, feat_micro, feat_macro):
        combined = torch.cat([feat_micro, feat_macro], dim=1)
        weights = self.attention(combined.transpose(1, 2))
        w_micro = weights[:, :, 0].unsqueeze(1)
        w_macro = weights[:, :, 1].unsqueeze(1)
        return w_micro * feat_micro + w_macro * feat_macro

class CGAMSGNet(nn.Module):
    def __init__(self, input_dim=10, num_classes=2, k_micro=32, k_macro=128):
        super(CGAMSGNet, self).__init__()
        self.k_micro = k_micro
        self.k_macro = k_macro
        self.conv_micro = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv_macro = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.attention = MSGAttention(64)
        self.threshold_mlp = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, num_classes, kernel_size=1)
        )

    def get_graph_feature(self, x, k, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        device = x.device
        if idx is None:
            dist = -2 * torch.matmul(x[:, :3, :].transpose(2, 1), x[:, :3, :])
            xx = torch.sum(x[:, :3, :]**2, dim=1, keepdim=True)
            dist = -xx - dist - xx.transpose(2, 1)
            idx = dist.topk(k=k, dim=-1)[1]
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)
        x_trans = x.transpose(2, 1).contiguous()
        feature = x_trans.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, -1)
        x_expanded = x_trans.view(batch_size, num_points, 1, -1).repeat(1, 1, k, 1)
        delta_rgb = x_expanded[:, :, :, 3:6] - feature[:, :, :, 3:6]
        std_rgb = torch.std(feature[:, :, :, 3:6], dim=2, keepdim=True).repeat(1, 1, k, 1)
        delta_xyz = x_expanded[:, :, :, 0:3] - feature[:, :, :, 0:3]
        global_theta = x_expanded[:, :, :, 6:7]
        f = torch.cat([delta_rgb, std_rgb, delta_xyz, global_theta], dim=-1)
        return f.permute(0, 3, 1, 2).contiguous()

    def forward(self, x, idx_micro=None, idx_macro=None):
        f_micro = self.get_graph_feature(x, k=self.k_micro, idx=idx_micro)
        x_micro = self.conv_micro(f_micro)
        x_micro = x_micro.max(dim=-1, keepdim=False)[0]
        f_macro = self.get_graph_feature(x, k=self.k_macro, idx=idx_macro)
        x_macro = self.conv_macro(f_macro)
        x_macro = x_macro.max(dim=-1, keepdim=False)[0]
        feat_fused = self.attention(x_micro, x_macro)
        feat_adaptive = self.threshold_mlp(feat_fused)
        return self.classifier(feat_adaptive)

# ============================================
# 3. 特征计算工具函数
# ============================================
def compute_stage1_5_features(m_xyz, m_rgb, m_labels):
    """计算 14+1 维特征矩阵"""
    # 1. Local XYZ
    new_center = np.mean(m_xyz, axis=0)
    local_xyz = m_xyz - new_center
    
    # 2. RGB Norm
    m_rgb_norm = m_rgb / 255.0
    
    # 3. New Theta
    new_theta = np.arctan2(local_xyz[:, 1], local_xyz[:, 0]).reshape(-1, 1) / np.pi
    
    # 4. Color Variance (16 neighbors)
    m_tree = cKDTree(m_xyz)
    _, m_indices = m_tree.query(m_xyz, k=16)
    m_neighbor_rgb = m_rgb_norm[m_indices]
    m_color_var = np.var(m_neighbor_rgb, axis=1)
    
    # 5. Normals & Curvature (Open3D)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(m_xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    m_normals = np.asarray(pcd.normals)
    
    # Curvature calculation
    m_neighbor_xyz = m_xyz[m_indices]
    m_neighbor_xyz_centered = m_neighbor_xyz - m_xyz.reshape(-1, 1, 3)
    cov = np.matmul(m_neighbor_xyz_centered.transpose(0, 2, 1), m_neighbor_xyz_centered) / 16.0
    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = np.sort(eigenvalues, axis=-1)
    sum_eigen = np.sum(eigenvalues, axis=-1)
    valid_mask = sum_eigen > 1e-6
    m_curvature = np.zeros((len(m_xyz), 1))
    m_curvature[valid_mask, 0] = eigenvalues[valid_mask, 0] / sum_eigen[valid_mask]
    
    # Concatenate [3+3+1+3+3+1 + 1] = 15 columns
    combined = np.concatenate([
        local_xyz, m_rgb_norm, new_theta, m_color_var, m_normals, m_curvature, m_labels.reshape(-1, 1)
    ], axis=1)
    return combined.astype(np.float32)

def process_file_list(file_list, output_dir, model, cfg):
    """核心处理逻辑：推理 -> 筛选 -> 特征计算 -> 保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    for file_path in tqdm(file_list, desc=f"Processing {os.path.basename(output_dir)}", leave=False):
        try:
            # 1. 加载数据
            if file_path.endswith('.npy'):
                data = np.load(file_path)
            else:
                data = np.loadtxt(file_path)
            
            num_cols = data.shape[1]
            label_idx = 6 if num_cols >= 7 else num_cols - 1
            xyz_raw = data[:, :3].copy().astype(np.float32)
            rgb_raw = data[:, 3:6].copy().astype(np.float32)
            labels = data[:, label_idx].copy().astype(np.int64)
            labels = np.clip(labels, 0, 1)

            # 2. 推理获取概率 (采样 32768 点提高速度)
            num_pred_points = 32768
            if len(xyz_raw) > num_pred_points:
                indices = np.random.choice(len(xyz_raw), num_pred_points, replace=False)
            else:
                indices = np.arange(len(xyz_raw))
            
            s1_xyz = xyz_raw[indices]
            s1_rgb = rgb_raw[indices]
            s1_labels = labels[indices]
            
            center = np.mean(s1_xyz, axis=0)
            rel_points = s1_xyz - center
            theta = np.arctan2(rel_points[:, 1], rel_points[:, 0]).reshape(-1, 1) / np.pi
            s1_input = np.concatenate([s1_xyz, s1_rgb / 255.0, theta], axis=1)
            
            tree = cKDTree(s1_xyz)
            _, idx_micro = tree.query(s1_xyz, k=cfg.K_MICRO, workers=-1)
            _, idx_macro = tree.query(s1_xyz, k=cfg.K_MACRO, workers=-1)
            
            with torch.no_grad():
                feat_tensor = torch.from_numpy(s1_input).float().transpose(0, 1).unsqueeze(0).to(cfg.DEVICE)
                idx_micro_t = torch.from_numpy(idx_micro).long().unsqueeze(0).to(cfg.DEVICE)
                idx_macro_t = torch.from_numpy(idx_macro).long().unsqueeze(0).to(cfg.DEVICE)
                logits = model(feat_tensor, idx_micro_t, idx_macro_t)
                probs = F.softmax(logits, dim=1)[0, 1, :].cpu().numpy()
            
            # 3. 截取 Top-M 点 (M=8192)
            m_points = cfg.M_POINTS
            if len(probs) >= m_points:
                top_indices = np.argsort(probs)[-m_points:]
            else:
                top_indices = np.random.choice(len(probs), m_points, replace=True)
            
            m_xyz = s1_xyz[top_indices]
            m_rgb = s1_rgb[top_indices]
            m_labels = s1_labels[top_indices]
            
            # 4. 计算 15 维特征
            combined_stage2 = compute_stage1_5_features(m_xyz, m_rgb, m_labels)
            
            # 5. 保存
            file_name = os.path.basename(file_path).replace(".txt", ".npy")
            save_path = os.path.join(output_dir, file_name)
            np.save(save_path, combined_stage2)
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

# ============================================
# 4. 主程序
# ============================================
def main():
    cfg = Config()
    print(f"🚀 Starting 06_Generate_Stage1.5_Offline_Data pipeline...")
    
    # 清理输出根目录
    if os.path.exists(cfg.OUTPUT_ROOT):
        print(f"🧹 Cleaning up existing output root: {cfg.OUTPUT_ROOT}")
        shutil.rmtree(cfg.OUTPUT_ROOT, ignore_errors=True)
    os.makedirs(cfg.OUTPUT_ROOT, exist_ok=True)

    # 1. 加载模型
    model = CGAMSGNet(input_dim=cfg.INPUT_CHANNELS, 
                      num_classes=cfg.NUM_CLASSES,
                      k_micro=cfg.K_MICRO,
                      k_macro=cfg.K_MACRO).to(cfg.DEVICE)
    
    if os.path.exists(cfg.STAGE1_CHECKPOINT):
        model.load_state_dict(torch.load(cfg.STAGE1_CHECKPOINT, map_location=cfg.DEVICE))
        print(f"✅ Loaded Stage 1 weights from {cfg.STAGE1_CHECKPOINT}")
    else:
        print(f"❌ Error: Stage 1 checkpoint not found at {cfg.STAGE1_CHECKPOINT}")
        return
    model.eval()

    # 2. 处理消融实验仿真数据 (Ablation Sim Data)
    print("\n--- Processing Ablation Sim Data ---")
    for mode in cfg.SIM_MODES:
        input_dir = os.path.join(cfg.SIM_INPUT_ROOT, mode)
        if not os.path.exists(input_dir):
            print(f"⚠️ Skip: {input_dir} not found.")
            continue
        
        output_dir = os.path.join(cfg.OUTPUT_ROOT, mode)
        file_list = glob.glob(os.path.join(input_dir, "*.npy"))
        print(f"📦 Mode {mode}: Found {len(file_list)} files.")
        process_file_list(file_list, output_dir, model, cfg)

    # 3. 处理真实点云数据 (Real Shoe Data)
    print("\n--- Processing Real Shoe Data ---")
    for entity_name, input_dir in cfg.REAL_INPUT_PATHS.items():
        if not os.path.exists(input_dir):
            print(f"⚠️ Skip: {input_dir} not found.")
            continue
        
        # 真实数据保存在公共目录下，供 run_ablation 和 run_benchmark 复用
        output_dir = os.path.join(cfg.OUTPUT_ROOT, entity_name)
        file_list = glob.glob(os.path.join(input_dir, "*.txt"))
        print(f"📦 Entity {entity_name}: Found {len(file_list)} files.")
        process_file_list(file_list, output_dir, model, cfg)

    print(f"\n✅ All Stage 1.5 offline data generated at: {cfg.OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
