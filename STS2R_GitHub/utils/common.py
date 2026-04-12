import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset

# ==========================================
# DATASET
# ==========================================
def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_distance = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    if max_distance > 0:
        points_normalized = points_centered / max_distance
    else:
        points_normalized = points_centered
    return points_normalized, centroid, max_distance


class ShoeTrajectoryDataset(Dataset):
    def __init__(self, real_path, sim_path, num_points=8192, is_train=True, oversample_real_factor=15):
        self.num_points = num_points
        self.is_train = is_train

        def safe_glob_files(folder_path):
            if not folder_path or not os.path.isdir(folder_path):
                return []
            npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
            if len(npy_files) > 0:
                npy_files.sort()
                return npy_files
            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
            txt_files.sort()
            return txt_files

        self.real_files = safe_glob_files(real_path)
        self.sim_files = safe_glob_files(sim_path)

        if self.is_train:
            if oversample_real_factor and oversample_real_factor > 1:
                oversampled_real = self.real_files * oversample_real_factor if self.real_files else []
                self.file_list = self.sim_files + oversampled_real
                print(f"训练数据集: {len(self.sim_files)}虚拟 + {len(oversampled_real)}真实(过采样)")
            else:
                self.file_list = self.sim_files + self.real_files
                print(f"训练数据集: {len(self.sim_files)}虚拟 + {len(self.real_files)}真实")
        else:
            self.file_list = self.real_files
            print(f"测试数据集: {len(self.real_files)}真实")

        random.shuffle(self.file_list)

    def load_point_cloud(self, file_path):
        if file_path.endswith(".txt"):
            data = np.loadtxt(file_path)
        elif file_path.endswith(".npy"):
            data = np.load(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    def apply_data_augmentation(self, points):
        """轻度数据增强以防止过拟合"""
        # 1. 随机微小平移 [-0.05, 0.05]
        translation = np.random.uniform(-0.05, 0.05, size=(1, 3))
        points = points + translation

        # 2. 随机抖动 (Jittering) N(0, 0.01) clip to [-0.03, 0.03]
        jitter = np.clip(np.random.normal(0.0, 0.01, size=points.shape), -0.03, 0.03)
        points = points + jitter

        # 3. 微小偏航角旋转 [-5度, +5度] 绕 Z 轴
        angle = np.random.uniform(-5, 5) * np.pi / 180.0
        cos_val, sin_val = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0],
            [0, 0, 1]
        ])
        points = np.dot(points, rot_matrix.T)

        return points

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            data = self.load_point_cloud(file_path)

            if data.shape[1] >= 7:
                full_points = data[:, :3]
                full_colors = data[:, 3:6]
                full_labels = data[:, 6].astype(np.int64)
            elif data.shape[1] >= 4:
                full_points = data[:, :3]
                full_colors = np.ones((len(full_points), 3)) * 128
                full_labels = data[:, 3].astype(np.int64)
            else:
                full_points = data[:, :3]
                full_colors = np.ones((len(full_points), 3)) * 128
                full_labels = np.zeros(len(full_points), dtype=np.int64)

            full_labels = np.clip(full_labels, 0, 1)

            # --- 核心对齐逻辑 1: 全量点云的绝对坐标去中心化 ---
            centroid = np.mean(full_points, axis=0)
            full_points_centered = full_points - centroid
            
            # --- 核心对齐逻辑 2: 计算全量点云的最大尺度用于 Global 归一化 ---
            global_scale = np.max(np.sqrt(np.sum(full_points_centered ** 2, axis=1)))
            if global_scale == 0:
                global_scale = 1.0
                
            # 计算全量点云的全局特征 (Global_Norm_XYZ)
            full_global_norm = full_points_centered / global_scale
            
            # --- 轻量级数据增强 (仅训练时) ---
            if self.is_train:
                full_points_centered = self.apply_data_augmentation(full_points_centered)
                # 增强后重新计算 global norm 保持一致性
                full_global_norm = full_points_centered / global_scale

            # --- 采样 N 个点 ---
            if len(full_points_centered) >= self.num_points:
                indices = np.random.choice(len(full_points_centered), self.num_points, replace=False)
            else:
                indices = np.random.choice(len(full_points_centered), self.num_points, replace=True)

            points = full_points_centered[indices]
            colors = full_colors[indices]
            labels = full_labels[indices]
            global_norm = full_global_norm[indices]

            # --- 核心对齐逻辑 3: 构建 Local_XYZ ---
            # 减去局部 N 个点的均值，做局部去中心化
            local_centroid = np.mean(points, axis=0)
            points_local = points - local_centroid
            # 局部归一化 (可选，这里与原来 normalize_point_cloud 逻辑一致)
            local_scale = np.max(np.sqrt(np.sum(points_local ** 2, axis=1)))
            if local_scale > 0:
                points_local = points_local / local_scale

            # --- 核心对齐逻辑 4: 构建 Theta ---
            # 根据 Global_Norm_XYZ 的 X 和 Y 计算反正切角度
            theta = np.arctan2(global_norm[:, 1], global_norm[:, 0]) / np.pi
            theta = theta.reshape(-1, 1)

            # RGB 归一化
            rgb = colors / 255.0

            # 拼接 10 维特征: [Local_XYZ(3), RGB(3), Global_Norm_XYZ(3), Theta(1)]
            features = np.concatenate([points_local, rgb, global_norm, theta], axis=1)

            features = torch.FloatTensor(features.T)
            labels = torch.LongTensor(labels)
            points_tensor = torch.FloatTensor(points_local.T)
            points_original = torch.FloatTensor(points.T)

            return features, labels, points_tensor, points_original, {}

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return (
                torch.zeros(10, self.num_points),
                torch.zeros(self.num_points, dtype=torch.long),
                torch.zeros(3, self.num_points),
                torch.zeros(3, self.num_points),
                {},
            )

    def __len__(self):
        return len(self.file_list)
