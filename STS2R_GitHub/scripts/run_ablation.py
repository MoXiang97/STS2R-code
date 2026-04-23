import os
import sys
import glob
import time
import random
import shutil
import logging
import numpy as np

# Add project root and src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# 解决 OpenMP 重复初始化导致程序崩溃的问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import recall_score, precision_score, f1_score

# 导入 Stage 2 专用模型 (从 src/models 文件夹导入标准版本)
try:
    from src.models.model_dgcnn import DGCNN
    from src.models.model_pointnet2 import PointNet2
    from src.models.model_point_transformer import PointTransformerSeg
    from src.models.model_pointmlp import PointMLP
except ImportError as e:
    print(f"Warning: Standard models in src/models/ not found. Error: {e}")

# ============================================
# 1. Config 管理超参数 (工业级面向对象设计)
# ============================================
class Config:
    # 消融实验模式控制 (Synthetic Dataset Variants)
    # '01_Base'   - V-Base (基础版本)
    # '02_Geo'    - +Geom (几何增强)
    # '03_Phys'   - +Geom+Phys (几何+物理退化)
    # '04_STS2R' - Full STS2R (最终完整版)
    # 'all'       - 依次运行所有消融版本的训练
    MODE = 'all' 
    
    # 阶段控制开关: 
    # 2   - 仅运行 Stage 2 消融训练与评估
    RUN_STAGE = 2
    
    # 消融实验工程管理
    EXP_NAME = f"ablation_{MODE}"
    
    # Stage 2 离线数据存储根目录 (由 06_Generate_Stage1.5_Offline_Data.py 生成)
    STAGE2_OFFLINE_ROOT = r"outputs\stage1.5_offline_data"

    # 针对当前模式的离线数据路径
    STAGE2_OFFLINE_DIR = os.path.join(STAGE2_OFFLINE_ROOT, MODE)
  
    # 结果输出根目录
    OUTPUT_ROOT = os.path.join("./experiments/ablation_results", EXP_NAME)
    
    MODEL_SAVE_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
    VIZ_DIR = os.path.join(OUTPUT_ROOT, "visualizations")
    LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")
    LOG_FILE = os.path.join(LOG_DIR, f"{EXP_NAME}_stage2_train.log")
    
    @classmethod
    def update_mode(cls, new_mode):
        """动态更新模式及相关路径"""
        cls.MODE = new_mode
        cls.EXP_NAME = f"ablation_{new_mode}"
        cls.STAGE2_OFFLINE_DIR = os.path.join(cls.STAGE2_OFFLINE_ROOT, new_mode)
        cls.OUTPUT_ROOT = os.path.join("./experiments/ablation_results", cls.EXP_NAME)
        cls.MODEL_SAVE_DIR = os.path.join(cls.OUTPUT_ROOT, "checkpoints")
        cls.VIZ_DIR = os.path.join(cls.OUTPUT_ROOT, "visualizations")
        cls.LOG_DIR = os.path.join(cls.OUTPUT_ROOT, "logs")
        cls.LOG_FILE = os.path.join(cls.LOG_DIR, f"{cls.EXP_NAME}_stage2_train.log")
    
    # 硬件与通用参数
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4 
    
    # Stage 2 训练参数 (固定 14 维特征协议)
    STAGE2_MODEL_TYPES = ['pointnet2', 'pointmlp', 'dgcnn'] 
    STAGE2_EPOCHS = 100
    STAGE2_BATCH_SIZE = 8
    STAGE2_PATIENCE = 10 
    STAGE2_INPUT_CHANNELS = 14 # [Local_XYZ(3), RGB_Norm(3), Theta(1), Var_RGB(3), Normal(3), Curvature(1)]
    STAGE2_USE_ONLINE_AUGMENT = True
    
    # 在线增强模式: 'none', 'old', 'mirror', 'full'
    # 'none':   不做任何增强
    # 'old':    包含 rotation, scale, jitter, color (保持几何一致性但无镜像和重算)
    # 'mirror': 'old' + Y-axis mirror (左右镜像)
    # 'full':   'mirror' + theta recomputation (完整协议)
    STAGE2_AUG_MODE = 'none' 
    
    # Stage 2 模型专属超参数 (保持与主实验一致)
    STAGE2_MODEL_CONFIGS = {
        'dgcnn': {'lr': 5e-4,       'pos_weight': 4.0,        'weight_decay': 1e-3,     'use_smooth_loss': True,   'smooth_weight': 0.01  },
        'pointnet2': {'lr': 5e-4,    'pos_weight': 3.0,       'weight_decay': 1e-4,     'use_smooth_loss': False,  'smooth_weight': 0.0   },
        'transformer': {'lr': 2e-4,   'pos_weight': 2.5,      'weight_decay': 1e-3,     'use_smooth_loss': True,   'smooth_weight': 0.005 },
        'pointmlp': {    'lr': 1e-3,  'pos_weight': 2.0,      'weight_decay': 2e-4,     'use_smooth_loss': True,    'smooth_weight': 0.01 }
    }

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

class Stage2Dataset(Dataset):
    """
    第二步：重构 Stage2Dataset (适配消融实验)
    """
    def __init__(self, offline_dir, split="train", augment=False):
        self.offline_dir = offline_dir
        self.split = split
        self.augment = augment 
        
        # 1. 收集仿真训练集文件 (位于当前消融目录下)
        sim_train_path = offline_dir
        self.sim_files = glob.glob(os.path.join(sim_train_path, "*.npy"))
        
        # 2. 收集真实验证/测试集资源池 (由 06 脚本生成的 Real Shoe 数据)
        eval_pool = []
        real_entities = ['shoe_a', 'shoe_b', 'shoe_c']
        for ent in real_entities:
            # 真实数据保存在公共根目录下
            path = os.path.join(Config.STAGE2_OFFLINE_ROOT, ent)
            eval_pool.extend(glob.glob(os.path.join(path, "*.npy")))
        
        # 排序并使用固定种子打乱，确保分片的一致性
        eval_pool.sort()
        random.seed(Config.SEED)
        random.shuffle(eval_pool)
        
        # 固定 50/50 混合切分验证和测试集 (其中 Shoe A, B, C 均只作为评估用)
        split_idx = int(len(eval_pool) * 0.5)
        self.val_files = eval_pool[:split_idx]
        self.test_files = eval_pool[split_idx:]

        if split == "train":
            # 消融实验训练集必须是 Pure Synthetic Only，不允许混入真实数据
            self.file_list = self.sim_files
            print(f"[Stage2Dataset] Train Split (Pure Sim): {len(self.file_list)} files from {sim_train_path}")
        elif split == "val":
            # 验证集必须是 Pure Real Only
            self.file_list = self.val_files
            print(f"[Stage2Dataset] Val Split (Pure Real): {len(self.file_list)} files")
        else: # test
            # 测试集必须是 Pure Real Only
            self.file_list = self.test_files
            print(f"[Stage2Dataset] Test Split (Pure Real): {len(self.file_list)} files")
        
        if len(self.file_list) == 0:
             print(f"Warning: No data found for {split} in {offline_dir}. Check STAGE1.5 results.")
             
    def __len__(self):
        return len(self.file_list)

    def _augment(self, feat):
        """
        统一 Stage 2 在线增强协议 (根据 STAGE2_AUG_MODE 切换)
        支持模式: 'none', 'old', 'mirror', 'full'
        """
        mode = Config.STAGE2_AUG_MODE
        if mode == 'none':
            return feat

        # feat: (C, N)
        # 协议索引: 0-2: XYZ, 3-5: RGB, 6: Theta, 7-9: Var_RGB, 10-12: Normal, 13: Curvature
        xyz = feat[:3, :]
        rgb = feat[3:6, :]
        
        # --- 1. 左右镜像 (Y-axis Mirror: y -> -y) ---
        # 仅在 'mirror' 和 'full' 模式下开启
        if mode in ['mirror', 'full']:
            if np.random.rand() > 0.5:
                # 翻转 xyz[1, :]
                xyz[1, :] = -xyz[1, :]
                # 同步翻转法向量的 Y 分量 (Index 11)
                if feat.shape[0] >= 13:
                    feat[11, :] = -feat[11, :]

        # --- 2. 几何变换 (Rotation, Scale, Jitter) ---
        # 在 'old', 'mirror', 'full' 模式下都开启
        
        # 2.1 随机旋转 (绕 Z 轴)
        angle = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        xyz = np.dot(rot_matrix, xyz)
        # 法向量必须随同一个 rotation matrix 一起旋转以保持几何一致性
        if feat.shape[0] >= 13:
            normal = feat[10:13, :]
            normal = np.dot(rot_matrix, normal)
            feat[10:13, :] = normal

        # 2.2 随机缩放 (0.9 - 1.1)
        s = np.random.uniform(0.9, 1.1)
        xyz = xyz * s

        # 2.3 随机抖动 (Jitter)
        xyz += np.random.normal(0, 0.005, size=xyz.shape)

        # --- 3. 颜色增强 (Brightness/Contrast) ---
        # 在 'old', 'mirror', 'full' 模式下都开启
        a = np.random.uniform(0.8, 1.2)
        rgb = np.clip(rgb * a, 0.0, 1.0)
        # 同步更新颜色方差 Var_RGB (Index 7-9)
        if feat.shape[0] >= 10:
            feat[7:10, :] = feat[7:10, :] * (a**2)

        # --- 4. Theta 重计算 ---
        # 仅在 'full' 模式下，在所有空间变换结束后重算
        if mode == 'full':
            new_center = np.mean(xyz, axis=1, keepdims=True)
            rel_xyz = xyz - new_center
            new_theta = np.arctan2(rel_xyz[1, :], rel_xyz[0, :]) / np.pi
            feat[6, :] = new_theta
        # 注意: 'old' 和 'mirror' 模式下不重算 theta，保留原始离线生成的 theta

        # 更新回特征矩阵
        feat[:3, :] = xyz
        feat[3:6, :] = rgb
        return feat

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path).astype(np.float32)
        
        # 严格校验 15 维数据: [14维特征 + 1维标签]
        if data.shape[1] == 15:
            feat = data[:, :14].T
            label = data[:, 14].astype(np.int64)
        else:
            raise ValueError(f"Ablation pipeline requires 15-column data, but got {data.shape[1]}. Path: {file_path}")
        
        if self.augment:
            feat = self._augment(feat)
        
        return torch.from_numpy(feat).float(), torch.from_numpy(label).long(), file_path


class DiceLoss(nn.Module):
    """
    Dice Loss 针对拓扑结构 (如圆环) 非常有效，它关注预测区域与真实区域的重合度
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: (B, 2, N), target: (B, N)
        probs = F.softmax(logits, dim=1)[:, 1, :] # (B, N)
        target = target.float()
        
        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class ColorAwareSmoothnessLoss(nn.Module):
    """
    颜色感知平滑损失 (Color-Aware Smoothness Loss)
    逻辑：如果邻居点 j 与中心点 i 颜色极度相似，但模型预测差异大，则施加惩罚。
    """
    def __init__(self, k=16):
        super(ColorAwareSmoothnessLoss, self).__init__()
        self.k = k

    def forward(self, logits, xyz, rgb):
        # logits: (B, 2, N), xyz: (B, 3, N), rgb: (B, 3, N)
        batch_size, _, num_points = xyz.size()
        
        # 1. 寻找空间 K 近邻
        dist = -2 * torch.matmul(xyz.transpose(2, 1), xyz)
        xx = torch.sum(xyz**2, dim=1, keepdim=True)
        dist = xx + dist + xx.transpose(2, 1)
        idx = dist.topk(k=self.k, dim=-1, largest=False)[1] # (B, N, K)
        
        # 2. 获取预测概率
        probs = F.softmax(logits, dim=1) # (B, 2, N)
        
        # 3. 提取邻居特征
        def gather_neighbors(feat, idx):
            B, C, N = feat.size()
            K = idx.size(2)
            idx_base = torch.arange(0, B, device=feat.device).view(-1, 1, 1) * N
            idx_flat = (idx + idx_base).view(-1)
            feat_t = feat.transpose(2, 1).contiguous()
            neighbor_feat = feat_t.view(B * N, C)[idx_flat, :]
            return neighbor_feat.view(B, N, K, C)

        neighbor_rgb = gather_neighbors(rgb, idx)      # (B, N, K, 3)
        neighbor_probs = gather_neighbors(probs, idx)  # (B, N, K, 2)
        
        center_rgb = rgb.transpose(2, 1).unsqueeze(2)      # (B, N, 1, 3)
        center_probs = probs.transpose(2, 1).unsqueeze(2)  # (B, N, 1, 2)
        
        # 4. 计算 RGB 距离
        rgb_dist = torch.norm(center_rgb - neighbor_rgb, dim=-1) # (B, N, K)
        
        # 5. 计算预测概率差异 (L1)
        prob_diff = torch.abs(center_probs - neighbor_probs).sum(dim=-1) # (B, N, K)
        
        # 6. 权重计算：颜色越接近 (rgb_dist 越小)，权重越大
        # 使用高斯核映射：weight = exp(-rgb_dist^2 / sigma)
        weight = torch.exp(-rgb_dist * 10.0)
        
        loss = (weight * prob_diff).mean()
        return loss

class Stage2Loss(nn.Module):
    """
    组合损失函数：CE_Loss + 0.5 * Dice_Loss + (optional) Smooth_Loss
    """
    def __init__(self, pos_weight=2.0, use_smooth=True, smooth_weight=0.01):
        super(Stage2Loss, self).__init__()
        self.register_buffer('weight', torch.tensor([1.0, pos_weight]))
        self.ce = nn.CrossEntropyLoss(weight=self.weight)
        self.dice = DiceLoss()
        self.use_smooth = use_smooth
        self.smooth_weight = smooth_weight
        if use_smooth:
            self.smooth = ColorAwareSmoothnessLoss()

    def forward(self, logits, target, xyz, rgb):
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        loss = ce_loss + 0.5 * dice_loss
        
        if self.use_smooth:
            smooth_loss = self.smooth(logits, xyz, rgb)
            loss += self.smooth_weight * smooth_loss
            
        return loss

class Stage2Trainer:
    """
    Stage 2 精细分割训练引擎
    1. 支持 4 种 Baseline 模型切换
    2. 使用标准加权交叉熵损失
    3. 动态日志管理
    """
    def __init__(self, config, current_model_type=None):
        self.cfg = config
        # 如果传入了 model_type 则使用，否则使用配置中的第一个
        if current_model_type:
            self.model_type = current_model_type.lower()
        else:
            self.model_type = config.STAGE2_MODEL_TYPES[0].lower()
            
        # 加载专属配置 (如果找不到则使用 dgcnn 作为默认值)
        self.m_cfg = config.STAGE2_MODEL_CONFIGS.get(self.model_type, config.STAGE2_MODEL_CONFIGS['dgcnn'])
        
        # 确保日志目录存在
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.log_file = os.path.join(config.LOG_DIR, f"stage2_{self.model_type}_train.log")
        self.setup_logging()
        
        # 初始化模型 (全部使用标准版 Baseline，确保公平对比)
        if self.model_type == 'dgcnn':
            self.model = DGCNN(num_classes=2, input_channels=config.STAGE2_INPUT_CHANNELS).to(config.DEVICE)
        elif self.model_type == 'pointnet2':
            self.model = PointNet2(num_classes=2, input_channels=config.STAGE2_INPUT_CHANNELS).to(config.DEVICE)
        elif self.model_type == 'transformer':
            self.model = PointTransformerSeg(num_classes=2, in_channels=config.STAGE2_INPUT_CHANNELS).to(config.DEVICE)
        elif self.model_type == 'pointmlp':
            self.model = PointMLP(num_classes=2, input_channels=config.STAGE2_INPUT_CHANNELS).to(config.DEVICE)
        else:
            raise ValueError(f"Unknown Stage 2 model type: {self.model_type}")

        # 使用专属的学习率和权重衰减
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.m_cfg['lr'], 
                                    weight_decay=self.m_cfg['weight_decay'])
        # 引入学习率调度器：增加 T_max 周期，配合大数据量
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.STAGE2_EPOCHS)
        
        # 使用组合损失函数 (专属配置)
        self.criterion = Stage2Loss(
            pos_weight=self.m_cfg['pos_weight'],
            use_smooth=self.m_cfg['use_smooth_loss'],
            smooth_weight=self.m_cfg['smooth_weight']
        ).to(config.DEVICE)
        
        self.best_iou = 0.0
        self.history = defaultdict(list)

    def setup_logging(self):
        # 移除已有的 handlers 避免冲突
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"Stage2_{self.model_type}")

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        pbar = tqdm(loader, desc=f"Stage 2 {self.model_type} Epoch {epoch+1}")
        
        for feat, target, _ in pbar:
            feat, target = feat.to(self.cfg.DEVICE), target.to(self.cfg.DEVICE)
            # 提取 XYZ 和 RGB 用于计算平滑损失
            xyz, rgb = feat[:, :3, :], feat[:, 3:6, :]
            
            self.optimizer.zero_grad()
            outputs = self.model(feat) # 处理兼容性
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = self.criterion(logits, target, xyz, rgb)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy().flatten())
            
        avg_loss = total_loss / len(loader)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        self.logger.info(f"Train - Loss: {avg_loss:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
        return avg_loss

    def evaluate(self, loader, desc="Eval"):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for feat, target, _ in tqdm(loader, desc=desc):
                feat, target = feat.to(self.cfg.DEVICE), target.to(self.cfg.DEVICE)
                xyz, rgb = feat[:, :3, :], feat[:, 3:6, :]
                
                outputs = self.model(feat) # 处理兼容性
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self.criterion(logits, target, xyz, rgb)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(target.cpu().numpy().flatten())
                
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        intersection = np.sum((all_preds == 1) & (all_labels == 1))
        union = np.sum((all_preds == 1) | (all_labels == 1))
        iou = intersection / union if union > 0 else 0
        
        self.logger.info(f"{desc} - Recall: {recall:.4f}, Precision: {precision:.4f}, IoU: {iou:.4f}, F1: {f1:.4f}")
        return iou, {"recall": recall, "precision": precision, "f1": f1, "iou": iou}

    def run(self, train_loader, val_loader, test_loader):
        self.logger.info(f"Starting Stage 2 Training with {self.model_type}...")
        patience_counter = 0
        
        # 记录最佳模型对应的指标
        self.best_epoch = 0
        self.best_val_metrics = None
        self.best_test_metrics = None
        
        for epoch in range(self.cfg.STAGE2_EPOCHS):
            _ = self.train_epoch(train_loader, epoch)
            val_iou, val_metrics = self.evaluate(val_loader, "Val")
            test_iou, test_metrics = self.evaluate(test_loader, "Test")
            
            # 步进学习率
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1} LR: {curr_lr:.6f}")
            
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.best_epoch = epoch + 1
                self.best_val_metrics = val_metrics
                self.best_test_metrics = test_metrics
                
                torch.save(self.model.state_dict(), os.path.join(self.cfg.MODEL_SAVE_DIR, f"best_stage2_{self.model_type}.pth"))
                # 明确标注 Val 和 Test IoU
                self.logger.info(f">>> Best Model Saved (Val IoU: {val_iou:.4f}, Test IoU: {test_iou:.4f})")
                patience_counter = 0 # 重置计数器
            else:
                patience_counter += 1
                self.logger.info(f"Patience counter: {patience_counter}/{self.cfg.STAGE2_PATIENCE}")
            
            if patience_counter >= self.cfg.STAGE2_PATIENCE:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # 训练结束后打印 Stage 2 模型汇总信息
        print("\n" + "="*60)
        print(f"*** Stage 2 {self.model_type.upper()} BEST MODEL SUMMARY (Epoch {self.best_epoch}) ***")
        print("="*60)
        if self.best_val_metrics:
            print(f"[Val Dataset]   Recall: {self.best_val_metrics['recall']:.4f}, IoU: {self.best_val_metrics['iou']:.4f}")
        if self.best_test_metrics:
            print(f"[Test Dataset]  Recall: {self.best_test_metrics['recall']:.4f}, IoU: {self.best_test_metrics['iou']:.4f}")
        print("="*60 + "\n")
        
        return self.best_val_metrics, self.best_test_metrics

# ============================================
# 5. 主程序入口 (消融实验专用版)
# ============================================
def main():
    config = Config()
    
    # 确定要处理的模式列表
    modes_to_process = [config.MODE]
    if config.MODE == 'all':
        modes_to_process = ["01_Base", "02_Geo", "03_Phys", "04_STS2R"]
        print(f">>> BATCH MODE: Processing all variants: {modes_to_process}")

    # 初始化全局汇总字典 (仅在 all 模式下用于汇总)
    global_ablation_results = {}

    # 循环处理每个消融版本
    for current_mode in modes_to_process:
        # 动态更新 Config 状态 (路径、实验名等)
        config.update_mode(current_mode)
        print("\n" + "#"*60)
        print(f"### RUNNING ABLATION MODE: {current_mode.upper()} ###")
        print("#"*60 + "\n")
        
        # ==========================================================
        # 0. 实验目录准备
        # ==========================================================
        def prepare_directory(path, desc, clean=False):
            if clean and os.path.exists(path):
                print(f">>> Cleaning up existing {desc} directory: {path}")
                shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)

        # 只要涉及 Stage 2 训练，就清理当前模式的结果目录
        if config.RUN_STAGE == 2:
            prepare_directory(config.OUTPUT_ROOT, f"Results ({current_mode})", clean=True)
            for sub in [config.MODEL_SAVE_DIR, config.VIZ_DIR, config.LOG_DIR]:
                os.makedirs(sub, exist_ok=True)

        # ==========================================================
        # 2. Stage 2: Fine Segmentation (精细分割消融训练)
        # ==========================================================
        all_summary_results = {} # 每个模式独立的汇总
        if config.RUN_STAGE == 2:
            print("\n" + "="*50)
            print(f">>> [Stage 2] Starting Fine Segmentation Ablation: {current_mode}")
            print("="*50)
            
            # 实例化数据集：训练只用当前模式的 Sim，验证/测试用 Shoe A, B, C 的混合
            train_ds = Stage2Dataset(config.STAGE2_OFFLINE_DIR, split="train", augment=config.STAGE2_USE_ONLINE_AUGMENT)
            val_ds = Stage2Dataset(config.STAGE2_OFFLINE_DIR, split="val", augment=False)
            test_ds = Stage2Dataset(config.STAGE2_OFFLINE_DIR, split="test", augment=False)
            
            if len(train_ds) == 0:
                print(f"Error: No training data found for {current_mode}. Please run 06 script first.")
                if config.MODE != 'all': return
                else: continue

            # 依次训练列表中的所有模型 (PointMLP / DGCNN / PointNet2)
            for model_name in config.STAGE2_MODEL_TYPES:
                # [环境隔离] 每次训练前重置种子并重建 DataLoader，确保独立运行的一致性
                set_seed(config.SEED)
                train_loader = DataLoader(train_ds, batch_size=config.STAGE2_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
                val_loader = DataLoader(val_ds, batch_size=config.STAGE2_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
                test_loader = DataLoader(test_ds, batch_size=config.STAGE2_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
                print(f"\n" + "-"*60)
                print(f"[Stage2] Dataset Variant: {current_mode} | Backbone: {model_name} | Aug Mode: {config.STAGE2_AUG_MODE}")
                print("-"*60)
                try:
                    trainer = Stage2Trainer(config, current_model_type=model_name)
                    s2_val, s2_test = trainer.run(train_loader, val_loader, test_loader)
                    
                    # 记录结果
                    all_summary_results[model_name] = {'val': s2_val, 'test': s2_test}
                except Exception as e:
                    print(f"!!! Error training {model_name}: {e}")
                    continue

            print(f"\n>>> All Stage 2 tasks for {current_mode} have been completed.")
            
            # 将当前模式的结果存入全局汇总 (深拷贝以防引用问题)
            import copy
            global_ablation_results[current_mode] = copy.deepcopy(all_summary_results)
        
        # ==========================================================
        # 3. 生成当前模式的最终汇总报告
        # ==========================================================
        if all_summary_results:
            summary_path = os.path.join(config.OUTPUT_ROOT, "final_ablation_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("="*60 + "\n")
                f.write("       ABLATION STUDY SUMMARY REPORT (DATASET VARIANT)\n")
                f.write("="*60 + "\n")
                f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Ablation Version: {current_mode}\n")
                f.write("-" * 60 + "\n\n")
                
                for m_name, results in all_summary_results.items():
                    f.write(f"BACKBONE: {m_name.upper()}\n")
                    if results['val']:
                        v = results['val']
                        f.write(f"  [Validation - Real] IoU: {v['iou']:.4f}, F1: {v['f1']:.4f}, Recall: {v['recall']:.4f}, Precision: {v['precision']:.4f}\n")
                    if results['test']:
                        t = results['test']
                        f.write(f"  [Test - Real]       IoU: {t['iou']:.4f}, F1: {t['f1']:.4f}, Recall: {t['recall']:.4f}, Precision: {t['precision']:.4f}\n")
                    f.write("-" * 30 + "\n")
            
            print("\n" + "*"*60)
            print(f">>> Ablation evaluation summary for {current_mode} saved to:\n{os.path.abspath(summary_path)}")
            print("*"*60 + "\n")

    # ==========================================================
    # 4. [新增] 如果是 all 模式，生成全局汇总报告
    # ==========================================================
    if len(modes_to_process) > 1 and global_ablation_results:
        global_summary_path = os.path.join("./experiments/ablation_results", "global_ablation_summary.txt")
        with open(global_summary_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("             GLOBAL ABLATION STUDY CROSS-MODE SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ablation Versions Run: {', '.join(modes_to_process)}\n")
            f.write("-" * 80 + "\n\n")
            
            for mode_name, mode_res in global_ablation_results.items():
                f.write(f"### ABLATION VARIANT: {mode_name.upper()} ###\n")
                for m_name, results in mode_res.items():
                    f.write(f"  BACKBONE: {m_name.upper()}\n")
                    if results['test']:
                        t = results['test']
                        f.write(f"    [Test Results - Real] IoU: {t['iou']:.4f}, F1: {t['f1']:.4f}, Recall: {t['recall']:.4f}, Precision: {t['precision']:.4f}\n")
                f.write("\n" + "-" * 40 + "\n\n")
        
        print("\n" + "#"*60)
        print(f">>> GLOBAL Ablation summary saved to:\n{os.path.abspath(global_summary_path)}")
        print("#"*60 + "\n")

    if config.RUN_STAGE != 2:
        print(f"Invalid RUN_STAGE: {config.RUN_STAGE}. Only Stage 2 training is supported in this script.")
    else:
        print("\n" + "="*50)
        print(">>> All requested Ablation Pipeline tasks completed.")
        print("="*50)

if __name__ == "__main__":
    main()
