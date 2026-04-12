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
import matplotlib.pyplot as plt
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
    # 实验模式控制:
    # 'full'      - 全量数据训练 (Sim + Real)
    # 'real_only' - 仅使用真实数据 (Real Only)
    # 'sim_only'  - 仅使用仿真数据 (Sim Only)
    # 'all'       - 依次运行以上所有模式
    MODE = 'all' 
    
    # 阶段控制开关: 
    # 2   - 仅运行精细分割 (Stage 2)
    RUN_STAGE = 2
    
    # 基准测试工程管理 (自动根据模式生成实验名)
    EXP_NAME = f"exp_{MODE}"
    
    # Stage 2 离线数据存储根目录 (由 06 脚本生成)
    STAGE2_OFFLINE_ROOT = r"outputs\stage1.5_offline_data"

    # Stage 2 离线数据存储 (所有实验模式复用)
    STAGE2_OFFLINE_DIR = os.path.join(STAGE2_OFFLINE_ROOT, "04_STS2R")
  
    OUTPUT_ROOT = os.path.join("./experiments/benchmark_results", EXP_NAME)
    
    MODEL_SAVE_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
    VIZ_DIR = os.path.join(OUTPUT_ROOT, "visualizations")
    LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")
    
    LOG_FILE = os.path.join(LOG_DIR, f"{EXP_NAME}_stage2_train.log")

    @classmethod
    def update_mode(cls, new_mode):
        """动态更新模式及相关路径"""
        cls.MODE = new_mode
        cls.EXP_NAME = f"exp_{new_mode}"
        cls.OUTPUT_ROOT = os.path.join("./experiments/benchmark_results", cls.EXP_NAME)
        cls.MODEL_SAVE_DIR = os.path.join(cls.OUTPUT_ROOT, "checkpoints")
        cls.VIZ_DIR = os.path.join(cls.OUTPUT_ROOT, "visualizations")
        cls.LOG_DIR = os.path.join(cls.OUTPUT_ROOT, "logs")
        cls.LOG_FILE = os.path.join(cls.LOG_DIR, f"{cls.EXP_NAME}_stage2_train.log")
    
    # 动态混合验证和测试集的比例 (仅在 sim_only 模式下生效)
    SIM_ONLY_VAL_RATIO = 0.5
    
    # 硬件与通用参数
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4 
    NUM_CLASSES = 2
    
    # Stage 2 训练参数
    STAGE2_MODEL_TYPES = [ 'pointmlp', 'dgcnn', 'pointnet2', ] # 依次训练这些模型
    STAGE2_EPOCHS = 100
    STAGE2_BATCH_SIZE = 8
    STAGE2_PATIENCE = 10 # 早停
    M_POINTS = 8192
    STAGE2_INPUT_CHANNELS = 14 # 升级为 14维特征: [XYZ(3), RGB(3), Theta(1), Var_RGB(3), Normal(3), Curvature(1)]
    STAGE2_REAL_OVERSAMPLE_RATIO = 10
    STAGE2_USE_ONLINE_AUGMENT = False # 已设为 None 模式: 不使用在线增强效果最佳
    
    # 模型专属配置 (解耦学习率、权重和平滑损失)
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
    第二步：重构 Stage2Dataset (三折验证适配版)
    """
    def __init__(self, offline_dir, train_dirs=None, eval_dirs=None, split="train", augment=False):
        self.offline_dir = offline_dir
        self.split = split
        self.augment = augment 
        
        # 1. 收集训练集文件
        train_files = []
        sim_train_files = []
        real_train_files = []
        if train_dirs:
            for d in train_dirs:
                # 兼容 06 脚本生成的路径结构
                path = os.path.join(Config.STAGE2_OFFLINE_ROOT, d)
                
                files = glob.glob(os.path.join(path, "*.npy"))
                train_files.extend(files)
                if d.lower() in ["01_base", "02_geo", "03_phys", "04_sts2r", "sim"]:
                    sim_train_files.extend(files)
                else:
                    real_train_files.extend(files)
        
        # 2. 收集验证/测试集资源池
        eval_pool = []
        if eval_dirs:
            for d in eval_dirs:
                path = os.path.join(Config.STAGE2_OFFLINE_ROOT, d)
                eval_pool.extend(glob.glob(os.path.join(path, "*.npy")))
            # 排序并使用固定种子打乱，确保分片的一致性
            eval_pool.sort()
            random.seed(Config.SEED)
            random.shuffle(eval_pool)
            
        if split == "train":
            oversample_ratio = max(int(getattr(Config, "STAGE2_REAL_OVERSAMPLE_RATIO", 1)), 1)
            self.file_list = sim_train_files + real_train_files * oversample_ratio
            print(f"[Stage2Dataset] Train Split: {len(sim_train_files)} Sim + {len(real_train_files)} Real (Oversampled {oversample_ratio}x)")
        else:
            # 50/50 混合切分验证和测试集 (保持与 Config 比例一致)
            val_ratio = getattr(Config, "SIM_ONLY_VAL_RATIO", 0.5)
            split_idx = int(len(eval_pool) * val_ratio)
            if split == "val":
                self.file_list = eval_pool[:split_idx]
            else: # test
                self.file_list = eval_pool[split_idx:]
            print(f"[Stage2Dataset] {split.capitalize()} Split: {len(self.file_list)} files (Mixed from {len(eval_pool)} eval resources, ratio={val_ratio})")
        
        if len(self.file_list) == 0:
             print(f"Warning: No data found for {split} in {offline_dir}. Check STAGE1.5 results.")
             
    def __len__(self):
        return len(self.file_list)

    def _augment(self, feat):
        """
        Stage 2 在线增强 (None 模式)
        根据消融实验结论，不进行任何在线增强效果最佳。
        """
        return feat

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # 兼容性读取：如果数据没更新，自动兼容旧的 11 维数据
        data = np.load(file_path).astype(np.float32)
        
        if data.shape[1] == 15:
            # 15 维数据: [14维特征 + 1维标签]
            feat = data[:, :14].T
            label = data[:, 14].astype(np.int64)
        elif data.shape[1] == 11:
            # 11 维数据: [10维特征 + 1维标签]
            feat = data[:, :10].T
            label = data[:, 10].astype(np.int64)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape} in {file_path}")
        
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

# ============================================
# 4. 训练引擎 (已隐藏)
# ============================================


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
# 5. 主程序入口
# ============================================
def main():
    config = Config()
    
    # 定义要运行的模式列表
    if config.MODE == 'all':
        modes_to_run = ['real_only', 'full', 'sim_only']
    else:
        modes_to_run = [config.MODE]

    # 初始化全局汇总字典 (仅在 all 模式下用于汇总)
    global_benchmark_results = {}

    for current_mode in modes_to_run:
        # 动态更新当前实验模式
        config.update_mode(current_mode)
        print("\n" + "#"*60)
        print(f"### RUNNING BENCHMARK MODE: {current_mode.upper()} ###")
        print("#"*60 + "\n")

        all_summary_results = {} # 用于记录当前模式下所有模型的最终指标
        
        # 获取运行模式
        run_all = (config.RUN_STAGE == 0)

        # ==========================================================
        # 0. 实验目录清理 (确保完全覆盖，无残留)
        # ==========================================================
        def cleanup_directory(path, desc):
            if os.path.exists(path):
                print(f">>> Cleaning up existing {desc} directory: {path}")
                shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)

        # 只要运行阶段包含训练 (2)，就清理结果目录
        if run_all or config.RUN_STAGE == 2:
            cleanup_directory(config.OUTPUT_ROOT, f"Results ({current_mode})")
            # 重新创建必要的子目录 (Trainer 初始化时也会创建，这里双重保险)
            for sub in [config.MODEL_SAVE_DIR, config.VIZ_DIR, config.LOG_DIR]:
                os.makedirs(sub, exist_ok=True)

        # ==========================================================
        # 2. Stage 2: Fine Segmentation (精细分割) - 支持连续训练多个模型
        # ==========================================================
        if run_all or config.RUN_STAGE == 2:
            print("\n" + "="*50)
            print(f">>> [Stage 2] Initializing Fine Segmentation for {current_mode}...")
            print("="*50)
            
            # 定义不同模式下的分折策略
            fold_configs = []
            if config.MODE == 'real_only':
                # 三折验证逻辑
                fold_configs = [
                    {'name': 'Fold0', 'train': ['shoe_a'], 'eval': ['shoe_b', 'shoe_c']},
                    {'name': 'Fold1', 'train': ['shoe_b'], 'eval': ['shoe_a', 'shoe_c']},
                    {'name': 'Fold2', 'train': ['shoe_c'], 'eval': ['shoe_a', 'shoe_b']}
                ]
            elif config.MODE == 'full':
                fold_configs = [{'name': 'Full', 'train': ['04_STS2R', 'shoe_a'], 'eval': ['shoe_b', 'shoe_c']}]
            elif config.MODE == 'sim_only':
                fold_configs = [{'name': 'SimOnly', 'train': ['04_STS2R'], 'eval': ['shoe_a', 'shoe_b', 'shoe_c']}]

            fold_results = {} # 存储每一折的结果

            for fold in fold_configs:
                fold_name = fold['name']
                print(f"\n>>> [Fold: {fold_name}] Train Dirs: {fold['train']}, Eval Dirs: {fold['eval']}")
                
                # 实例化数据集
                train_ds = Stage2Dataset(config.STAGE2_OFFLINE_DIR, train_dirs=fold['train'], split="train", augment=config.STAGE2_USE_ONLINE_AUGMENT)
                val_ds = Stage2Dataset(config.STAGE2_OFFLINE_DIR, eval_dirs=fold['eval'], split="val", augment=False)
                test_ds = Stage2Dataset(config.STAGE2_OFFLINE_DIR, eval_dirs=fold['eval'], split="test", augment=False)
                
                if len(train_ds) == 0:
                    print(f"Error: No training data found for {fold_name}. Skipping.")
                    continue

                train_loader = DataLoader(train_ds, batch_size=config.STAGE2_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
                val_loader = DataLoader(val_ds, batch_size=config.STAGE2_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
                test_loader = DataLoader(test_ds, batch_size=config.STAGE2_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
                
                # 依次训练列表中的所有模型
                for model_name in config.STAGE2_MODEL_TYPES:
                    print(f"\n--- Fold: {fold_name} | Model: {model_name.upper()} ---")
                    try:
                        # 为了避免不同 Fold 的权重覆盖，在保存时加入 Fold 名
                        trainer = Stage2Trainer(config, current_model_type=model_name)
                        # 修改 trainer 的保存路径 (Hack: 临时修改 cfg.MODEL_SAVE_DIR 以区分折数)
                        original_save_dir = config.MODEL_SAVE_DIR
                        config.MODEL_SAVE_DIR = os.path.join(original_save_dir, fold_name)
                        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
                        
                        s2_val, s2_test = trainer.run(train_loader, val_loader, test_loader)
                        
                        # 恢复原始路径
                        config.MODEL_SAVE_DIR = original_save_dir
                        
                        # 记录结果
                        res_key = f"{fold_name}_{model_name}"
                        fold_results[res_key] = {'val': s2_val, 'test': s2_test}
                        all_summary_results[res_key] = fold_results[res_key]
                    except Exception as e:
                        print(f"!!! Error training {model_name} in {fold_name}: {e}")
                        continue
                        
            # 如果是 real_only 模式，生成额外的三折汇总报告
            if config.MODE == 'real_only' and fold_results:
                summary_3fold_path = os.path.join(config.OUTPUT_ROOT, "real_only_3fold_summary.txt")
                with open(summary_3fold_path, "w", encoding="utf-8") as f:
                    f.write("="*60 + "\n")
                    f.write("       REAL_ONLY 3-FOLD CROSS VALIDATION REPORT\n")
                    f.write("="*60 + "\n")
                    
                    # 按模型分类计算平均值
                    for m_name in config.STAGE2_MODEL_TYPES:
                        f.write(f"\nMODEL: {m_name.upper()}\n")
                        f.write("-" * 30 + "\n")
                        metrics = ['iou', 'f1', 'recall', 'precision']
                        sum_test = {k: 0.0 for k in metrics}
                        count = 0
                        
                        for f_name in ['Fold0', 'Fold1', 'Fold2']:
                            key = f"{f_name}_{m_name}"
                            if key in fold_results and fold_results[key]['test']:
                                res = fold_results[key]['test']
                                f.write(f"  {f_name} -> IoU: {res['iou']:.4f}, F1: {res['f1']:.4f}, R: {res['recall']:.4f}, P: {res['precision']:.4f}\n")
                                for k in metrics: sum_test[k] += res[k]
                                count += 1
                        
                        if count > 0:
                            avg_test = {k: v / count for k, v in sum_test.items()}
                            f.write(f"  [AVERAGE] -> IoU: {avg_test['iou']:.4f}, F1: {avg_test['f1']:.4f}, R: {avg_test['recall']:.4f}, P: {avg_test['precision']:.4f}\n")
                    
                print(f"\n>>> 3-Fold Cross Validation Summary saved to: {summary_3fold_path}")

            print(f"\n>>> All requested Stage 2 training tasks for {current_mode} have been completed.")
            
            # 将当前模式的结果存入全局汇总 (深拷贝以防引用问题)
            import copy
            global_benchmark_results[current_mode] = copy.deepcopy(all_summary_results)
        
        if config.RUN_STAGE != 2:
            print(f"Invalid RUN_STAGE: {config.RUN_STAGE}. Only Stage 2 training is supported in this script.")
        else:
            # ==========================================================
            # 4. 生成当前模式的汇总报告
            # ==========================================================
            if all_summary_results:
                summary_path = os.path.join(config.OUTPUT_ROOT, "final_benchmark_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write("="*60 + "\n")
                    f.write("       FINAL BENCHMARK SUMMARY REPORT\n")
                    f.write("="*60 + "\n")
                    f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Experiment Mode: {config.MODE}\n")
                    f.write("-" * 60 + "\n\n")
                    
                    for model_id, results in all_summary_results.items():
                        f.write(f"MODEL: {model_id.upper()}\n")
                        if results['val']:
                            v = results['val']
                            f.write(f"  [Validation] IoU: {v['iou']:.4f}, F1: {v['f1']:.4f}, Recall: {v['recall']:.4f}, Precision: {v['precision']:.4f}\n")
                        if results['test']:
                            t = results['test']
                            f.write(f"  [Test]       IoU: {t['iou']:.4f}, F1: {t['f1']:.4f}, Recall: {t['recall']:.4f}, Precision: {t['precision']:.4f}\n")
                        f.write("-" * 30 + "\n")
                
                print("\n" + "*"*60)
                print(f">>> All models evaluation summary for {current_mode} saved to:\n{os.path.abspath(summary_path)}")
                print("*"*60 + "\n")

    # ==========================================================
    # 5. [新增] 如果是 all 模式，生成全局汇总报告
    # ==========================================================
    if len(modes_to_run) > 1 and global_benchmark_results:
        global_summary_path = os.path.join("./experiments/benchmark_results", "global_benchmark_summary.txt")
        with open(global_summary_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("             GLOBAL BENCHMARK CROSS-MODE SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modes Run: {', '.join(modes_to_run)}\n")
            f.write("-" * 80 + "\n\n")
            
            for mode_name, mode_res in global_benchmark_results.items():
                f.write(f"### MODE: {mode_name.upper()} ###\n")
                for model_id, results in mode_res.items():
                    f.write(f"  MODEL: {model_id.upper()}\n")
                    if results['test']:
                        t = results['test']
                        f.write(f"    [Test Results] IoU: {t['iou']:.4f}, F1: {t['f1']:.4f}, Recall: {t['recall']:.4f}, Precision: {t['precision']:.4f}\n")
                f.write("\n" + "-" * 40 + "\n\n")
        
        print("\n" + "#"*60)
        print(f">>> GLOBAL Benchmark summary saved to:\n{os.path.abspath(global_summary_path)}")
        print("#"*60 + "\n")

    print("\n" + "="*50)
    print(">>> All Benchmark Pipeline Tasks Completed Successfully.")
    print("="*50)

if __name__ == "__main__":
    main()
