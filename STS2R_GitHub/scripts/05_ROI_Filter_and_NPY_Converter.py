import os
import sys
import glob
import time
import shutil
import numpy as np
from tqdm import tqdm

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pc_io import load_txt_pointcloud

# ============================================
# 1. Config
# ============================================
class Config:
    # 输入目录 (由 01-04 脚本生成)
    INPUT_PATHS = {
        "01_Base": r"outputs\Generated_ablation_data\01_Base",
        "02_Geo": r"outputs\Generated_ablation_data\02_Geo",
        "03_Phys": r"outputs\Generated_ablation_data\03_Phys",
        "04_STS2R": r"outputs\Generated_ablation_data\04_STS2R",
    }
    
    # 输出目录 (ROI 过滤后的 NPY 数据)
    OUTPUT_ROOT = r"outputs\Ablation_ROI_NPY"
    
    # ROI 过滤比例 (保留 Z 轴底部 50%)
    FILTER_RATIO = 0.5

# ============================================
# 2. ROI 过滤逻辑
# ============================================
def pass_through_filter_z(data, filter_ratio=0.5):
    """
    直通滤波：保留 z 轴从最小值开始的指定比例高度范围
    """
    if data is None or len(data) == 0:
        return None
    
    # z 轴在第 3 列 (索引 2)
    z_coords = data[:, 2].astype(np.float64)
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    
    # 计算滤波阈值
    z_filter_upper = z_min + (z_max - z_min) * filter_ratio
    
    # 筛选符合条件的点
    mask = (z_coords >= z_min) & (z_coords <= z_filter_upper)
    filtered_data = data[mask]
    
    return filtered_data

# ============================================
# 3. 主流程
# ============================================
def main():
    cfg = Config()
    print(f"🚀 Starting 05_ROI_Filter_and_NPY_Converter pipeline...")
    print(f"Filtering ratio: {cfg.FILTER_RATIO * 100:.0f}% (Keep bottom part of Z-axis)")
    
    # 清理输出根目录
    if os.path.exists(cfg.OUTPUT_ROOT):
        print(f"🧹 Cleaning up existing output root: {cfg.OUTPUT_ROOT}")
        shutil.rmtree(cfg.OUTPUT_ROOT, ignore_errors=True)
    os.makedirs(cfg.OUTPUT_ROOT, exist_ok=True)

    start_time = time.time()
    total_processed = 0
    total_success = 0
    
    for mode_name, input_dir in cfg.INPUT_PATHS.items():
        print(f"\n{'=' * 70}")
        print(f"Processing mode: {mode_name}")
        print(f"Input dir : {input_dir}")
        
        if not os.path.exists(input_dir):
            print(f"⚠️ Warning: Input directory {input_dir} not found. Skipping...")
            continue
            
        output_dir = os.path.join(cfg.OUTPUT_ROOT, mode_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有 txt 文件
        file_list = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
        print(f"Total files: {len(file_list)}")
        
        if len(file_list) == 0:
            continue
            
        success_count = 0
        fail_count = 0
        
        for file_path in tqdm(file_list, desc=f"Converting {mode_name}"):
            total_processed += 1
            try:
                # 1. 加载数据
                data = load_txt_pointcloud(file_path)
                if data is None:
                    fail_count += 1
                    continue
                
                # 2. ROI 过滤 (Z-axis Pass-through Filter)
                filtered_data = pass_through_filter_z(data, cfg.FILTER_RATIO)
                if filtered_data is None or len(filtered_data) == 0:
                    fail_count += 1
                    continue
                
                # 3. 保存为 NPY
                file_name = os.path.splitext(os.path.basename(file_path))[0] + ".npy"
                save_path = os.path.join(output_dir, file_name)
                np.save(save_path, filtered_data.astype(np.float32))
                
                success_count += 1
                total_success += 1
                
            except Exception as e:
                fail_count += 1
                print(f"\n❌ Error processing {file_path}: {e}")
                
        print(f"Done: {mode_name} | success={success_count}, fail={fail_count}")
        print(f"Results saved to: {output_dir}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"✅ All tasks completed in {total_time:.2f} seconds.")
    print(f"Total processed: {total_processed}, Success: {total_success}, Fail: {total_processed - total_success}")
    print(f"Final output root: {cfg.OUTPUT_ROOT}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
