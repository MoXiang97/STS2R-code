import os
import sys
import shutil
import numpy as np
from tqdm import tqdm

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pc_io import load_txt_pointcloud, save_txt_pointcloud
from modules.aug_shape import apply as shape_augment
from modules.aug_line import apply as line_augment
from modules.aug_general import apply as general_augment

# Configuration
INPUT_DIR = r"assets\Base_Perfect_scan_60CAD"
OUTPUT_DIR = r"outputs\Generated_ablation_data\02_Geo"
N_VARIANTS = 50

def main():
    print(f"🚀 Starting 02_Generate_V_Geo pipeline...")
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} not found.")
        return

    # 清理并创建输出目录
    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 Cleaning up existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng()
    # 递归获取所有 .txt 文件路径
    files = []
    for root, _, filenames in os.walk(INPUT_DIR):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                files.append(os.path.join(root, filename))
    
    for input_path in tqdm(files, desc="Processing Files"):
        seed_data = load_txt_pointcloud(input_path)
        if seed_data is None: continue
        
        # 获取相对于 INPUT_DIR 的相对路径，并将路径分隔符替换为下划线，以避免文件名冲突并扁平化输出
        rel_path = os.path.relpath(input_path, INPUT_DIR)
        base_name = os.path.splitext(rel_path.replace(os.sep, '_'))[0]
        
        for i in range(1, N_VARIANTS + 1):
            try:
                lam = float(rng.uniform(0.1, 1.0))
                aug_data = shape_augment(seed_data, rng, lam)
                aug_data = line_augment(aug_data, rng, lam)
                if aug_data is None:
                    continue
                aug_data = general_augment(aug_data, rng, lam)
                
                output_name = f"{base_name}_aug_{i}_l{lam:.3f}.txt"
                save_txt_pointcloud(os.path.join(OUTPUT_DIR, output_name), aug_data)
            except Exception as e:
                print(f"Error augmenting {filename} variant {i}: {e}")
                continue

    print(f"✅ Pipeline 02 finished. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
