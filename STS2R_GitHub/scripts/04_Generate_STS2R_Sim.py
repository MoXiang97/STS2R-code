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
from modules.aug_physical import apply as physical_augment
from modules.aug_panel import apply as panel_augment, load_image_rgb255, load_image_rgba255
from modules.aug_general import apply as general_augment

# Configuration
INPUT_DIR = r"assets\Base_Perfect_scan_60CAD"
OUTPUT_DIR = r"outputs\Generated_ablation_data\04_STS2R"
IMAGES_DIR = r"assets\Textures"
SOLE_IMAGES_DIR = r"assets\Textures_sole"
LOGO_IMAGES_DIR = r"assets\Textures_logo"
N_VARIANTS = 50
MAX_IMAGE_EDGE = 768

def collect_image_paths(folder):
    out = []
    if not os.path.exists(folder):
        return out
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                out.append(os.path.join(root, fn))
    out.sort()
    return out

def main():
    print(f"🚀 Starting 04_Generate_STS2R_Sim pipeline...")
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} not found.")
        return

    # 清理并创建输出目录
    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 Cleaning up existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng()
    
    # 全局提前加载图片
    print(f"📦 Loading background textures from {IMAGES_DIR}...")
    img_paths = collect_image_paths(IMAGES_DIR)
    image_list = []
    if img_paths:
        for p in tqdm(img_paths, desc="Loading BG Images"):
            img = load_image_rgb255(p, MAX_IMAGE_EDGE)
            if img is not None:
                image_list.append(img)
    print(f"✅ Loaded {len(image_list)} background images.")

    print(f"📦 Loading sole textures from {SOLE_IMAGES_DIR}...")
    sole_img_paths = collect_image_paths(SOLE_IMAGES_DIR)
    sole_image_list = []
    if sole_img_paths:
        for p in tqdm(sole_img_paths, desc="Loading Sole Images"):
            img = load_image_rgb255(p, MAX_IMAGE_EDGE)
            if img is not None:
                sole_image_list.append(img)
    print(f"✅ Loaded {len(sole_image_list)} sole images.")

    print(f"📦 Loading logo textures from {LOGO_IMAGES_DIR}...")
    logo_img_paths = collect_image_paths(LOGO_IMAGES_DIR)
    logo_image_list = []
    if logo_img_paths:
        for p in tqdm(logo_img_paths, desc="Loading Logo Images"):
            img = load_image_rgba255(p, MAX_IMAGE_EDGE)
            if img is not None:
                logo_image_list.append(img)
    print(f"✅ Loaded {len(logo_image_list)} logo images.")

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
                # 阻断内存污染，确保每个变体都在干净的原数据上操作
                aug_data = np.copy(seed_data)
                
                lam = float(rng.uniform(0.1, 1.0))
                
                # 强度解耦采样，并随机化目标鞋码
                target_eu_size = rng.integers(33, 49)  # 随机选择一个目标鞋码 (范围是 [33, 49)，即 33 到 48)
                aug_data = shape_augment(aug_data, rng, float(rng.uniform(0.1, 1.0)), eu_size=target_eu_size)
                aug_data = line_augment(aug_data, rng, float(rng.uniform(0.1, 1.0)))
                if aug_data is None:
                    continue
                
                # 面片增强传入预加载的图片列表，新增 sole_image_list, logo_image_list
                aug_data = panel_augment(aug_data, rng, image_list=image_list, sole_image_list=sole_image_list, logo_image_list=logo_image_list, lam=lam)
                
                # 物理增强内部已处理去 lam 逻辑 (在重新贴图上色后执行，防止污渍被覆盖)
                aug_data = physical_augment(aug_data, rng, 1.0)
                
                aug_data = general_augment(aug_data, rng, float(rng.uniform(0.1, 1.0)))
                
                output_name = f"{base_name}_aug_{i}_l{lam:.3f}.txt"
                save_txt_pointcloud(os.path.join(OUTPUT_DIR, output_name), aug_data)
            except Exception as e:
                print(f"Error augmenting {filename} variant {i}: {e}")
                continue

    print(f"✅ Pipeline 04 finished. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
