import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def general_augment_real(data, rng):
    """
    100% 触发。
    对点云的 XYZ 坐标进行全局等比例缩放，缩放因子在 0.8 到 1.2 之间。
    严禁旋转。
    """
    scale = float(rng.uniform(0.8, 1.2))
    out = data.copy()
    out[:, :3] *= scale
    return out

def color_augment_real(data, rng):
    """
    50% 独立触发。
    全局环境光偏移：R、G、B 分别在 [-30, +30] 之间随机偏移。
    HSV 抖动：亮度 (V) 和饱和度 (S) 在 0.8 到 1.2 倍随机缩放。
    最后将结果转换回 RGB，并严格截断（Clip）在 [0, 255] 范围内。
    """
    if rng.random() >= 0.5:
        return data
    
    out = data.copy()
    rgb = out[:, 3:6].astype(np.float32)
    
    # 1. 全局环境光偏移 (Ambient Shift)
    rgb_offset = rng.integers(-30, 31, size=3).astype(np.float32)
    rgb += rgb_offset
    
    # 2. HSV 抖动
    rgb_01 = np.clip(rgb / 255.0, 0.0, 1.0)
    hsv = rgb_to_hsv(rgb_01)
    
    # S 和 V 缩放 (0.8 - 1.2)
    s_scale = float(rng.uniform(0.8, 1.2))
    v_scale = float(rng.uniform(0.8, 1.2))
    hsv[:, 1] = np.clip(hsv[:, 1] * s_scale, 0.0, 1.0)
    hsv[:, 2] = np.clip(hsv[:, 2] * v_scale, 0.0, 1.0)
    
    rgb_back = hsv_to_rgb(hsv) * 255.0
    out[:, 3:6] = np.clip(rgb_back, 0.0, 255.0).astype(np.float32)
    
    return out

def physical_augment_real(data, rng):
    """
    30% 独立触发。
    绝对禁止强破坏操作。
    1. 全局随机丢点 (Uniform Dropout)：随机丢弃 1% 到 3% 的点。
    2. 极微小高斯噪声：std 在 0.01 到 0.03 之间，截断在 ±3σ。
    """
    if rng.random() >= 0.3:
        return data
    
    # 1. 全局随机丢点 (Uniform Dropout)
    drop_rate = float(rng.uniform(0.01, 0.03))
    num_points = len(data)
    keep_mask = rng.random(num_points) >= drop_rate
    out = data[keep_mask].copy()
    
    # 2. 极微小高斯噪声
    if len(out) > 0:
        std = float(rng.uniform(0.01, 0.03))
        noise = rng.normal(0.0, std, size=(len(out), 3))
        # 截断在 ±3σ，防止出现极端飞点
        noise = np.clip(noise, -3*std, 3*std)
        out[:, :3] += noise.astype(np.float32)
    
    return out
