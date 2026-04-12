import numpy as np

# ==========================================
# 全局形变参数配置 (已放宽截断限制，确保形变可见)
# ==========================================
W_L_MIN = 0.30
W_L_MAX = 0.44
H_L_MIN = 0.16
H_L_MAX = 0.38
SIZE_SCALE_MIN = 0.95  
SIZE_SCALE_MAX = 1.30  
EU_MIN_SIZE = 33
EU_MAX_SIZE = 48
EU_LENGTH_MM = {
    33: 212.0, 34: 219.0, 35: 223.0, 36: 230.0, 37: 237.0, 38: 243.0,
    39: 250.0, 40: 257.0, 41: 263.0, 42: 270.0, 43: 277.0, 44: 283.0,
    45: 290.0, 46: 297.0, 47: 303.0, 48: 310.0
}
RATIO_JITTER = 0.005
ALLOW_MM = 12.0
RATIO_STEP_LIMIT_W = 0.05  
RATIO_STEP_LIMIT_H = 0.03  
GLOBAL_Y_MAX_SCALE_DELTA = 0.10 
GLOBAL_Z_MAX_SCALE_DELTA = 0.12 
MICRO_DELTA_MAX_Y = 0.12
MICRO_DELTA_MAX_Z = 0.10
VISUAL_WIDENING_FACTOR = 0.25


def _parametric_shoe_deformation(points, params):
    """底层的分区形变应用逻辑"""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    xmin = np.min(x)
    xmax = np.max(x)
    x_norm = (x - xmin) / (xmax - xmin + 1e-8)
    
    # 分区掩膜计算
    mu = np.array([0.10, 0.60, 0.95], dtype=np.float32)
    sigma = np.array([0.08, 0.08, 0.05], dtype=np.float32)
    masks = [np.exp(-0.5 * ((x_norm - mu[i]) / (sigma[i] + 1e-8)) ** 2) for i in range(3)]
    masks = np.stack(masks, axis=1)
    denom = np.sum(masks, axis=1, keepdims=True) + 1e-8
    weights = masks / denom
    
    scale_x = float(params.get("scale_x", 1.0))
    global_y = float(params.get("global_y", 1.0))
    global_z = float(params.get("global_z", 1.0))
    delta_y = np.array(params.get("delta_y", np.zeros(3, dtype=np.float32)), dtype=np.float32)
    delta_z = np.array(params.get("delta_z", np.zeros(3, dtype=np.float32)), dtype=np.float32)
    
    # 计算切片中心
    bins = 64
    edges = np.linspace(xmin, xmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_cent_vals = np.zeros(bins, dtype=np.float32)
    z_cent_vals = np.zeros(bins, dtype=np.float32)
    y_mean = float(np.mean(y))
    z_mean = float(np.mean(z))
    
    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i+1])
        if np.any(m):
            y_cent_vals[i] = np.mean(y[m])
            z_cent_vals[i] = np.mean(z[m])
        else:
            y_cent_vals[i] = y_mean
            z_cent_vals[i] = z_mean
            
    # 平滑中心线
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    for _ in range(3):
        y_cent_vals = np.convolve(y_cent_vals, k, mode='same')
        z_cent_vals = np.convolve(z_cent_vals, k, mode='same')
    y0_x = np.interp(x, centers, y_cent_vals)
    z0_x = np.interp(x, centers, z_cent_vals)
    
    # 局部缩放场
    micro_y = np.sum(weights * delta_y[None, :], axis=1)
    micro_z = np.sum(weights * delta_z[None, :], axis=1)
    scale_y = global_y * (1.0 + micro_y)
    scale_z = global_z * (1.0 + micro_z)
    
    scale_y = np.clip(scale_y, global_y * (1.0 - MICRO_DELTA_MAX_Y), global_y * (1.0 + MICRO_DELTA_MAX_Y))
    scale_z = np.clip(scale_z, global_z * (1.0 - MICRO_DELTA_MAX_Z), global_z * (1.0 + MICRO_DELTA_MAX_Z))
    
    y_scale_bins = np.zeros(bins, dtype=np.float32)
    z_scale_bins = np.zeros(bins, dtype=np.float32)
    y_scale_mean = float(np.mean(scale_y))
    z_scale_mean = float(np.mean(scale_z))
    
    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i+1])
        if np.any(m):
            y_scale_bins[i] = np.mean(scale_y[m])
            z_scale_bins[i] = np.mean(scale_z[m])
        else:
            y_scale_bins[i] = y_scale_mean
            z_scale_bins[i] = z_scale_mean
            
    for _ in range(2):
        y_scale_bins = np.convolve(y_scale_bins, k, mode='same')
        z_scale_bins = np.convolve(z_scale_bins, k, mode='same')
    scale_y = np.interp(x, centers, y_scale_bins)
    scale_z = np.interp(x, centers, z_scale_bins)
    
    # 执行初级坐标映射
    x_new = xmin + (x - xmin) * scale_x
    y_new = y0_x + (y - y0_x) * scale_y
    z_new = z0_x + (z - z0_x) * scale_z
    
    # 全局比例最后强制微调（使用放宽的裁剪参数，允许 0.7 到 1.3）
    tgt_wr = float(params.get("target_w_ratio", 0.0))
    tgt_hr = float(params.get("target_h_ratio", 0.0))
    if tgt_wr > 0.0:
        l_after = (np.max(x_new) - np.min(x_new)) + 1e-8
        w_after = (np.max(y_new) - np.min(y_new)) + 1e-8
        corr_y = (tgt_wr * l_after) / w_after
        corr_y = np.clip(corr_y, 0.70, 1.30)
        y_new = y0_x + (y_new - y0_x) * corr_y
    if tgt_hr > 0.0:
        l_after = (np.max(x_new) - np.min(x_new)) + 1e-8
        h_after = (np.max(z_new) - np.min(z_new)) + 1e-8
        corr_z = (tgt_hr * l_after) / h_after
        corr_z = np.clip(corr_z, 0.70, 1.30)
        z_new = z0_x + (z_new - z0_x) * corr_z
        
    out = points.copy()
    out[:, 0] = x_new
    out[:, 1] = y_new
    out[:, 2] = z_new
    return out


def _sample_params(points, rng, lam, eu_size):
    """计算形变所需的缩放比例等参数字典"""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    L = np.max(x) - np.min(x) + 1e-8
    W = np.max(y) - np.min(y) + 1e-8
    H = np.max(z) - np.min(z) + 1e-8
    
    orig_wr = W / (L + 1e-8)
    orig_hr = H / (L + 1e-8)
    
    # 单位自动修正（如果长度小于5米，视为单位为米，转为毫米匹配鞋码表）
    L_mm = L * 1000.0 if L < 5.0 else L
    base_size = min(EU_LENGTH_MM.keys(), key=lambda s: abs(EU_LENGTH_MM[s] - L_mm))
    
    # 如果没有指定 eu_size，随机选一个目标码数
    if eu_size is None:
        delta = int(rng.integers(-3, 4))
        eu_size = np.clip(base_size + delta, EU_MIN_SIZE, EU_MAX_SIZE)
        
    target_len = EU_LENGTH_MM.get(eu_size, float(EU_LENGTH_MM[base_size]))
    s_x_full = float((target_len + ALLOW_MM) / (L_mm + 1e-8))
    s_x_full = float(np.clip(s_x_full, SIZE_SCALE_MIN, SIZE_SCALE_MAX))
    
    alpha = (eu_size - EU_MIN_SIZE) / float(EU_MAX_SIZE - EU_MIN_SIZE)
    
    # 这里我们把 lam（0到1之间）作为形变程度的权重系数
    lam = float(np.clip(lam, 0.0, 1.0)) if lam is not None else 1.0
    
    base_wr = np.clip(W_L_MIN + (W_L_MAX - W_L_MIN) * alpha + rng.normal(0.0, RATIO_JITTER), W_L_MIN, W_L_MAX)
    base_hr = np.clip(H_L_MIN + (H_L_MAX - H_L_MIN) * (0.4 + 0.4 * alpha) + rng.normal(0.0, RATIO_JITTER), H_L_MIN, H_L_MAX)
    
    w_boost = VISUAL_WIDENING_FACTOR * max(0.0, s_x_full - 1.0)
    base_wr = float(np.clip(base_wr + w_boost, W_L_MIN, W_L_MAX))
    
    target_w_ratio_full = orig_wr + np.clip(base_wr - orig_wr, -RATIO_STEP_LIMIT_W, RATIO_STEP_LIMIT_W)
    target_h_ratio_full = orig_hr + np.clip(base_hr - orig_hr, -RATIO_STEP_LIMIT_H, RATIO_STEP_LIMIT_H)
    
    s_needed_y_full = (target_w_ratio_full * L * s_x_full) / W
    s_needed_z_full = (target_h_ratio_full * L * s_x_full) / H
    s_needed_y_full = float(np.clip(s_needed_y_full, 1.0 - GLOBAL_Y_MAX_SCALE_DELTA, 1.0 + GLOBAL_Y_MAX_SCALE_DELTA))
    s_needed_z_full = float(np.clip(s_needed_z_full, 1.0 - GLOBAL_Z_MAX_SCALE_DELTA, 1.0 + GLOBAL_Z_MAX_SCALE_DELTA))
    
    def d_y(): return rng.uniform(-MICRO_DELTA_MAX_Y, MICRO_DELTA_MAX_Y)
    def d_z(): return rng.uniform(-MICRO_DELTA_MAX_Z, MICRO_DELTA_MAX_Z)
    
    pat = int(rng.integers(0, 6))
    if pat == 0:
        dy_full = np.array([d_y(), d_y(), abs(d_y())], dtype=np.float32)
        dz_full = np.array([d_z(), d_z(), d_z()], dtype=np.float32)
    elif pat == 1:
        dy_full = np.array([d_y(), d_y(), d_y()], dtype=np.float32)
        dz_full = np.array([d_z(), abs(d_z()), d_z()], dtype=np.float32)
    elif pat == 2:
        dy_full = np.array([d_y(), d_y(), d_y()], dtype=np.float32)
        dz_full = np.array([abs(d_z()), d_z(), d_z()], dtype=np.float32)
    elif pat == 3:
        dy_full = np.array([abs(d_y()), -abs(d_y()), abs(d_y())], dtype=np.float32)
        dz_full = np.array([d_z(), d_z(), d_z()], dtype=np.float32)
    elif pat == 4:
        dy_full = np.array([d_y(), -abs(d_y()), d_y()], dtype=np.float32)
        dz_full = np.array([d_z(), d_z(), d_z()], dtype=np.float32)
    else:
        dy_full = np.array([d_y(), d_y(), d_y()], dtype=np.float32)
        dz_full = np.array([d_z(), d_z(), d_z()], dtype=np.float32)
        
    # 应用 lam 插值控制强度
    scale_x = 1.0 + lam * (s_x_full - 1.0)
    global_y = 1.0 + lam * (s_needed_y_full - 1.0)
    global_z = 1.0 + lam * (s_needed_z_full - 1.0)
    delta_y = dy_full * lam
    delta_z = dz_full * lam
    target_w_ratio = orig_wr + lam * (target_w_ratio_full - orig_wr)
    target_h_ratio = orig_hr + lam * (target_h_ratio_full - orig_hr)

    return {
        "scale_x": float(scale_x), 
        "global_y": float(global_y), 
        "global_z": float(global_z), 
        "delta_y": delta_y, 
        "delta_z": delta_z, 
        "target_w_ratio": float(target_w_ratio), 
        "target_h_ratio": float(target_h_ratio)
    }

def apply(data, rng, lam=1.0, eu_size=None):
    """
    模块对外的统一调用接口
    :param data: NxC 的 numpy array，前三列必须是 xyz
    :param rng: numpy 随机数生成器
    :param lam: 0~1的系数，控制增强的强度
    :param eu_size: 目标尺码，如果不传则自动随机浮动
    :return: 变形后的数据
    """
    # 确保不修改原始数据地址
    out = data.astype(np.float32, copy=True)
    xyz = out[:, :3]
    
    # 1. 计算形变参数
    params = _sample_params(xyz, rng, lam, eu_size)
    
    # 2. 执行空间变形并写回
    out[:, :3] = _parametric_shoe_deformation(xyz, params)
    
    return out