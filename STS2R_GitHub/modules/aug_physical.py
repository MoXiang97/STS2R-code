import numpy as np
from scipy.spatial import cKDTree


def _apply_spatial_smudge_noise(xyz, rgb_01, rng, num_patches=(10, 50)):
    """
    高级空间连续做旧：模拟真实的污渍、按痕、褪色斑块
    半径在 0.2mm 到 2.0mm 之间，颜色呈高斯平滑衰减，绝不产生雪花点。
    """
    tree = cKDTree(xyz)
    out_rgb = rgb_01.copy()
    
    n_patches = rng.integers(num_patches[0], num_patches[1])
    centers_idx = rng.choice(len(xyz), size=min(n_patches, len(xyz)), replace=False)
    
    for idx in centers_idx:
        center_pt = xyz[idx]
        radius = float(rng.uniform(0.2, 2.0))
        
        neighbors = tree.query_ball_point(center_pt, r=radius)
        if len(neighbors) == 0:
            continue
            
        neighbors = np.array(neighbors)
        dists = np.linalg.norm(xyz[neighbors] - center_pt, axis=1)
        
        sigma = radius / 3.0
        weights = np.exp(- (dists**2) / (2 * sigma**2 + 1e-8)).reshape(-1, 1)
        
        darken_factor = rng.uniform(0.6, 0.9)  
        dust_shift = rng.uniform(-0.05, 0.05, size=3)
        
        original_color = out_rgb[neighbors]
        target_color = np.clip(original_color * darken_factor + dust_shift, 0.0, 1.0)
        
        out_rgb[neighbors] = original_color * (1.0 - weights) + target_color * weights
        
    return out_rgb


def _cartesian_to_polar_xy(coords, center):
    rel = coords[:, :2] - center[:2]
    r = np.linalg.norm(rel, axis=1)
    theta = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi)
    return r, theta


def apply(data, rng, lam):
    # 虽然 lam 还在 signature 中，但后续严重破坏型增强将不再受其线性支配，且触发比例控制在 20%
    xyz = data[:, :3].astype(np.float32, copy=False)
    label = data[:, 6].astype(np.int32, copy=False)
    keep_mask = np.ones(len(data), dtype=bool)

    if len(xyz) == 0:
        return data

    centroid = np.mean(xyz, axis=0)
    r, theta = _cartesian_to_polar_xy(xyz, centroid)
    mean_r = float(np.mean(r))

    # 1. 极坐标扇形阴影缺失 
    if rng.random() < 0.05:
        num_shadows = rng.integers(1, 3)
        width = np.deg2rad(rng.uniform(15.0, 30.0))
        bg_keep_shadow = rng.uniform(0.20, 0.40)
        trace_keep_shadow = float(np.clip(bg_keep_shadow * 2.0, 0.35, 1.0))

        for _ in range(num_shadows):
            start_angle = float(rng.uniform(0.0, 2 * np.pi))
            end_angle = float((start_angle + width) % (2 * np.pi))
            if start_angle < end_angle:
                sector_mask = (theta >= start_angle) & (theta <= end_angle)
            else:
                sector_mask = (theta >= start_angle) | (theta <= end_angle)
            sector_mask &= r > mean_r * 0.6
            idxs = np.where(sector_mask & keep_mask)[0]
            if len(idxs) == 0:
                continue
            rv = rng.random(len(idxs))
            lbls = label[idxs]
            drop = np.zeros(len(idxs), dtype=bool)
            drop[lbls == 0] = rv[lbls == 0] > bg_keep_shadow
            drop[lbls == 1] = rv[lbls == 1] > trace_keep_shadow
            keep_mask[idxs[drop]] = False

    # 2. KDTree 随机斑块缺失 
    if rng.random() < 0.1:
        tree = cKDTree(xyz)
        num_patches = rng.integers(2, 5)
        valid_indices = np.where(keep_mask)[0]
        if len(valid_indices) > 0:
            patch_centers = rng.choice(valid_indices, size=min(num_patches, len(valid_indices)), replace=False)
            for center_idx in patch_centers:
                center_point = xyz[int(center_idx)]
                radius = float(rng.uniform(10.0, 20.0))
                patch_idxs = np.asarray(tree.query_ball_point(center_point, r=radius), dtype=np.int64)
                if len(patch_idxs) == 0:
                    continue
                patch_idxs = patch_idxs[keep_mask[patch_idxs]]
                if len(patch_idxs) == 0:
                    continue
                base_keep = float(rng.uniform(0.1, 0.3))
                keep_bg = base_keep
                keep_trace = float(np.clip(keep_bg * 2.0, 0.10, 1.0))
                rv = rng.random(len(patch_idxs))
                lbls = label[patch_idxs]
                drop = np.zeros(len(patch_idxs), dtype=bool)
                drop[lbls == 0] = rv[lbls == 0] > keep_bg
                drop[lbls == 1] = rv[lbls == 1] > keep_trace
                keep_mask[patch_idxs[drop]] = False

    # 3. 轴对齐扫描线缺失
    if rng.random() < 0.05:
        num_scanlines = rng.integers(1, 3)
        width_mm = rng.uniform(1.0, 3.0)
        for _ in range(num_scanlines):
            axis = int(rng.integers(0, 2))
            min_val = float(np.min(xyz[:, axis]))
            max_val = float(np.max(xyz[:, axis]))
            pos = float(rng.uniform(min_val, max_val))
            band_min = pos - width_mm / 2.0
            band_max = pos + width_mm / 2.0
            in_band = (xyz[:, axis] >= band_min) & (xyz[:, axis] <= band_max) & keep_mask
            idxs = np.where(in_band)[0]
            if len(idxs) == 0:
                continue
            rv = rng.random(len(idxs))
            lbls = label[idxs]
            drop = np.zeros(len(idxs), dtype=bool)
            drop[lbls == 0] = True
            trace_keep = float(rng.uniform(0.50, 0.80))
            drop[lbls == 1] = rv[lbls == 1] > trace_keep
            keep_mask[idxs[drop]] = False

    kept = data[keep_mask].astype(np.float32, copy=True)
    if len(kept) == 0:
        return kept

    # 4. 扫描仪仿真与物理污渍做旧 (Trigger 20%)
    if rng.random() < 0.2:
        # 坐标微小高斯抖动 (模拟扫描仪精度误差)
        jitter_std = rng.uniform(0.02, 0.05)
        kept[:, :3] = kept[:, :3] + rng.normal(0.0, jitter_std, size=kept[:, :3].shape).astype(np.float32)

        # 高级空间斑块做旧 (替换原有的全局独立 RGB 噪声)
        if kept.shape[1] >= 6:
            rgb_01 = np.clip(kept[:, 3:6] / 255.0, 0.0, 1.0)
            # 根据点云规模动态决定斑块数量
            patch_min = max(5, int(len(kept) / 10000))
            patch_max = max(15, int(len(kept) / 2000))
            
            new_rgb_01 = _apply_spatial_smudge_noise(kept[:, :3], rgb_01, rng, num_patches=(patch_min, patch_max))
            kept[:, 3:6] = np.clip(new_rgb_01 * 255.0, 0.0, 255.0)

    return kept
