import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

PSEUDO_PANEL_BINS = 16

INDUSTRIAL_COLORS_RGB255 = np.array(
    [
        [255, 255, 255],
        [255, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 0],
    ],
    dtype=np.float32,
)

def _resize_image_rgb255(img_rgb255, max_edge):
    h, w = img_rgb255.shape[:2]
    if max(h, w) <= max_edge:
        return img_rgb255
    scale = float(max_edge) / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if cv2 is not None:
        return cv2.resize(img_rgb255, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    if Image is not None:
        pil = Image.fromarray(img_rgb255.astype(np.uint8), mode="RGB")
        pil = pil.resize((nw, nh), resample=Image.BILINEAR)
        return np.asarray(pil, dtype=np.float32)
    return img_rgb255

def _resize_image_rgba255(img_rgba255, max_edge):
    h, w = img_rgba255.shape[:2]
    if max(h, w) <= max_edge:
        return img_rgba255
    scale = float(max_edge) / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if cv2 is not None:
        return cv2.resize(img_rgba255, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    if Image is not None:
        pil = Image.fromarray(img_rgba255.astype(np.uint8), mode="RGBA")
        pil = pil.resize((nw, nh), resample=Image.BILINEAR)
        return np.asarray(pil, dtype=np.float32)
    return img_rgba255

def load_image_rgb255(path, max_edge=768):
    img = None
    if cv2 is not None:
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            if buf.size > 0:
                dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if dec is not None:
                    img = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)
        except Exception:
            img = None

    if img is None and Image is not None:
        try:
            img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        except Exception:
            img = None

    if img is None:
        return None
    return _resize_image_rgb255(img, max_edge)

def load_image_rgba255(path, max_edge=768):
    img = None
    if cv2 is not None:
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            if buf.size > 0:
                dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                if dec is not None:
                    if dec.ndim == 3 and dec.shape[2] == 4:
                        img = cv2.cvtColor(dec, cv2.COLOR_BGRA2RGBA).astype(np.float32)
                    elif dec.ndim == 3 and dec.shape[2] == 3:
                        rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)
                        a = np.full((rgb.shape[0], rgb.shape[1], 1), 255.0, dtype=np.float32)
                        img = np.concatenate([rgb, a], axis=2)
        except Exception:
            img = None

    if img is None and Image is not None:
        try:
            img = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32)
        except Exception:
            img = None

    if img is None:
        return None
    return _resize_image_rgba255(img, max_edge)

def _sample_image_nn_rgb01(img_rgb255, uv01):
    h, w = img_rgb255.shape[:2]
    u = np.clip(uv01[:, 0], 0.0, 1.0)
    v = np.clip(uv01[:, 1], 0.0, 1.0)
    xs = (u * (w - 1)).astype(np.int32)
    ys = (v * (h - 1)).astype(np.int32)
    return (img_rgb255[ys, xs, :3] / 255.0).astype(np.float32)

def _sample_image_nn_rgba01(img_rgba255, uv01):
    h, w = img_rgba255.shape[:2]
    u = np.clip(uv01[:, 0], 0.0, 1.0)
    v = np.clip(uv01[:, 1], 0.0, 1.0)
    xs = (u * (w - 1)).astype(np.int32)
    ys = (v * (h - 1)).astype(np.int32)
    if img_rgba255.ndim == 3 and img_rgba255.shape[2] >= 4:
        return (img_rgba255[ys, xs, :4] / 255.0).astype(np.float32)
    rgb01 = (img_rgba255[ys, xs, :3] / 255.0).astype(np.float32)
    a = np.ones((rgb01.shape[0], 1), dtype=np.float32)
    return np.concatenate([rgb01, a], axis=1)

def project_image_to_points(xyz, image_array, projection="auto"):
    pts = xyz.astype(np.float32, copy=False)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    nrm = (pts - mn) / span

    if projection == "triplanar":
        uv_xy = nrm[:, [0, 1]]
        uv_yz = nrm[:, [1, 2]]
        uv_zx = nrm[:, [2, 0]]
        c_xy = _sample_image_nn_rgb01(image_array, uv_xy)
        c_yz = _sample_image_nn_rgb01(image_array, uv_yz)
        c_zx = _sample_image_nn_rgb01(image_array, uv_zx)
        ex, ey, ez = span.astype(np.float32)
        w_xy, w_yz, w_zx = ez, ex, ey
        w_sum = max(w_xy + w_yz + w_zx, 1e-6)
        return (c_xy * w_xy + c_yz * w_yz + c_zx * w_zx) / w_sum

    ex, ey, ez = span.astype(np.float32)
    area_xy, area_yz, area_zx = ex * ey, ey * ez, ez * ex
    if projection == "xy" or (projection == "auto" and area_xy >= area_yz and area_xy >= area_zx):
        uv = nrm[:, [0, 1]]
    elif projection == "yz" or (projection == "auto" and area_yz >= area_xy and area_yz >= area_zx):
        uv = nrm[:, [1, 2]]
    else:
        uv = nrm[:, [2, 0]]
    return _sample_image_nn_rgb01(image_array, uv)

def project_image_to_points_rgba(xyz, image_array, projection="auto"):
    pts = xyz.astype(np.float32, copy=False)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    nrm = (pts - mn) / span

    if projection == "triplanar":
        uv_xy = nrm[:, [0, 1]]
        uv_yz = nrm[:, [1, 2]]
        uv_zx = nrm[:, [2, 0]]
        c_xy = _sample_image_nn_rgba01(image_array, uv_xy)
        c_yz = _sample_image_nn_rgba01(image_array, uv_yz)
        c_zx = _sample_image_nn_rgba01(image_array, uv_zx)
        ex, ey, ez = span.astype(np.float32)
        w_xy, w_yz, w_zx = ez, ex, ey
        w_sum = max(w_xy + w_yz + w_zx, 1e-6)
        return (c_xy * w_xy + c_yz * w_yz + c_zx * w_zx) / w_sum

    ex, ey, ez = span.astype(np.float32)
    area_xy, area_yz, area_zx = ex * ey, ey * ez, ez * ex
    if projection == "xy" or (projection == "auto" and area_xy >= area_yz and area_xy >= area_zx):
        uv = nrm[:, [0, 1]]
    elif projection == "yz" or (projection == "auto" and area_yz >= area_xy and area_yz >= area_zx):
        uv = nrm[:, [1, 2]]
    else:
        uv = nrm[:, [2, 0]]
    return _sample_image_nn_rgba01(image_array, uv)

def _build_pseudo_panel_model(xyz_bg, k):
    pts = xyz_bg.astype(np.float32, copy=False)
    if pts.shape[0] < 4:
        mu = pts.mean(axis=0, keepdims=True) if pts.shape[0] else np.zeros((1, 3), dtype=np.float32)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        edges = np.array([0.0, 1.0], dtype=np.float32)
        return mu.astype(np.float32), axis, edges
    mu = pts.mean(axis=0, keepdims=True)
    x = pts - mu
    cov = (x.T @ x) / max(1, pts.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov.astype(np.float64))
    axis = evecs[:, int(np.argmax(evals))].astype(np.float32)
    t = (x @ axis.reshape(3, 1)).reshape(-1).astype(np.float32, copy=False)
    qs = np.linspace(0.0, 1.0, int(k) + 1, dtype=np.float32)
    edges = np.quantile(t.astype(np.float64), qs).astype(np.float32)
    if np.unique(edges).size < edges.size:
        mn, mx = float(t.min()), float(t.max())
        edges = np.linspace(mn, mx, int(k) + 1, dtype=np.float32) if mx - mn > 1e-6 else np.linspace(0.0, 1.0, int(k) + 1, dtype=np.float32)
    return mu.astype(np.float32), axis, edges

def _apply_pseudo_panel_model(xyz, mu, axis, edges):
    pts = xyz.astype(np.float32, copy=False)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    x = pts - mu.astype(np.float32, copy=False)
    t = (x @ axis.reshape(3, 1)).reshape(-1).astype(np.float32, copy=False)
    if edges.size < 3:
        return np.ones((pts.shape[0],), dtype=np.int32)
    bins = np.searchsorted(edges[1:-1], t, side="right").astype(np.int32)
    return bins + 1

def _assign_textures_by_panel(rgb01, xyz, labels, panels, image_list, rng, panel_ids, projection):
    out = rgb01.copy()
    if not image_list:
        return out
    panel_ids = np.asarray(panel_ids, dtype=np.int32)
    if panel_ids.size == 0:
        return out
    
    if len(image_list) >= panel_ids.size:
        chosen = rng.choice(len(image_list), size=panel_ids.size, replace=False)
    else:
        chosen = rng.choice(len(image_list), size=panel_ids.size, replace=True)
        
    for pid, img_idx in zip(panel_ids.tolist(), chosen.tolist()):
        mask = (labels == 0) & (panels == int(pid))
        if not np.any(mask):
            continue
        img = image_list[int(img_idx)]
        out[mask] = project_image_to_points(xyz[mask], img, projection=projection)
    return out


def apply(data, rng, image_list=None, sole_image_list=None, logo_image_list=None, projection="auto", lam=None):
    if data.shape[1] < 8:
        return data
    aug = data.astype(np.float32, copy=True)
    xyz = aug[:, :3].astype(np.float32, copy=False)
    rgb01 = np.clip(aug[:, 3:6].astype(np.float32, copy=False) / 255.0, 0.0, 1.0)
    labels = aug[:, 6].astype(np.int32, copy=False)
    panels = aug[:, 7].astype(np.int32, copy=False)

    bg_mask = labels == 0
    trace_mask = labels == 1

    panel_work = panels.copy()
    if np.any(bg_mask):
        bg_panels_real = np.unique(panels[bg_mask])
        valid_panels = bg_panels_real[bg_panels_real != 0]
        
        if valid_panels.size == 0:
            # 原有的伪面板生成逻辑
            bg_count = int(np.sum(bg_mask))
            k = int(PSEUDO_PANEL_BINS)
            k = max(2, min(k, max(2, bg_count // 20000)))
            mu, axis, edges = _build_pseudo_panel_model(xyz[bg_mask], k)
            panel_work[bg_mask] = _apply_pseudo_panel_model(xyz[bg_mask], mu, axis, edges)
            if np.any(trace_mask):
                panel_work[trace_mask] = _apply_pseudo_panel_model(xyz[trace_mask], mu, axis, edges)
        else:
            # 【核心修复】：利用 KD-Tree 抹平旧边界的 Panel=0 幽灵
            from scipy.spatial import cKDTree
            invalid_mask = bg_mask & (panel_work == 0)
            valid_mask = bg_mask & (panel_work != 0)
            
            if np.any(invalid_mask) and np.any(valid_mask):
                # 找到无效点最近的有效点，将其 panel_id 同化
                tree = cKDTree(xyz[valid_mask])
                _, nearest_idx = tree.query(xyz[invalid_mask])
                panel_work[invalid_mask] = panel_work[valid_mask][nearest_idx]

    out_rgb = rgb01.copy()
    
    if np.any(bg_mask):
        bg_panels = np.unique(panel_work[bg_mask])
        bg_panels = bg_panels[bg_panels != 0]
        bg_panels_list = bg_panels.tolist()

        # 【新增】Z轴寻靶与中底布拦截
        bottom_panel_id = None
        if len(bg_panels_list) > 0:
            min_mean_z = float('inf')
            for pid in bg_panels_list:
                mask = bg_mask & (panel_work == pid)
                if np.any(mask):
                    mean_z = np.mean(xyz[mask, 2])
                    if mean_z < min_mean_z:
                        min_mean_z = mean_z
                        bottom_panel_id = pid
        
        # 100% 强制拦截 (鞋底贴图)
        if bottom_panel_id is not None and sole_image_list:
            sole_img = sole_image_list[rng.choice(len(sole_image_list))]
            mask_sole = bg_mask & (panel_work == bottom_panel_id)
            out_rgb[mask_sole] = project_image_to_points(xyz[mask_sole], sole_img, projection=projection)
            
            # 将 bottom_panel_id 从 bg_panels 列表中移除
            bg_panels_list.remove(bottom_panel_id)

        # 【重构】剩余背景的 4 种贴图模式
        if len(bg_panels_list) > 0:
            remaining_bg_mask = bg_mask & np.isin(panel_work, bg_panels_list)
            if np.any(remaining_bg_mask):
                mode = float(rng.random())
                
                # 模式 1 (30% 概率，mode < 0.30)：全局统一材质
                if mode < 0.30 and image_list:
                    img = image_list[rng.choice(len(image_list))]
                    out_rgb[remaining_bg_mask] = project_image_to_points(xyz[remaining_bg_mask], img, projection=projection)
                
                # 模式 2 (20% 概率，0.30 <= mode < 0.50)：两组拼色
                elif mode < 0.50 and image_list:
                    rng.shuffle(bg_panels_list)
                    split = max(1, len(bg_panels_list) // 2)
                    group_a = bg_panels_list[:split]
                    group_b = bg_panels_list[split:] if split < len(bg_panels_list) else group_a
                    
                    if len(image_list) >= 2:
                        i1, i2 = rng.choice(len(image_list), size=2, replace=False)
                        img_a, img_b = image_list[i1], image_list[i2]
                    else:
                        img_a = img_b = image_list[0]
                        
                    mask_a = remaining_bg_mask & np.isin(panel_work, group_a)
                    if np.any(mask_a):
                        out_rgb[mask_a] = project_image_to_points(xyz[mask_a], img_a, projection=projection)
                    mask_b = remaining_bg_mask & np.isin(panel_work, group_b)
                    if np.any(mask_b):
                        out_rgb[mask_b] = project_image_to_points(xyz[mask_b], img_b, projection=projection)
                
                # 模式 3 (30% 概率，0.50 <= mode < 0.80)：独立材质
                elif mode < 0.80 and image_list:
                    out_rgb = _assign_textures_by_panel(out_rgb, xyz, labels, panel_work, image_list, rng, bg_panels_list, projection)
                
                # 模式 4 (20% 概率，mode >= 0.80 或图库为空)：带波动的纯色背景
                else:
                    for pid in bg_panels_list:
                        m = remaining_bg_mask & (panel_work == pid)
                        if np.any(m):
                            if rng.random() < 0.5:
                                # 50% 黑/白灰度
                                if rng.random() < 0.5:
                                    # 黑底: 0.0 ~ 0.25 统一值
                                    val = float(rng.uniform(0.0, 0.25))
                                else:
                                    # 白底: 0.75 ~ 1.0 统一值
                                    val = float(rng.uniform(0.75, 1.0))
                                color = np.array([val, val, val], dtype=np.float32)
                            else:
                                # 50% 随机彩色: 0.1 ~ 0.9 范围
                                color = rng.uniform(0.1, 0.9, size=3).astype(np.float32)
                            out_rgb[m] = color

    # 【新增】局部 Logo 贴花（Logo Decal）等比例增强环节
    if logo_image_list and rng.random() < 0.5:
        # 找出有效区域：属于背景，且不属于被保护的中底布
        valid_logo_mask = bg_mask.copy()
        if bottom_panel_id is not None:
            valid_logo_mask = valid_logo_mask & (panel_work != bottom_panel_id)
            
        if np.any(valid_logo_mask):
            # 1. 随机选择一个中心点
            valid_indices = np.where(valid_logo_mask)[0]
            candidates = valid_indices
            if np.any(trace_mask):
                trace_z = xyz[trace_mask, 2]
                z_min = float(np.min(trace_z))
                z_max = float(np.max(trace_z))
                z_thr = z_max - 0.1 * (z_max - z_min)
                cand_mask = xyz[valid_indices, 2] >= z_thr
                if np.any(cand_mask):
                    candidates = valid_indices[cand_mask]
            center_idx = rng.choice(candidates)
            center_xyz = xyz[center_idx]
            
            # 2. 提前抽取 Logo 图片以获取真实长宽比
            logo_img = logo_image_list[rng.choice(len(logo_image_list))]
            img_h, img_w = logo_img.shape[:2]
            aspect_ratio = img_h / float(img_w) if img_w > 0 else 1.0
            
            # 3. 设定物理尺寸 (毫米)
            physical_w = float(rng.uniform(20.0, 60.0))  # 贴纸的物理宽度 20~60mm
            physical_h = physical_w * aspect_ratio       # 等比例计算物理高度
            
            # 4. 随机投影方向映射 (打乱XYZ轴来模拟不同角度的贴纸)
            axes = rng.permutation([0, 1, 2])
            
            # 计算所有有效点到中心点的各轴独立距离
            dist_w = np.abs(xyz[:, axes[0]] - center_xyz[axes[0]])
            dist_h = np.abs(xyz[:, axes[1]] - center_xyz[axes[1]])
            dist_d = np.abs(xyz[:, axes[2]] - center_xyz[axes[2]])
            
            # 5. 长方体包围盒截取 (宽度、高度范围内，且深度渗透不超过 30mm 确保贴在表面)
            local_mask = valid_logo_mask & (dist_w < physical_w / 2.0) & (dist_h < physical_h / 2.0) & (dist_d < 30.0)
            
            # 6. 执行投影贴图
            if np.sum(local_mask) > 50:
                if logo_img.ndim == 3 and logo_img.shape[2] == 3:
                    a = np.full((logo_img.shape[0], logo_img.shape[1], 1), 255.0, dtype=np.float32)
                    logo_img = np.concatenate([logo_img.astype(np.float32, copy=False), a], axis=2)
                logo_rgba = project_image_to_points_rgba(xyz[local_mask], logo_img, projection=projection)
                a = np.clip(logo_rgba[:, 3:4], 0.0, 1.0)
                a = a * float(rng.uniform(0.7, 1.0))
                logo_rgb = np.clip(logo_rgba[:, :3], 0.0, 1.0)
                bg_rgb = out_rgb[local_mask]
                out_rgb[local_mask] = logo_rgb * a + bg_rgb * (1.0 - a)

    # 目标区域（trace_mask）纯色实涂逻辑：寻找局部最大反差颜色
    if np.any(trace_mask):
        local_mean_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        best_rgb01 = None
        
        if np.any(bg_mask):
            from scipy.spatial import cKDTree
            trace_xyz = xyz[trace_mask]
            bg_xyz = xyz[bg_mask]
            
            tree_bg = cKDTree(bg_xyz)
            # 查找 3mm 范围内的邻居
            neighbors = tree_bg.query_ball_point(trace_xyz, r=3.0)
            
            # 扁平化并去重所有邻居索引
            local_bg_indices = np.unique(np.concatenate([n for n in neighbors if len(n) > 0] + [np.array([], dtype=int)]))
            
            if local_bg_indices.size > 0:
                local_bg_indices = local_bg_indices.astype(int)
                bg_xyz_local = bg_xyz[local_bg_indices]
                bg_colors = out_rgb[bg_mask][local_bg_indices]
                local_mean_rgb = np.median(bg_colors, axis=0)
                n_local = int(bg_xyz_local.shape[0])
                if n_local >= 8:
                    tree_local = cKDTree(bg_xyz_local)
                    k = min(16, n_local)
                    d_knn, i_knn = tree_local.query(bg_xyz_local, k=k)
                    if k == 1:
                        d_knn = d_knn[:, None]
                        i_knn = i_knn[:, None]
                    labels_local = -np.ones(n_local, dtype=np.int32)
                    clusters = []
                    spatial_r = 5.0
                    color_thr = 0.12
                    cid = 0
                    for seed in range(n_local):
                        if labels_local[seed] != -1:
                            continue
                        stack = [seed]
                        labels_local[seed] = cid
                        members = []
                        while stack:
                            p = stack.pop()
                            members.append(p)
                            p_rgb = bg_colors[p]
                            neigh = i_knn[p]
                            neigh_d = d_knn[p]
                            for q, dq in zip(neigh, neigh_d):
                                if dq > spatial_r:
                                    continue
                                if labels_local[q] != -1:
                                    continue
                                if np.linalg.norm(bg_colors[q] - p_rgb) > color_thr:
                                    continue
                                labels_local[q] = cid
                                stack.append(int(q))
                        clusters.append(np.asarray(members, dtype=np.int32))
                        cid += 1
                    if clusters:
                        sizes = np.array([c.size for c in clusters], dtype=np.int32)
                        min_size = max(20, int(round(0.01 * float(n_local))))
                        kept = [clusters[i] for i in range(len(clusters)) if sizes[i] >= min_size]
                        if not kept:
                            kept = [clusters[int(np.argmax(sizes))]]
                        reps = []
                        for c in kept:
                            med = np.median(bg_colors[c], axis=0)
                            j = int(np.argmin(np.sum((bg_colors[c] - med) ** 2, axis=1)))
                            reps.append(bg_colors[c[j]])
                        if reps:
                            reps = np.stack(reps, axis=0).astype(np.float32)
                            industrial_colors_01 = INDUSTRIAL_COLORS_RGB255 / 255.0
                            dist_mat = np.linalg.norm(industrial_colors_01[:, None, :] - reps[None, :, :], axis=2)
                            scores = np.min(dist_mat, axis=1)
                            best_idx = int(np.argmax(scores))
                            best_rgb01 = industrial_colors_01[best_idx]
        
        if best_rgb01 is None:
            industrial_colors_01 = INDUSTRIAL_COLORS_RGB255 / 255.0
            dists = np.linalg.norm(industrial_colors_01 - local_mean_rgb, axis=1)
            best_idx = int(np.argmax(dists))
            best_rgb01 = industrial_colors_01[best_idx]
        
        # 1. 模拟老化/批次色偏 (在基准色上加一个全局随机偏移)
        aging_offset = rng.uniform(-0.05, 0.05, size=3).astype(np.float32)
        base_color = np.clip(best_rgb01 + aging_offset, 0.0, 1.0)
        
        # 2. 模拟墨水厚度/覆盖度 (透明度 Alpha: 0.7 ~ 1.0)
        alpha_base = float(rng.uniform(0.7, 1.0))
        
        # 3. 空间平滑波动 (方案B：模拟“成片波动”的斑块感)
        trace_count = int(np.sum(trace_mask))
        trace_xyz = xyz[trace_mask]
        
        # 生成原始高频噪点
        raw_noise = rng.normal(0.0, 1.0, size=(trace_count,)).astype(np.float32)
        
        # 利用 cKDTree 进行空间平滑 (平滑半径 10mm)
        from scipy.spatial import cKDTree
        tree_trace = cKDTree(trace_xyz)
        # 查找 10mm 范围内的邻居点进行均值滤波
        neighbors_noise = tree_trace.query_ball_point(trace_xyz, r=10.0)
        
        smooth_noise = np.zeros(trace_count, dtype=np.float32)
        for i, indices in enumerate(neighbors_noise):
            if len(indices) > 0:
                smooth_noise[i] = np.mean(raw_noise[indices])
        
        # 将平滑后的噪声归一化并映射到 alpha 波动 (波动范围 ±0.15)
        if np.std(smooth_noise) > 1e-5:
            smooth_noise = (smooth_noise - np.mean(smooth_noise)) / np.std(smooth_noise)
        alpha_noise = smooth_noise * 0.15
        
        # 最终 Alpha 混合系数 (基础值 + 空间波动)
        alpha_final = np.clip(alpha_base + alpha_noise, 0.0, 1.0)[:, np.newaxis]
        
        # 4. 逐点微观波动 (保留一点细微颗粒感: 标准差从 0.08 降到 0.03)
        ink_rough = rng.normal(0.0, 0.03, size=(trace_count, 3)).astype(np.float32)
        
        # 5. 执行混合 (模拟墨水与背景的融合)
        ink_pixels = np.clip(base_color + ink_rough, 0.0, 1.0)
        bg_pixels = out_rgb[trace_mask]
        out_rgb[trace_mask] = ink_pixels * alpha_final + bg_pixels * (1.0 - alpha_final)

    aug[:, 3:6] = np.clip(out_rgb * 255.0, 0.0, 255.0)
    return aug
