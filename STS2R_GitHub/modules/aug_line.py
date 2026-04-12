import numpy as np
import scipy.spatial
from scipy.interpolate import CubicSpline

INPAINT_K = 10
INPAINT_RADIUS = 3.0
Z_CLEARANCE = 0.5


def _cartesian_to_polar_xy(coords, center):
    rel = coords[:, :2] - center[:2]
    r = np.linalg.norm(rel, axis=1)
    theta = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi)
    return r, theta


def _generate_spline_offsets(thetas, rng, anchor_count, z_offset):
    anchor_count = int(anchor_count)
    # 基础均匀切分
    base_anchors = np.linspace(0, 2 * np.pi, anchor_count + 1)
    step = 2 * np.pi / anchor_count

    # 为中间的锚点加入随机抖动 (Jitter)，最大范围不超过标准间距的 30%
    # 保持首尾 (0 和 2*pi) 严格对齐
    jitters = rng.uniform(-0.3 * step, 0.3 * step, size=anchor_count - 1)
    anchors_theta = base_anchors.copy()
    anchors_theta[1:-1] += jitters
    anchors_theta = np.sort(anchors_theta)  # 确保单调递增

    anchor_offsets_z = rng.uniform(-z_offset, z_offset, size=anchor_count).astype(np.float32)
    anchor_offsets_z = np.append(anchor_offsets_z, anchor_offsets_z[0])
    cs_z = CubicSpline(anchors_theta, anchor_offsets_z, bc_type="periodic")
    return cs_z(thetas).astype(np.float32)


def _build_index_set(list_of_lists):
    if list_of_lists is None:
        return np.array([], dtype=np.int64)
    if len(list_of_lists) == 0:
        return np.array([], dtype=np.int64)
    merged = []
    for item in list_of_lists:
        if item:
            merged.append(np.asarray(item, dtype=np.int64))
    if not merged:
        return np.array([], dtype=np.int64)
    conc = np.concatenate(merged)
    if len(conc) == 0:
        return np.array([], dtype=np.int64)
    return np.unique(conc)


def _inpaint_region(xyz, rgb, label, tree_full, trace_xyz, smear_indices, rng):
    dirty_indices = _build_index_set(tree_full.query_ball_point(trace_xyz, r=float(INPAINT_RADIUS)))
    clean_mask = (label == 0).copy()
    if len(dirty_indices) > 0:
        clean_mask[dirty_indices] = False
    source_xyz = xyz[clean_mask]
    source_rgb = rgb[clean_mask]
    if len(source_xyz) <= int(INPAINT_K):
        return rgb
    tree_clean = scipy.spatial.cKDTree(source_xyz)
    points_to_inpaint = xyz[smear_indices]
    _, neighbors = tree_clean.query(points_to_inpaint, k=int(INPAINT_K))
    neighbor_colors = source_rgb[neighbors]
    mean_colors = np.mean(neighbor_colors, axis=1)
    std_colors = np.std(neighbor_colors, axis=1)
    noise = rng.standard_normal(mean_colors.shape).astype(np.float32) * std_colors
    new_colors = np.clip(mean_colors + noise, 0, 255)
    rgb_out = rgb.copy()
    rgb_out[smear_indices] = new_colors
    return rgb_out


def apply(data, rng, lam, erase_original=False):
    xyz = data[:, :3].astype(np.float32, copy=False)
    rgb = data[:, 3:6].astype(np.float32, copy=False)
    label = data[:, 6].astype(np.int32, copy=False)

    trace_mask = label == 1
    trace_indices = np.where(trace_mask)[0]
    if len(trace_indices) < 10:
        return None

    trace_xyz = xyz[trace_indices]
    tree_full = scipy.spatial.cKDTree(xyz)

    lam = float(lam)
    smear_radius = 1.5 + 0.5 * lam
    neigh_lists = tree_full.query_ball_point(trace_xyz, r=float(smear_radius))
    smear_set = set(trace_indices.tolist())
    for t_i, neigh in enumerate(neigh_lists):
        if not neigh:
            continue
        neigh_arr = np.asarray(neigh, dtype=np.int64)
        dz = np.abs(xyz[neigh_arr, 2] - trace_xyz[t_i, 2])
        ok = neigh_arr[dz <= float(Z_CLEARANCE)]
        if len(ok) > 0:
            smear_set.update(ok.tolist())
    smear_indices = np.array(sorted(smear_set), dtype=np.int64)

    label_clean = label.copy()
    if erase_original:
        label_clean[:] = 0
    else:
        label_clean[smear_indices] = 0

    rgb_clean = _inpaint_region(xyz, rgb, label, tree_full, trace_xyz, smear_indices, rng)

    # 保持原有的形变随机幅度，保证数据多样性
    anchor_count = int(8 + np.floor(12 * lam))
    z_offset = 2.0 + 8.0 * lam

    trace_centroid = np.mean(trace_xyz, axis=0)
    _, theta = _cartesian_to_polar_xy(trace_xyz, trace_centroid)
    offsets_z = _generate_spline_offsets(theta, rng, anchor_count, z_offset)

    # 任务二：在 X 轴边缘加入空间衰减遮罩 (Attenuation Mask)
    x_coords = trace_xyz[:, 0]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    x_span = x_max - x_min
    if x_span > 0:
        # 0% -> 10%: 0.0 -> 1.0; 90% -> 100%: 1.0 -> 0.0
        mask = np.ones_like(x_coords)
        left_bound = x_min + 0.10 * x_span
        right_bound = x_max - 0.10 * x_span
        
        # 使用 np.where 实现分段线性权重
        mask = np.where(x_coords < left_bound, (x_coords - x_min) / (0.10 * x_span), mask)
        mask = np.where(x_coords > right_bound, (x_max - x_coords) / (0.10 * x_span), mask)
        mask = np.clip(mask, 0.0, 1.0)
        offsets_z *= mask

    virtual_xyz = trace_xyz.copy()
    virtual_xyz[:, 2] = virtual_xyz[:, 2] + offsets_z

    # 【核心修复】：移除糟糕的 proj_radius 范围搜索和 argmax(dists) 逻辑。
    # 直接使用最近邻 (k=1) 将平滑的 virtual_xyz 吸附到真实点云表面。
    # 这彻底解决了线发散（锯齿化）和错误吸附到内壁的问题。
    _, nearest_indices = tree_full.query(virtual_xyz, k=1)
    new_trace_indices = np.unique(np.asarray(nearest_indices, dtype=np.int64))

    trace_color = np.mean(rgb[trace_mask], axis=0)
    label_out = label_clean
    rgb_out = rgb_clean
    label_out[new_trace_indices] = 1
    rgb_out[new_trace_indices] = trace_color

    out = data.astype(np.float32, copy=True)
    out[:, 3:6] = rgb_out
    out[:, 6] = label_out.astype(np.float32)
    return out