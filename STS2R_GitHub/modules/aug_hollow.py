import numpy as np

TRACE_DROP_PROB = 0.05
HOLLOW_TRIGGER_PROB = 0.20  # 核心解耦：只有 20% 的概率会发生遮挡

def apply(data, rng, lam):
    # 1. 触发概率过滤（彻底解耦）
    if float(rng.random()) > HOLLOW_TRIGGER_PROB:
        return data  # 80% 的数据直接原样返回，没有任何洞！

    xyz = data[:, :3].astype(np.float32, copy=False)
    label = data[:, 6].astype(np.int32, copy=False)
    keep_mask = np.ones(len(data), dtype=bool)
    lam = float(lam)

    # 2. 约束挖洞位置：只在“非轨迹区（背景）”生成遮挡中心
    bg_indices = np.where(label == 0)[0]
    if len(bg_indices) == 0:
        return data

    # 决定有几块遮挡区域（通常手或夹具只有 1 到 2 处）
    num_blobs = int(rng.integers(1, 3)) 
    base_radius = 5.0 + 15.0 * lam  # lam 越大，整体遮挡范围越大

    # 3. 构造不规则边缘：融合斑块 (Metaballs/Cluster) 思路
    for _ in range(num_blobs):
        # 从背景点中随机挑一个点作为“主遮挡中心”
        main_center = xyz[rng.choice(bg_indices)]
        
        # 在主中心周围，生成 3 到 6 个随机偏移的“子中心”
        num_sub_centers = int(rng.integers(3, 7))
        # 子中心在主中心附近随机游走，游走范围与 base_radius 相关
        sub_offsets = rng.normal(0, base_radius * 0.4, size=(num_sub_centers, 3))
        sub_centers = main_center + sub_offsets

        # 将这几个重叠的小椭球融合在一起，就会形成像云朵、手掌一样极其不规则的 3D 边缘！
        for sc in sub_centers:
            # 每个子球都有随机的形变比例
            scales = rng.uniform(0.6, 1.5, size=3).astype(np.float32)
            # 子球的半径是主半径的一部分
            sub_radius = base_radius * float(rng.uniform(0.4, 0.8))

            distorted_pos = (xyz - sc) / scales
            dist_sq = np.sum(distorted_pos * distorted_pos, axis=1)
            in_sub_ellipsoid = dist_sq < (sub_radius * sub_radius)

            if not np.any(in_sub_ellipsoid):
                continue
                
            indices_in_hole = np.where(in_sub_ellipsoid)[0]
            labels_in_hole = label[indices_in_hole]
            rv = rng.random(len(indices_in_hole))

            # 遮挡逻辑：背景点 100% 遮挡，轨迹点仅有 5% 概率被波及
            drop_bg = labels_in_hole == 0
            drop_trace = (labels_in_hole == 1) & (rv < float(TRACE_DROP_PROB))
            
            keep_mask[indices_in_hole[drop_bg | drop_trace]] = False

    return data[keep_mask]