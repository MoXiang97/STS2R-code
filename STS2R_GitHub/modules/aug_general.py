import numpy as np

TRACE_DROP_PROB = 0.05
HOLLOW_TRIGGER_PROB = 0.20
KEEP_RATIO_STD = 0.05


def apply(data, rng, lam):
    lam = float(lam)
    t = float(np.clip((lam - 0.1) / 0.9, 0.0, 1.0))
    base_keep = 0.95 - 0.60 * t
    keep_ratio = float(np.clip(rng.normal(base_keep, KEEP_RATIO_STD), 0.35, 1.0))

    label = data[:, 6].astype(np.int32, copy=False)
    trace_mask = label == 1
    bg_mask = label == 0
    other_mask = ~(trace_mask | bg_mask)

    rv = rng.random(len(data))
    keep_density = trace_mask | other_mask | (bg_mask & (rv < keep_ratio))
    data_out = data[keep_density]

    if data_out is None or len(data_out) == 0:
        return data_out
    if float(rng.random()) >= float(HOLLOW_TRIGGER_PROB):
        return data_out

    xyz = data_out[:, :3].astype(np.float32, copy=False)
    lbl = data_out[:, 6].astype(np.int32, copy=False)
    bg_indices = np.where(lbl == 0)[0]
    if bg_indices.size == 0:
        return data_out

    n_main = int(rng.integers(1, 3))
    n_main = int(min(n_main, int(bg_indices.size)))
    main_centers = xyz[rng.choice(bg_indices, size=n_main, replace=False)]

    inside = np.zeros(len(data_out), dtype=bool)
    for c in main_centers:
        base_radius = float(rng.uniform(10.0, 25.0))
        n_sub = int(rng.integers(3, 7))

        offsets = rng.normal(0.0, base_radius * 0.4, size=(n_sub, 3)).astype(np.float32)
        sub_centers = c.reshape(1, 3).astype(np.float32) + offsets
        scales = rng.uniform(0.6, 1.5, size=(n_sub, 3)).astype(np.float32)
        sub_radii = (base_radius * rng.uniform(0.4, 0.8, size=(n_sub,))).astype(np.float32)

        for j in range(n_sub):
            d = (xyz - sub_centers[j]) / scales[j]
            inside |= (np.sum(d * d, axis=1) < float(sub_radii[j] * sub_radii[j]))

    if not np.any(inside):
        return data_out

    keep = np.ones(len(data_out), dtype=bool)
    keep[inside & (lbl == 0)] = False
    trace_inside = inside & (lbl == 1)
    if np.any(trace_inside):
        idxs = np.where(trace_inside)[0]
        keep_trace = rng.random(int(idxs.size)) >= float(TRACE_DROP_PROB)
        keep[idxs] &= keep_trace

    return data_out[keep]
