import numpy as np


def apply(data, rng, lam):
    lam = float(lam)
    t = (lam - 0.1) / 0.9
    t = float(np.clip(t, 0.0, 1.0))
    keep_ratio = 0.95 - 0.60 * t
    label = data[:, 6].astype(np.int32, copy=False)
    trace_mask = label == 1
    bg_mask = label == 0
    rv = rng.random(len(data))
    keep = trace_mask.copy()
    keep |= bg_mask & (rv < keep_ratio)
    return data[keep]
