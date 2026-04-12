import os
import numpy as np


def load_txt_pointcloud(path):
    try:
        data = np.loadtxt(path, dtype=np.float32)
    except Exception:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        return None
    if data.shape[1] > 8:
        data = data[:, :8]
    return data


def save_txt_pointcloud(path, data):
    if data is None or len(data) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = data[:, :3].astype(np.float32, copy=False)
    rgb = np.clip(data[:, 3:6], 0, 255).astype(np.int32, copy=False)
    lbl = data[:, 6].astype(np.int32, copy=False).reshape(-1, 1)
    if data.shape[1] >= 8:
        pnl = data[:, 7].astype(np.int32, copy=False).reshape(-1, 1)
        combined = np.concatenate([xyz, rgb.astype(np.float32), lbl.astype(np.float32), pnl.astype(np.float32)], axis=1)
        fmt = "%.6f %.6f %.6f %d %d %d %d %d"
    else:
        combined = np.concatenate([xyz, rgb.astype(np.float32), lbl.astype(np.float32)], axis=1)
        fmt = "%.6f %.6f %.6f %d %d %d %d"
    np.savetxt(path, combined, fmt=fmt)
