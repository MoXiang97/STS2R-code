import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

class AblationAugmentor:
    """
    Modular 3D Point Cloud Augmentor for Sim2Real Ablation Studies.
    All methods are driven by a continuous intensity variable 'lam' (0.1 - 1.0).
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def load_txt(path):
        """Loads point cloud from txt. Returns N x 7 or N x 8 array or None if invalid."""
        try:
            data = np.loadtxt(path, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data
        except Exception as e:
            print(f"Skipping {os.path.basename(path)}: {e}")
            return None

    @staticmethod
    def save_txt(path, data):
        """Saves point cloud to txt. Automatically handles 7 or 8 columns."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = data[:, :3]
        rgb = np.clip(data[:, 3:6], 0, 255).astype(np.int32)
        lbl = data[:, 6].astype(np.int32)
        
        if data.shape[1] > 7:
            pnl = data[:, 7].astype(np.int32)
            fmt = "%.6f %.6f %.6f %d %d %d %d %d"
            combined = np.column_stack([xyz, rgb, lbl, pnl])
        else:
            fmt = "%.6f %.6f %.6f %d %d %d %d"
            combined = np.column_stack([xyz, rgb, lbl])
        
        np.savetxt(path, combined, fmt=fmt)

    def geo_deform(self, data, lam):
        """Parametric width/height scaling using Gaussian soft masks."""
        xyz = data[:, :3].copy()
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        xmin, xmax = np.min(x), np.max(x)
        x_norm = (x - xmin) / (xmax - xmin + 1e-8)

        # Soft mask anchors (Heel, Arch, Toe)
        mu = np.array([0.15, 0.55, 0.90], dtype=np.float32)
        sigma = np.array([0.10, 0.10, 0.08], dtype=np.float32)
        
        # Scaling parameters driven by lam
        scale_y_base = 1.0 + self.rng.uniform(-0.1, 0.1) * lam
        scale_z_base = 1.0 + self.rng.uniform(-0.08, 0.08) * lam
        
        delta_y = self.rng.uniform(-0.12, 0.12, size=3) * lam
        delta_z = self.rng.uniform(-0.10, 0.10, size=3) * lam

        masks = np.exp(-0.5 * ((x_norm[:, None] - mu) / sigma)**2)
        weights = masks / (np.sum(masks, axis=1, keepdims=True) + 1e-8)
        
        sy = scale_y_base * (1.0 + np.sum(weights * delta_y, axis=1))
        sz = scale_z_base * (1.0 + np.sum(weights * delta_z, axis=1))
        
        # Local centers via binning
        bins = 32
        edges = np.linspace(xmin, xmax, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        y_means, z_means = np.zeros(bins), np.zeros(bins)
        for i in range(bins):
            m = (x >= edges[i]) & (x < edges[i+1])
            y_means[i] = np.mean(y[m]) if np.any(m) else np.mean(y)
            z_means[i] = np.mean(z[m]) if np.any(m) else np.mean(z)
        
        y0 = np.interp(x, centers, y_means)
        z0 = np.interp(x, centers, z_means)
        
        xyz[:, 1] = y0 + (y - y0) * sy
        xyz[:, 2] = z0 + (z - z0) * sz
        
        out = data.copy()
        out[:, :3] = xyz
        return out

    def spline_jitter(self, data, lam, erase_original=False):
        """Polar periodic Cubic Spline offset in Z and surface projection."""
        xyz = data[:, :3].copy()
        rgb = data[:, 3:6].copy()
        lbl = data[:, 6].copy().astype(np.int32)
        
        trace_mask = (lbl == 1)
        if np.sum(trace_mask) < 10: return data
        
        trace_xyz = xyz[trace_mask]
        centroid = np.mean(trace_xyz, axis=0)
        
        # Polar conversion
        rel = trace_xyz[:, :2] - centroid[:2]
        theta = np.arctan2(rel[:, 1], rel[:, 0])
        
        # Periodic Spline
        n_anchors = int(6 + 10 * lam)
        anchors_theta = np.linspace(-np.pi, np.pi, n_anchors + 1)
        z_jitter = self.rng.uniform(-5.0, 5.0, size=n_anchors) * lam
        z_jitter = np.append(z_jitter, z_jitter[0])
        cs = CubicSpline(anchors_theta, z_jitter, bc_type='periodic')
        
        new_trace_xyz = trace_xyz.copy()
        new_trace_xyz[:, 2] += cs(theta).astype(np.float32)
        
        # Surface projection onto BG
        bg_mask = (lbl == 0)
        if not np.any(bg_mask): return data
        
        tree_bg = cKDTree(xyz[bg_mask])
        _, idx = tree_bg.query(new_trace_xyz, k=1)
        proj_indices = np.where(bg_mask)[0][idx]
        
        if erase_original:
            lbl[:] = 0
            lbl[proj_indices] = 1
        else:
            trace_color = np.mean(rgb[trace_mask], axis=0)
            lbl[trace_mask] = 0
            lbl[proj_indices] = 1
            rgb[proj_indices] = trace_color
            
        out = data.copy()
        out[:, :3] = xyz
        out[:, 3:6] = rgb
        out[:, 6] = lbl
        return out

    def pc_phys_degradation(self, data, lam):
        """Simulates physical scanning artifacts using normal-based ray dropout and distance decay."""
        xyz = data[:, :3].copy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normals = np.asarray(pcd.normals)
        
        centroid = np.mean(xyz, axis=0)
        cam_pos = centroid + np.array([0, 0, 150]) # Virtual overhead camera
        
        # 1. Cosine dropout
        rays = cam_pos - xyz
        rays /= (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-8)
        cos_theta = np.abs(np.sum(normals * rays, axis=1))
        cos_threshold = self.rng.uniform(0.05, 0.10)
        keep_mask = cos_theta > cos_threshold
        
        # 2. Distance dropout P = 1 - exp(-d/lambda)
        dists = np.linalg.norm(cam_pos - xyz, axis=1)
        d_norm = dists / (np.max(dists) + 1e-8)
        lam_decay = 0.5 + 2.0 * lam
        p_dropout = 1.0 - np.exp(-d_norm / lam_decay)
        keep_mask &= (self.rng.random(len(xyz)) > p_dropout)
        
        # 3. Tangential Speckle Noise
        jitter_std = 0.05 * lam
        rand_jitter = self.rng.normal(0, jitter_std, size=xyz.shape)
        tangential_jitter = rand_jitter - (np.sum(rand_jitter * normals, axis=1, keepdims=True)) * normals
        xyz += tangential_jitter
        
        out = data[keep_mask].copy()
        out[:, :3] = xyz[keep_mask]
        return out

    def panel_texturing(self, data, lam):
        """Random HSV texture assignment based on Panel_ID."""
        if data.shape[1] < 8: return data
        out = data.copy()
        pids = out[:, 7].astype(np.int32)
        for pid in np.unique(pids):
            if pid == 0: continue
            h, s, v = self.rng.uniform(0, 1), self.rng.uniform(0.3, 0.8), self.rng.uniform(0.4, 0.9)
            out[pids == pid, 3:6] = hsv_to_rgb([h, s, v]) * 255.0
        return out

    def real_color_jitter(self, data, lam):
        """Continuous HSV saturation/value shifts."""
        out = data.copy()
        hsv = rgb_to_hsv(np.clip(out[:, 3:6] / 255.0, 0, 1))
        hsv[:, 1] = np.clip(hsv[:, 1] + self.rng.uniform(-0.2, 0.2) * lam, 0, 1)
        hsv[:, 2] = np.clip(hsv[:, 2] + self.rng.uniform(-0.15, 0.15) * lam, 0, 1)
        out[:, 3:6] = hsv_to_rgb(hsv) * 255.0
        return out

    def standard_augs(self, data, lam):
        """Final pipeline step: Random density downsampling and ellipsoidal hollowing."""
        # 1. Density: Background ~ U(0.35, 0.95), Trace 100%
        lbl = data[:, 6].astype(np.int32)
        bg_keep_rate = self.rng.uniform(0.35, 0.95)
        bg_mask = (lbl == 0)
        keep = np.ones(len(data), dtype=bool)
        keep[bg_mask] = self.rng.random(np.sum(bg_mask)) < bg_keep_rate
        data = data[keep]
        
        # 2. Irregular Hollowing
        xyz, lbl = data[:, :3], data[:, 6].astype(np.int32)
        n_holes = self.rng.integers(1, 4)
        radius = 5.0 + 15.0 * lam
        keep = np.ones(len(data), dtype=bool)
        if len(xyz) > 0:
            centers = xyz[self.rng.choice(len(xyz), size=min(n_holes, len(xyz)), replace=False)]
            for c in centers:
                scales = self.rng.uniform(0.5, 3.0, size=3)
                in_hole = np.sum(((xyz - c) / scales)**2, axis=1) < radius**2
                keep[in_hole & (lbl == 0)] = False
                # Trace drop 5%
                trace_in_hole = in_hole & (lbl == 1)
                keep[trace_in_hole] &= (self.rng.random(np.sum(trace_in_hole)) >= 0.05)
        return data[keep]
