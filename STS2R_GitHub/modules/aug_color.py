import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

PATTERN_S_THRESHOLD = 0.15

TRACE_COLORS = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def apply(data, rng, lam):
    out = data.astype(np.float32, copy=True)
    rgb01 = np.clip(out[:, 3:6] / 255.0, 0.0, 1.0).astype(np.float32, copy=False)
    label = out[:, 6].astype(np.int32, copy=False)
    hsv = rgb_to_hsv(rgb01)

    lam = float(lam)
    bg_mask = label == 0
    trace_mask = label == 1

    if np.any(bg_mask):
        bg_indices = np.where(bg_mask)[0]
        bg_s = hsv[bg_indices, 1]
        is_pattern = bg_s > float(PATTERN_S_THRESHOLD)
        pattern_indices = bg_indices[is_pattern]
        base_indices = bg_indices[~is_pattern]

        h_shift = lam * 0.5
        perturb = 0.10 + 0.40 * lam

        if len(base_indices) > 0:
            base_s_target = float(rng.uniform(0.3, 0.7))
            hsv[base_indices, 1] = base_s_target
            base_h_start = float(rng.uniform(0.0, 1.0))
            hsv[base_indices, 0] = (base_h_start + h_shift) % 1.0

        if len(pattern_indices) > 0:
            vibrancy = float(rng.uniform(1.2, 1.5))
            hsv[pattern_indices, 1] = np.clip(hsv[pattern_indices, 1] * vibrancy, 0, 1)
            jitter = rng.uniform(-0.05, 0.05, size=len(pattern_indices)).astype(np.float32)
            hsv[pattern_indices, 0] = (hsv[pattern_indices, 0] + h_shift + 0.1 + jitter) % 1.0

        s_factors = rng.uniform(1.0 - perturb, 1.0 + perturb, size=len(bg_indices)).astype(np.float32)
        hsv[bg_indices, 1] = np.clip(hsv[bg_indices, 1] * s_factors, 0, 1)

        v_factor = float(rng.uniform(1.0 - perturb / 2.0, 1.0 + perturb / 2.0))
        hsv[bg_indices, 2] = np.clip(hsv[bg_indices, 2] * v_factor, 0, 1)

    if np.any(trace_mask):
        bg_mean_v = float(np.mean(hsv[bg_mask, 2])) if np.any(bg_mask) else 0.5
        bg_mean_s = float(np.mean(hsv[bg_mask, 1])) if np.any(bg_mask) else 0.0

        target_rgb = TRACE_COLORS[int(rng.integers(0, len(TRACE_COLORS)))]
        target_hsv = rgb_to_hsv(target_rgb.reshape(1, 1, 3)).reshape(-1)

        blend_alpha = 0.85 * lam
        new_v = float(target_hsv[2] * (1 - blend_alpha) + bg_mean_v * blend_alpha)
        new_s = float(target_hsv[1] * (1 - blend_alpha) + bg_mean_s * blend_alpha)

        trace_indices = np.where(trace_mask)[0]
        hsv[trace_indices, 0] = float(target_hsv[0])
        hsv[trace_indices, 1] = np.clip(new_s, 0, 1)
        hsv[trace_indices, 2] = np.clip(new_v, 0, 1)

        v_noise = rng.uniform(0.95, 1.05, size=len(trace_indices)).astype(np.float32)
        hsv[trace_indices, 2] = np.clip(hsv[trace_indices, 2] * v_noise, 0, 1)

    rgb_aug = hsv_to_rgb(hsv)
    if float(rng.random()) < 0.10:
        gray = np.dot(rgb_aug, np.array([0.299, 0.587, 0.114], dtype=np.float32))
        rgb_aug = np.stack([gray, gray, gray], axis=1)

    out[:, 3:6] = np.clip(rgb_aug * 255.0, 0, 255)
    return out
