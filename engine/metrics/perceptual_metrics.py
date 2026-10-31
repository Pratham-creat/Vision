from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_metrics(original: np.ndarray, modified: np.ndarray) -> dict[str, float | int]:
    if original.shape != modified.shape:
        raise ValueError("Images must have identical dimensions for comparison.")
    original_float = original.astype(np.float32) / 255.0
    modified_float = modified.astype(np.float32) / 255.0
    difference = np.abs(original.astype(np.int16) - modified.astype(np.int16))
    delta_e = deltaE_ciede2000(rgb2lab(original_float), rgb2lab(modified_float))
    return {
        "psnr": float(peak_signal_noise_ratio(original, modified, data_range=255)),
        "ssim": float(structural_similarity(original, modified, channel_axis=2, data_range=255)),
        "delta_e_2000": float(np.mean(delta_e)),
        "mean_absolute_rgb_difference": float(np.mean(difference)),
        "maximum_pixel_difference": int(np.max(difference)),
        "changed_pixel_percentage": float(np.mean(np.any(difference > 0, axis=2)) * 100),
    }
