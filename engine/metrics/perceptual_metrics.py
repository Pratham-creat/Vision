from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _ssim(original: np.ndarray, modified: np.ndarray) -> float:
    height, width = original.shape[:2]
    smallest = min(height, width)
    win_size = min(7, smallest if smallest % 2 else smallest - 1)

    # SSIM requires an odd window of at least 3. Edge-pad extremely small
    # images so 1x1 and 2xN inputs remain measurable instead of raising.
    if win_size < 3:
        pad_h = max(0, 3 - height)
        pad_w = max(0, 3 - width)
        pad = ((0, pad_h), (0, pad_w), (0, 0))
        original = np.pad(original, pad, mode="edge")
        modified = np.pad(modified, pad, mode="edge")
        win_size = 3

    return float(structural_similarity(
        original, modified, channel_axis=2, data_range=255, win_size=win_size
    ))


def calculate_metrics(original: np.ndarray, modified: np.ndarray) -> dict[str, float | int]:
    if original.shape != modified.shape:
        raise ValueError("Images must have identical dimensions for comparison.")
    if original.ndim != 3 or original.shape[2] != 3 or original.size == 0:
        raise ValueError("Images must be non-empty RGB arrays.")

    original_float = original.astype(np.float32) / 255.0
    modified_float = modified.astype(np.float32) / 255.0
    difference = np.abs(original.astype(np.int16) - modified.astype(np.int16))
    delta_e = deltaE_ciede2000(rgb2lab(original_float), rgb2lab(modified_float))
    identical = not np.any(difference)
    psnr = 100.0 if identical else float(
        peak_signal_noise_ratio(original, modified, data_range=255)
    )
    return {
        "psnr": psnr,
        "ssim": _ssim(original, modified),
        "delta_e_2000": float(np.mean(delta_e)),
        "mean_absolute_rgb_difference": float(np.mean(difference)),
        "maximum_pixel_difference": int(np.max(difference)),
        "changed_pixel_percentage": float(np.mean(np.any(difference > 0, axis=2)) * 100),
    }
