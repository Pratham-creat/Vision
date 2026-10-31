from __future__ import annotations

import numpy as np
from PIL import Image


def make_difference_map(original: np.ndarray, modified: np.ndarray) -> Image.Image:
    diff = np.abs(original.astype(np.int16) - modified.astype(np.int16)).max(axis=2)
    normalized = np.clip(diff * 8, 0, 255).astype(np.uint8)
    heatmap = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    heatmap[:, :, 0] = normalized
    heatmap[:, :, 1] = np.where(normalized > 80, 255 - normalized, normalized // 3)
    return Image.fromarray(heatmap, mode="RGB")
