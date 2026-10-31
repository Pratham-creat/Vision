from __future__ import annotations

import cv2
import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)


def ycrcb_to_rgb(ycrcb: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(ycrcb.astype(np.float32), cv2.COLOR_YCrCb2RGB), 0, 255).astype(np.uint8)
