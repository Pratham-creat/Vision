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
    # OpenCV's YCrCb float conversion expects normalized [0, 1] values.
    # These values use the uint8 [0, 255] encoding, so convert back to uint8.
    encoded = np.clip(np.rint(ycrcb), 0, 255).astype(np.uint8)
    return cv2.cvtColor(encoded, cv2.COLOR_YCrCb2RGB)
