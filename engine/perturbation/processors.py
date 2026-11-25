from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from engine.color_space.conversions import lab_to_rgb, rgb_to_lab, rgb_to_ycrcb, ycrcb_to_rgb


@dataclass(frozen=True)
class PerturbationConfig:
    method: str = "perceptual"
    budget: float = 4.0
    iterations: int = 1
    seed: int = 42


def controlled_rgb(rgb: np.ndarray, config: PerturbationConfig) -> np.ndarray:
    if config.budget <= 0:
        return rgb.copy()
    rng = np.random.default_rng(config.seed)
    result = rgb.astype(np.int16).copy()
    amplitude = max(0.0, min(config.budget, 32.0))
    for _ in range(max(1, config.iterations)):
        noise = rng.uniform(-amplitude, amplitude, rgb.shape)
        result = np.clip(result + np.rint(noise), 0, 255)
    return result.astype(np.uint8)


def chrominance(rgb: np.ndarray, config: PerturbationConfig) -> np.ndarray:
    if config.budget <= 0:
        return rgb.copy()
    rng = np.random.default_rng(config.seed)
    ycrcb = rgb_to_ycrcb(rgb)
    amplitude = max(0.0, min(config.budget * 0.8, 24.0))
    for _ in range(max(1, config.iterations)):
        noise = rng.uniform(-amplitude, amplitude, ycrcb[:, :, 1:].shape)
        ycrcb[:, :, 1:] = np.clip(ycrcb[:, :, 1:] + noise, 0, 255)
    return ycrcb_to_rgb(ycrcb)


def perceptually_constrained(rgb: np.ndarray, config: PerturbationConfig) -> np.ndarray:
    if config.budget <= 0:
        return rgb.copy()
    rng = np.random.default_rng(config.seed)
    lab = rgb_to_lab(rgb)
    # OpenCV's Lab uses L in [0,255]; skimage's CIE Lab uses L in [0,100].
    target = min(config.budget, 20.0)
    for _ in range(max(1, config.iterations)):
        candidate = lab.copy()
        candidate[:, :, 1:] += rng.uniform(-target, target, candidate[:, :, 1:].shape)
        candidate_rgb = lab_to_rgb(candidate)
        actual = deltaE_ciede2000(rgb2lab(rgb / 255.0), rgb2lab(candidate_rgb / 255.0))
        scale = np.minimum(1.0, target / np.maximum(actual, 1e-6))
        candidate = lab + (candidate - lab) * scale[:, :, None]
        lab = candidate
    return lab_to_rgb(lab)


def combined(rgb: np.ndarray, config: PerturbationConfig) -> np.ndarray:
    """Fuse three proposals under a final per-channel RGB displacement budget."""
    if config.budget <= 0:
        return rgb.copy()
    proposals = []
    for method in ("rgb", "chrominance", "perceptual"):
        proposal = apply_perturbation(
            rgb,
            PerturbationConfig(method=method, budget=config.budget,
                               iterations=config.iterations, seed=config.seed),
        )
        proposals.append(proposal.astype(np.float32) - rgb.astype(np.float32))
    fused_delta = np.mean(np.stack(proposals, axis=0), axis=0)
    fused_delta = np.clip(fused_delta, -config.budget, config.budget)
    return np.clip(np.rint(rgb.astype(np.float32) + fused_delta), 0, 255).astype(np.uint8)


def apply_perturbation(rgb: np.ndarray, config: PerturbationConfig) -> np.ndarray:
    methods = {
        "rgb": controlled_rgb,
        "chrominance": chrominance,
        "perceptual": perceptually_constrained,
        "combined": combined,
    }
    try:
        return methods[config.method](rgb, config)
    except KeyError as exc:
        raise ValueError(f"Unknown perturbation method: {config.method}") from exc
