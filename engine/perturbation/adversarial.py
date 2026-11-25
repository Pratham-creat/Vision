"""Optional model-aware privacy perturbation using local PyTorch surrogate models.

Adapters expose two differentiable callables:
- identity_model(BCHW RGB float in [0, 1]) -> identity embedding tensor
- geometry_model(BCHW RGB float in [0, 1]) -> landmarks or face-structure tensor

This is a research optimizer, not a universal defense. Effectiveness is bounded
by the supplied surrogate models and must be measured against held-out systems.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ProtectionModels(Protocol):
    identity_model: object
    geometry_model: object


@dataclass(frozen=True)
class ProtectionResult:
    image: np.ndarray
    max_channel_delta: int
    objective_before: float
    objective_after: float


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Model-aware protection requires PyTorch. Install "
            "backend/requirements-protection.txt."
        ) from exc
    return torch, F


def _view(image, scale: float, brightness: float, contrast: float, F):
    """Apply one sampled differentiable transform consistently to image pairs."""
    _, _, height, width = image.shape
    new_h = max(8, round(height * scale))
    new_w = max(8, round(width * scale))
    resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    resized = F.interpolate(resized, size=(height, width), mode="bilinear", align_corners=False)
    mean = resized.mean(dim=(2, 3), keepdim=True)
    return ((resized - mean) * contrast + mean).mul(brightness).clamp(0, 1)


def _outputs(model, image, torch, F):
    output = model(image)
    if not isinstance(output, torch.Tensor):
        raise TypeError("Protection model adapters must return a single torch.Tensor.")
    if output.ndim < 2 or output.shape[0] != image.shape[0]:
        raise ValueError("Protection model output must have a batch dimension.")
    return output.flatten(start_dim=1)


def _objective(identity_model, geometry_model, source, candidate, torch, F):
    source_identity = F.normalize(_outputs(identity_model, source, torch, F), dim=1)
    candidate_identity = F.normalize(_outputs(identity_model, candidate, torch, F), dim=1)
    identity_similarity = (source_identity * candidate_identity).sum(dim=1).mean()

    source_geometry = _outputs(geometry_model, source, torch, F)
    candidate_geometry = _outputs(geometry_model, candidate, torch, F)
    geometry_scale = source_geometry.detach().std().clamp_min(1e-3)
    geometry_divergence = ((candidate_geometry - source_geometry) / geometry_scale).square().mean()
    return identity_similarity - 0.25 * geometry_divergence


def optimize_protection(
    rgb: np.ndarray,
    models: ProtectionModels,
    *,
    budget: float = 4.0,
    iterations: int = 20,
    seed: int = 42,
    eot_samples: int = 3,
) -> ProtectionResult:
    """Optimize bounded changes against identity and facial-geometry surrogates.

    Budget is an L-infinity bound in 8-bit RGB levels. The supplied models must
    be differentiable and return embeddings / landmarks or face-structure tensors.
    EOT applies small differentiable resize and photometric changes during
    optimization to reduce brittleness to simple preprocessing.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 NumPy array.")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("rgb must be a non-empty H×W×3 image.")
    if not np.isfinite(budget) or budget < 0 or budget > 32:
        raise ValueError("budget must be between 0 and 32.")
    if iterations < 1 or iterations > 200:
        raise ValueError("iterations must be between 1 and 200.")
    if eot_samples < 1 or eot_samples > 16:
        raise ValueError("eot_samples must be between 1 and 16.")
    if budget == 0:
        return ProtectionResult(rgb.copy(), 0, 0.0, 0.0)

    torch, F = _require_torch()
    identity_model = models.identity_model
    geometry_model = models.geometry_model
    device = next(identity_model.parameters(), torch.empty(0)).device
    identity_model = identity_model.to(device).eval()
    geometry_model = geometry_model.to(device).eval()
    for model in (identity_model, geometry_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    source = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)
    source = source.permute(2, 0, 1).unsqueeze(0) / 255.0
    epsilon = float(budget) / 255.0
    delta = torch.zeros_like(source, requires_grad=True)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    with torch.no_grad():
        before = float(_objective(identity_model, geometry_model, source, source, torch, F).item())

    step_size = max(epsilon / max(1, iterations // 2), 1.0 / 255.0)
    for _ in range(iterations):
        candidate = (source + delta).clamp(0, 1)
        losses = []
        for _sample in range(eot_samples):
            # Use matching randomization for source and candidate.
            scale = float(torch.empty((), device=source.device).uniform_(0.92, 1.08, generator=generator))
            brightness = float(torch.empty((), device=source.device).uniform_(0.97, 1.03, generator=generator))
            contrast = float(torch.empty((), device=source.device).uniform_(0.97, 1.03, generator=generator))
            view_source = _view(source, scale, brightness, contrast, F)
            view_candidate = _view(candidate, scale, brightness, contrast, F)
            losses.append(_objective(identity_model, geometry_model, view_source, view_candidate, torch, F))
        objective = torch.stack(losses).mean()
        tv = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
        tv = tv + (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
        loss = objective + 0.015 * tv
        loss.backward()
        with torch.no_grad():
            delta -= step_size * delta.grad.sign()
            delta.clamp_(-epsilon, epsilon)
            delta.copy_((source + delta).clamp(0, 1) - source)
        delta.grad = None

    with torch.no_grad():
        protected = (source + delta).clamp(0, 1)
        after = float(_objective(identity_model, geometry_model, source, protected, torch, F).item())
        output = protected.squeeze(0).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    max_delta = int(np.abs(output.astype(np.int16) - rgb.astype(np.int16)).max())
    return ProtectionResult(output, max_delta, before, after)
