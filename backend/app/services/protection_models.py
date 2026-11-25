"""Load an explicitly configured local Vision protection-model adapter."""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProtectionModelBundle:
    identity_model: Any
    manipulation_model: Any


def load_protection_models() -> ProtectionModelBundle:
    target = os.getenv("VISION_PROTECTION_ADAPTER", "").strip()
    if not target or ":" not in target:
        raise RuntimeError(
            "Model-aware protection is not configured. Set "
            "VISION_PROTECTION_ADAPTER=your_adapter_module:load_models and install "
            "the adapter's local model weights."
        )
    module_name, attribute = target.split(":", 1)
    if not module_name or not attribute:
        raise RuntimeError("VISION_PROTECTION_ADAPTER must use module:callable syntax.")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute, None)
        if not callable(factory):
            raise RuntimeError(f"Protection adapter factory '{target}' is not callable.")
        bundle = factory()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not initialize protection adapter '{target}': {exc}") from exc

    if not hasattr(bundle, "identity_model") or not hasattr(bundle, "manipulation_model"):
        raise RuntimeError(
            "Adapter must return an object exposing identity_model and manipulation_model."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Model-aware protection requires PyTorch. Install backend/requirements-protection.txt."
        ) from exc
    if not isinstance(bundle.identity_model, torch.nn.Module) or not isinstance(
        bundle.manipulation_model, torch.nn.Module
    ):
        raise RuntimeError("identity_model and manipulation_model must be torch.nn.Module instances.")
    return ProtectionModelBundle(bundle.identity_model, bundle.manipulation_model)
