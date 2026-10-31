from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float | None = None


class VisionModelAdapter(Protocol):
    """Contract for an optional local or remote image classifier."""

    identifier: str

    def predict(self, image_path: Path) -> Prediction:
        """Return a model prediction for one image."""
