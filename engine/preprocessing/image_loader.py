from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


def load_image(source: bytes | str | Path) -> Image.Image:
    """Load an image as an RGB Pillow image without mutating the source."""
    try:
        image = Image.open(BytesIO(source) if isinstance(source, bytes) else source)
        if image.format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image.format or 'unknown'}")
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid supported image.") from exc


def image_metadata(image: Image.Image, file_size: int | None = None) -> dict[str, int | str | None]:
    return {
        "width": image.width,
        "height": image.height,
        "format": image.format or "PNG",
        "file_size": file_size,
    }
