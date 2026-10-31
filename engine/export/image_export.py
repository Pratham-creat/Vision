from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def encode_png(rgb: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()


def encode_image(rgb: np.ndarray, image_format: str = "PNG") -> bytes:
    output = BytesIO()
    image = Image.fromarray(rgb, mode="RGB")
    fmt = image_format.upper()
    if fmt in {"JPG", "JPEG"}:
        image.save(output, format="JPEG", quality=95, optimize=True)
    elif fmt == "WEBP":
        image.save(output, format="WEBP", quality=95)
    else:
        image.save(output, format="PNG", optimize=True)
    return output.getvalue()
