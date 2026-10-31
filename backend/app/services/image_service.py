from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.experiment import Experiment
from engine.export.image_export import encode_png
from engine.metrics.difference_map import make_difference_map
from engine.metrics.perceptual_metrics import calculate_metrics
from engine.perturbation.processors import PerturbationConfig, apply_perturbation
from engine.preprocessing.image_loader import load_image


def path_for(image_id: str, suffix: str) -> Path:
    return settings.storage_dir / f"{image_id}_{suffix}.png"


def save_upload(data: bytes, filename: str) -> dict[str, str | int]:
    image = load_image(data)
    image_id = str(uuid.uuid4())
    original_path = path_for(image_id, "original")
    original_path.write_bytes(encode_png(np.asarray(image)))
    return {
        "image_id": image_id,
        "filename": Path(filename).name,
        "width": image.width,
        "height": image.height,
        "file_size": len(data),
        "original_path": str(original_path),
    }


def process_image(db: Session, image_id: str, filename: str, config: PerturbationConfig) -> dict:
    original_path = path_for(image_id, "original")
    if not original_path.exists():
        raise FileNotFoundError("Image was not found.")
    original = np.asarray(load_image(original_path))
    started = time.perf_counter()
    modified = apply_perturbation(original, config)
    duration_ms = (time.perf_counter() - started) * 1000
    modified_path = path_for(image_id, "modified")
    difference_path = path_for(image_id, "difference")
    modified_path.write_bytes(encode_png(modified))
    difference = make_difference_map(original, modified)
    difference.save(difference_path, format="PNG", optimize=True)
    metrics = calculate_metrics(original, modified)
    record = Experiment(
        id=image_id,
        filename=filename,
        method=config.method,
        budget=config.budget,
        duration_ms=duration_ms,
        original_path=str(original_path),
        modified_path=str(modified_path),
        metrics_json=json.dumps(metrics),
        width=original.shape[1],
        height=original.shape[0],
    )
    db.merge(record)
    db.commit()
    return {
        "image_id": image_id,
        "filename": filename,
        "original_url": f"/api/v1/images/{image_id}/download?variant=original",
        "modified_url": f"/api/v1/images/{image_id}/download?variant=modified",
        "difference_url": f"/api/v1/images/{image_id}/download?variant=difference",
        "width": original.shape[1],
        "height": original.shape[0],
        "original_size": original_path.stat().st_size,
        "modified_size": modified_path.stat().st_size,
        "processing_time_ms": duration_ms,
        "metrics": metrics,
    }
