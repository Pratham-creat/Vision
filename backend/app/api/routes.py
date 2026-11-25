from __future__ import annotations

import json
import logging
import re
import time

import numpy as np

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import get_db
from backend.app.models.experiment import Experiment
from backend.app.schemas.api import AIRequest, AIResponse, ProcessRequest, ProcessResponse, ProtectionRequest, ProtectionResponse
from backend.app.services.image_service import path_for, process_image, save_upload
from backend.app.services.protection_models import load_protection_models
from engine.export.image_export import encode_png
from engine.metrics.difference_map import make_difference_map
from engine.metrics.perceptual_metrics import calculate_metrics
from engine.preprocessing.image_loader import load_image
from engine.perturbation.adversarial import optimize_protection
from evaluation.adapters.unconfigured import unavailable_message
from engine.perturbation.processors import PerturbationConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")
_IMAGE_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$")


def validate_image_id(image_id: str) -> str:
    if not _IMAGE_ID_RE.fullmatch(image_id):
        raise HTTPException(400, "Invalid image_id format.")
    return image_id


@router.get("/capabilities")
def capabilities() -> dict:
    return {"methods": ["rgb", "chrominance", "perceptual", "combined"], "protection": ["adversarial"], "formats": ["jpg", "jpeg", "png", "webp"], "ai_evaluation": False, "protection_models_configured": bool(__import__("os").getenv("VISION_PROTECTION_ADAPTER"))}


@router.post("/images/upload")
async def upload_image(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        raise HTTPException(415, "Only JPG, PNG, and WebP files are supported.")
    data = await file.read()
    if len(data) > settings.max_upload_size:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_size // (1024 * 1024)} MB upload limit.")
    try:
        return save_upload(data, file.filename or "image")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/images/process", response_model=ProcessResponse)
def process(request: ProcessRequest, db: Session = Depends(get_db)) -> dict:
    validate_image_id(request.image_id)
    try:
        return process_image(db, request.image_id, request.filename or request.image_id, PerturbationConfig(
            method=request.method, budget=request.budget, iterations=request.iterations, seed=request.seed
        ))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/protection/optimize", response_model=ProtectionResponse)
def optimize_image_protection(request: ProtectionRequest, db: Session = Depends(get_db)) -> dict:
    validate_image_id(request.image_id)
    original_path = path_for(request.image_id, "original")
    if not original_path.exists():
        raise HTTPException(404, "Image was not found.")
    try:
        models = load_protection_models()
        original = np.asarray(load_image(original_path))
        started = time.perf_counter()
        result = optimize_protection(
            original, models, budget=request.budget, iterations=request.iterations,
            seed=request.seed, eot_samples=request.eot_samples,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        modified = result.image
        modified_path = path_for(request.image_id, "modified")
        difference_path = path_for(request.image_id, "difference")
        modified_path.write_bytes(encode_png(modified))
        make_difference_map(original, modified).save(difference_path, format="PNG", optimize=True)
        metrics = calculate_metrics(original, modified)
        filename = request.filename or request.image_id
        record = Experiment(
            id=request.image_id, filename=filename, method="adversarial",
            budget=request.budget, duration_ms=duration_ms,
            original_path=str(original_path), modified_path=str(modified_path),
            metrics_json=json.dumps(metrics), width=original.shape[1], height=original.shape[0],
        )
        db.merge(record)
        db.commit()
        return {
            "image_id": request.image_id, "filename": filename,
            "original_url": f"/api/v1/images/{request.image_id}/download?variant=original",
            "modified_url": f"/api/v1/images/{request.image_id}/download?variant=modified",
            "difference_url": f"/api/v1/images/{request.image_id}/download?variant=difference",
            "width": original.shape[1], "height": original.shape[0],
            "original_size": original_path.stat().st_size,
            "modified_size": modified_path.stat().st_size,
            "processing_time_ms": duration_ms, "metrics": metrics,
            "configured_models": ["identity", "facial_geometry"],
            "objective_before": result.objective_before,
            "objective_after": result.objective_after,
            "max_channel_delta": result.max_channel_delta,
        }
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/images/history")
def history(db: Session = Depends(get_db)) -> list[dict]:
    records = db.scalars(select(Experiment).order_by(Experiment.created_at.desc())).all()
    return [{"id": item.id, "filename": item.filename, "date": item.created_at, "method": item.method, "budget": item.budget,
             "duration_ms": item.duration_ms, "metrics": json.loads(item.metrics_json), "width": item.width, "height": item.height} for item in records]


@router.get("/images/{image_id}")
def get_image(image_id: str, db: Session = Depends(get_db)) -> dict:
    validate_image_id(image_id)
    record = db.get(Experiment, image_id)
    if not record:
        raise HTTPException(404, "Experiment not found.")
    return {"id": record.id, "filename": record.filename, "metrics": json.loads(record.metrics_json)}


@router.get("/images/{image_id}/download")
def download(image_id: str, variant: str = Query("modified", pattern="^(original|modified|difference)$")):
    validate_image_id(image_id)
    path = path_for(image_id, variant)
    if not path.exists():
        raise HTTPException(404, "Image variant not found.")
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"vision-{variant}.png",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"},
    )


@router.post("/images/compare")
def compare(request: ProcessRequest, db: Session = Depends(get_db)) -> dict:
    validate_image_id(request.image_id)
    record = db.get(Experiment, request.image_id)
    if not record:
        raise HTTPException(404, "Experiment not found.")
    return {"metrics": json.loads(record.metrics_json), "image_id": request.image_id}


@router.post("/evaluation/metrics")
def metrics(request: ProcessRequest, db: Session = Depends(get_db)) -> dict:
    return compare(request, db)


@router.post("/evaluation/ai", response_model=AIResponse)
def ai_evaluation(request: AIRequest) -> AIResponse:
    return AIResponse(configured=False, message=unavailable_message())
