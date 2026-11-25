from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import get_db
from backend.app.models.experiment import Experiment
from backend.app.schemas.api import AIRequest, AIResponse, ProcessRequest, ProcessResponse
from backend.app.services.image_service import path_for, process_image, save_upload
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
    return {"methods": ["rgb", "chrominance", "perceptual", "combined"], "formats": ["jpg", "jpeg", "png", "webp"], "ai_evaluation": False}


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
