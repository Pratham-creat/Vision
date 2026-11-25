from typing import Any, Literal

from pydantic import BaseModel, Field


Method = Literal["rgb", "chrominance", "perceptual", "combined"]


class ProcessRequest(BaseModel):
    image_id: str
    filename: str | None = Field(default=None, max_length=255)
    method: Method = "perceptual"
    budget: float = Field(default=4.0, ge=0, le=32)
    iterations: int = Field(default=1, ge=1, le=20)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class ProtectionRequest(BaseModel):
    image_id: str
    filename: str | None = Field(default=None, max_length=255)
    budget: float = Field(default=4.0, ge=0, le=16)
    iterations: int = Field(default=20, ge=1, le=200)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    eot_samples: int = Field(default=3, ge=1, le=16)


class MetricsResponse(BaseModel):
    psnr: float
    ssim: float
    delta_e_2000: float
    mean_absolute_rgb_difference: float
    maximum_pixel_difference: int
    changed_pixel_percentage: float


class ProcessResponse(BaseModel):
    image_id: str
    filename: str
    original_url: str
    modified_url: str
    difference_url: str
    width: int
    height: int
    original_size: int
    modified_size: int
    processing_time_ms: float
    metrics: MetricsResponse


class ProtectionResponse(ProcessResponse):
    configured_models: list[str]
    objective_before: float
    objective_after: float
    max_channel_delta: int


class AIRequest(BaseModel):
    image_id: str
    model_identifier: str = Field(min_length=1, max_length=200)
    endpoint: str | None = None


class AIResponse(BaseModel):
    configured: bool
    message: str
    original: dict[str, Any] | None = None
    modified: dict[str, Any] | None = None
    agreement: bool | None = None
