from datetime import datetime
from typing import Any, List, Optional
import uuid

from pydantic import BaseModel, Field

class AnalysisResponseSegment(BaseModel):
    from_: float = Field(..., alias="from")
    to: float
    deepfake_probability: float = Field(..., alias="deepfakeProbability", ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class VideoTooLongErrorResponse(BaseModel):
    className: str = "VideoTooLongErrorResponse"
    message: str
    durationSeconds: float
    maxDurationSeconds: float


class NoFaceDetectedErrorResponse(BaseModel):
    className: str = "NoFaceDetectedErrorResponse"
    message: str

    def __init__(self, message: str = "No face detected in the video."):
        Exception.__init__(self, message)
        BaseModel.__init__(self, message=message)


class JobCreatedResponse(BaseModel):
    jobId: uuid.UUID
    claimToken: str


class JobStatusResponse(BaseModel):
    jobId: uuid.UUID
    status: str  # pending | running | done | failed
    filename: Optional[str] = None
    createdAt: Optional[datetime] = None
    result: Optional[List[AnalysisResponseSegment]] = None
    error: Optional[Any] = None  # structured dict with className + message, or None