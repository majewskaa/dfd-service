from pydantic import BaseModel, Field


class AnalysisSegment(BaseModel):
    from_: float = Field(..., alias="from")
    to: float
    deepfake_probability: float = Field(..., alias="deepfakeProbability", ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class VideoTooLongError(BaseModel):
    message: str
    durationSeconds: float
    maxDurationSeconds: float


class NoFaceDetectedError(BaseModel):
    className: str = "NoFaceDetectedError"
    message: str
