import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import bcrypt
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, select

from service.src.auth import jwt as jwt_utils
from service.src.db.models import Job, User
from service.src.db.session import get_session
from service.src.lab_service import video_analyzer_runner as job_runner
from service.src.lab_service.errors import NoFaceDetectedError
from service.src.main.helpers.response_helper import get_video_duration
from service.src.schemas.responses import (
    AnalysisResponseSegment,
    JobCreatedResponse,
    JobStatusResponse,
    NoFaceDetectedErrorResponse,
    VideoTooLongErrorResponse,
)

log = logging.getLogger(__name__)
router = APIRouter()
_bearer = HTTPBearer(auto_error=False)

# Populated at startup via configure().
_analyzers: dict = {}
_max_video_duration_seconds: Optional[float] = None
_uploads_dir: str = tempfile.gettempdir()


def configure(analyzers: dict, max_duration: Optional[float], uploads_dir: str) -> None:
    global _analyzers, _max_video_duration_seconds, _uploads_dir
    _analyzers = analyzers
    _max_video_duration_seconds = max_duration
    _uploads_dir = uploads_dir


# ── auth helpers ──────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[uuid.UUID]:
    if credentials is None:
        return None
    return jwt_utils.decode_access_token(credentials.credentials)


def require_current_user_id(
    user_id: Optional[uuid.UUID] = Depends(get_current_user_id),
) -> uuid.UUID:
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user_id


# ── schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    notify_email: bool = False
    claim_token: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    notify_email: bool


class UserSettingsRequest(BaseModel):
    notify_email: bool


class UpdateEmailRequest(BaseModel):
    email: EmailStr
    password: str  # require current password to confirm the change


# ── job helpers ───────────────────────────────────────────────────────────────

def _job_to_response(job: Job) -> JobStatusResponse:
    result = None
    if job.result_json:
        result = [AnalysisResponseSegment(**s) for s in json.loads(job.result_json)]

    error = None
    if job.error:
        try:
            error = json.loads(job.error)
        except (json.JSONDecodeError, ValueError):
            error = {"className": "UnexpectedError", "message": job.error}

    return JobStatusResponse(
        jobId=job.id,
        status=job.status,
        algorithm=job.algorithm or "xception",
        filename=job.filename,
        createdAt=job.created_at,
        result=result,
        error=error,
    )


# ── auth endpoints ────────────────────────────────────────────────────────────

@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED, tags=["auth"])
def register(body: RegisterRequest, session: Session = Depends(get_session)):
    if session.exec(select(User).where(User.email == body.email)).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(email=body.email, password_hash=_hash(body.password), notify_email=body.notify_email)
    session.add(user)
    session.flush()

    if body.claim_token:
        job = session.exec(select(Job).where(Job.claim_token == body.claim_token)).first()
        if job and job.user_id is None:
            job.user_id = user.id

    session.commit()
    return TokenResponse(access_token=jwt_utils.create_access_token(user.id))


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(body: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == body.email)).first()
    if user is None or not _verify(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return TokenResponse(access_token=jwt_utils.create_access_token(user.id))


@router.get("/auth/me", response_model=UserResponse, tags=["auth"])
def me(
    user_id: uuid.UUID = Depends(require_current_user_id),
    session: Session = Depends(get_session),
):
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(id=user.id, email=user.email, notify_email=user.notify_email)


@router.patch("/auth/email", response_model=UserResponse, tags=["auth"])
def update_email(
    body: UpdateEmailRequest,
    user_id: uuid.UUID = Depends(require_current_user_id),
    session: Session = Depends(get_session),
):
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not _verify(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")
    if session.exec(select(User).where(User.email == body.email, User.id != user_id)).first():
        raise HTTPException(status_code=409, detail="Email already in use")
    user.email = body.email
    session.commit()
    session.refresh(user)
    return UserResponse(id=user.id, email=user.email, notify_email=user.notify_email)


@router.patch("/auth/settings", response_model=UserResponse, tags=["auth"])
def update_settings(
    body: UserSettingsRequest,
    user_id: uuid.UUID = Depends(require_current_user_id),
    session: Session = Depends(get_session),
):
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user.notify_email = body.notify_email
    session.commit()
    session.refresh(user)
    return UserResponse(id=user.id, email=user.email, notify_email=user.notify_email)


# ── job endpoints ─────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["jobs"],
    responses={
        413: {"model": VideoTooLongErrorResponse},
        422: {"description": "Unknown algorithm"},
        503: {"description": "Model not loaded"},
    },
)
async def submit_analysis(
    video: UploadFile = File(...),
    algorithm: str = Form(..., description="Detection algorithm to use. Available: xception, cnn_lstm, avff"),
    user_id: Optional[uuid.UUID] = Depends(get_current_user_id),
    session: Session = Depends(get_session),
):
    available = list(_analyzers.keys()) or list(_max_durations.keys())
    if algorithm not in available:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown algorithm '{algorithm}'. Available: {available}",
        )

    analyzer = _analyzers.get(algorithm)
    if analyzer is None:
        raise HTTPException(status_code=503, detail=f"Model for algorithm '{algorithm}' is not loaded")

    log.info("[analyze] user_id=%s algorithm=%s", user_id, algorithm)
    content = await video.read()

    Path(_uploads_dir).mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=_uploads_dir)
    try:
        with open(tmp_fd, "wb") as f:
            f.write(content)

        if _max_video_duration_seconds is not None:
            duration = get_video_duration(tmp_path)
            if duration > _max_video_duration_seconds:
                Path(tmp_path).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=VideoTooLongErrorResponse(
                        message=f"Video is too long ({duration:.1f}s). Maximum allowed duration is {_max_video_duration_seconds:.0f}s.",
                        durationSeconds=round(duration, 1),
                        maxDurationSeconds=_max_video_duration_seconds,
                    ).model_dump(),
                )

        job = Job(user_id=user_id, filename=video.filename, upload_path=tmp_path, status="pending", algorithm=algorithm)
        session.add(job)
        session.commit()
        session.refresh(job)

        asyncio.create_task(job_runner.run_analysis(job.id, tmp_path, analyzer))
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    return JobCreatedResponse(jobId=job.id, claimToken=job.claim_token)


@router.get("/jobs", response_model=List[JobStatusResponse], tags=["jobs"])
def list_jobs(
    user_id: uuid.UUID = Depends(require_current_user_id),
    session: Session = Depends(get_session),
):
    jobs = session.exec(
        select(Job).where(Job.user_id == user_id).order_by(Job.created_at.desc())
    ).all()
    return [_job_to_response(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
def get_job(
    job_id: uuid.UUID,
    claim_token: Optional[str] = None,
    user_id: Optional[uuid.UUID] = Depends(get_current_user_id),
    session: Session = Depends(get_session),
):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    is_owner = user_id is not None and job.user_id == user_id
    has_claim = claim_token is not None and claim_token == job.claim_token
    if not is_owner and not has_claim:
        raise HTTPException(status_code=403, detail="Forbidden")

    return _job_to_response(job)
