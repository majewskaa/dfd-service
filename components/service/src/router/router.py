"""
Job endpoints:

  POST /analyze         – submit a video; returns job_id + claim_token
  GET  /jobs            – list jobs for the authenticated user
  GET  /jobs/{job_id}   – poll a single job (auth optional; claim_token allowed)
"""

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session, select

from service.src.auth.router import get_current_user_id, require_current_user_id
from service.src.db.models import Job
from service.src.db.session import get_session
from service.src.lab_service import video_analyzer_runner as job_runner
from service.src.schemas.responses import AnalysisResponseSegment, NoFaceDetectedErrorResponse, VideoTooLongErrorResponse, JobCreatedResponse, JobStatusResponse
from service.src.main.helpers.response_helper import get_video_duration
from service.src.lab_service.errors import NoFaceDetectedError

router = APIRouter(tags=["jobs"])

# Set from service.py at startup.
_analyzer = None
_max_video_duration_seconds: Optional[float] = None
_uploads_dir: str = tempfile.gettempdir()


def configure(analyzer, max_duration: Optional[float], uploads_dir: str) -> None:
    global _analyzer, _max_video_duration_seconds, _uploads_dir
    _analyzer = analyzer
    _max_video_duration_seconds = max_duration
    _uploads_dir = uploads_dir

def _job_to_response(job: Job) -> JobStatusResponse:
    result = None
    if job.result_json:
        raw = json.loads(job.result_json)
        result = [AnalysisResponseSegment(**s) for s in raw]

    error = None
    if job.error:
        try:
            error = json.loads(job.error)
        except (json.JSONDecodeError, ValueError):
            error = {"className": "UnexpectedError", "message": job.error}

    return JobStatusResponse(
        jobId=job.id,
        status=job.status,
        filename=job.filename,
        result=result,
        error=error,
    )

@router.post(
    "/analyze",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        413: {"model": VideoTooLongErrorResponse},
        400: {"model": NoFaceDetectedErrorResponse},
        503: {"description": "Model not loaded"},
    },
)
async def submit_analysis(
    video: UploadFile = File(...),
    user_id: Optional[uuid.UUID] = Depends(get_current_user_id),
    session: Session = Depends(get_session),
):
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    log.info("[analyze] user_id from token: %s", user_id)
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
                raise HTTPException(status_code=413, detail=VideoTooLongErrorResponse(
                    message=(
                        f"Video is too long ({duration:.1f}s). "
                        f"Maximum allowed duration is {_max_video_duration_seconds:.0f}s."
                    ),
                    durationSeconds=round(duration, 1),
                    maxDurationSeconds=_max_video_duration_seconds,
                ))

        job = Job(
            user_id=user_id,
            filename=video.filename,
            upload_path=tmp_path,
            status="pending",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        try:
            asyncio.create_task(
                job_runner.run_analysis(job.id, tmp_path, _analyzer)
            )
        except NoFaceDetectedError as e:
            Path(tmp_path).unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=NoFaceDetectedErrorResponse(message=e.message))
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise 

    return JobCreatedResponse(jobId=job.id, claimToken=job.claim_token)


@router.get("/jobs", response_model=List[JobStatusResponse])
def list_jobs(
    user_id: uuid.UUID = Depends(require_current_user_id),
    session: Session = Depends(get_session),
):
    jobs = session.exec(
        select(Job).where(Job.user_id == user_id).order_by(Job.created_at.desc())
    ).all()
    return [_job_to_response(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(
    job_id: uuid.UUID,
    claim_token: Optional[str] = None,
    user_id: Optional[uuid.UUID] = Depends(get_current_user_id),
    session: Session = Depends(get_session),
):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Access allowed if: authenticated owner OR valid claim_token.
    is_owner = user_id is not None and job.user_id == user_id
    has_claim = claim_token is not None and claim_token == job.claim_token
    if not is_owner and not has_claim:
        raise HTTPException(status_code=403, detail="Forbidden")

    return _job_to_response(job)
