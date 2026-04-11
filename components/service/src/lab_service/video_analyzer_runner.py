"""
Background task that runs deepfake analysis and persists the result.

The task is fire-and-forget; it is started with asyncio.create_task() from the
/analyze endpoint.  All DB work uses a fresh Session so it is safe to run
concurrently with other requests.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from sqlmodel import Session, select

from service.src.db.models import Job, User
from service.src.db.session import get_engine
from service.src.lab_service.errors import NoFaceDetectedError

if TYPE_CHECKING:
    from service.src.lab_service.video_analyzer import VideoAnalyzer

log = logging.getLogger(__name__)

# Populated at startup (see service.py).
_smtp_config: Optional[dict] = None


def configure_smtp(cfg: dict) -> None:
    global _smtp_config
    _smtp_config = cfg


async def run_analysis(
    job_id: uuid.UUID,
    video_path: str,
    analyzer: "VideoAnalyzer",
) -> None:
    log.info("[runner] job=%s starting", job_id)

    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None:
            log.error("[runner] job=%s not found in DB", job_id)
            return
        job.status = "running"
        session.commit()

    try:
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(None, analyzer.analyze, video_path)
        result = json.dumps([s.model_dump(by_alias=True) for s in segments])
        _update_job(job_id, status="done", result_json=result)
        log.info("[runner] job=%s done  segments=%d", job_id, len(segments))
        await _maybe_notify(job_id)
    except NoFaceDetectedError as exc:
        log.warning("[runner] job=%s no face detected", job_id)
        error_json = json.dumps({
            "className": "NoFaceDetectedErrorResponse",
            "message": str(exc),
        })
        _update_job(job_id, status="failed", error=error_json)
    except Exception as exc:
        log.exception("[runner] job=%s failed", job_id)
        _update_job(job_id, status="failed", error=json.dumps({
            "className": "UnexpectedError",
            "message": str(exc),
        }))
    finally:
        Path(video_path).unlink(missing_ok=True)


def _update_job(job_id: uuid.UUID, **kwargs) -> None:
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        for key, value in kwargs.items():
            setattr(job, key, value)
        job.completed_at = datetime.now(timezone.utc)
        session.commit()


async def _maybe_notify(job_id: uuid.UUID) -> None:
    if not _smtp_config or not _smtp_config.get("enabled"):
        return

    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None or job.user_id is None:
            return
        user = session.get(User, job.user_id)
        if user is None or not user.notify_email:
            return
        email = user.email

    try:
        import aiosmtplib
        from email.mime.text import MIMEText

        cfg = _smtp_config
        msg = MIMEText(
            f"Your deepfake analysis (job {job_id}) has finished. "
            "Log in to review the results."
        )
        msg["Subject"] = "Your analysis is ready"
        msg["From"] = cfg["from_address"]
        msg["To"] = email

        await aiosmtplib.send(
            msg,
            hostname=cfg["host"],
            port=cfg.get("port", 587),
            username=cfg.get("username"),
            password=cfg.get("password"),
            use_tls=cfg.get("use_tls", False),
            start_tls=cfg.get("start_tls", True),
        )
        log.info("[runner] notification sent to %s for job=%s", email, job_id)
    except Exception:
        log.exception("[runner] failed to send notification for job=%s", job_id)
