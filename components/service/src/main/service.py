import sys
from pathlib import Path

_COMPONENTS_DIR = str(Path(__file__).parents[3])  # …/components/
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).parent.resolve():
    sys.path[0] = _COMPONENTS_DIR
elif _COMPONENTS_DIR not in sys.path:
    sys.path.insert(0, _COMPONENTS_DIR)

import logging
import os
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.src.auth import jwt as jwt_utils
from service.src.auth.router import router as auth_router
from service.src.db.session import init_db
from service.src.inference.video_analyzer import VideoAnalyzer
from service.src.jobs import router as jobs_router
from service.src.jobs import runner as job_runner

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "service.yaml"
_config_path: Path = Path(os.environ.get("DFD_CONFIG", str(_DEFAULT_CONFIG_PATH)))

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(_config_path) as f:
        config = yaml.safe_load(f)

    # ── database ──────────────────────────────────────────────────────────────
    db_path_raw = config.get("database", {}).get("path", "data/dfd_service.db")
    db_path = Path(db_path_raw)
    if not db_path.is_absolute():
        db_path = (_config_path.parent / db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(f"sqlite:///{db_path}")

    # ── auth ──────────────────────────────────────────────────────────────────
    auth_cfg = config.get("auth", {})
    jwt_utils.configure(
        secret_key=auth_cfg.get("jwt_secret", "change-me-in-production"),
        expire_minutes=auth_cfg.get("jwt_expire_minutes", 1440),
    )

    # ── email ─────────────────────────────────────────────────────────────────
    job_runner.configure_smtp(config.get("email", {}))

    # ── model ─────────────────────────────────────────────────────────────────
    raw_checkpoint = config.get("evaluation", {}).get("checkpoint_path", "")
    checkpoint = Path(raw_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (_config_path.parent / checkpoint).resolve()

    max_duration = config.get("evaluation", {}).get("max_video_duration_seconds")

    analyzer = None
    if checkpoint.exists():
        try:
            analyzer = VideoAnalyzer(config)
            log.info("Model loaded from %s", checkpoint)
        except Exception as exc:
            log.warning("failed to load model — /analyze will return 503. Reason: %s", exc)
    else:
        log.warning("checkpoint not found at '%s' — /analyze will return 503.", checkpoint)

    # ── jobs router ───────────────────────────────────────────────────────────
    uploads_dir = str(db_path.parent / "uploads")
    jobs_router.configure(
        analyzer=analyzer,
        max_duration=max_duration,
        uploads_dir=uploads_dir,
    )

    yield

    log.info("Shutting down")


dfd_service = FastAPI(lifespan=lifespan)

dfd_service.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dfd_service.include_router(auth_router)
dfd_service.include_router(jobs_router.router)


@dfd_service.get("/health")
def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="DFD deepfake detection service")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG_PATH),
                        help="Path to service.yaml config file")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["DFD_CONFIG"] = str(Path(args.config).resolve())

    uvicorn.run(
        "service.src.main.service:dfd_service",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
