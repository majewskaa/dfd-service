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
from pathlib import Path as _Path

from dotenv import load_dotenv
load_dotenv(_Path(__file__).parents[4] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.src.auth import jwt as jwt_utils
from service.src.db.session import init_db
from service.src.lab_service.video_analyzer import VideoAnalyzer
from service.src.router import router as api_router
from service.src.lab_service import video_analyzer_runner as job_runner

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "service.yaml"
_config_path: Path = Path(os.environ.get("DFD_CONFIG", str(_DEFAULT_CONFIG_PATH)))

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(_config_path) as f:
        config = yaml.safe_load(f)

    algorithm_configs: dict = config.pop("algorithms", {})
    config.pop("default_algorithm", None)

    uploads_base = (_config_path.parent / "data").resolve()

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        log.info("Using database from DATABASE_URL env var")
    else:
        db_path_raw = config.get("database", {}).get("path", "data/dfd_service.db")
        db_path = Path(db_path_raw)
        if not db_path.is_absolute():
            db_path = (_config_path.parent / db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        uploads_base = db_path.parent
        database_url = f"sqlite:///{db_path}"
        log.info("Using local SQLite database at %s", db_path)
    init_db(database_url)

    # ── auth ──────────────────────────────────────────────────────────────────
    auth_cfg = config.get("auth", {})
    jwt_utils.configure(
        secret_key=auth_cfg.get("jwt_secret", "change-me-in-production"),
        expire_minutes=auth_cfg.get("jwt_expire_minutes", 1440),
    )

    # ── email ─────────────────────────────────────────────────────────────────
    email_cfg = dict(config.get("email", {}))
    smtp_password = os.environ.get("SMTP_PASSWORD")
    if smtp_password:
        email_cfg["password"] = smtp_password
    job_runner.configure_smtp(email_cfg)

    # ── models — one VideoAnalyzer per algorithm, all must load ──────────────
    max_duration = config.get("evaluation", {}).get("max_video_duration_seconds")
    analyzers: dict = {}
    for algo_name, algo_cfg in algorithm_configs.items():
        merged = {**config, **algo_cfg}
        raw_ckpt = merged.get("evaluation", {}).get("checkpoint_path", "")
        ckpt = Path(raw_ckpt)
        if not ckpt.is_absolute():
            ckpt = (_config_path.parent / ckpt).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(
                f"algorithm '{algo_name}': checkpoint not found at '{ckpt}'"
            )
        analyzers[algo_name] = VideoAnalyzer(merged)
        log.info("algorithm '%s': model loaded from %s", algo_name, ckpt)

    # ── jobs router ───────────────────────────────────────────────────────────
    uploads_dir = str(uploads_base / "uploads")
    api_router.configure(
        analyzers=analyzers,
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

dfd_service.include_router(api_router.router)


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
