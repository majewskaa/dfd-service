import sys
from pathlib import Path

_COMPONENTS_DIR = str(Path(__file__).parents[3])  # …/components/
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).parent.resolve():
    sys.path[0] = _COMPONENTS_DIR
elif _COMPONENTS_DIR not in sys.path:
    sys.path.insert(0, _COMPONENTS_DIR)

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from service.src.inference.video_analyzer import VideoAnalyzer, NoFaceDetected
from service.src.schemas.analysis import AnalysisSegment, VideoTooLongError, NoFaceDetectedError
from service.src.main.helpers.response_helper import get_video_duration

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "service.yaml"

_config_path: Path = Path(os.environ.get("DFD_CONFIG", str(_DEFAULT_CONFIG_PATH)))

_analyzer: VideoAnalyzer | None = None
_max_video_duration_seconds: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _analyzer, _max_video_duration_seconds
    with open(_config_path) as f:
        config = yaml.safe_load(f)

    raw_checkpoint = config.get("evaluation", {}).get("checkpoint_path", "")
    checkpoint = Path(raw_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (_config_path.parent / checkpoint).resolve()

    _max_video_duration_seconds = config.get("evaluation", {}).get("max_video_duration_seconds")

    if checkpoint.exists():
        try:
            _analyzer = VideoAnalyzer(config)
        except Exception as exc:
            print(f"WARNING: failed to load model — /analyze will return 503. Reason: {exc}")
    else:
        print(f"WARNING: checkpoint not found at '{checkpoint}' — /analyze will return 503.")
    yield
    _analyzer = None
    _max_video_duration_seconds = None


dfd_service = FastAPI(lifespan=lifespan)

dfd_service.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dfd_service.get("/health")
def healthcheck():
    return {"message": "Hello, World!"}


@dfd_service.post(
    "/analyze",
    response_model=list[AnalysisSegment],
    responses={
        413: {"model": VideoTooLongError},
        422: {"model": NoFaceDetectedError},
    },
)
async def analyze_video(video: UploadFile = File(...)):
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await video.read()
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        if _max_video_duration_seconds is not None:
            duration = get_video_duration(tmp_path)
            if duration > _max_video_duration_seconds:
                return JSONResponse(
                    status_code=413,
                    content=VideoTooLongError(
                        message=f"Video is too long ({duration:.1f}s). Maximum allowed duration is {_max_video_duration_seconds:.0f}s.",
                        durationSeconds=round(duration, 1),
                        maxDurationSeconds=_max_video_duration_seconds,
                    ).model_dump(),
                )

        try:
            loop = asyncio.get_event_loop()
            segments = await loop.run_in_executor(None, _analyzer.analyze, tmp_path)
        except NoFaceDetected as exc:
            return JSONResponse(
                status_code=422,
                content=NoFaceDetectedError(message=str(exc)).model_dump(),
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return segments


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
