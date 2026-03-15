import sys
from pathlib import Path

# When this file is executed directly (python service.py), Python inserts the
# script's own directory into sys.path[0].  That makes `import service` resolve
# to this file instead of the `service` package under components/.  Fix by
# swapping that entry for the `components/` directory so all package imports
# work the same way whether the file is run as a script or imported by uvicorn.
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

from service.src.inference.video_analyzer import VideoAnalyzer
from service.src.schemas.analysis import AnalysisSegment

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "service.yaml"

# The config path is communicated across the uvicorn module re-import via an
# environment variable.  The __main__ block sets it before calling uvicorn.run.
_config_path: Path = Path(os.environ.get("DFD_CONFIG", str(_DEFAULT_CONFIG_PATH)))

_analyzer: VideoAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _analyzer
    with open(_config_path) as f:
        config = yaml.safe_load(f)

    raw_checkpoint = config.get("evaluation", {}).get("checkpoint_path", "")
    # Resolve relative paths against the config file's directory so the path
    # works regardless of where the service is launched from.
    checkpoint = Path(raw_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (_config_path.parent / checkpoint).resolve()

    if checkpoint.exists():
        try:
            _analyzer = VideoAnalyzer(config)
        except Exception as exc:
            print(f"WARNING: failed to load model — /analyze will return 503. Reason: {exc}")
    else:
        print(f"WARNING: checkpoint not found at '{checkpoint}' — /analyze will return 503.")
    yield
    _analyzer = None


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


@dfd_service.post("/analyze", response_model=list[AnalysisSegment])
async def analyze_video(video: UploadFile = File(...)):
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await video.read()
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(None, _analyzer.analyze, tmp_path)
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

    # Set env var BEFORE uvicorn.run — uvicorn re-imports this module in a fresh
    # interpreter context, so module-level variable assignments made here are
    # lost.  The env var survives and is picked up by the module-level read above.
    os.environ["DFD_CONFIG"] = str(Path(args.config).resolve())

    uvicorn.run(
        "service.src.main.service:dfd_service",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
