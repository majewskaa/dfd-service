"""
ModelBackend — the seam between VideoAnalyzer and any concrete scoring model.

Adding a new model means writing a class that satisfies ModelBackend.score_batch
and wiring it into VideoAnalyzer via dependency injection.

Included implementations
------------------------
DeepfakeTaskBackend   – wraps the existing PyTorch-Lightning DeepfakeTask.
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import List, Protocol, runtime_checkable

import torch

log = logging.getLogger(__name__)

# Make lab source importable (mirrors the path setup in video_analyzer.py)
_LAB_SRC = str(Path(__file__).parents[3] / "lab")
if _LAB_SRC not in sys.path:
    sys.path.insert(0, _LAB_SRC)

from src.models.base import BaseDetector
from src.training.lightning_module import DeepfakeTask


@runtime_checkable
class ModelBackend(Protocol):
    """Scoring contract that every backend must satisfy.

    A backend is responsible for everything model-specific:
      - how a batch dict is preprocessed for this model
      - how the forward pass is invoked
      - how raw outputs are converted to a deepfake probability in [0, 1]

    Parameters
    ----------
    batch:
        Dict with optional keys ``"video_frames"`` (B, T, H, W, C) and
        ``"audio_frames"`` (B, T, H, W), plus a ``"label"`` key that some
        Lightning modules require during the forward pass.

    Returns
    -------
    List[float]
        One probability per item in the batch, in the same order.
    """

    def score_batch(self, batch: dict) -> List[float]: ...


class DeepfakeTaskBackend:
    """ModelBackend backed by a PyTorch-Lightning DeepfakeTask checkpoint.

    This is a direct extraction of the scoring logic that previously lived
    inside VideoAnalyzer._score_clips.  Softmax over two logits is assumed;
    index 1 is the "fake" class.
    """

    def __init__(self, config: dict, device: torch.device) -> None:
        self._device = device
        model = self._build_model(config)
        log.info("Loading checkpoint: %s", config["evaluation"]["checkpoint_path"])
        self._task = DeepfakeTask.load_from_checkpoint(
            config["evaluation"]["checkpoint_path"],
            model=model,
            config=config,
            strict=False,
        )
        self._task.eval()
        self._task.to(device)
        log.info("DeepfakeTaskBackend ready on device: %s", device)

    # ------------------------------------------------------------------
    # ModelBackend implementation
    # ------------------------------------------------------------------

    def score_batch(self, batch: dict) -> List[float]:
        with torch.no_grad():
            video, audio, _ = self._task._prepare_batch(batch)
            logits = self._task(video, audio)
        probs = torch.softmax(logits, dim=1)
        return [float(probs[i, 1].item()) for i in range(probs.size(0))]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(config: dict) -> BaseDetector:
        model_cfg = config["model"]
        module = importlib.import_module(model_cfg["module_path"])
        ModelClass = getattr(module, model_cfg["name"])
        return ModelClass(
            num_classes=model_cfg["num_classes"],
            **(model_cfg.get("model_params") or {}),
        )
