"""
Deepfake video analysis for the service endpoint.

Replicates the preprocessing pipeline from components/lab so that raw MP4
files can be scored without going through the offline shard-writing step.
The model weights and configuration are loaded once at startup and reused
for every request.
"""

import importlib
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchaudio

log = logging.getLogger(__name__)

# Make lab source importable (src.* namespace mirrors the lab component layout)
# Path: components/service/src/inference/ → parents[3] = components/
_LAB_SRC = str(Path(__file__).parents[3] / "lab")
if _LAB_SRC not in sys.path:
    sys.path.insert(0, _LAB_SRC)

from src.models.base import BaseDetector  # noqa: E402
from src.training.lightning_module import DeepfakeTask  # noqa: E402

from service.src.schemas.analysis import AnalysisSegment  # noqa: E402

# Number of keyframes sampled to find a representative face bounding box.
_FACE_DETECTION_KEYFRAMES = 10
# Maximum clips to score in a single forward pass.
_INFERENCE_BATCH_SIZE = 8


class VideoAnalyzer:
    """Loads a trained deepfake-detection checkpoint and scores an MP4 file.

    The video is split into non-overlapping (or strided) clips of
    `frames_per_clip` frames.  Each clip is scored independently and the
    result is returned as a list of time-stamped AnalysisSegment objects.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._prep = config.get("preprocessing", {})

        face_cfg = self._prep.get("face_detection", {})
        self._min_face_size = int(face_cfg.get("min_face_size", 100))
        self._face_margin = float(face_cfg.get("margin", 0.2))

        img_cfg = self._prep.get("image_processing", {})
        self._target_size: Tuple[int, int] = tuple(img_cfg.get("target_size", [299, 299]))

        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        device_name = config["evaluation"].get("device", "cpu")
        self._device = torch.device("cpu" if device_name == "auto" else device_name)

        log.info("Loading checkpoint: %s", config["evaluation"]["checkpoint_path"])
        model = self._build_model()
        self._task = DeepfakeTask.load_from_checkpoint(
            config["evaluation"]["checkpoint_path"],
            model=model,
            config=config,
            strict=False,
        )
        self._task.eval()
        self._task.to(self._device)
        log.info("Model ready on device: %s", self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, video_path: str) -> List[AnalysisSegment]:
        """Score a video file and return per-clip deepfake probabilities."""
        t0 = time.perf_counter()
        log.info("[analyze] start  file=%s", video_path)

        fps = self._get_fps(video_path)

        log.info("[analyze] decoding frames …")
        frames = self._extract_frames(video_path)
        if not frames:
            log.warning("[analyze] no frames decoded — returning empty")
            return []
        log.info("[analyze] decoded %d frames at %.1f fps  (%.1fs)",
                 len(frames), fps, time.perf_counter() - t0)

        log.info("[analyze] detecting face box from keyframes …")
        face_box = self._detect_face_box(frames)
        if face_box is None:
            log.warning("[analyze] no face detected — returning empty")
            return []
        log.info("[analyze] face box %s  (%.1fs)", face_box, time.perf_counter() - t0)

        log.info("[analyze] cropping & resizing all frames …")
        cropped = [_crop_and_resize(f, face_box, self._target_size) for f in frames]

        log.info("[analyze] extracting audio mel spectrogram …")
        audio_mel_full = self._extract_audio_mel(video_path)
        if audio_mel_full is None:
            log.warning("[analyze] audio extraction failed — returning empty")
            return []
        log.info("[analyze] audio mel shape=%s  (%.1fs)",
                 audio_mel_full.shape, time.perf_counter() - t0)

        data_cfg = self._config["data"]
        frames_per_clip: int = data_cfg.get("frames_per_clip", 32)
        clip_stride: int = self._prep.get("clip_stride", frames_per_clip)

        audio_cfg = self._prep.get("audio_processing", {})
        sr = int(audio_cfg.get("sample_rate", 16000))
        hop = int(audio_cfg.get("hop_length", 256))
        mel_per_second = sr / hop
        overlap_ratio = float(audio_cfg.get("mel_overlap_ratio", 0.1))
        target_mel_size: Tuple[int, int] = tuple(audio_cfg.get("target_mel_size", [299, 299]))

        total_frames = len(cropped)
        starts = list(range(0, total_frames - frames_per_clip + 1, clip_stride))
        if not starts:
            log.warning("[analyze] video too short for even one clip — returning empty")
            return []

        n_clips = len(starts)
        n_batches = (n_clips + _INFERENCE_BATCH_SIZE - 1) // _INFERENCE_BATCH_SIZE
        log.info("[analyze] running inference: %d clips in %d batches …", n_clips, n_batches)

        segments: List[AnalysisSegment] = []
        for batch_idx, batch_starts in enumerate(_batched(starts, _INFERENCE_BATCH_SIZE)):
            log.debug("[analyze] batch %d/%d", batch_idx + 1, n_batches)
            video_batch, audio_batch = [], []
            for start in batch_starts:
                clip = cropped[start : start + frames_per_clip]
                video_batch.append(torch.from_numpy(np.stack(clip)))
                mel = _extract_frame_level_mel(
                    audio_mel_full, start, frames_per_clip,
                    fps, mel_per_second, overlap_ratio, target_mel_size,
                )
                audio_batch.append(torch.from_numpy(mel).float())

            batch = {
                "video_frames": torch.stack(video_batch).to(self._device),
                "audio_frames": torch.stack(audio_batch).to(self._device),
                "label": torch.zeros(len(batch_starts), dtype=torch.long, device=self._device),
            }

            with torch.no_grad():
                video, audio, _ = self._task._prepare_batch(batch)
                outputs = self._task(video, audio)
                probs = torch.softmax(outputs, dim=1)

            for i, start in enumerate(batch_starts):
                segments.append(AnalysisSegment(**{
                    "from": start / fps,
                    "to": (start + frames_per_clip) / fps,
                    "deepfakeProbability": float(probs[i, 1].item()),
                }))

        log.info("[analyze] done  segments=%d  total=%.1fs",
                 len(segments), time.perf_counter() - t0)
        return segments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> BaseDetector:
        model_cfg = self._config["model"]
        module = importlib.import_module(model_cfg["module_path"])
        ModelClass = getattr(module, model_cfg["name"])
        return ModelClass(
            num_classes=model_cfg["num_classes"],
            **(model_cfg.get("model_params") or {}),
        )

    def _get_fps(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 0 else 25.0

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Decode every frame as RGB. No face detection yet."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _detect_face_box(
        self, frames: List[np.ndarray]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Run face detection on a small set of keyframes and return the
        median bounding box (x_min, y_min, x_max, y_max).

        Sampling avoids running the expensive Haar cascade on every frame.
        """
        n = len(frames)
        keyframe_indices = np.linspace(0, n - 1, min(_FACE_DETECTION_KEYFRAMES, n), dtype=int)

        boxes = []
        for idx in keyframe_indices:
            box = _find_face_box(
                frames[idx], self._face_detector,
                self._min_face_size, self._face_margin,
            )
            if box is not None:
                boxes.append(box)

        if not boxes:
            return None

        median = np.median(np.array(boxes), axis=0).astype(int)
        return tuple(median)

    def _extract_audio_mel(self, video_path: str) -> Optional[np.ndarray]:
        audio_cfg = self._prep.get("audio_processing", {})
        if not audio_cfg.get("enabled", True):
            return None
        try:
            import imageio_ffmpeg

            sr = int(audio_cfg.get("sample_rate", 16000))
            n_mels = int(audio_cfg.get("n_mels", 64))
            n_fft = int(audio_cfg.get("n_fft", 1024))
            hop = int(audio_cfg.get("hop_length", 256))

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-i", video_path,
                "-f", "s16le", "-acodec", "pcm_s16le",
                "-ac", "1", "-ar", str(sr), "-",
            ]
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True
            )
            y = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
            if y.size == 0:
                return None

            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, win_length=n_fft,
                hop_length=hop, n_mels=n_mels, normalized=True,
            )
            S = mel_transform(torch.from_numpy(y).unsqueeze(0))
            S_dB = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)(S)
            return S_dB.squeeze(0).numpy().astype(np.float32)

        except Exception as exc:
            print(f"Warning: audio extraction failed for {video_path}: {exc}")
            return None


# ------------------------------------------------------------------
# Module-level pure functions (stateless, easier to test)
# ------------------------------------------------------------------


def _find_face_box(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
    min_face_size: int,
    margin: float,
) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    x_min = max(0, int(x - w * margin))
    y_min = max(0, int(y - h * margin))
    x_max = min(frame.shape[1], int(x + w * (1 + margin)))
    y_max = min(frame.shape[0], int(y + h * (1 + margin)))
    return x_min, y_min, x_max, y_max


def _crop_and_resize(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    x_min, y_min, x_max, y_max = box
    face = frame[y_min:y_max, x_min:x_max]
    return cv2.resize(face, target_size)


def _extract_frame_level_mel(
    audio_mel_full: np.ndarray,
    start_frame: int,
    clip_len: int,
    fps: float,
    mel_per_second: float,
    overlap_ratio: float,
    target_mel_size: Tuple[int, int],
) -> np.ndarray:
    """Build a per-frame mel-spectrogram slice tensor for one clip.

    Mirrors DataPreprocessor.extract_frame_level_mel so that the audio
    representation at inference time matches what the model was trained on.

    Returns shape: (clip_len, target_h, target_w) as float16.
    """
    mel_fpvf = max(1, int(round(mel_per_second / fps)))
    overlap = max(0, min(int(round(mel_fpvf * overlap_ratio)), mel_fpvf - 1))
    total_mel_w = mel_fpvf + 2 * overlap

    target_h, target_w = target_mel_size
    result = np.zeros((clip_len, target_h, target_w), dtype=np.float16)

    for i in range(clip_len):
        t_sec = (start_frame + i) / fps
        center = int(np.floor(t_sec * mel_per_second))
        m_start = max(0, center - overlap)
        m_end = min(audio_mel_full.shape[1], center + mel_fpvf + overlap)

        if m_end <= m_start:
            continue

        sl = audio_mel_full[:, m_start:m_end]
        if sl.shape[1] < total_mel_w:
            sl = np.pad(sl, ((0, 0), (0, total_mel_w - sl.shape[1])))
        elif sl.shape[1] > total_mel_w:
            sl = sl[:, :total_mel_w]

        result[i] = _resize_mel(sl, target_h, target_w)

    return result


def _resize_mel(mel: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    mel_min, mel_max = mel.min(), mel.max()
    rng = mel_max - mel_min
    if rng > 1e-8:
        uint8 = ((mel - mel_min) / rng * 255).astype(np.uint8)
        resized = cv2.resize(uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return (resized.astype(np.float32) / 255.0 * rng + mel_min).astype(np.float16)
    return np.full((target_h, target_w), mel_min, dtype=np.float16)


def _batched(items: list, size: int):
    """Yield successive chunks of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
