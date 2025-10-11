import io
import json
import os
import tarfile
import time
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np


class ShardWriter:
    """Writer that packs samples into tar shards with compressed image frames.

    Each sample is stored as a directory inside the tar with files:
      - frame_000.webp (or .jpg), frame_001.webp, ...
      - meta.json

    Rotation happens before writing a sample so a sample never spans shards.
    """

    def __init__(
            self,
            output_dir: str,
            shard_prefix: str,
            max_shard_size_bytes: int = 1024 * 1024 * 1024,
            image_codec: str = "webp",
            image_quality: int = 90,
            index_filename: str = "index.csv",
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.shard_prefix = shard_prefix
        self.max_shard_size_bytes = max_shard_size_bytes
        self.image_codec = image_codec.lower()
        self.image_quality = int(image_quality)
        self.index_path = os.path.join(self.output_dir, index_filename)

        # State
        self._shard_idx = 0
        self._tar = None  # type: ignore
        self._tar_path = None  # type: ignore
        self._bytes_written = 0

        # Prepare index file header if new
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w", encoding="utf-8") as f:
                f.write("sample_id,shard,dir,num_frames,label,metadata\n")

        self._open_new_shard()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _build_metadata(sample_id, frames_rgb, label, sample_metadata):
        meta = {
            "id": sample_id,
            "label": int(label),
            "num_frames": int(frames_rgb.shape[0]),
        }
        meta.update(sample_metadata or {})
        return meta

    @staticmethod
    def _estimate_mel_size(mel_frames: np.ndarray) -> int:
        mel_bytes = mel_frames.astype(np.float16).tobytes(order="C")
        return len(mel_bytes) + 128  # header estimate

    def close(self):
        if self._tar is not None:
            self._tar.close()
            self._tar = None
            self._tar_path = None

    def add_sample(
            self,
            sample_id: str,
            frames_rgb: np.ndarray,
            label: int,
            sample_metadata: Dict[str, Any],
            *,
            mel_frames: np.ndarray | None = None,
    ) -> None:
        """Add a sample consisting of frames and metadata."""
        assert frames_rgb.dtype == np.uint8 and frames_rgb.ndim == 4 and frames_rgb.shape[-1] == 3

        meta = self._build_metadata(sample_id, frames_rgb, label, sample_metadata)

        estimated_size, buffers, ext = self._estimate_sample_size(frames_rgb, meta)
        if mel_frames is not None:
            estimated_size += self._estimate_mel_size(mel_frames)

        self._rotate_if_needed(estimated_size)

        sample_dir = f"sample_{sample_id}"
        mtime = int(time.time())

        self._write_frames(sample_dir, buffers, ext, mtime)
        self._write_meta(sample_dir, meta, mtime)

        if mel_frames is not None:
            self._write_mel(sample_dir, mel_frames, mtime)

        self._update_index(sample_id, sample_dir, frames_rgb, label, meta)

    def _open_new_shard(self):
        self.close()
        shard_name = f"{self.shard_prefix}_{self._shard_idx:05d}.tar"
        path = os.path.join(self.output_dir, shard_name)
        # Use non-compressed tar for maximum read throughput; images are already compressed
        self._tar = tarfile.open(path, mode="w")
        self._tar_path = path
        self._bytes_written = 0
        self._shard_idx += 1

    def _encode_frame(self, frame_rgb: np.ndarray) -> Tuple[bytes, str]:
        # Convert RGB -> BGR for OpenCV encoding
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self.image_codec == "webp":
            ext = ".webp"
            params = [cv2.IMWRITE_WEBP_QUALITY, self.image_quality]
        else:
            # default to JPEG
            ext = ".jpg"
            params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
        ok, buf = cv2.imencode(ext, bgr, params)
        if not ok:
            raise RuntimeError("Failed to encode frame")
        return buf.tobytes(), ext

    def _estimate_sample_size(self, frames_rgb: np.ndarray, metadata: Dict[str, Any]) -> Tuple[int, List[bytes], str]:
        # Pre-encode frames to know total size and reuse buffers for write
        buffers: List[bytes] = []
        ext = None
        total = 0
        for i in range(frames_rgb.shape[0]):
            data, ext = self._encode_frame(frames_rgb[i])
            buffers.append(data)
            total += len(data)
        meta_bytes = json.dumps(metadata).encode("utf-8")
        total += len(meta_bytes)
        return total, buffers, ext or ".jpg"

    def _rotate_if_needed(self, estimated_size: int):
        if self._bytes_written + estimated_size > self.max_shard_size_bytes:
            self._open_new_shard()

    def _write_frames(self, sample_dir: str, buffers: List[bytes], ext: str, mtime: int):
        for i, data in enumerate(buffers):
            name = f"{sample_dir}/frame_{i:03d}{ext}"
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = mtime
            self._tar.addfile(info, io.BytesIO(data))
            self._bytes_written += info.size

    def _write_meta(self, sample_dir: str, meta: Dict[str, Any], mtime: int):
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        info = tarfile.TarInfo(name=f"{sample_dir}/meta.json")
        info.size = len(meta_bytes)
        info.mtime = mtime
        self._tar.addfile(info, io.BytesIO(meta_bytes))
        self._bytes_written += info.size

    def _write_mel(self, sample_dir: str, mel_frames: np.ndarray, mtime: int):
        arr = mel_frames.astype(np.float16, copy=False)
        raw = arr.tobytes(order="C")
        info = tarfile.TarInfo(name=f"{sample_dir}/audio_mel_frames.f16")
        info.size = len(raw)
        info.mtime = mtime
        self._tar.addfile(info, io.BytesIO(raw))
        self._bytes_written += info.size

        shape = {"shape": list(arr.shape), "dtype": "float16", "layout": "[frames,mels,mel_frames_per_frame]"}
        shape_bytes = json.dumps(shape).encode("utf-8")
        info = tarfile.TarInfo(name=f"{sample_dir}/audio_mel_frames.json")
        info.size = len(shape_bytes)
        info.mtime = mtime
        self._tar.addfile(info, io.BytesIO(shape_bytes))
        self._bytes_written += info.size

    def _update_index(self, sample_id, sample_dir, frames_rgb, label, meta):
        rel_shard = os.path.basename(self._tar_path)
        with open(self.index_path, "a", encoding="utf-8") as f:
            meta_copy = dict(meta)
            meta_copy.pop("num_frames", None)
            f.write(
                f"{sample_id},{rel_shard},{sample_dir},{frames_rgb.shape[0]},{label},{json.dumps(meta_copy, ensure_ascii=False)}\n"
            )
