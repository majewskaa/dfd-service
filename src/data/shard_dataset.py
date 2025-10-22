import json
import os
import tarfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset


def _decode_image(buf: bytes) -> np.ndarray:
    """Decode image buffer to RGB numpy array."""
    nparr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode image from buffer")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class ShardClipDataset(IterableDataset):
    """Streams tar shards of video samples with frames and metadata.

    Yields batches with:
    - video_frames: Tensor[batch_size, T, H, W, 3] - video frames
    - label: Tensor[batch_size] - binary labels
    - metadata: List[dict] - sample metadata
    - audio_frames: Tensor[batch_size, T, n_mels, mel_frames_per_frame] (optional)
    """

    def __init__(
            self,
            shards_dir: str,
            index_filename: str = "index.csv",
            target_device: str = "cpu",
            max_samples: Optional[int] = None,
            batch_size: int = 1,
    ):
        super().__init__()
        self.shards_dir = shards_dir
        self.index_path = os.path.join(shards_dir, index_filename)
        self.target_device = target_device
        self.max_samples = max_samples
        self.batch_size = batch_size

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        self._entries = self._load_index()

    def __iter__(self):
        """Iterate through batches of samples."""
        for shard_name, entries in self._group_entries_by_shard().items():
            shard_path = os.path.join(self.shards_dir, shard_name)

            with tarfile.open(shard_path, mode="r") as tar:
                members = {m.name: m for m in tar.getmembers()}

                batch = []
                for sample_id, _, sample_dir, num_frames, label, meta_json in entries:
                    sample = self._load_sample(tar, members, sample_id, sample_dir, num_frames, label, meta_json)
                    batch.append(sample)

                    if len(batch) == self.batch_size:
                        yield ShardClipDataset._collate_batch(batch)
                        batch = []

                if batch:
                    yield ShardClipDataset._collate_batch(batch)

    def _load_index(self) -> List[Tuple[str, str, str, int, int, str]]:
        """Load and parse the CSV index file."""
        entries = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.rstrip("\n").split(",", 5)
                if len(parts) == 6:
                    sample_id, shard, sample_dir, num_frames, label, meta_json = parts
                    entries.append((sample_id, shard, sample_dir, int(num_frames), int(label), meta_json))
        return entries

    def _group_entries_by_shard(self) -> Dict[str, List[Tuple[str, str, str, int, int, str]]]:
        """Group entries by shard to minimize tar file operations."""
        entries = self._entries[:self.max_samples] if self.max_samples else self._entries
        shard_to_entries = {}
        for entry in entries:
            shard_to_entries.setdefault(entry[1], []).append(entry)
        return shard_to_entries

    def _load_sample(
            self,
            tar: tarfile.TarFile,
            members: Dict[str, tarfile.TarInfo],
            sample_id: str,
            sample_dir: str,
            num_frames: int,
            label: int,
            meta_json: str
    ) -> Dict[str, Any]:
        """Load a single sample from the tar file."""
        # Load video frames
        frames = []
        for i in range(num_frames):
            for ext in (".webp", ".jpg"):
                frame_name = f"{sample_dir}/frame_{i:03d}{ext}"
                if frame_name in members:
                    buf = tar.extractfile(members[frame_name]).read()
                    frames.append(_decode_image(buf))
                    break
            else:
                raise FileNotFoundError(f"Missing frame {i} for {sample_id}")

        # Load metadata
        meta = json.loads(meta_json)
        meta.update({"id": sample_id, "num_frames": num_frames})

        # Load audio features (optional)
        audio_mel = ShardClipDataset._load_audio_features(tar, members, sample_dir)

        # Create sample
        sample = {
            "video_frames": torch.from_numpy(np.stack(frames).astype(np.uint8)).to(self.target_device),
            "label": label,
            "metadata": meta,
        }

        if audio_mel is not None:
            sample["audio_frames"] = audio_mel.to(self.target_device)

        return sample

    @staticmethod
    def _load_audio_features(
            tar: tarfile.TarFile,
            members: Dict[str, tarfile.TarInfo],
            sample_dir: str
    ) -> Optional[torch.Tensor]:
        """Load optional audio mel features."""
        shape_file = f"{sample_dir}/audio_mel_frames.json"
        data_file = f"{sample_dir}/audio_mel_frames.f16"

        if shape_file not in members or data_file not in members:
            return None

        try:
            # Load shape info
            shape_info = json.loads(tar.extractfile(members[shape_file]).read().decode("utf-8"))
            shape = shape_info.get("shape", [])

            if len(shape) != 3:
                return None

            # Load data
            data_bytes = tar.extractfile(members[data_file]).read()
            data = np.frombuffer(data_bytes, dtype=np.float16)

            # Reshape and convert to tensor
            expected_size = shape[0] * shape[1] * shape[2]
            if data.size == expected_size:
                # Make array writable before converting to tensor
                data_reshaped = data.reshape(shape).copy()
                return torch.from_numpy(data_reshaped)
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            # Log specific error for debugging if needed
            print(f"Failed to load audio features: {e}")
            pass

        return None

    @staticmethod
    def _collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate samples into a batch."""
        if not samples:
            raise ValueError("Cannot collate empty batch")

        # Stack video data
        video_data = torch.stack([sample["video_frames"] for sample in samples])
        labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
        metadata = [sample["metadata"] for sample in samples]

        batch = {
            "video_frames": video_data,
            "label": labels,
            "metadata": metadata,
        }

        # Stack audio features if all samples have them
        if all("audio_frames" in sample for sample in samples):
            batch["audio_frames"] = torch.stack([sample["audio_frames"] for sample in samples])

        return batch
