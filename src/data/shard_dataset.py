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

    Yields individual samples (not batches):
    - video_frames: Tensor[T, H, W, 3] - video frames
    - label: int - binary label
    - metadata: dict - sample metadata
    - audio_frames: Tensor[T, n_mels, mel_frames_per_frame] (optional)
    """

    def __init__(
            self,
            shards_dir: str,
            index_filename: str = "index.csv",
            target_device: str = "cpu",
            max_samples: Optional[int] = None,
            frames_per_clip: int = 32,
    ):
        super().__init__()
        self.shards_dir = shards_dir
        self.index_path = os.path.join(shards_dir, index_filename)
        self.target_device = target_device
        self.max_samples = max_samples
        self.frames_per_clip = frames_per_clip

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        self._entries = self._load_index()

    def __len__(self) -> int:
        """Return the number of samples."""
        if self.max_samples:
            return min(len(self._entries), self.max_samples)
        return len(self._entries)

    def __iter__(self):
        """Iterate through samples."""
        worker_info = torch.utils.data.get_worker_info()
        
        # Group entries by shard
        shard_to_entries = self._group_entries_by_shard()
        shards = list(shard_to_entries.keys())
        
        # Shuffle shards to randomize order
        import random
        random.shuffle(shards)
        
        # Handle multi-processing
        if worker_info is not None:
            # Split shards among workers
            per_worker = int(np.ceil(len(shards) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(shards))
            shards = shards[iter_start:iter_end]
        
        for shard_name in shards:
            entries = shard_to_entries[shard_name]
            # Shuffle entries within the shard
            random.shuffle(entries)
            
            shard_path = os.path.join(self.shards_dir, shard_name)
            
            with tarfile.open(shard_path, mode="r") as tar:
                # Lazy member lookup: only create dict when needed, use getmember() for individual lookups
                # This avoids loading all members into memory at once
                _members_cache = {}

                for sample_id, _, sample_dir, num_frames, label, meta_json in entries:
                    try:
                        sample = self._load_sample(tar, _members_cache, sample_id, sample_dir, num_frames, label, meta_json)
                        yield sample
                    except Exception as e:
                        print(f"Error loading sample {sample_id}: {e}")
                        continue
                        
                    # Clear cache periodically to free memory
                    if len(_members_cache) > 1000:
                        _members_cache.clear()

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

    def _sample_frame_indices(self, num_frames: int) -> np.ndarray:
        """Sample indices for frames."""
        if num_frames <= self.frames_per_clip:
            return np.arange(num_frames)
        
        # Uniform sampling
        indices = np.linspace(0, num_frames - 1, self.frames_per_clip, dtype=int)
        return indices

    def _load_sample(
            self,
            tar: tarfile.TarFile,
            members_cache: Dict[str, tarfile.TarInfo],
            sample_id: str,
            sample_dir: str,
            num_frames: int,
            label: int,
            meta_json: str
    ) -> Dict[str, Any]:
        """Load a single sample from the tar file."""
        # Determine which frames to load
        frame_indices = self._sample_frame_indices(num_frames)
        actual_num_frames = len(frame_indices)
        
        # Load video frames directly into pre-allocated array
        frames_array = None
        
        for i, frame_idx in enumerate(frame_indices):
            frame_found = False
            for ext in (".webp", ".jpg"):
                frame_name = f"{sample_dir}/frame_{frame_idx:03d}{ext}"
                # Lazy member lookup with caching
                if frame_name not in members_cache:
                    try:
                        member = tar.getmember(frame_name)
                        members_cache[frame_name] = member
                    except KeyError:
                        continue
                
                member = members_cache[frame_name]
                buf = tar.extractfile(member).read()
                decoded = _decode_image(buf)
                
                if frames_array is None:
                    # First frame: get dimensions and pre-allocate array
                    H, W, C = decoded.shape
                    frames_array = np.empty((actual_num_frames, H, W, C), dtype=np.uint8)
                    frames_array[0] = decoded
                else:
                    # Subsequent frames: fill pre-allocated array
                    frames_array[i] = decoded
                
                frame_found = True
                break
            
            if not frame_found:
                raise FileNotFoundError(f"Missing frame {frame_idx} for {sample_id}")
        
        # Load metadata
        meta = json.loads(meta_json)
        meta.update({"id": sample_id, "num_frames": num_frames, "sampled_frames": actual_num_frames})

        # Load audio features
        audio_mel = ShardClipDataset._load_audio_features(tar, members_cache, sample_dir)
        
        # If audio exists, we need to sample it too to match video temporal dimension if needed
        # For now, we assume audio features are already aligned or model handles it
        # But if audio has temporal dimension matching video frames, we should sample it
        if audio_mel is not None and audio_mel.shape[0] == num_frames:
             audio_mel = audio_mel[frame_indices]

        # Convert directly from pre-allocated array to tensor (single copy)
        sample = {
            "video_frames": torch.from_numpy(frames_array).to(self.target_device),
            "label": label,
            "metadata": meta,
        }

        if audio_mel is not None:
            sample["audio_frames"] = audio_mel.to(self.target_device)

        return sample

    @staticmethod
    def _load_audio_features(
            tar: tarfile.TarFile,
            members_cache: Dict[str, tarfile.TarInfo],
            sample_dir: str
    ) -> Optional[torch.Tensor]:
        """Load optional audio mel features."""
        shape_file = f"{sample_dir}/audio_mel_frames.json"
        data_file = f"{sample_dir}/audio_mel_frames.f16"

        # Lazy member lookup with caching
        try:
            if shape_file not in members_cache:
                members_cache[shape_file] = tar.getmember(shape_file)
            if data_file not in members_cache:
                members_cache[data_file] = tar.getmember(data_file)
        except KeyError:
            return None

        try:
            # Load shape info
            shape_info = json.loads(tar.extractfile(members_cache[shape_file]).read().decode("utf-8"))
            shape = shape_info.get("shape", [])

            if len(shape) != 3:
                return None

            # Load data
            data_bytes = tar.extractfile(members_cache[data_file]).read()
            data = np.frombuffer(data_bytes, dtype=np.float16)

            # Reshape and convert to tensor
            expected_size = shape[0] * shape[1] * shape[2]
            if data.size == expected_size:
                # Make array writable before converting to tensor
                data_reshaped = data.reshape(shape).copy()
                return torch.from_numpy(data_reshaped)
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            print(f"Failed to load audio features: {e}")

        return None

    @staticmethod
    def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate samples into a batch."""
        if not samples:
            raise ValueError("Cannot collate empty batch")

        video_data = torch.stack([sample["video_frames"] for sample in samples])
        labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
        metadata = [sample["metadata"] for sample in samples]

        batch = {
            "video_frames": video_data,
            "label": labels,
            "metadata": metadata,
        }

        if all("audio_frames" in sample for sample in samples):
            batch["audio_frames"] = torch.stack([sample["audio_frames"] for sample in samples])

        return batch
