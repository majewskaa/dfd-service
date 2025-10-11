import os
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

import cv2
import librosa
import numpy as np

from src.data.shard_writer import ShardWriter


class DataPreprocessor(ABC):
    """Abstract base class for preprocessing deepfake detection data."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.face_detector = self.initialize_face_detector()
        self.sample_index = 0

    @staticmethod
    def initialize_face_detector():
        """Initialize OpenCV face detector."""
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    @staticmethod
    def get_video_fps(video_path: str) -> float:
        """Get frames-per-second (FPS) of the input video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        # Fallback to a sensible default if FPS could not be determined
        return float(fps if fps and fps > 0 else 25.0)

    @staticmethod
    def extract_frames(video_path: str) -> List[np.ndarray]:
        """Extract frames from video.

        Args:
            video_path: Path to the video file

        Returns:
            List of extracted frames
        """
        cap = cv2.VideoCapture(video_path)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames

    @staticmethod
    def extract_audio_mel(video_path: str, sr: int, n_mels: int, n_fft: int, hop_length: int) -> Optional[
        np.ndarray]:
        """Extract mel spectrogram using ffmpeg (PCM s16le) + librosa.

        Returns a numpy array [n_mels, time] in dB scale, or None on failure.
        """
        try:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(sr),
                "-",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            raw_audio = proc.stdout
            if not raw_audio:
                return None
            y = np.frombuffer(raw_audio, np.int16).astype(np.float32) / 32768.0
            if y.size == 0:
                return None
            mel_kwargs = {"y": y, "sr": sr, "n_mels": n_mels, "n_fft": n_fft, "hop_length": hop_length}
            S = librosa.feature.melspectrogram(**mel_kwargs)
            S_dB = librosa.power_to_db(S, ref=np.max)
            return S_dB.astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def extract_frame_level_mel(
            audio_mel_full: np.ndarray,
            start_frame: int,
            clip_len: int,
            fps: float,
            mel_per_second: float,
            overlap_ratio: float = 0.5
    ) -> np.ndarray:
        """Extract mel spectrograms for each frame in the clip with overlap for continuity.

        Args:
            audio_mel_full: Full mel spectrogram [n_mels, time_frames]
            start_frame: Starting frame index in the video
            clip_len: Number of frames in the clip
            fps: Frames per second of the video
            mel_per_second: Mel frames per second
            overlap_ratio: Ratio of overlap between consecutive mel spectrograms (0.0-1.0)

        Returns:
            Frame-level mel spectrograms [clip_len, n_mels, mel_frames_per_frame_with_overlap]
        """
        n_mels = audio_mel_full.shape[0]
        mel_frames_per_frame = max(1, int(round(mel_per_second / fps)))
        
        # Calculate overlap in mel frames
        overlap_frames = int(round(mel_frames_per_frame * overlap_ratio))
        overlap_frames = max(0, min(overlap_frames, mel_frames_per_frame - 1))
        
        # Calculate the total size including overlap (previous + current + next)
        mel_frames_per_frame_with_overlap = mel_frames_per_frame + 2 * overlap_frames

        # Initialize output array for frame-level mel spectrograms with overlap
        mel_frames = np.zeros((clip_len, n_mels, mel_frames_per_frame_with_overlap), dtype=np.float16)

        for i in range(clip_len):
            frame_idx = start_frame + i
            # Calculate time for this frame
            frame_time = frame_idx / fps
            # Map to mel spectrogram indices
            mel_start = int(np.floor(frame_time * mel_per_second))
            
            # Add overlap by extending the window backwards and forwards
            mel_start_with_overlap = mel_start - overlap_frames
            mel_end_with_overlap = mel_start + mel_frames_per_frame + overlap_frames

            # Ensure bounds
            mel_start_with_overlap = max(0, mel_start_with_overlap)
            mel_end_with_overlap = min(audio_mel_full.shape[1], mel_end_with_overlap)

            if mel_end_with_overlap > mel_start_with_overlap:
                # Extract mel spectrogram for this frame with overlap
                mel_frame = audio_mel_full[:, mel_start_with_overlap:mel_end_with_overlap]
                
                # Pad or truncate to ensure consistent shape
                if mel_frame.shape[1] < mel_frames_per_frame_with_overlap:
                    # Pad with zeros
                    pad_width = mel_frames_per_frame_with_overlap - mel_frame.shape[1]
                    mel_frame = np.pad(mel_frame, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                elif mel_frame.shape[1] > mel_frames_per_frame_with_overlap:
                    # Truncate
                    mel_frame = mel_frame[:, :mel_frames_per_frame_with_overlap]

                mel_frames[i] = mel_frame.astype(np.float16)

        return mel_frames

    @staticmethod
    def save_dataset_statistics(stats: Dict[str, Any], output_dir: str):
        """Save dataset statistics.

        Args:
            stats: Statistics dictionary
            output_dir: Directory to save statistics
        """
        import json
        stats_path = os.path.join(output_dir, 'dataset_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

    @abstractmethod
    def process_dataset(self, input_dir: str, output_dir: str):
        """Process entire dataset. Must be implemented by subclasses.

        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save processed data
        """
        pass

    def process_video(self, video_path: str, label: int) -> Dict[str, Any]:
        """Process a single video.

        Args:
            video_path: Path to the video file
            label: Label of the video (0 for real, 1 for fake)

        Returns:
            Dictionary containing processed data and metadata
        """
        # Extract frames and fps
        frames = self.extract_frames(video_path)
        fps = self.get_video_fps(video_path)

        # Process frames
        processed_frames = []
        for frame in frames:
            processed_frame = self._preprocess_frame(frame)
            if processed_frame is not None:
                processed_frames.append(processed_frame)

        if not processed_frames:
            return None

        result: Dict[str, Any] = {
            "data": np.stack(processed_frames),
            "label": label,
            "metadata": {
                "num_frames": len(processed_frames),
                "video_path": video_path,
                "fps": float(fps),
            },
        }

        # Optionally extract audio and compute full mel spectrogram (for later clip slicing)
        audio_cfg = self.config.get("preprocessing", {}).get("audio_processing", {})
        if audio_cfg and audio_cfg.get("enabled", False):
            sr = int(audio_cfg.get("sample_rate", 16000))
            n_mels = int(audio_cfg.get("n_mels", 80))
            n_fft = int(audio_cfg.get("n_fft", 2048))
            hop = int(audio_cfg.get("hop_length", 512))
            # Use librosa default upper frequency (Nyquist)
            mel_db = self.extract_audio_mel(video_path, sr, n_mels, n_fft, hop)
            if mel_db is not None and mel_db.size > 0:
                # Store as numpy float16 for compactness
                result["audio_mel_full"] = mel_db.astype(np.float16)
                result["metadata"].update({
                    "audio_sample_rate": sr,
                    "mel_hop_length": hop,
                    "mel_n_mels": n_mels,
                })

        return result

    def _detect_face(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Detect face in frame using OpenCV.

        Args:
            frame: Input frame

        Returns:
            Tuple of (cropped face, bounding box)
        """
        min_face_size = self.config["preprocessing"]["face_detection"]["min_face_size"]
        margin = self.config["preprocessing"]["face_detection"]["margin"]

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size)
        )

        if len(faces) == 0:
            return None, None

        x, y, w, h = faces[0]
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h

        # Add margin
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, int(x_min - width * margin))
        y_min = max(0, int(y_min - height * margin))
        x_max = min(frame.shape[1], int(x_max + width * margin))
        y_max = min(frame.shape[0], int(y_max + height * margin))

        face = frame[y_min:y_max, x_min:x_max]
        return face, (x_min, y_min, x_max, y_max)

    def _initialize_output_storage(self, output_dir: str):
        """Initialize shards-only storage."""
        wds_cfg = self.config["output"]["webdataset"]
        shard_dir = os.path.join(output_dir, 'shards')
        self.shard_writer = ShardWriter(
            output_dir=shard_dir,
            shard_prefix=wds_cfg.get("prefix", "shard"),
            max_shard_size_bytes=int(wds_cfg.get("max_shard_size_mb", 1024)) * 1024 * 1024,
            image_codec=wds_cfg.get("image_codec", "webp"),
            image_quality=int(wds_cfg.get("image_quality", 90)),
            index_filename=wds_cfg.get("index_filename", "index.csv"),
        )

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame.

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        # Detect and crop face
        face, bbox = self._detect_face(frame)
        if face is None:
            return None

        # Resize to target size
        target_size = self.config["preprocessing"]["image_processing"]["target_size"]
        face = cv2.resize(face, target_size)

        # Keep as uint8 for compact storage; defer normalization to loader
        return face

    def _save_incremental(self, result: Dict[str, Any], output_dir: str):
        """Save a single result incrementally (shards-only)."""
        # Save as clip samples with frame-level audio mel-spectrograms
        wds_cfg = self.config["output"]["webdataset"]
        clip_len = int(wds_cfg.get("clip_length", 16))
        clip_stride = int(wds_cfg.get("clip_stride", clip_len))
        frames: np.ndarray = result["data"]  # (T,H,W,3) uint8
        label = int(result["label"])
        meta_base = result["metadata"].copy()

        t = frames.shape[0]
        fps = float(meta_base.get("fps", 25.0))
        audio_mel_full = result.get("audio_mel_full")  # [n_mels, time] float16 np.ndarray or None
        audio_cfg = self.config.get("preprocessing", {}).get("audio_processing", {})
        sr = int(audio_cfg.get("sample_rate", 16000))
        hop = int(audio_cfg.get("hop_length", 512))

        # If audio present, compute mapping from frame indices to mel frame indices via time
        mel_per_second = sr / hop if hop > 0 else 0.0
        
        # Get overlap ratio from audio processing config
        overlap_ratio = float(audio_cfg.get("mel_overlap_ratio", 0.5))

        start = 0
        while start + clip_len <= t:
            clip = frames[start:start + clip_len]
            sample_id = f"{self.sample_index:06d}_{start:06d}"
            meta = meta_base.copy()
            meta["clip_start_frame"] = int(start)
            meta["clip_length"] = int(clip_len)

            # Extract frame-level mel spectrograms
            mel_frames: np.ndarray | None = None
            if audio_mel_full is not None and mel_per_second > 0 and fps > 0:
                mel_frames = self.extract_frame_level_mel(
                    audio_mel_full, start, clip_len, fps, mel_per_second, overlap_ratio
                )

            self.shard_writer.add_sample(sample_id, clip, label, meta, mel_frames=mel_frames)
            start += clip_stride
        self.sample_index += 1

    def _finalize_output_storage(self, output_dir: str):
        """Finalize shards storage and save any remaining data."""
        self.shard_writer.close()
        # After writing shards and index.csv, generate stratified train/val index files
        try:
            self._stratified_split_webdataset_indexes(output_dir)
        except Exception as e:
            print(f"Warning: failed to create train/val indexes: {e}")

    def _stratified_split_webdataset_indexes(self, output_dir: str) -> None:
        """Create index_train.csv and index_val.csv stratified by label.

        Uses data.train_split and data.val_split from config. The two ratios
        are renormalized to sum to 1.0. Randomness is controlled by data.random_seed.
        """
        import random

        shard_dir = os.path.join(output_dir, 'shards')
        wds_cfg = self.config["output"]["webdataset"]
        index_filename = wds_cfg.get("index_filename", "index.csv")
        index_path = os.path.join(shard_dir, index_filename)

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"WebDataset index not found: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            header = f.readline()
            lines = [line.rstrip("\n") for line in f]

        # Group by label (5th column: sample_id,shard,dir,num_frames,label,metadata)
        by_label: Dict[int, List[str]] = {}
        for line in lines:
            parts = line.split(",", 5)
            if len(parts) < 6:
                continue
            try:
                label = int(parts[4])
            except ValueError:
                # Skip malformed line
                continue
            by_label.setdefault(label, []).append(line)

        train_ratio_cfg = float(self.config["data"].get("train_split", 0.8))
        val_ratio_cfg = float(self.config["data"].get("val_split", 0.2))
        total = max(train_ratio_cfg + val_ratio_cfg, 1e-9)
        val_ratio = val_ratio_cfg / total
        seed = int(self.config["data"].get("random_seed", 42))
        rng = random.Random(seed)

        train_lines: List[str] = []
        val_lines: List[str] = []
        for _, group in by_label.items():
            rng.shuffle(group)
            n_val = int(round(len(group) * val_ratio))
            val_lines.extend(group[:n_val])
            train_lines.extend(group[n_val:])

        # Write outputs
        train_out = os.path.join(shard_dir, "index_train.csv")
        val_out = os.path.join(shard_dir, "index_val.csv")
        with open(train_out, "w", encoding="utf-8") as f:
            f.write(header)
            for line in train_lines:
                f.write(line + "\n")
        with open(val_out, "w", encoding="utf-8") as f:
            f.write(header)
            for line in val_lines:
                f.write(line + "\n")
        print(
            f"Created stratified indexes: {os.path.basename(train_out)} ({len(train_lines)}), {os.path.basename(val_out)} ({len(val_lines)})")

    def _update_statistics(self, stats: Dict[str, Any], metadata: Dict[str, Any]):
        """Update statistics with metadata from a processed sample.

        Args:
            stats: Current statistics dictionary
            metadata: Metadata from processed sample
        """
        # This is a default implementation that can be overridden by subclasses
        pass
