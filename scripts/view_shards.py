import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch

# Ensure project root is on sys.path so 'src' is importable when run from anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.shard_dataset import ShardClipDataset


def draw_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    overlay = frame.copy()
    y0 = 24
    cv2.rectangle(overlay, (5, 5), (5 + 520, 5 + 28), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(
        frame,
        text,
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def play_clip(clip: torch.Tensor, title: str, fps: int = 20) -> None:
    # clip: Tensor[T,H,W,3], uint8 on any device
    t = int(1000 / max(1, fps))
    clip_np = clip.detach().cpu().numpy()
    for i, frame in enumerate(clip_np):
        frame = frame.astype(np.uint8)
        bgr = to_bgr(frame)
        info = f"{title} | frame {i + 1}/{len(clip_np)} | q:quit n:next p:pause"
        bgr = draw_overlay(bgr, info)
        cv2.imshow("shard_viewer", bgr)
        key = cv2.waitKey(t) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        if key == ord('p'):
            # pause until any key
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 in (ord('p'), ord('n'), ord('q')):
                    if key2 == ord('q'):
                        raise KeyboardInterrupt
                    break


def main(shards_dir: str, index_filename: str, device: str, max_samples: Optional[int], fps: int) -> int:
    if not os.path.isdir(shards_dir):
        print(f"Shards directory not found: {shards_dir}")
        return 1

    ds = ShardClipDataset(
        shards_dir=shards_dir,
        index_filename=index_filename,
        target_device=device,
        max_samples=max_samples,
    )

    print(f"Loaded index with {len(ds._entries)} entries from {ds.index_path}")

    try:
        for idx, sample in enumerate(ds):
            has_audio = "audio_mel_frames" in sample
            data: torch.Tensor = sample["data"]
            label = int(sample["label"]) if not isinstance(sample["label"], torch.Tensor) else int(
                sample["label"].item())
            meta = sample.get("metadata", {})
            print(sample)
            title = f"sample {meta.get('id', idx)} | label={label} | frames={meta.get('num_frames', data.shape[0])} | audio={'yes' if has_audio else 'no'}"
            play_clip(data, title=title, fps=fps)
            # After clip ends, wait for next or quit
            print(title)
            print("Press 'n' for next, 'q' to quit, 'p' to replay")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
                if key == ord('n'):
                    break
                if key == ord('p'):
                    play_clip(data, title=title, fps=fps)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View processed shard samples")
    parser.add_argument("shards_dir", type=str, help="Directory containing tar shards and index.csv")
    parser.add_argument("--index", type=str, default="index.csv", help="Index CSV filename")
    parser.add_argument("--device", type=str, default="cpu", help="Target device for tensors: cpu or cuda")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to view")
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS")
    args = parser.parse_args()

    raise SystemExit(
        main(
            shards_dir=args.shards_dir,
            index_filename=args.index,
            device=args.device,
            max_samples=args.max_samples,
            fps=args.fps,
        )
    )
