# WebDataset Implementation in Deepfake Detection Pipeline

This document describes the WebDataset implementation used in the deepfake detection preprocessing pipeline. The implementation provides efficient storage and streaming of video clips with associated metadata and optional audio features.

## Overview

The WebDataset implementation consists of three main components:

1. **ShardWriter** - Writes processed video clips to compressed tar shards
2. **ShardClipDataset** - Streams data from tar shards for training/inference
3. **DataPreprocessor** - Orchestrates the preprocessing and shard creation process

### Storage Format

Each processed sample is stored as a directory within a tar shard containing:

```
sample_000001_000000/
├── frame_000.webp    # Compressed video frames
├── frame_001.webp
├── ...
├── frame_015.webp
├── meta.json         # Sample metadata
├── audio_mel.f16     # Optional: Audio mel-spectrogram (float16)
└── audio_mel.json    # Optional: Audio metadata
```

## Components

### 1. ShardWriter

**Location**: `src/data/shard_writer.py`

The `ShardWriter` class handles writing processed video clips to compressed tar shards.

#### Key Features

- **Automatic shard rotation**: Creates new shards when size limit is reached
- **Image compression**: Supports WebP and JPEG compression with configurable quality
- **Incremental writing**: Samples are written immediately to avoid memory issues
- **Audio support**: Optional mel-spectrogram storage in float16 format
- **Index generation**: Creates CSV index for efficient sample lookup

#### Configuration

```yaml
output:
  webdataset:
    prefix: "shard"              # Shard filename prefix
    max_shard_size_mb: 1024      # Maximum shard size in MB
    image_codec: "webp"          # Image compression format
    image_quality: 90            # Compression quality (1-100)
    index_filename: "index.csv"  # Index filename
    clip_length: 16              # Number of frames per clip
    clip_stride: 8               # Stride between clips
```

#### Usage Example

```python
from src.data.shard_writer import ShardWriter

writer = ShardWriter(
    output_dir="data/processed/shard",
    shard_prefix="shard",
    max_shard_size_bytes=1024 * 1024 * 1024,  # 1GB
    image_codec="webp",
    image_quality=90
)

# Add a sample
writer.add_sample(
    sample_id="000001_000000",
    frames_rgb=frames_array,  # Shape: (T, H, W, 3), dtype: uint8
    label=1,
    sample_metadata={"fps": 25.0, "source": "fake"},
    mel_clip=audio_mel_array  # Optional: (n_mels, time), dtype: float16
)

writer.close()
```

### 2. ShardClipDataset

**Location**: `src/data/shard_dataset.py`

The `ShardClipDataset` class provides efficient streaming access to shard data for training and inference.

#### Key Features

- **Lazy loading**: Frames are decoded on-demand from compressed shards
- **Memory efficient**: Only loads required samples into memory
- **GPU support**: Can load tensors directly to specified device
- **Index-based access**: Uses CSV index for efficient sample lookup

#### Usage Example

```python
from src.data.shard_dataset import ShardClipDataset
import torch

# Create dataset
dataset = ShardClipDataset(
    shards_dir="data/processed/FakeAVCeleb_v1.2/shard",
    index_filename="index.csv",
    target_device="cuda",
    max_samples=1000  # Optional: limit number of samples
)

# Iterate through samples
for sample in dataset:
    frames = sample["data"]        # Tensor[T, H, W, 3] on target device
    label = sample["label"]        # int
    metadata = sample["metadata"]  # dict
    audio_mel = sample.get("audio_mel")  # Optional Tensor[n_mels, time]
    
    # Process sample...
```

### 3. DataPreprocessor Integration

**Location**: `src/data/base_preprocessor.py`

The base preprocessor integrates WebDataset functionality into the preprocessing pipeline.


#### Processing Pipeline

1. **Video Processing**: Extract frames, detect faces, resize to target size
2. **Audio Processing**: Extract mel-spectrograms with configurable parameters
3. **Clip Creation**: Slice videos into fixed-length clips with configurable stride
4. **Shard Writing**: Write clips to compressed tar shards with metadata
5. **Index Generation**: Create CSV index and stratified train/val splits

## Configuration

### Preprocessing Configuration

```yaml
data:
  train_split: 0.8      # Training set ratio
  val_split: 0.2        # Validation set ratio
  random_seed: 42       # Random seed for splitting

preprocessing:
  face_detection:
    detector: "opencv"  # Face detection method
    min_face_size: 100  # Minimum face size in pixels
    margin: 0.2         # Margin around detected face

  image_processing:
    target_size: [224, 224]  # Target frame size

  audio_processing:
    enabled: true
    sample_rate: 16000  # Audio sample rate
    n_mels: 80          # Number of mel bins
    n_fft: 2048         # FFT window size
    hop_length: 512     # Hop length for mel computation

output:
  webdataset:
    prefix: "shard"
    max_shard_size_mb: 1024
    image_codec: "webp"
    image_quality: 90
    clip_length: 16
    clip_stride: 8
```

## File Structure

After preprocessing, the output directory structure is:

```
data/processed/FakeAVCeleb_v1.2/
├── shards/
│   ├── shard_00000.tar    # Compressed tar shards
│   ├── shard_00001.tar
│   ├── ...
│   ├── index.csv          # Complete dataset index
│   ├── index_train.csv    # Training set index
│   └── index_val.csv      # Validation set index
└── dataset_statistics.json # Dataset statistics
```

## Index Format

The CSV index contains the following columns:

- `sample_id`: Unique identifier for the sample
- `shard`: Name of the tar shard containing the sample
- `dir`: Directory name within the shard
- `num_frames`: Number of frames in the clip
- `label`: Class label (0=real, 1=fake)
- `metadata`: JSON string containing sample metadata

Example index entry:
```csv
000001_000000,shard_00000.tar,sample_000001_000000,16,1,"{""fps"":25.0,""source"":""fake""}"
```

## Usage Examples

### Running Preprocessing

```bash
# Process FakeAVCeleb dataset
python preprocess.py --dataset fakeavceleb
```

### Viewing Processed Data

```bash
# View samples from shards
python scripts/view_shards.py data/processed/FakeAVCeleb_v1.2/shard --index index.csv --device cuda
```

### Using in Training

```python
from src.data.shard_dataset import ShardClipDataset
from torch.utils.data import DataLoader

# Create dataset
train_dataset = ShardClipDataset(
    shards_dir="data/processed/FakeAVCeleb_v1.2/shard",
    index_filename="index_train.csv",
    target_device="cuda"
)

# Create data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for batch in train_loader:
    frames = batch["data"]      # [B, T, H, W, 3]
    labels = batch["label"]     # [B]
    # ... training code
```
