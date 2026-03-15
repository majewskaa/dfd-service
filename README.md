# Deepfake Detection Lab

A modular platform for training, evaluating, and comparing deepfake detectors across different modalities. Currently focused on image-based detection with plans to extend to video and audio modalities.

## Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) to manage this Python project. 
Simply run the following in root directory:

1. Run a script
```bash
uv run ./src/main/preprocess.py --dataset fakeavceleb
```

2. Update projects environment
```bash
uv sync
```

## Usage

### Data Preprocessing
```bash
python ./src/main/preprocess.py --dataset fakeavceleb
```

### Training
```bash
python ./src/main/train.py
```

### Evaluation
```bash
python ./src/main/evaluate.py
```

### Service
```bash
v run ./components/service/src/main/service.py --config ./components/service/configs/service.yaml
```

## Configuration

The project uses YAML configuration files to manage different aspects:
- `preprocessing.yaml`: Data preprocessing parameters
- `training.yaml`: Model training parameters
- `evaluation.yaml`: Evaluation parameters
