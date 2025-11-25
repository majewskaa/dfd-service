#!/usr/bin/env python3
"""
Evaluation Script for Deepfake Detection Models (PyTorch Lightning)

Usage:
    python evaluate.py
    python evaluate.py --config configs/my_eval_config.yaml
"""

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
import importlib
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.shard_dataset import ShardClipDataset
from src.training.lightning_module import DeepfakeTask
from src.models.base import BaseDetector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    return parser.parse_args()

def load_model_class(module_path: str, class_name: str):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load model class '{class_name}' from '{module_path}': {e}")

def create_model(config: dict) -> BaseDetector:
    model_config = config["model"]
    ModelClass = load_model_class(model_config["module_path"], model_config["name"])
    model = ModelClass(num_classes=model_config["num_classes"], **model_config.get("model_params", {}))
    return model

def main():
    args = parse_args()
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Overrides
    checkpoint_path = args.checkpoint or config["evaluation"].get("checkpoint_path")
    if not checkpoint_path:
        print("Error: No checkpoint path specified in config or arguments.")
        sys.exit(1)
        
    # Optimization
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(config.get("seed", 42))

    # Data
    data_config = config["data"]
    print(f"Loading test data using index: {data_config['test_index']}")
    
    test_dataset = ShardClipDataset(
        shards_dir=data_config["shards_dir"],
        index_filename=data_config["test_index"],
        target_device="cpu", 
        max_samples=None, 
        frames_per_clip=data_config.get("frames_per_clip", 32)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.get("batch_size", 16),
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=ShardClipDataset.collate_fn,
        drop_last=False
    )

    # Model
    print(f"Loading model from {checkpoint_path}")
    base_model = create_model(config)
    
    # Load from checkpoint
    task = DeepfakeTask.load_from_checkpoint(
        checkpoint_path, 
        model=base_model, 
        config=config,
        strict=False 
    )

    # Trainer
    eval_config = config["evaluation"]
    device = eval_config.get("device", "auto")
    accelerator = device if device != "auto" else "auto"
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False, 
        enable_checkpointing=False,
        limit_test_batches=eval_config.get("limit_batches")
    )

    # Evaluate
    print("Starting evaluation...")
    trainer.test(task, dataloaders=test_loader)

if __name__ == "__main__":
    main()