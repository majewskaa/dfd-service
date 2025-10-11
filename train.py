#!/usr/bin/env python3
"""
Universal Training Pipeline for Deepfake Detection Models

This script provides a flexible training pipeline that works with any model
extending the BaseDetector class. It supports multi-modal (video + audio) 
deepfake detection using preprocessed shard datasets.

Usage:
    python train.py --config configs/training.yaml
    
    # With custom configuration
    python train.py --config configs/custom_training.yaml
    
    # Resume from checkpoint
    python train.py --config configs/training.yaml --resume checkpoints/best_model.pth
"""

import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path
import sys
import os
import importlib
from typing import Dict, Any

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.shard_dataset import ShardClipDataset
from src.models.base import BaseDetector
from src.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Universal training pipeline for deepfake detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration file (default: configs/training.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config). Options: cuda, cpu, cuda:0, etc."
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_class(module_path: str, class_name: str):
    """Dynamically load model class from module path.
    
    Args:
        module_path: Python import path (e.g., "src.models.xception")
        class_name: Name of the class to import
        
    Returns:
        Model class
    """
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load model class '{class_name}' from '{module_path}': {e}")


def create_model(config: Dict[str, Any]) -> BaseDetector:
    """Create model instance based on configuration.
    
    This function dynamically loads the model class specified in the config,
    making it flexible to support any BaseDetector subclass.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Initialized model instance
    """
    model_config = config["model"]
    model_name = model_config["name"]
    module_path = model_config["module_path"]
    num_classes = model_config["num_classes"]
    
    # Load model class dynamically
    print(f"Loading model: {model_name} from {module_path}")
    ModelClass = load_model_class(module_path, model_name)
    
    # Get model-specific parameters
    model_params = model_config.get("model_params", {})
    
    # Create model instance
    model = ModelClass(num_classes=num_classes, **model_params)
    
    # Verify it's a BaseDetector subclass
    if not isinstance(model, BaseDetector):
        raise TypeError(f"Model {model_name} must inherit from BaseDetector")
    
    print(f"✓ Model created successfully: {model_name}")
    return model


def create_data_loaders(config: Dict[str, Any], device: str):
    """Create training and validation data loaders from shard datasets.
    
    Args:
        config: Training configuration dictionary
        device: Device for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config["data"]
    shards_dir = data_config["shards_dir"]
    
    print(f"Loading datasets from: {shards_dir}")
    
    # Create training dataset
    train_dataset = ShardClipDataset(
        shards_dir=shards_dir,
        index_filename=data_config["train_index"],
        target_device=device,
        max_samples=data_config.get("max_train_samples")
    )
    
    # Create validation dataset
    val_dataset = ShardClipDataset(
        shards_dir=shards_dir,
        index_filename=data_config["val_index"],
        target_device=device,
        max_samples=data_config.get("max_val_samples")
    )
    
    print(f"✓ Training dataset loaded")
    print(f"✓ Validation dataset loaded")
    
    # Note: IterableDataset doesn't need DataLoader for batching
    # We return the datasets directly as "loaders" since Trainer expects iterables
    return train_dataset, val_dataset


def load_checkpoint(checkpoint_path: str, model: BaseDetector, trainer: Trainer = None):
    """Load model and training state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load state into
        trainer: Optional trainer instance to restore training state
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load trainer state if provided
    if trainer is not None:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint.get("epoch", 0)
        trainer.best_metric = checkpoint.get("best_metric", float("-inf"))
    
    print(f"✓ Checkpoint loaded (epoch {checkpoint.get('epoch', 0)})")


def print_config(config: Dict[str, Any]):
    """Print training configuration in a readable format."""
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['data']['shards_dir']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['optimizer']['learning_rate']}")
    print(f"Optimizer: {config['training']['optimizer']['name']}")
    print(f"Scheduler: {config['training']['scheduler']['name']}")
    print(f"Loss: {config['loss']['name']}")
    print(f"Device: {config['device']}")
    print(f"Checkpoints: {config['checkpointing']['dir']}")
    print(f"Logs: {config['logging']['log_dir']}")
    print("="*70 + "\n")


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if args.device:
        config["device"] = args.device
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    print(f"✓ Random seed set to {config.get('seed', 42)}")
    
    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    print(f"✓ Using device: {device}")
    
    # Print configuration
    print_config(config)
    
    # Create model
    print("Initializing model...")
    model = create_model(config)
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_data_loaders(config, device)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(args.resume, model, trainer)
    
    # Train model
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    try:
        trainer.train()
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nBest model saved to: {trainer.checkpoint_dir / 'best_model.pth'}")
        print(f"Logs saved to: {trainer.log_dir}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Latest checkpoint saved to: {trainer.checkpoint_dir}")
        
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 