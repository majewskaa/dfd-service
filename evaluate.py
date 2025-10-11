#!/usr/bin/env python3
"""
Evaluation Script for Deepfake Detection Models

This script evaluates trained models on test datasets and generates
comprehensive metrics and visualizations.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    
    # With custom test configuration
    python evaluate.py --checkpoint checkpoints/best_model.pth --config configs/evaluation.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import os
import json
import importlib
from typing import Dict, Any
from tqdm import tqdm

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.shard_dataset import ShardClipDataset
from src.models.base import BaseDetector
from src.training.metrics import MetricsCalculator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation.yaml",
        help="Path to evaluation configuration file (default: configs/evaluation.yaml)"
    )
    parser.add_argument(
        "--test-index",
        type=str,
        default="index_val.csv",
        help="Test index file name (default: uses validation set)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config). Options: cuda, cpu, cuda:0, etc."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    return parser.parse_args()


def load_model_class(module_path: str, class_name: str):
    """Dynamically load model class from module path."""
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load model class '{class_name}' from '{module_path}': {e}")


def load_model(checkpoint_path: str, device: str) -> BaseDetector:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get model configuration from checkpoint
    config = checkpoint["config"]
    model_config = config["model"]
    
    # Load model class dynamically
    ModelClass = load_model_class(model_config["module_path"], model_config["name"])
    
    # Create model instance
    model_params = model_config.get("model_params", {})
    model = ModelClass(num_classes=model_config["num_classes"], **model_params)
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {model_config['name']}")
    print(f"✓ Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"✓ Best metric: {checkpoint.get('best_metric', 'unknown')}")
    
    return model


def prepare_batch(sample: Dict[str, Any], device: str):
    """Prepare batch data for multi-modal model."""
    # Extract video frames
    video_frames = sample["data"]
    if video_frames.dim() == 4:
        video_frames = video_frames.permute(0, 3, 1, 2)
        video_frames = video_frames.unsqueeze(0)
    video_frames = video_frames.float() / 255.0
    
    # Extract audio mel spectrogram
    audio_mel = sample["audio_mel_frames"]
    if audio_mel.dim() == 3:
        audio_mel = audio_mel.unsqueeze(0).unsqueeze(2)
    
    # Extract label
    label = torch.tensor([sample["label"]], dtype=torch.long)
    
    return video_frames.to(device), audio_mel.to(device), label.to(device)


def evaluate_model(model: BaseDetector, test_loader, device: str, config: Dict[str, Any]):
    """Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        config: Evaluation configuration
        
    Returns:
        Dictionary of metrics and predictions
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Evaluating"):
            # Prepare batch
            image_input, audio_input, target = prepare_batch(sample, device)
            
            # Forward pass
            output = model(image_input, audio_input)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            # Collect predictions
            all_preds.append(pred.cpu())
            all_targets.append(target.squeeze().cpu())
            all_probs.append(probs.cpu())
    
    # Concatenate all predictions
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, average="binary", zero_division=0),
        "recall": recall_score(all_targets, all_preds, average="binary", zero_division=0),
        "f1_score": f1_score(all_targets, all_preds, average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(all_targets, all_preds).tolist()
    }
    
    # Add AUC-ROC if probabilities available
    try:
        metrics["auc_roc"] = roc_auc_score(all_targets, all_probs[:, 1])
    except:
        metrics["auc_roc"] = 0.0
    
    return metrics, all_preds, all_targets, all_probs


def save_results(metrics: Dict[str, Any], output_dir: Path):
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_file}")


def print_metrics(metrics: Dict[str, Any]):
    """Print evaluation metrics in a readable format."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            print(f"{metric:15s}: {value:.4f}")
    
    if "confusion_matrix" in metrics:
        print("\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        print(f"  TN: {cm[0][0]:4d}  |  FP: {cm[0][1]:4d}")
        print(f"  FN: {cm[1][0]:4d}  |  TP: {cm[1][1]:4d}")
    print("="*70 + "\n")


def main():
    """Main evaluation pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load evaluation configuration (if exists)
    config = {}
    if Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    
    # Set device
    device = args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    print(f"✓ Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load test dataset
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    data_config = checkpoint["config"]["data"]
    shards_dir = data_config["shards_dir"]
    
    print(f"\nLoading test dataset from: {shards_dir}")
    test_loader = ShardClipDataset(
        shards_dir=shards_dir,
        index_filename=args.test_index,
        target_device=device,
        max_samples=None
    )
    print("✓ Test dataset loaded")
    
    # Evaluate model
    metrics, preds, targets, probs = evaluate_model(model, test_loader, device, config)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    output_dir = Path(args.output_dir)
    save_results(metrics, output_dir)
    
    print(f"Evaluation completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 