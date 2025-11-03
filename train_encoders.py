#!/usr/bin/env python3
"""
Training Pipeline for AVFF Encoder Pretraining

This script provides a training pipeline specifically for the EncoderPretrain model
which trains audio and video encoders using contrastive learning and reconstruction.
It uses the EncoderTrainer class for GAN-style training with discriminators.

Usage:
    python train_encoders.py --config configs/encoder_training.yaml
    
    # With custom configuration
    python train_encoders.py --config configs/custom_encoder_training.yaml
    
    # Resume from checkpoint
    python train_encoders.py --config configs/encoder_training.yaml --resume checkpoints/best_encoder.pth
"""

import argparse
import random
from typing import Dict, Any
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.data.shard_dataset import ShardClipDataset
from src.models.AVFF_encoder import EncoderPretrain, EncoderTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training pipeline for AVFF encoder pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/encoder_training.yaml",
        help="Path to training configuration file"
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


def create_encoder_model(config: Dict[str, Any]) -> EncoderPretrain:
    """Create EncoderPretrain model instance based on configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Initialized EncoderPretrain model instance
    """
    model_config = config["model"]
    
    print(f"Creating EncoderPretrain model with parameters:")
    print(f"  embed_dim: {model_config['embed_dim']}")
    print(f"  video_in_channels: {model_config['video_in_channels']}")
    print(f"  audio_in_channels: {model_config['audio_in_channels']}")
    print(f"  video_patch: {model_config['video_patch']}")
    print(f"  audio_patch: {model_config['audio_patch']}")
    print(f"  num_slices: {model_config['num_slices']}")
    print(f"  encoder_layers: {model_config['encoder_layers']}")
    
    model = EncoderPretrain(
        embed_dim=model_config["embed_dim"],
        video_in_channels=model_config["video_in_channels"],
        audio_in_channels=model_config["audio_in_channels"],
        video_patch=tuple(model_config["video_patch"]),
        audio_patch=tuple(model_config["audio_patch"]),
        num_slices=model_config["num_slices"],
        encoder_layers=model_config["encoder_layers"]
    )
    
    print(f"✓ EncoderPretrain model created successfully")
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
        max_samples=data_config.get("max_train_samples"),
        batch_size=data_config.get("batch_size", 16)
    )

    # Create validation dataset
    val_dataset = ShardClipDataset(
        shards_dir=shards_dir,
        index_filename=data_config["val_index"],
        target_device=device,
        max_samples=data_config.get("max_val_samples"),
        batch_size=data_config.get("batch_size", 16)
    )

    print(f"✓ Training dataset loaded (batch_size={data_config.get('batch_size', 16)})")
    print(f"✓ Validation dataset loaded (batch_size={data_config.get('batch_size', 16)})")

    return train_dataset, val_dataset


def prepare_batch_for_encoder(batch: Dict[str, Any], device: str):
    """Prepare batch data for encoder training.
    
    Args:
        batch: Batch dictionary from dataset
        device: Device to move tensors to
        
    Returns:
        Tuple of (video_frames, audio_mel_frames)
    """
    video_frames = batch["data"]
    if video_frames.dim() == 5:
        # Convert from (B, T, H, W, C) to (B, C, T, H, W)
        video_frames = video_frames.permute(0, 4, 1, 2, 3)
    video_frames = video_frames.float() / 255.0

    audio_mel = batch["audio_mel_frames"]
    if audio_mel.dim() == 4:
        # Add channel dimension if needed: (B, T, H, W) -> (B, T, C, H, W)
        audio_mel = audio_mel.unsqueeze(2)
    
    return video_frames.to(device), audio_mel.to(device)


def load_checkpoint(checkpoint_path: str, trainer: EncoderTrainer):
    """Load training state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        trainer: EncoderTrainer instance to restore state into
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer states
    trainer.opt_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    trainer.opt_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    
    # Load training state
    trainer.current_epoch = checkpoint.get("epoch", 0)
    trainer.best_loss = checkpoint.get("best_loss", float("inf"))

    print(f"✓ Checkpoint loaded (epoch {checkpoint.get('epoch', 0)})")


def print_config(config: Dict[str, Any]):
    """Print training configuration in a readable format."""
    print("\n" + "=" * 70)
    print("ENCODER TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model: EncoderPretrain")
    print(f"Embed Dim: {config['model']['embed_dim']}")
    print(f"Dataset: {config['data']['shards_dir']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Generator LR: {config['training']['generator_lr']}")
    print(f"Discriminator LR: {config['training']['discriminator_lr']}")
    print(f"N Critic: {config['training']['n_critic']}")
    print(f"Device: {config['device']}")
    print("=" * 70 + "\n")


class EncoderTrainingLoop:
    """Training loop for encoder pretraining."""
    
    def __init__(self, trainer: EncoderTrainer, config: Dict[str, Any]):
        self.trainer = trainer
        self.config = config
        self.device = trainer.device
        
        # Setup checkpointing and logging
        self.checkpoint_dir = Path(config["checkpointing"]["dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.training_history = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.trainer.model.train()
        total_losses = {"g": {}, "d": {}}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch in pbar:
            video_frames, audio_frames = prepare_batch_for_encoder(batch, self.device)
            
            # Train one batch
            logs = self.trainer.train_batch(video_frames, audio_frames)
            
            # Accumulate losses
            for key, value in logs["g"].items():
                if key not in total_losses["g"]:
                    total_losses["g"][key] = 0.0
                total_losses["g"][key] += value.item()
            
            for key, value in logs["d"].items():
                if key not in total_losses["d"]:
                    total_losses["d"][key] = 0.0
                total_losses["d"][key] += value.item()
            
            num_batches += 1
            
            # Update progress bar
            current_g_loss = logs["g"]["total"].item()
            current_d_loss = logs["d"]["d_loss"].item()
            pbar.set_postfix({
                "g_loss": f"{current_g_loss:.4f}",
                "d_loss": f"{current_d_loss:.4f}"
            })
        
        # Average losses
        for category in total_losses:
            for key in total_losses[category]:
                total_losses[category][key] /= num_batches
        
        return total_losses
    
    def validate(self, val_loader):
        """Validate for one epoch."""
        self.trainer.model.eval()
        total_losses = {"g": {}, "d": {}}
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            for batch in pbar:
                video_frames, audio_frames = prepare_batch_for_encoder(batch, self.device)
                
                # Validate one batch (no optimizer updates)
                logs = self.trainer.train_batch(video_frames, audio_frames)
                
                # Accumulate losses
                for key, value in logs["g"].items():
                    if key not in total_losses["g"]:
                        total_losses["g"][key] = 0.0
                    total_losses["g"][key] += value.item()
                
                for key, value in logs["d"].items():
                    if key not in total_losses["d"]:
                        total_losses["d"][key] = 0.0
                    total_losses["d"][key] += value.item()
                
                num_batches += 1
                
                # Update progress bar
                current_g_loss = logs["g"]["total"].item()
                current_d_loss = logs["d"]["d_loss"].item()
                pbar.set_postfix({
                    "g_loss": f"{current_g_loss:.4f}",
                    "d_loss": f"{current_d_loss:.4f}"
                })
        
        # Average losses
        for category in total_losses:
            for key in total_losses[category]:
                total_losses[category][key] /= num_batches
        
        return total_losses
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("STARTING ENCODER TRAINING")
        print("=" * 70 + "\n")
        
        try:
            for epoch in range(self.config["training"]["epochs"]):
                self.current_epoch = epoch
                
                # Training
                train_losses = self.train_epoch(train_loader)
                
                # Validation
                val_losses = self.validate(val_loader)
                
                # Logging and checkpointing
                self._log_epoch(train_losses, val_losses)
                self._save_checkpoint(val_losses)
                
                # Early stopping check
                if self._should_stop_early(val_losses):
                    print("Early stopping triggered")
                    break
            
            print("\n" + "=" * 70)
            print("ENCODER TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"\nBest model saved to: {self.checkpoint_dir / 'best_encoder.pth'}")
            print(f"Logs saved to: {self.log_dir}")
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print(f"Latest checkpoint saved to: {self.checkpoint_dir}")
        
        except Exception as e:
            print(f"\n\nTraining failed with error: {e}")
            raise
    
    def _log_epoch(self, train_losses, val_losses):
        """Log epoch results."""
        print(f"\n{'=' * 70}")
        print(f"Epoch {self.current_epoch + 1}/{self.config['training']['epochs']}")
        print(f"{'=' * 70}")
        
        print("Training losses:")
        print("  Generator:")
        for key, value in train_losses["g"].items():
            print(f"    {key:12s}: {value:.4f}")
        print("  Discriminator:")
        for key, value in train_losses["d"].items():
            print(f"    {key:12s}: {value:.4f}")
        
        print("\nValidation losses:")
        print("  Generator:")
        for key, value in val_losses["g"].items():
            print(f"    {key:12s}: {value:.4f}")
        print("  Discriminator:")
        for key, value in val_losses["d"].items():
            print(f"    {key:12s}: {value:.4f}")
        print(f"{'=' * 70}\n")
        
        # Save to history
        epoch_data = {
            "epoch": self.current_epoch + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "timestamp": torch.utils.data.get_worker_info()
        }
        self.training_history.append(epoch_data)
    
    def _save_checkpoint(self, val_losses):
        """Save checkpoint if this is the best model."""
        current_loss = val_losses["g"]["total"]
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            
            checkpoint_path = self.checkpoint_dir / "best_encoder.pth"
            checkpoint_data = {
                "epoch": self.current_epoch + 1,
                "model_state_dict": self.trainer.model.state_dict(),
                "optimizer_g_state_dict": self.trainer.opt_g.state_dict(),
                "optimizer_d_state_dict": self.trainer.opt_d.state_dict(),
                "best_loss": self.best_loss,
                "val_losses": val_losses,
                "config": self.config
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"✓ Saved best encoder model (epoch {self.current_epoch + 1}, loss={current_loss:.4f})")
        
        # Save regular checkpoint
        if (self.current_epoch + 1) % self.config["logging"]["save_frequency"] == 0:
            checkpoint_path = self.checkpoint_dir / f"encoder_checkpoint_epoch_{self.current_epoch + 1}.pth"
            checkpoint_data = {
                "epoch": self.current_epoch + 1,
                "model_state_dict": self.trainer.model.state_dict(),
                "optimizer_g_state_dict": self.trainer.opt_g.state_dict(),
                "optimizer_d_state_dict": self.trainer.opt_d.state_dict(),
                "best_loss": self.best_loss,
                "val_losses": val_losses,
                "config": self.config
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {self.current_epoch + 1}")
    
    def _should_stop_early(self, val_losses):
        """Check if early stopping should be triggered."""
        # Simple early stopping based on validation loss
        current_loss = val_losses["g"]["total"]
        
        # This is a simplified early stopping - you might want to implement
        # more sophisticated logic based on your needs
        return False


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()

    # Load configuration
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
    print("Initializing encoder model...")
    model = create_encoder_model(config)

    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_data_loaders(config, device)

    # Create trainer
    print("\nInitializing encoder trainer...")
    trainer = EncoderTrainer(
        model=model,
        lr_g=config["training"]["generator_lr"],
        lr_d=config["training"]["discriminator_lr"],
        device=device,
        n_critic=config["training"]["n_critic"]
    )

    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(args.resume, trainer)

    # Create training loop
    training_loop = EncoderTrainingLoop(trainer, config)

    # Train model
    training_loop.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
