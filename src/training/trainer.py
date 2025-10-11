from typing import Dict, Any, Optional, List, Tuple
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torchvision.transforms as transforms


class Trainer:
    """Universal trainer for multi-modal deepfake detection models (BaseDetector subclasses)."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the trainer.
        
        Args:
            model: Model to train (must inherit from BaseDetector)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_criterion()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config["checkpointing"]["dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logging directory
        self.log_dir = Path(config["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training state
        self.current_epoch = 0
        self.best_metric = float("-inf") if config["checkpointing"]["mode"] == "max" else float("inf")
        self.early_stopping_counter = 0
        
        # Initialize training history
        self.history = {"train": [], "val": []}
        
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on configuration."""
        optimizer_config = self.config["training"]["optimizer"]
        if optimizer_config["name"] == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config["learning_rate"],
                weight_decay=optimizer_config["weight_decay"]
            )
        elif optimizer_config["name"] == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_config["learning_rate"],
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_config = self.config["training"]["scheduler"]
        if scheduler_config["name"] == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["epochs"],
                eta_min=scheduler_config["min_lr"]
            )
        elif scheduler_config["name"] == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_config["name"] == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.1,
                patience=5,
                min_lr=scheduler_config["min_lr"]
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_config = self.config["loss"]
        if loss_config["name"] == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_config["name"] == "focal":
            return FocalLoss(gamma=loss_config["focal_gamma"])
        else:
            raise ValueError(f"Unsupported loss function: {loss_config['name']}")
    
    def _prepare_batch(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data for multi-modal model.
        
        Args:
            sample: Sample dictionary containing 'data' (video frames), 'audio_mel_frames', and 'label'
            
        Returns:
            Tuple of (image_input, audio_input, labels)
        """
        # Extract video frames (T, H, W, 3) -> (1, T, 3, H, W)
        video_frames = sample["data"]  # Already a tensor from ShardClipDataset
        if video_frames.dim() == 4:  # (T, H, W, 3)
            video_frames = video_frames.permute(0, 3, 1, 2)  # (T, 3, H, W)
            video_frames = video_frames.unsqueeze(0)  # (1, T, 3, H, W)
        
        # Normalize video frames to [0, 1]
        video_frames = video_frames.float() / 255.0
        
        # Extract audio mel spectrogram (T, n_mels, mel_frames) -> (1, T, 1, n_mels, mel_frames)
        audio_mel = sample["audio_mel_frames"]  # (T, n_mels, mel_frames)
        if audio_mel.dim() == 3:
            audio_mel = audio_mel.unsqueeze(0).unsqueeze(2)  # (1, T, 1, n_mels, mel_frames)
        
        # Extract label
        label = torch.tensor([sample["label"]], dtype=torch.long)
        
        return video_frames.to(self.device), audio_mel.to(self.device), label.to(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for sample in pbar:
            # Prepare batch data
            image_input, audio_input, target = self._prepare_batch(sample)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(image_input, audio_input)
            loss = self.criterion(output, target.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional, helps with stability)
            if self.config["training"].get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip"])
            
            self.optimizer.step()
            
            # Collect predictions and targets for metrics
            with torch.no_grad():
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.append(pred.cpu())
                all_targets.append(target.squeeze().cpu())
                all_probs.append(probs.cpu())
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                current_acc = (pred == target.squeeze()).float().mean().item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{current_acc:.4f}"
                })
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_targets, all_probs)
        metrics["loss"] = total_loss / num_batches
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            for sample in pbar:
                # Prepare batch data
                image_input, audio_input, target = self._prepare_batch(sample)
                
                # Forward pass
                output = self.model(image_input, audio_input)
                loss = self.criterion(output, target.squeeze())
                
                # Collect predictions
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.append(pred.cpu())
                all_targets.append(target.squeeze().cpu())
                all_probs.append(probs.cpu())
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                current_acc = (pred == target.squeeze()).float().mean().item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{current_acc:.4f}"
                })
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_targets, all_probs)
        metrics["loss"] = total_loss / num_batches
        
        return metrics
    
    def _calculate_metrics(self, preds: torch.Tensor, targets: torch.Tensor, probs: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics.
        
        Args:
            preds: Predicted labels
            targets: Ground truth labels
            probs: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        probs_np = probs.numpy()
        
        metrics = {}
        
        # Calculate metrics from config
        for metric_name in self.config["logging"]["metrics"]:
            if metric_name == "accuracy":
                metrics[metric_name] = (preds_np == targets_np).mean()
            elif metric_name == "precision":
                metrics[metric_name] = precision_score(targets_np, preds_np, average="binary", zero_division=0)
            elif metric_name == "recall":
                metrics[metric_name] = recall_score(targets_np, preds_np, average="binary", zero_division=0)
            elif metric_name == "f1_score":
                metrics[metric_name] = f1_score(targets_np, preds_np, average="binary", zero_division=0)
            elif metric_name == "auc_roc":
                # Need probability of positive class
                try:
                    metrics[metric_name] = roc_auc_score(targets_np, probs_np[:, 1])
                except:
                    metrics[metric_name] = 0.0
        
        return metrics
    
    def train(self):
        """Train the model."""
        for epoch in range(self.config["training"]["epochs"]):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.config["checkpointing"]["monitor"]])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(val_metrics)
            
            # Early stopping
            if self._should_stop_early(val_metrics):
                print("Early stopping triggered")
                break
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log training and validation metrics.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Log to console
        print(f"\n{'='*70}")
        print(f"Epoch {self.current_epoch + 1}/{self.config['training']['epochs']}")
        print(f"{'='*70}")
        print("Training metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric:12s}: {value:.4f}")
        print("\nValidation metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric:12s}: {value:.4f}")
        print(f"{'='*70}\n")
        
        # Store in history
        self.history["train"].append(train_metrics)
        self.history["val"].append(val_metrics)
        
        # Log to file
        log_file = self.log_dir / "training_log.json"
        log_data = {
            "epoch": self.current_epoch + 1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "timestamp": datetime.now().isoformat()
        }
        
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_data)
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
        
        # Save history
        history_file = self.log_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def _save_checkpoint(self, val_metrics: Dict[str, float]):
        """Save model checkpoint.
        
        Args:
            val_metrics: Validation metrics
        """
        monitor_metric = self.config["checkpointing"]["monitor"]
        current_metric = val_metrics.get(monitor_metric, val_metrics.get("accuracy", 0.0))
        
        is_best = (
            current_metric > self.best_metric
            if self.config["checkpointing"]["mode"] == "max"
            else current_metric < self.best_metric
        )
        
        if is_best:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
            
            # Save best model
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": self.current_epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_metric": self.best_metric,
                "val_metrics": val_metrics,
                "config": self.config
            }, checkpoint_path)
            print(f"✓ Saved best model (epoch {self.current_epoch + 1}, {monitor_metric}={current_metric:.4f})")
        else:
            self.early_stopping_counter += 1
        
        # Save regular checkpoint if configured
        if (
            not self.config["checkpointing"]["save_best_only"]
            and (self.current_epoch + 1) % self.config["logging"]["save_frequency"] == 0
        ):
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"
            torch.save({
                "epoch": self.current_epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_metric": self.best_metric,
                "val_metrics": val_metrics,
                "config": self.config
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {self.current_epoch + 1}")
    
    def _should_stop_early(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should be stopped early.
        
        Args:
            val_metrics: Validation metrics
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.early_stopping_counter >= self.config["training"]["early_stopping"]["patience"]:
            return True
        return False

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0):
        """Initialize focal loss.
        
        Args:
            gamma: Focusing parameter
        """
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            input: Model predictions
            target: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss 