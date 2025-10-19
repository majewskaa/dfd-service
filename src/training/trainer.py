import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Universal trainer for multi-modal deepfake detection models."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            config: Dict[str, Any],
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()

        self.checkpoint_dir = Path(config["checkpointing"]["dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.best_metric = float("-inf") if config["checkpointing"]["mode"] == "max" else float("inf")
        self.early_stopping_counter = 0
        self.history = {"train": [], "val": []}

    def _create_optimizer(self) -> Optimizer:
        optimizer_config = self.config["training"]["optimizer"]
        name = optimizer_config["name"]

        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config["learning_rate"],
                weight_decay=optimizer_config["weight_decay"]
            )
        elif name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_config["learning_rate"],
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"]
            )

        raise ValueError(f"Unsupported optimizer: {name}")

    def _create_scheduler(self) -> CosineAnnealingLR | StepLR | ReduceLROnPlateau | None:
        scheduler_config = self.config["training"]["scheduler"]
        name = scheduler_config["name"]

        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["epochs"],
                eta_min=scheduler_config["min_lr"]
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.1,
                patience=5,
                min_lr=scheduler_config["min_lr"]
            )

        return None

    def _create_criterion(self) -> nn.Module:
        loss_config = self.config["loss"]
        name = loss_config["name"]

        if name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif name == "focal":
            return FocalLoss(gamma=loss_config["focal_gamma"])

        raise ValueError(f"Unsupported loss function: {name}")

    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_frames = batch["data"]
        if video_frames.dim() == 5:
            video_frames = video_frames.permute(0, 1, 4, 2, 3)
        video_frames = video_frames.float() / 255.0

        audio_mel = batch["audio_mel_frames"]
        if audio_mel.dim() == 4:
            audio_mel = audio_mel.unsqueeze(2)

        labels = batch["label"]

        return video_frames.to(self.device), audio_mel.to(self.device), labels.to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        all_preds, all_targets, all_probs = [], [], []
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch in pbar:
            image_input, audio_input, target = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            output = self.model(image_input, audio_input)
            loss = self.criterion(output, target)
            loss.backward()

            if self.config["training"].get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip"])

            self.optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)

                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                all_probs.append(probs.cpu())

                total_loss += loss.item()
                num_batches += 1

                current_acc = (pred == target).float().mean().item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{current_acc:.4f}"
                })

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)

        metrics = self._calculate_metrics(all_preds, all_targets, all_probs)
        metrics["loss"] = total_loss / num_batches

        return metrics

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_targets, all_probs = [], [], []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            for batch in pbar:
                image_input, audio_input, target = self._prepare_batch(batch)

                output = self.model(image_input, audio_input)
                loss = self.criterion(output, target)

                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)

                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                all_probs.append(probs.cpu())

                total_loss += loss.item()
                num_batches += 1

                current_acc = (pred == target).float().mean().item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{current_acc:.4f}"
                })

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)

        metrics = self._calculate_metrics(all_preds, all_targets, all_probs)
        metrics["loss"] = total_loss / num_batches

        return metrics

    def _calculate_metrics(self, preds: torch.Tensor, targets: torch.Tensor, probs: torch.Tensor) -> Dict[str, float]:
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        probs_np = probs.numpy()

        metrics = {}
        metric_calculators = {
            "accuracy": lambda: (preds_np == targets_np).mean(),
            "precision": lambda: precision_score(targets_np, preds_np, average="binary", zero_division=0),
            "recall": lambda: recall_score(targets_np, preds_np, average="binary", zero_division=0),
            "f1_score": lambda: f1_score(targets_np, preds_np, average="binary", zero_division=0),
            "auc_roc": lambda: self._safe_auc_roc(targets_np, probs_np)
        }

        for metric_name in self.config["logging"]["metrics"]:
            if metric_name in metric_calculators:
                metrics[metric_name] = metric_calculators[metric_name]()

        return metrics

    def _safe_auc_roc(self, targets: np.ndarray, probs: np.ndarray) -> float:
        try:
            return roc_auc_score(targets, probs[:, 1])
        except:
            return 0.0

    def train(self):
        for epoch in range(self.config["training"]["epochs"]):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self._update_scheduler(val_metrics)
            self._log_metrics(train_metrics, val_metrics)
            self._save_checkpoint(val_metrics)

            if self._should_stop_early(val_metrics):
                print("Early stopping triggered")
                break

    def _update_scheduler(self, val_metrics: Dict[str, float]):
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_metrics[self.config["checkpointing"]["monitor"]])
        else:
            self.scheduler.step()

    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        self._print_metrics(train_metrics, val_metrics)
        self._update_history(train_metrics, val_metrics)
        self._save_logs(train_metrics, val_metrics)

    def _print_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        print(f"\n{'=' * 70}")
        print(f"Epoch {self.current_epoch + 1}/{self.config['training']['epochs']}")
        print(f"{'=' * 70}")
        print("Training metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric:12s}: {value:.4f}")
        print("\nValidation metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric:12s}: {value:.4f}")
        print(f"{'=' * 70}\n")

    def _update_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        self.history["train"].append(train_metrics)
        self.history["val"].append(val_metrics)

    def _save_logs(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        log_data = {
            "epoch": self.current_epoch + 1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "timestamp": datetime.now().isoformat()
        }

        log_file = self.log_dir / "training_log.json"
        logs = self._load_existing_logs(log_file)
        logs.append(log_data)

        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

        history_file = self.log_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def _load_existing_logs(self, log_file: Path) -> list:
        if log_file.exists():
            with open(log_file, "r") as f:
                return json.load(f)
        return []

    def _save_checkpoint(self, val_metrics: Dict[str, float]):
        monitor_metric = self.config["checkpointing"]["monitor"]
        current_metric = val_metrics.get(monitor_metric, val_metrics.get("accuracy", 0.0))

        is_best = self._is_best_metric(current_metric)

        if is_best:
            self._save_best_model(val_metrics, current_metric, monitor_metric)
        else:
            self.early_stopping_counter += 1

        if self._should_save_regular_checkpoint():
            self._save_regular_checkpoint(val_metrics)

    def _is_best_metric(self, current_metric: float) -> bool:
        if self.config["checkpointing"]["mode"] == "max":
            return current_metric > self.best_metric
        return current_metric < self.best_metric

    def _save_best_model(self, val_metrics: Dict[str, float], current_metric: float, monitor_metric: str):
        self.best_metric = current_metric
        self.early_stopping_counter = 0

        checkpoint_path = self.checkpoint_dir / "best_model.pth"
        checkpoint_data = self._create_checkpoint_data(val_metrics)
        torch.save(checkpoint_data, checkpoint_path)
        print(f"✓ Saved best model (epoch {self.current_epoch + 1}, {monitor_metric}={current_metric:.4f})")

    def _should_save_regular_checkpoint(self) -> bool:
        return (
                not self.config["checkpointing"]["save_best_only"]
                and (self.current_epoch + 1) % self.config["logging"]["save_frequency"] == 0
        )

    def _save_regular_checkpoint(self, val_metrics: Dict[str, float]):
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        checkpoint_data = self._create_checkpoint_data(val_metrics)
        torch.save(checkpoint_data, checkpoint_path)
        print(f"✓ Saved checkpoint at epoch {self.current_epoch + 1}")

    def _create_checkpoint_data(self, val_metrics: Dict[str, float]) -> Dict[str, Any]:
        return {
            "epoch": self.current_epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "val_metrics": val_metrics,
            "config": self.config
        }

    def _should_stop_early(self) -> bool:
        return self.early_stopping_counter >= self.config["training"]["early_stopping"]["patience"]


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
