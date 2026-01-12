from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class DeepfakeTask(pl.LightningModule):
    """
    PyTorch Lightning Module for Deepfake Detection.
    Encapsulates model, loss, optimization, and metrics.
    """

    def __init__(
            self,
            model: nn.Module,
            config: Dict[str, Any]
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = self._create_criterion()

        # Save hyperparameters (excluding model to avoid pickling issues if complex)
        self.save_hyperparameters(ignore=["model"])

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2)

    def forward(self, video_frames, audio_frames):
        return self.model(video_frames, audio_frames)

    def _create_criterion(self) -> nn.Module:
        loss_config = self.config["loss"]
        name = loss_config["name"]
        if name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif name == "focal":
            return FocalLoss(gamma=loss_config.get("focal_gamma", 2.0))
        raise ValueError(f"Unsupported loss function: {name}")

    def _prepare_batch(self, batch):
        video_frames = batch["video_frames"]
        # Ensure correct dimensions: (B, C, T, H, W)
        if video_frames.dim() == 5:
            video_frames = video_frames.permute(0, 1, 4, 2, 3)
        video_frames = video_frames.float() / 255.0
        
        # Normalization from config
        norm_config = self.config["data"].get("normalization", {})
        mean_val = norm_config.get("mean", [0.485, 0.456, 0.406])
        std_val = norm_config.get("std", [0.229, 0.224, 0.225])
        
        # Shape: (1, 1, 3, 1, 1) to match (B, T, C, H, W)
        mean = torch.tensor(mean_val, device=video_frames.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(std_val, device=video_frames.device).view(1, 1, 3, 1, 1)
        video_frames = (video_frames - mean) / std

        audio_mel = batch["audio_frames"]
        if audio_mel.dim() == 4:
            audio_mel = audio_mel.unsqueeze(2)

        labels = batch["label"]
        return video_frames, audio_mel.float(), labels

    def training_step(self, batch, batch_idx):
        video, audio, targets = self._prepare_batch(batch)
        outputs = self(video, audio)
        loss = self.criterion(outputs, targets)

        # Log metrics
        batch_size = video.shape[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Calculate metrics
        self.train_acc(outputs, targets)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        video, audio, targets = self._prepare_batch(batch)
        outputs = self(video, audio)
        loss = self.criterion(outputs, targets)

        # Log loss
        batch_size = video.shape[0]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Update metrics
        self.val_acc(outputs, targets)
        self.val_f1(outputs, targets)
        self.val_auroc(outputs, targets)

        # Log metrics
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_auroc", self.val_auroc, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        video, audio, targets = self._prepare_batch(batch)
        print(f"DEBUG: video shape: {video.shape}, audio shape: {audio.shape}")
        outputs = self(video, audio)
        probs = torch.softmax(outputs, dim=1)
        return {"probs": probs, "targets": targets}

    def configure_optimizers(self):
        optimizer_config = self.config["training"]["optimizer"]
        name = optimizer_config["name"]
        lr = float(optimizer_config["learning_rate"])
        weight_decay = float(optimizer_config["weight_decay"])

        if name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

        scheduler_config = self.config["training"]["scheduler"]
        scheduler_name = scheduler_config["name"]

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config["training"]["epochs"],
                eta_min=float(scheduler_config.get("min_lr", 0.0))
            )
            return [optimizer], [scheduler]
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=5,
                min_lr=float(scheduler_config.get("min_lr", 1e-6))
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_f1"  # Assumes we compute this
                }
            }

        return optimizer


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
