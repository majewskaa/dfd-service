#!/usr/bin/env python3
"""
Universal Training Pipeline for Multimodal Deepfake Detection Models

Usage:
    python train.py --config configs/training.yaml
"""

import argparse
import importlib
from typing import Dict, Any

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch

from src.data.shard_dataset import ShardClipDataset
from src.models.base import BaseDetector
from src.training.lightning_module import DeepfakeTask
from src.callbacks.memory_monitor import CUDAMemoryMonitor



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_model_class(module_path: str, class_name: str):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load model class '{class_name}' from '{module_path}': {e}")


def create_model(config: Dict[str, Any]) -> BaseDetector:
    model_config = config["model"]
    ModelClass = load_model_class(model_config["module_path"], model_config["name"])
    model = ModelClass(num_classes=model_config["num_classes"], **(model_config.get("model_params") or {}))

    # Optional: Load pre-trained encoder weights if specified
    if "encoder_checkpoint" in model_config:
        ckpt_path = model_config["encoder_checkpoint"]
        if hasattr(model, "load_encoders"):
            print(f"Loading encoder weights from {ckpt_path}...")
            # We assume the model's load_encoders method handles the loading logic
            # Some models might require a device argument, but let's try default first or check signature if needed.
            # Based on AVFF.py, load_encoders(path, device="cpu") is the signature.
            # We can pass "cpu" safely here as we are just initializing, or let it default.
            try:
                model.load_encoders(ckpt_path)
                print("Encoder weights loaded successfully.")
            except Exception as e:
                print(f"WARNING: Failed to load encoder weights: {e}")
        else:
            print(f"WARNING: 'encoder_checkpoint' specified but model '{model_config['name']}' does not have 'load_encoders' method.")
    
    return model


def create_data_loaders(config: Dict[str, Any]):
    data_config = config["data"]
    shards_dir = data_config["shards_dir"]
    frames_per_clip = data_config.get("frames_per_clip", 32)
    num_workers = data_config.get("num_workers", 4)
    batch_size = data_config.get("batch_size", 16)

    train_dataset = ShardClipDataset(
        shards_dir=shards_dir,
        index_filename=data_config["train_index"],
        target_device="cpu",
        max_samples=data_config.get("max_train_samples"),
        frames_per_clip=frames_per_clip
    )

    val_dataset = ShardClipDataset(
        shards_dir=shards_dir,
        index_filename=data_config["val_index"],
        target_device="cpu",
        max_samples=data_config.get("max_val_samples"),
        frames_per_clip=frames_per_clip
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=ShardClipDataset.collate_fn,
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=ShardClipDataset.collate_fn,
        drop_last=False,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Optimization for Tensor Cores
    torch.set_float32_matmul_precision('medium')

    pl.seed_everything(config.get("seed", 42))

    # Data
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(config)

    # Model
    print("Initializing model...")
    base_model = create_model(config)
    task = DeepfakeTask(base_model, config)

    # Callbacks

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        CUDAMemoryMonitor()
    ]

    if config.get("checkpointing", {}).get("enabled", True):
        callbacks.append(ModelCheckpoint(
            dirpath=config["checkpointing"]["dir"],
            filename="best-{epoch}-{val_loss:.2f}",
            monitor=config["checkpointing"]["monitor"],
            mode=config["checkpointing"]["mode"],
            save_top_k=1,
            save_last=True
        ))

    if "early_stopping" in config["training"]:
        es_config = config["training"]["early_stopping"]
        callbacks.append(EarlyStopping(
            monitor=config["checkpointing"]["monitor"],
            patience=es_config["patience"],
            mode=config["checkpointing"]["mode"],
            min_delta=es_config.get("min_delta", 0.0),
            check_on_train_epoch_end=False
        ))

    # Logger
    wandb_config = config["logging"].get("wandb", {})
    logger = WandbLogger(
        project=wandb_config.get("project", "dfd-lab"),
        offline=wandb_config.get("offline", False),
        log_model=True
    )

    # Trainer
    trainer_config = config.get("trainer", {})
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        default_root_dir=config["logging"]["log_dir"],
        gradient_clip_val=config["training"].get("grad_clip", 0.0),
        **trainer_config
    )

    # Train
    # Determine checkpoint path
    ckpt_path = args.resume
    if ckpt_path is None:
        ckpt_path = config.get("checkpointing", {}).get("resume_from", None)
    
    if ckpt_path:
        print(f"Resuming training from checkpoint: {ckpt_path}")
    else:
        print("Starting training from scratch...")

    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
