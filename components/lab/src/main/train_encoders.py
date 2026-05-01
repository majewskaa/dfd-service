#!/usr/bin/env python3
"""
Training Script for avff Encoders (Pretraining)

Usage:
    python train_encoders.py --config configs/encoder_training.yaml
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
from src.models.avff_encoder import EncoderPretrain
from src.training.encoder_lightning_module import EncoderPretrainTask
from src.callbacks.memory_monitor import CUDAMemoryMonitor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/encoder_training.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def create_model(config: Dict[str, Any]) -> EncoderPretrain:
    model_config = config["model"]
    model = EncoderPretrain(
        embed_dim=model_config["embed_dim"],
        video_in_channels=model_config["video_in_channels"],
        audio_in_channels=model_config["audio_in_channels"],
        video_patch=tuple(model_config["video_patch"]),
        audio_patch=tuple(model_config["audio_patch"]),
        num_slices=model_config["num_slices"],
        encoder_layers=model_config["encoder_layers"]
    )
    return model


def create_data_loaders(config: Dict[str, Any]):
    data_config = config["data"]
    shards_dir = data_config["shards_dir"]
    # frames_per_clip is implicit in video_patch[0] * num_slices usually, 
    # but ShardClipDataset needs it. 
    # In avff_encoder, video_patch is (T_patch, H, W). 
    # Total frames = T_patch * num_slices? No, slice_pos_expand suggests num_slices is temporal.
    # Let's check how many frames we need.
    # The model takes (B, C, T, H, W).
    # video_patch is e.g. (2, 16, 16).
    # If we have num_slices=8, does that mean we need specific T?
    # The code doesn't strictly enforce T, but `slice_pos_expand` logic depends on token count.
    # Let's assume we want enough frames.
    # In `preprocessing.yaml`, `clip_length` is 25.
    # Let's use that or what's in config.
    
    frames_per_clip = data_config.get("frames_per_clip", 32) # Default to 32 if not specified
    num_workers = data_config.get("num_workers", 4)
    batch_size = data_config.get("batch_size", 16)

    train_dataset = ShardClipDataset(
        shards_dir=shards_dir,
        index_filename=data_config["train_index"],
        target_device="cpu",
        max_samples=data_config.get("max_train_samples"),
        frames_per_clip=frames_per_clip
    )

    # For pretraining, we might not have a validation set split in shards if we just processed `pretrain` folder.
    # But `ShardClipDataset` expects an index file.
    # If `pretrain_preprocessor` generated `index.csv`, we might need to split it or use it for both?
    # Usually `index.csv` contains everything.
    # If `train_index` and `val_index` are specified in config, we assume they exist.
    # If user only has `index.csv`, they might need to split it.
    # For now, let's assume the user handles index splitting or points to `index.csv` for both (bad practice but works for running).
    
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
    task = EncoderPretrainTask(base_model, config)

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        CUDAMemoryMonitor()
    ]

    if config.get("checkpointing", {}).get("enabled", True):
        callbacks.append(ModelCheckpoint(
            dirpath=config["checkpointing"]["dir"],
            filename="encoder-{epoch}-{val/total:.2f}",
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
            min_delta=es_config.get("min_delta", 0.0)
        ))

    # Logger
    wandb_config = config["logging"].get("wandb", {})
    logger = WandbLogger(
        project=wandb_config.get("project", "dfd-lab-pretrain"),
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
    print("Starting training...")
    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
