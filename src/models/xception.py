from typing import Tuple, Optional

import timm
import torch

from src.models.base import BaseDetector


class XceptionMaxFusionDetector(BaseDetector):
    """
    Deepfake detector utilizing Xception from torchvision for both video (image) and audio (spectrogram)
    modalities, with a late fusion strategy based on taking the maximum of their
    individual predicted logits.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__(num_classes)

        # --- Video Predictor (Xception from timm) ---
        self.video_model = timm.create_model('xception', pretrained=True, num_classes=num_classes)

        # --- Audio Predictor (Xception from timm) ---
        self.audio_model = timm.create_model('xception', pretrained=True, num_classes=num_classes, in_chans=1)

    def forward(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the XceptionMaxFusionDetector.

        Args:
            image_input (torch.Tensor): (B, T, 3, H, W) video frames
            audio_input (torch.Tensor): (B, T, 1, H, W) spectrogram frames

        Returns:
            torch.Tensor: (B, num_classes) fused logits
        """
        # Check input ranks
        if image_input.ndim != 5 or audio_input.ndim != 5:
            raise ValueError("Inputs must have shape (B, T, C, H, W)")

        B, T, C_v, H, W = image_input.shape
        _, _, C_a, _, _ = audio_input.shape  # just for clarity

        # Reshape: process all frames at once
        video_reshaped = image_input.view(B * T, C_v, H, W)  # (B*T, 3, H, W)
        audio_reshaped = audio_input.view(B * T, C_a, H, W)  # (B*T, 1, H, W)

        # Forward through backbones
        video_logits = self.video_model(video_reshaped)  # (B*T, num_classes)
        audio_logits = self.audio_model(audio_reshaped)  # (B*T, num_classes)

        # Late fusion (elementwise max across modalities)
        fused_logits = torch.max(video_logits, audio_logits)  # (B*T, num_classes)

        # Reshape back into sequence format
        fused_logits = fused_logits.view(B, T, self.num_classes)

        # Temporal pooling across frames (max-pooling)
        final_logits, _ = fused_logits.max(dim=1)  # (B, num_classes)

        return final_logits

    def predict(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """Get model predictions (single class) for multi-modal input.
        
        Args:
            image_input: Input tensor of shape (B, T, 3, H, W)
            audio_input: Input tensor of shape (B, T, 1, H, W) 
        Returns:
            Tensor of predicted classes (0 or 1) - one prediction per video in the batch
        """
        self.eval()
        with torch.no_grad():
            probabilities = torch.softmax(self.forward(image_input, audio_input), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            return predictions

    def get_confidence(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """Get prediction confidence scores."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(image_input, audio_input)
            probabilities = torch.softmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            return confidence_scores

    def get_modality_features(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Extract features from both modalities."""
        video_features = self.get_video_features(image_input)
        audio_features = self.get_audio_features(audio_input)
        return video_features, audio_features

    def get_video_features(self, image_input: torch.Tensor) -> torch.Tensor:
        """Extract video features using Xception backbone."""
        if image_input.ndim != 5:
            raise ValueError("Input must be a video sequence with shape (B, T, C, H, W)")

        B, T, C, H, W = image_input.shape
        video_reshaped = image_input.view(B * T, C, H, W)

        features = self.video_model.forward_features(video_reshaped)

        if features.dim() == 4:
            features = features.mean(dim=[-2, -1])

        features = features.view(B, T, -1)
        final_features = features.mean(dim=1)

        return final_features

    def get_audio_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        """Extract audio features using Xception backbone."""
        if audio_input.ndim != 5:
            raise ValueError("Input must be an audio sequence with shape (B, T, C, H, W)")

        B, T, C, H, W = audio_input.shape
        audio_reshaped = audio_input.view(B * T, C, H, W)

        features = self.audio_model.forward_features(audio_reshaped)

        if features.dim() == 4:
            features = features.mean(dim=[-2, -1])

        features = features.view(B, T, -1)
        final_features = features.mean(dim=1)

        return final_features

    def predict_single_modality(self, image_input: Optional[torch.Tensor] = None,
                                audio_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get predictions using only one modality."""

        if image_input is not None and audio_input is not None:
            raise ValueError("Only one modality can be provided at a time")

        if image_input is None and audio_input is None:
            raise ValueError("At least one modality must be provided")

        self.eval()
        with torch.no_grad():
            if image_input is not None:
                B, T, C, H, W = image_input.shape
                frames = image_input.view(B * T, C, H, W)
                logits = self.video_model(frames)
                logits = logits.view(B, T, -1)

            elif audio_input is not None:
                B, T, C, H, W = audio_input.shape
                frames = audio_input.view(B * T, C, H, W)
                logits = self.audio_model(frames)
                logits = logits.view(B, T, -1)

            final_logits, _ = logits.max(dim=1)
            probs = torch.softmax(final_logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

            return predictions
