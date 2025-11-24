import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
from base import BaseDetector


class FeatureExtractor(nn.Module):
    """Wrapper around torchvision models to extract features (no classifier)."""
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            layers = list(model.children())[:-1]   # drop final FC, keep avgpool
            self.backbone = nn.Sequential(*layers)
            self.out_dim = model.fc.in_features    # feature size (512)
        elif backbone == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            layers = list(model.children())[:-1]   # drop final FC, keep avgpool
            self.backbone = nn.Sequential(*layers)
            self.out_dim = model.fc.in_features    # feature size (2048)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        return feats


class CNNLSTMDetector(BaseDetector):
    def __init__(self, hidden_size=512, num_classes=2, backbone_video="resnet18", backbone_audio="resnet18"):
        super().__init__(num_classes)
        
        # Use torchvision CNNs for both modalities
        self.video_cnn = FeatureExtractor(backbone=backbone_video, pretrained=True)
        self.audio_cnn = FeatureExtractor(backbone=backbone_audio, pretrained=True)

        fused_dim = self.video_cnn.out_dim + self.audio_cnn.out_dim

        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        B, T, C_vid, H, W = image_input.shape
        _, _, C_aud, _, _ = audio_input.shape
        
        img_flat = image_input.view(B * T, C_vid, H, W)
        img_feats = self.video_cnn(img_flat).view(B, T, -1)
        
        aud_flat = audio_input.view(B * T, C_aud, H, W)
        aud_feats = self.audio_cnn(aud_flat).view(B, T, -1)
        
        fused_feats = torch.cat([img_feats, aud_feats], dim=-1)
        
        lstm_out, (h_n, _) = self.lstm(fused_feats)
        last_hidden = h_n[-1]
        
        logits = self.fc(last_hidden)
        return logits

    def predict(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(image_input, audio_input)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            return predictions

    def get_confidence(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(image_input, audio_input)
            probabilities = torch.softmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            return confidence_scores

    def get_modality_features(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        video_features = self.get_video_features(image_input)
        audio_features = self.get_audio_features(audio_input)
        return video_features, audio_features

    def get_video_features(self, image_input: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = image_input.shape
        
        video_flat = image_input.view(B * T, C, H, W)
        features = self.video_cnn(video_flat)
        features = features.view(B, T, -1)
        final_features = features.mean(dim=1)
        
        return final_features

    def get_audio_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = audio_input.shape
        
        audio_flat = audio_input.view(B * T, C, H, W)
        features = self.audio_cnn(audio_flat)
        features = features.view(B, T, -1)
        final_features = features.mean(dim=1)
        
        return final_features

    def predict_single_modality(self, image_input: Optional[torch.Tensor] = None, 
                                audio_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        if image_input is not None and audio_input is not None:
            raise ValueError("Only one modality can be provided at a time")
        
        if image_input is None and audio_input is None:
            raise ValueError("At least one modality must be provided")
        
        self.eval()
        with torch.no_grad():
            if image_input is not None:
                B, T, C, H, W = image_input.shape
                device = image_input.device
                
                img_flat = image_input.view(B * T, C, H, W)
                img_feats = self.video_cnn(img_flat).view(B, T, -1)
                aud_feats = torch.zeros(B, T, self.audio_cnn.out_dim, device=device)

            elif audio_input is not None:
                B, T, C, H, W = audio_input.shape
                device = audio_input.device
                
                aud_flat = audio_input.view(B * T, C, H, W)
                aud_feats = self.audio_cnn(aud_flat).view(B, T, -1)
                img_feats = torch.zeros(B, T, self.video_cnn.out_dim, device=device)

            fused_feats = torch.cat([img_feats, aud_feats], dim=-1)
            
            _, (h_n, _) = self.lstm(fused_feats)
            last_hidden = h_n[-1]
            
            logits = self.fc(last_hidden)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            return predictions
