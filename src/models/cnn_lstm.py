import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
from src.models.base import BaseDetector


class FeatureExtractor(nn.Module):
    """Wrapper around torchvision models to extract features (no classifier)."""
    def __init__(self, backbone="resnet18", pretrained=True, in_channels=3):
        super().__init__()
        if backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Handle input channels if not 3 (standard ImageNet)
        if in_channels != 3:
            # Get original layer
            original_conv = model.conv1
            
            # Create new layer with desired input channels
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize weights
            if pretrained:
                with torch.no_grad():
                    # Sum weights across channel dimension to preserve activation magnitude roughly
                    # Original: (Out, 3, K, K) -> Sum: (Out, 1, K, K) -> Expand if needed or just use
                    # For 1 channel, we can just sum. For N channels, it's more complex, but here we likely need 1.
                    if in_channels == 1:
                        new_conv.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)
                    else:
                        # Simple initialization for other cases
                        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            model.conv1 = new_conv

        layers = list(model.children())[:-1]   # drop final FC, keep avgpool
        self.backbone = nn.Sequential(*layers)
        self.out_dim = model.fc.in_features    # feature size

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        return feats


class CNNLSTMDetector(BaseDetector):
    def __init__(self, hidden_size=512, num_classes=2, backbone_video="resnet18", backbone_audio="resnet18", pos_freq=None):
        super().__init__(num_classes)
        
        # Use torchvision CNNs for both modalities
        self.video_cnn = FeatureExtractor(backbone=backbone_video, pretrained=True, in_channels=3)
        self.audio_cnn = FeatureExtractor(backbone=backbone_audio, pretrained=True, in_channels=1)

        fused_dim = self.video_cnn.out_dim + self.audio_cnn.out_dim

        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Bias Initialization for Imbalanced Classes
        if pos_freq is not None and num_classes == 2:
            import math
            # Assume class 1 is positive (Fake) and class 0 is negative (Real)
            # b1 - b0 = log(pi / (1 - pi))
            # We set b0 = 0, b1 = log(pos_freq / (1 - pos_freq))
            # Or symmetric: b1 = log(pi/(1-pi))/2, b0 = -b1
            
            bias_val = math.log(pos_freq / (1.0 - pos_freq))
            # Set bias for class 1
            with torch.no_grad():
                self.fc.bias[1] = bias_val / 2
                self.fc.bias[0] = -bias_val / 2
            print(f"Initialized output bias for pos_freq={pos_freq:.2f}: {self.fc.bias.data}")

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
