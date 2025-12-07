from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.models.AVFF_encoder import PatchTokenizer, CrossModalMapper, AdaptiveCrossModalMapper, TransformerDecoder, slice_pos_expand
from src.models.base import BaseDetector
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class AVClassifier(BaseDetector):
    def __init__(self,
                 num_classes: int = 2,
                 embed_dim: int = 256,
                 video_in_channels: int = 3,
                 audio_in_channels: int = 1,
                 video_patch: Tuple[int, int, int] = (2, 16, 16),
                 audio_patch: Tuple[int, int] = (16, 16),
                 num_slices: int = 8,
                 encoder_layers: int = 2,
                 freeze_encoders: bool = True,
                 pos_freq: float = 0.5):
        """
        Args:
          freeze_encoders: if True, freeze encoder weights initially (fine-tune later).
          pos_freq: Frequency of positive class in dataset. Used to init classifier bias.
        """
        super().__init__(num_classes=num_classes)
        self.embed_dim = embed_dim
        self.num_slices = num_slices

        # Tokenizers
        self.video_tokenizer = PatchTokenizer(video_in_channels, video_patch, embed_dim)
        self.audio_tokenizer = PatchTokenizer(audio_in_channels, audio_patch, embed_dim)

        # Encoders
        layer_v = TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4, batch_first=True)
        self.video_encoder = TransformerEncoder(layer_v, num_layers=encoder_layers)
        layer_a = TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4, batch_first=True)
        self.audio_encoder = TransformerEncoder(layer_a, num_layers=encoder_layers)

        # Cross-modal mappers (adaptive to handle different token counts)
        self.A2V = AdaptiveCrossModalMapper(embed_dim)
        self.V2A = AdaptiveCrossModalMapper(embed_dim)

        # classifier head: take concat of (a_pooled, av_pooled, v_pooled, va_pooled) -> 4*D
        in_dim = 4 * embed_dim
        hidden = max(256, in_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes)
        )
        
        # Init classifier bias for imbalanced data if pos_freq != 0.5
        if pos_freq != 0.5 and num_classes == 2:
            # We want log(p / (1-p)) for the positive class (index 1) relative to neg class (index 0)
            # bias[1] - bias[0] = log(pos_freq / (1 - pos_freq))
            # We can set bias[0] = 0 and bias[1] = ...
            bias_val = torch.log(torch.tensor(pos_freq / (1.0 - pos_freq)))
            # Access the last linear layer
            if hasattr(self.classifier[-1], 'bias') and self.classifier[-1].bias is not None:
                 self.classifier[-1].bias.data[0] = 0.0
                 self.classifier[-1].bias.data[1] = bias_val

        # slice pos param
        self.slice_pos = nn.Parameter(torch.randn(1, num_slices, embed_dim))

        # freeze encoders
        if freeze_encoders:
            self.freeze_encoders()

    def load_encoders(self, path: str, device: str = "cpu"):
        st = torch.load(path, map_location=device)
        if 'video_encoder' in st:
            self.video_encoder.load_state_dict(st['video_encoder'])
        if 'audio_encoder' in st:
            self.audio_encoder.load_state_dict(st['audio_encoder'])
        if 'video_tokenizer' in st:
            self.video_tokenizer.load_state_dict(st['video_tokenizer'])
        if 'audio_tokenizer' in st:
            self.audio_tokenizer.load_state_dict(st['audio_tokenizer'])
        if 'slice_pos' in st:
            self.slice_pos.data = st['slice_pos'].to(self.slice_pos.device)
        if 'A2V' in st:
            self.A2V.load_state_dict(st['A2V'])
        if 'V2A' in st:
            self.V2A.load_state_dict(st['V2A'])

    def freeze_encoders(self):
        for p in self.video_encoder.parameters():
            p.requires_grad = False
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        for p in self.video_tokenizer.parameters():
            p.requires_grad = False
        for p in self.audio_tokenizer.parameters():
            p.requires_grad = False
        for p in self.A2V.parameters():
            p.requires_grad = False
        for p in self.V2A.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        for p in self.video_encoder.parameters():
            p.requires_grad = True
        for p in self.audio_encoder.parameters():
            p.requires_grad = True
        for p in self.video_tokenizer.parameters():
            p.requires_grad = True
        for p in self.audio_tokenizer.parameters():
            p.requires_grad = True
        for p in self.A2V.parameters():
            p.requires_grad = True
        for p in self.V2A.parameters():
            p.requires_grad = True

    def _tokenize_and_encode(self, video: torch.Tensor, audio: torch.Tensor):
        """
        Returns:
          v_tokens (B, Nv, D), v_targets (None), v_feat (B, Nv, D)
          a_tokens (B, Na, D), a_targets(None), a_feat (B, Na, D)
        """
        v_tokens, _ = self.video_tokenizer(video, return_targets=False)
        a_tokens, _ = self.audio_tokenizer(audio, return_targets=False)
        # add slice-based pos embeddings
        v_pos = slice_pos_expand(v_tokens.size(1), self.num_slices, self.slice_pos).to(v_tokens.device)
        a_pos = slice_pos_expand(a_tokens.size(1), self.num_slices, self.slice_pos).to(a_tokens.device)
        v_tokens_pos = v_tokens + v_pos
        a_tokens_pos = a_tokens + a_pos
        v_feat = self.video_encoder(v_tokens_pos)
        a_feat = self.audio_encoder(a_tokens_pos)
        av_tokens = self.V2A(v_feat, target_token_count=a_feat.size(1))
        va_tokens = self.A2V(a_feat, target_token_count=v_feat.size(1))
        return (va_tokens, v_feat), (av_tokens, a_feat)

    def get_video_features(self, image_input: torch.Tensor) -> torch.Tensor:
        (_, v_feat), _ = self._tokenize_and_encode(image_input, image_input)
        return v_feat.mean(dim=1)

    def get_audio_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        _, (a_tokens_pos, a_feat) = self._tokenize_and_encode(audio_input, audio_input)
        return a_feat.mean(dim=1)

    def get_modality_features(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (v_tokens_pos, v_feat), (a_tokens_pos, a_feat) = self._tokenize_and_encode(image_input, audio_input)
        return v_feat.mean(dim=1), a_feat.mean(dim=1)

    def forward(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        (va_tokens, v_feat), (av_tokens, a_feat) = self._tokenize_and_encode(image_input, audio_input)

        # pool (mean over token dimension)
        a_unimodal = a_feat.mean(dim=1)
        v_unimodal = v_feat.mean(dim=1)
        a_cross = av_tokens.mean(dim=1)
        v_cross = va_tokens.mean(dim=1)

        # create modality feature vectors fa, fv and classifier input
        fa = torch.cat([a_unimodal, a_cross], dim=-1)
        fv = torch.cat([v_unimodal, v_cross], dim=-1)
        x = torch.cat([fa, fv], dim=-1)

        logits = self.classifier(x)
        return logits

    def get_confidence(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        logits = self.forward(image_input, audio_input)
        probs = F.softmax(logits, dim=-1)
        return probs

    def predict(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        logits = self.forward(image_input, audio_input)
        return torch.argmax(logits, dim=-1)

    def predict_single_modality(self, image_input: Optional[torch.Tensor] = None, 
                                audio_input: Optional[torch.Tensor] = None) -> int:
        if image_input is not None and audio_input is not None:
            # Both modalities provided, use full forward pass
            logits = self.forward(image_input, audio_input)
        elif image_input is not None:
            # Only video provided - use video features with dummy audio
            # Create dummy audio with same temporal dimension but single channel
            B, C, T, H, W = image_input.shape
            dummy_audio = torch.zeros(B, 1, T, H, W, device=image_input.device, dtype=image_input.dtype)
            logits = self.forward(image_input, dummy_audio)
        elif audio_input is not None:
            # Only audio provided - use audio features with dummy video
            # Create dummy video with same temporal dimension but 3 channels
            B, C, T, H, W = audio_input.shape
            dummy_video = torch.zeros(B, 3, T, H, W, device=audio_input.device, dtype=audio_input.dtype)
            logits = self.forward(dummy_video, audio_input)
        else:
            raise ValueError("At least one of image_input or audio_input must be provided")
        
        return torch.argmax(logits, dim=-1).item()

    # convenience: load encoders into this detector
    def load_trained_encoders(self, path: str, device: str = "cpu", load_a2v_v2a: bool = False):
        self.load_encoders(path, device=device)
