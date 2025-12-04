import math
from typing import Tuple, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def slice_pos_expand(token_len: int, num_slices: int, slice_pos: torch.Tensor) -> torch.Tensor:
    """
    Expand per-slice positional embeddings to per-token positional embeddings.
    Args:
        token_len: number of tokens (Na or Nv)
        num_slices: desired number of temporal slices
        slice_pos: (1, num_slices, D)
    Returns:
        pos (1, token_len, D)
    """
    device = slice_pos.device
    base = token_len // num_slices
    rem = token_len % num_slices
    counts = [base + (1 if i < rem else 0) for i in range(num_slices)]
    idxs = []
    for i, c in enumerate(counts):
        idxs += [i] * c
    idxs = torch.tensor(idxs, device=device, dtype=torch.long)
    pos = slice_pos[0, idxs]
    return pos.unsqueeze(0)


def info_nce_loss(a_feat: torch.Tensor, v_feat: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Bi-directional InfoNCE loss.
    a_feat, v_feat: (B, D) pooled features
    """
    B = a_feat.shape[0]
    a_norm = F.normalize(a_feat, dim=-1)
    v_norm = F.normalize(v_feat, dim=-1)
    logits = torch.matmul(a_norm, v_norm.t()) / temperature
    labels = torch.arange(B, device=a_feat.device)
    loss_a2v = F.cross_entropy(logits, labels)
    loss_v2a = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a2v + loss_v2a)


class PatchTokenizer(nn.Module):
    """
    Tokenize 5D inputs (video or audio) into non-overlapping 3D patches.
    Input: (B, T, C, H, W)
    Output: tokens (B, N, D), optional targets for reconstruction.
    """
    def __init__(self, in_channels: int, patch_size: Tuple[int, ...], embed_dim: int):
        super().__init__()
        if len(patch_size) == 3:
            # 3D conv (video)
            self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.patch_type = "3d"
        elif len(patch_size) == 2:
            # 2D conv (audio)
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.patch_type = "2d"
        else:
            raise ValueError("patch_size must be 2D or 3D tuple")

    def forward(self, x: torch.Tensor, return_targets: bool = False):
        if self.patch_type == "3d":
            # Expect (B, C, T, H, W)
            if x.ndim == 5:
                x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            p = self.proj(x)  # (B, D, T', H', W')
            B, D, T, H, W = p.shape
            tokens = p.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, D)
        else:
            # 2D patches per frame: (B, T, C, H, W) -> flatten batch and frames
            B, T, C, H, W = x.shape
            x = x.reshape(B*T, C, H, W)
            p = self.proj(x)  # (B*T, D, H', W')
            D, Hp, Wp = p.shape[1:]
            tokens = p.flatten(2).transpose(1, 2)  # (B*T, N, D)
            tokens = tokens.reshape(B, T*Hp*Wp, D)

        targets = tokens.detach().clone() if return_targets else None
        return (tokens, targets) if return_targets else (tokens, None)


class CrossModalMapper(nn.Module):
    def __init__(self, embed_dim: int, nhead: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4, batch_first=True)
        self.transformer = TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.mlp(x))

class AdaptiveCrossModalMapper(nn.Module):
    """
    Cross-modal mapper that can handle different token counts between modalities.
    Uses pooling and upsampling to match token dimensions.
    """
    def __init__(self, embed_dim: int, nhead: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4, batch_first=True)
        self.transformer = TransformerEncoder(layer, num_layers=1)
        
    def forward(self, source_features: torch.Tensor, target_token_count: int) -> torch.Tensor:
        """
        Map source features to target token count.
        
        Args:
            source_features: (B, N_source, D)
            target_token_count: desired number of output tokens
            
        Returns:
            mapped_features: (B, target_token_count, D)
        """
        B, N_source, D = source_features.shape
        
        if N_source == target_token_count:
            # Same size, just apply transformation
            return self.transformer(self.mlp(source_features))
        elif N_source > target_token_count:
            # Downsample: pool to target size
            pooled = source_features.mean(dim=1, keepdim=True)  # (B, 1, D)
            # Repeat to target size
            repeated = pooled.repeat(1, target_token_count, 1)  # (B, target_token_count, D)
            return self.transformer(self.mlp(repeated))
        else:
            # Upsample: interpolate to target size
            # Use linear interpolation along the sequence dimension
            source_features = source_features.transpose(1, 2)  # (B, D, N_source)
            interpolated = torch.nn.functional.interpolate(
                source_features, 
                size=target_token_count, 
                mode='linear', 
                align_corners=False
            )  # (B, D, target_token_count)
            interpolated = interpolated.transpose(1, 2)  # (B, target_token_count, D)
            return self.transformer(self.mlp(interpolated))

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4, batch_first=True)
        self.decoder = TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(embed_dim, embed_dim)  # output token embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.decoder(x))


class PatchDiscriminator(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = hidden_dim or (embed_dim // 2 if embed_dim >= 4 else embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        logits = logits.squeeze(-1).mean(dim=1)
        return logits


class EncoderPretrain(nn.Module):
    def __init__(self,
                 embed_dim: int = 256,
                 video_in_channels: int = 3,
                 audio_in_channels: int = 1,
                 video_patch: Tuple[int, int, int] = (2, 16, 16),
                 audio_patch: Tuple[int, int] = (16, 16),
                 num_slices: int = 8,
                 encoder_layers: int = 2):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_slices = num_slices

        # Tokenizers
        self.video_tokenizer = PatchTokenizer(video_in_channels, video_patch, embed_dim)
        self.audio_tokenizer = PatchTokenizer(audio_in_channels, audio_patch, embed_dim)

        # Encoders (small)
        layer_v = TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4, batch_first=True)
        self.video_encoder = TransformerEncoder(layer_v, num_layers=encoder_layers)
        layer_a = TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4, batch_first=True)
        self.audio_encoder = TransformerEncoder(layer_a, num_layers=encoder_layers)

        # Cross-modal mappers
        self.A2V = AdaptiveCrossModalMapper(embed_dim)
        self.V2A = AdaptiveCrossModalMapper(embed_dim)

        # Decoders
        self.video_decoder = TransformerDecoder(embed_dim)
        self.audio_decoder = TransformerDecoder(embed_dim)

        # Discriminators
        self.video_disc = PatchDiscriminator(embed_dim)
        self.audio_disc = PatchDiscriminator(embed_dim)

        # Slice-level positional embeddings (learnable)
        self.slice_pos = nn.Parameter(torch.randn(1, num_slices, embed_dim))

    def forward_encoders(self, video_input: torch.Tensor, audio_input: torch.Tensor):
        # Tokenize and get token-space targets
        v_tokens, v_targets = self.video_tokenizer(video_input, return_targets=True)
        a_tokens, a_targets = self.audio_tokenizer(audio_input, return_targets=True)
        # Add slice-based positional embeddings (expanded to tokens)
        v_pos = slice_pos_expand(v_tokens.size(1), self.num_slices, self.slice_pos).to(v_tokens.device)
        a_pos = slice_pos_expand(a_tokens.size(1), self.num_slices, self.slice_pos).to(a_tokens.device)
        v_tokens = v_tokens + v_pos
        a_tokens = a_tokens + a_pos
        # Encode
        v_feat = self.video_encoder(v_tokens)
        a_feat = self.audio_encoder(a_tokens)
        return (v_tokens, v_targets, v_feat), (a_tokens, a_targets, a_feat)

    def apply_complementary_masking(self, a_feat: torch.Tensor, v_feat: torch.Tensor):
        """
        Create complementary slice masks aligned by slice index (time).
        a_feat: (B, Na, D), v_feat: (B, Nv, D)
        Return:
            (a_vis, mask_a), (v_vis, mask_v) where
            a_vis/v_vis are same shape as inputs but masked positions zeroed,
            masks are boolean (B, Na)/(B, Nv) True -> masked
        """
        B, Na, D = a_feat.shape
        _, Nv, _ = v_feat.shape
        slice_len_a = Na // self.num_slices
        slice_len_v = Nv // self.num_slices
        # compute per-slice token ranges robustly
        a_ranges = []
        start = 0
        for s in range(self.num_slices):
            end = start + slice_len_a + (1 if s < (Na % self.num_slices) else 0)
            a_ranges.append((start, min(end, Na)))
            start = end
        v_ranges = []
        start = 0
        for s in range(self.num_slices):
            end = start + slice_len_v + (1 if s < (Nv % self.num_slices) else 0)
            v_ranges.append((start, min(end, Nv)))
            start = end

        mask_a = torch.zeros(B, Na, dtype=torch.bool, device=a_feat.device)
        mask_v = torch.zeros(B, Nv, dtype=torch.bool, device=v_feat.device)

        for i in range(B):
            # Ensure we mask exactly half the slices (or as close as possible)
            num_mask = self.num_slices // 2
            chosen = torch.randperm(self.num_slices, device=a_feat.device)[:num_mask]
            chosen_set = set(chosen.tolist())
            for s in range(self.num_slices):
                a_s, a_e = a_ranges[s]
                v_s, v_e = v_ranges[s]
                
                # Skip empty slices
                if a_s >= a_e and v_s >= v_e:
                    continue
                    
                if s in chosen_set:
                    # mask this slice in audio, keep video visible
                    if a_s < a_e:
                        mask_a[i, a_s:a_e] = True
                    # ensure video slice visible (explicit)
                    if v_s < v_e:
                        mask_v[i, v_s:v_e] = False
                else:
                    # mask this slice in video, ensure audio visible
                    if v_s < v_e:
                        mask_v[i, v_s:v_e] = True
                    if a_s < a_e:
                        mask_a[i, a_s:a_e] = False

        # create visible versions (zero-out masked tokens)
        a_vis = a_feat.masked_fill(mask_a.unsqueeze(-1), 0.0)
        v_vis = v_feat.masked_fill(mask_v.unsqueeze(-1), 0.0)
        return (a_vis, mask_a), (v_vis, mask_v)

    def pretrain_step(self,
                      video_input: torch.Tensor,
                      audio_input: torch.Tensor,
                      optimizer_g: torch.optim.Optimizer,
                      optimizer_va: Optional[torch.optim.Optimizer] = None,
                      optimizer_aa: Optional[torch.optim.Optimizer] = None,
                      temperature: float = 0.07) -> Dict[str, torch.Tensor]:
        """
        One generator step (compute losses but not update discriminators).
        For WGAN-GP training, discriminator updates are handled separately in EncoderTrainer.
        This function returns the computed losses (and applies generator update if optimizer_g provided).
        """
        device = next(self.parameters()).device
        (v_tokens, v_targets, v_feat), (a_tokens, a_targets, a_feat) = self.forward_encoders(video_input, audio_input)

        # Complementary masking
        (a_vis, mask_a), (v_vis, mask_v) = self.apply_complementary_masking(a_feat, v_feat)

        # Cross-modal predictions (only feed visible tokens)
        # Use adaptive mappers to handle different token counts
        a_cross = self.V2A(v_vis, target_token_count=a_vis.size(1))
        v_cross = self.A2V(a_vis, target_token_count=v_vis.size(1))

        # Mixed embeddings: replace masked positions with cross-modal predictions
        a_mixed = torch.where(mask_a.unsqueeze(-1), a_cross, a_vis)
        v_mixed = torch.where(mask_v.unsqueeze(-1), v_cross, v_vis)

        # Decode to token embedding space
        a_recon_tokens = self.audio_decoder(a_mixed)
        v_recon_tokens = self.video_decoder(v_mixed)

        # Reconstruction loss computed only on masked tokens against token targets (not raw pixels)
        loss_a_rec = F.mse_loss(a_recon_tokens[mask_a], a_targets[mask_a])
        loss_v_rec = F.mse_loss(v_recon_tokens[mask_v], v_targets[mask_v])

        # Contrastive InfoNCE on pooled encoder features (full features, not masked)
        a_pool = a_feat.mean(dim=1)
        v_pool = v_feat.mean(dim=1)
        loss_contrast = info_nce_loss(a_pool, v_pool, temperature=temperature)

        # Generator adversarial losses (use discriminators scoring of reconstructed tokens)
        adv_a = -self.audio_disc(a_recon_tokens).mean()
        adv_v = -self.video_disc(v_recon_tokens).mean()

        # Total generator loss
        lambda_rec = 1.0
        lambda_adv = 0.1
        lambda_contrast = 1.0
        total_loss = lambda_rec * (loss_a_rec + loss_v_rec) + lambda_adv * (adv_a + adv_v) + lambda_contrast * loss_contrast

        if optimizer_g is not None:
            optimizer_g.zero_grad()
            total_loss.backward()
            optimizer_g.step()

        return {
            "audio_rec": loss_a_rec.detach(),
            "video_rec": loss_v_rec.detach(),
            "adv_a": adv_a.detach(),
            "adv_v": adv_v.detach(),
            "contrast": loss_contrast.detach(),
            "total": total_loss.detach(),
            "loss_g": total_loss
        }

    def save_encoders(self, path: str):
        state = {
            "video_encoder": self.video_encoder.state_dict(),
            "audio_encoder": self.audio_encoder.state_dict(),
            "video_tokenizer": self.video_tokenizer.state_dict(),
            "audio_tokenizer": self.audio_tokenizer.state_dict(),
            "slice_pos": self.slice_pos.detach().cpu(),
            "A2V": self.A2V.state_dict(),
            "V2A": self.V2A.state_dict(),
            }
        torch.save(state, path)

    def load_encoders(self, path: str, device: str = "cpu"):
        state = torch.load(path, map_location=device)
        self.video_encoder.load_state_dict(state["video_encoder"])
        self.audio_encoder.load_state_dict(state["audio_encoder"])
        self.video_tokenizer.load_state_dict(state["video_tokenizer"])
        self.audio_tokenizer.load_state_dict(state["audio_tokenizer"])
        self.slice_pos.data = state["slice_pos"].to(self.slice_pos.device)
        self.A2V.load_state_dict(state["A2V"])
        self.V2A.load_state_dict(state["V2A"])


def compute_gradient_penalty(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor, device: torch.device, gp_weight: float = 10.0):
    """
    real, fake: (B, N, D) token embeddings
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = discriminator(interp) 
    grads = torch.autograd.grad(outputs=scores.sum(), inputs=interp, create_graph=True)[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp * gp_weight


class EncoderTrainer:
    def __init__(self, model: EncoderPretrain, lr_g: float = 1e-4, lr_d: float = 1e-4, device: str = "cuda", n_critic: int = 5):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.n_critic = n_critic
        # Optimizers
        self.opt_g = torch.optim.AdamW(
            list(self.model.audio_encoder.parameters()) +
            list(self.model.video_encoder.parameters()) +
            list(self.model.audio_decoder.parameters()) +
            list(self.model.video_decoder.parameters()) +
            list(self.model.A2V.parameters()) +
            list(self.model.V2A.parameters()) +
            [self.model.slice_pos],  # slice_pos is a single Parameter, not a module
            lr=lr_g
        )
        self.opt_d = torch.optim.AdamW(
            list(self.model.audio_disc.parameters()) + list(self.model.video_disc.parameters()),
            lr=lr_d
        )

    def discriminator_step(self, video_frames, audio_frames):
        (v_tokens, v_targets, v_feat), (a_tokens, a_targets, a_feat) = self.model.forward_encoders(video_frames, audio_frames)

        # Apply complementary masking to get visible features and masks
        (a_vis, mask_a), (v_vis, mask_v) = self.model.apply_complementary_masking(a_feat, v_feat)

        # cross modal predictions with adaptive token counts
        a_cross = self.model.V2A(v_vis, target_token_count=a_vis.size(1))  # Map video to audio token count
        v_cross = self.model.A2V(a_vis, target_token_count=v_vis.size(1))  # Map audio to video token count

        a_mixed = torch.where(mask_a.unsqueeze(-1), a_cross, a_vis)
        v_mixed = torch.where(mask_v.unsqueeze(-1), v_cross, v_vis)

        # fake recon tokens (detach for discriminator real/fake comparison)
        with torch.no_grad():
            a_fake = self.model.audio_decoder(a_mixed)
            v_fake = self.model.video_decoder(v_mixed)

        # real tokens for masked positions
        a_real = a_targets
        v_real = v_targets

        # discriminator update
        self.opt_d.zero_grad()

        real_v_score = self.model.video_disc(v_real)
        fake_v_score = self.model.video_disc(v_fake)
        loss_d_v = fake_v_score.mean() - real_v_score.mean()
        gp_v = compute_gradient_penalty(self.model.video_disc, v_real, v_fake, device=self.device)
        loss_d_v = loss_d_v + gp_v

        real_a_score = self.model.audio_disc(a_real)
        fake_a_score = self.model.audio_disc(a_fake)
        loss_d_a = fake_a_score.mean() - real_a_score.mean()
        gp_a = compute_gradient_penalty(self.model.audio_disc, a_real, a_fake, device=self.device)
        loss_d_a = loss_d_a + gp_a

        loss_d = loss_d_v + loss_d_a
        loss_d.backward()
        self.opt_d.step()

        return {"d_loss": loss_d.detach(), "d_v": loss_d_v.detach(), "d_a": loss_d_a.detach(), "gp_v": gp_v.detach(), "gp_a": gp_a.detach()}

    def train_batch(self, video_batch: torch.Tensor, audio_batch: torch.Tensor):
        """
        Perform n_critic discriminator steps, then one generator step.
        Accepts raw inputs:
         - video_batch: (B, C, T, H, W)
         - audio_batch: (B, C, T, F)
        """
        self.model.train()
        video_batch = video_batch.to(self.device)
        audio_batch = audio_batch.to(self.device)

        # For discriminator steps we provide tokenized targets; use tokenizer directly
        v_tokens, v_targets = self.model.video_tokenizer(video_batch, return_targets=True)
        a_tokens, a_targets = self.model.audio_tokenizer(audio_batch, return_targets=True)
        # Move token targets to device
        v_tokens = v_tokens.to(self.device); v_targets = v_targets.to(self.device)
        a_tokens = a_tokens.to(self.device); a_targets = a_targets.to(self.device)

        d_logs = {}
        for _ in range(self.n_critic):
            d_log = self.discriminator_step(video_batch, audio_batch)
            # aggregate logs (just keep last for simplicity)
            d_logs = d_log

        # Generator step (updates encoders, decoders, cross-mappers, slice_pos)
        g_logs = self.model.pretrain_step(video_batch, audio_batch, optimizer_g=self.opt_g, temperature=0.07)

        logs = {"d": d_logs, "g": g_logs}
        return logs

