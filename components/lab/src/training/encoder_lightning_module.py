import pytorch_lightning as pl
import torch
from typing import Dict, Any, Tuple, Optional

from src.models.AVFF_encoder import EncoderPretrain, compute_gradient_penalty

class EncoderPretrainTask(pl.LightningModule):
    def __init__(self, model: EncoderPretrain, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.config = config
        self.automatic_optimization = False # Manual optimization for GAN
        
        self.n_critic = config["training"].get("n_critic", 5)
        self.generator_lr = float(config["training"].get("generator_lr", 1e-4))
        self.discriminator_lr = float(config["training"].get("discriminator_lr", 1e-4))

    def forward(self, video, audio):
        return self.model.forward_encoders(video, audio)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            list(self.model.audio_encoder.parameters()) +
            list(self.model.video_encoder.parameters()) +
            list(self.model.audio_decoder.parameters()) +
            list(self.model.video_decoder.parameters()) +
            list(self.model.A2V.parameters()) +
            list(self.model.V2A.parameters()) +
            [self.model.slice_pos],
            lr=self.generator_lr
        )
        opt_d = torch.optim.AdamW(
            list(self.model.audio_disc.parameters()) + list(self.model.video_disc.parameters()),
            lr=self.discriminator_lr
        )
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        # batch is a dictionary
        # Video: (B, T, H, W, C) -> (B, T, C, H, W) and float [0, 1]
        video_batch = batch["video_frames"].float().permute(0, 1, 4, 2, 3) / 255.0
        
        # Audio: (B, T, H, W) -> (B, T, 1, H, W) and float
        audio_batch = batch["audio_frames"].float().unsqueeze(2)
        
        opt_g, opt_d = self.optimizers()
        
        # Discriminator steps
        # We perform n_critic updates to discriminator for every generator update
        # However, in PL manual optimization, we are inside a single training_step call.
        # If we loop here, we use the SAME batch for all n_critic updates, which is standard for WGAN-GP 
        # but sometimes people fetch new batches. Given the dataloader structure, reusing the batch is easiest 
        # and acceptable. Or we can just do 1 D step per training_step and control frequency via global_step, 
        # but the standard WGAN-GP loop is usually inner loop.
        
        # Let's stick to the logic in EncoderTrainer: n_critic steps then 1 generator step.
        # But wait, if we do n_critic steps on the SAME batch, the gradients might be similar.
        # Ideally we want different data. 
        # For simplicity and matching the provided `EncoderTrainer` logic (which takes a batch and loops),
        # I will follow that.
        
        # Tokenize targets once to save time if we were to loop, but `discriminator_step` does it internally.
        # We'll just call the model's helper methods or reimplement the logic here to be safe and clear.
        
        # Actually, `EncoderTrainer.train_batch` does exactly this: loops n_critic times on the SAME batch.
        # So I will replicate that behavior.
        
        d_logs = {}
        for _ in range(self.n_critic):
            # Discriminator step
            # We need to manually call the logic because `discriminator_step` in `EncoderTrainer` 
            # assumes it holds the optimizers. Here we have them from PL.
            
            # 1. Forward to get features
            (v_tokens, v_targets, v_feat), (a_tokens, a_targets, a_feat) = self.model.forward_encoders(video_batch, audio_batch)
            
            # 2. Complementary masking
            (a_vis, mask_a), (v_vis, mask_v) = self.model.apply_complementary_masking(a_feat, v_feat)
            
            # 3. Cross modal predictions
            a_cross = self.model.V2A(v_vis, target_token_count=a_vis.size(1))
            v_cross = self.model.A2V(a_vis, target_token_count=v_vis.size(1))
            
            a_mixed = torch.where(mask_a.unsqueeze(-1), a_cross, a_vis)
            v_mixed = torch.where(mask_v.unsqueeze(-1), v_cross, v_vis)
            
            # 4. Fake recon tokens
            with torch.no_grad():
                a_fake = self.model.audio_decoder(a_mixed)
                v_fake = self.model.video_decoder(v_mixed)
                
            # 5. Real tokens
            a_real = a_targets
            v_real = v_targets
            
            # 6. Update Discriminator
            opt_d.zero_grad()
            
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
            self.manual_backward(loss_d)
            opt_d.step()
            
            d_logs = {
                "d_loss": loss_d.detach(),
                "d_v": loss_d_v.detach(),
                "d_a": loss_d_a.detach(),
                "gp_v": gp_v.detach(),
                "gp_a": gp_a.detach()
            }

        # Generator step
        # The `pretrain_step` method in `EncoderPretrain` computes losses and does backward/step if optimizer provided.
        # We will use it but pass None for optimizer and do backward/step manually to be consistent with PL manual optimization.
        
        # Note: `pretrain_step` re-runs forward pass. This is good because we updated D, but G hasn't changed.
        # However, we might want to re-run forward to get fresh graph for G update.
        
        g_metrics = self.model.pretrain_step(
            video_batch, 
            audio_batch, 
            optimizer_g=None, # We handle optimization
            temperature=0.07
        )
        
        loss_g = g_metrics["loss_g"]
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()
        
        # Logging
        log_dict = {}
        for k, v in d_logs.items():
            log_dict[f"train/{k}"] = v
        for k, v in g_metrics.items():
            log_dict[f"train/{k}"] = v
            
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        video_batch = batch["video_frames"].float().permute(0, 1, 4, 2, 3) / 255.0
        audio_batch = batch["audio_frames"].float().unsqueeze(2)
        
        # For validation, we just compute generator losses to see how reconstruction/contrastive is doing
        # We don't update anything.
        
        g_metrics = self.model.pretrain_step(
            video_batch, 
            audio_batch, 
            optimizer_g=None,
            temperature=0.07
        )
        
        log_dict = {}
        for k, v in g_metrics.items():
            log_dict[f"val/{k}"] = v
            
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
