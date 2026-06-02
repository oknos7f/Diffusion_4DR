import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from ..encoders.pointnet_plusplus import PhysicsAwareRadarEncoder
from ..encoders.pointpillars import PointPillarsEncoder

class StableDiffusion3RadarConditioned(nn.Module):
    """
    4D Radar-Conditioned SD3 Model using MM-DiT and Flow Matching.
    Supports PointNet++ or PointPillars encoders.
    """
    def __init__(self, 
                 encoder_type: str = "pointnet", 
                 model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 unet_frozen: bool = True,
                 vae_frozen: bool = True,
                 train_cross_attention_only: bool = True):
        super().__init__()
        
        # Load pre-trained SD3 components
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.transformer = SD3Transformer2DModel.from_pretrained(model_id, subfolder="transformer")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # SD3 typical embedding dim is 1536 (context_dim).
        # We project our encoder output to match SD3's expectations.
        self.encoder_type = encoder_type
        if encoder_type == "pointnet":
            self.radar_encoder = PhysicsAwareRadarEncoder()
        else:
            self.radar_encoder = PointPillarsEncoder()
            
        # SD3 projection: 768 -> 1536 (context_dim)
        self.cond_projection = nn.Linear(768, 1536)
        
        # SD3 Pooled Projection: 1536 -> 2048 (pooled_projection_dim for SD3 Medium)
        self.pooled_projection = nn.Linear(1536, 2048)

        # Apply Freezing Strategy
        if vae_frozen:
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
        
        if unet_frozen:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
            
            if train_cross_attention_only:
                # Unfreeze DiT Joint Attention layers
                for name, module in self.transformer.named_modules():
                    if "attn" in name or "joint_transformer" in name.lower():
                        for param in module.parameters():
                            param.requires_grad = True
                self.transformer.train()

    def get_trainable_parameters(self) -> list:
        params = list(self.radar_encoder.parameters()) + list(self.cond_projection.parameters()) + list(self.pooled_projection.parameters())
        params += [p for p in self.transformer.parameters() if p.requires_grad]
        return params

    @torch.no_grad()
    def generate(self, radar_condition: torch.Tensor, num_inference_steps: int = 28):
        """
        Inference method for SD3 visual validation using Flow Matching.
        """
        device = radar_condition.device
        encoder_out = self.radar_encoder(radar_condition)
        encoder_hidden_states = self.cond_projection(encoder_out)
        pooled_out = encoder_hidden_states.mean(dim=1)
        pooled_projections = self.pooled_projection(pooled_out)
        
        # Initial noise
        latents = torch.randn((radar_condition.shape[0], 16, 64, 64), device=device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        for t in self.scheduler.timesteps:
            # Predict noise residual (flow)
            model_pred = self.transformer(
                hidden_states=latents,
                timestep=t.expand(latents.shape[0]),
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                return_dict=False
            )[0]
            
            # Step
            latents = self.scheduler.step(model_pred, t, latents).prev_sample
            
        # Decode
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, pixel_values: torch.Tensor, radar_condition: torch.Tensor) -> dict:
        """
        Flow Matching Loss Calculation.
        """
        device = pixel_values.device
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample timesteps for Flow Matching (u in [0, 1])
        u = torch.rand((bsz,), device=device)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device)

        # Mix noise and latents (Rectified Flow)
        # In SD3/Flow Matching, we use a linear interpolation between noise and latents
        # t=0: latents, t=1000: noise (depending on scheduler convention)
        # FlowMatchEulerDiscreteScheduler: sigmas = timesteps / 1000
        sigmas = u.view(bsz, 1, 1, 1)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        
        # Radar Conditioning
        encoder_out = self.radar_encoder(radar_condition) # [B, 77, 768]
        encoder_hidden_states = self.cond_projection(encoder_out) # [B, 77, 1536]
        
        # Pooled projections (e.g., mean pool over sequence length)
        pooled_out = encoder_hidden_states.mean(dim=1) # [B, 1536]
        pooled_projections = self.pooled_projection(pooled_out) # [B, 2048]

        # Predict flow/velocity
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=False
        )[0]

        # Target for flow matching is (noise - latents)
        target = noise - latents
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        return {"loss": loss}
