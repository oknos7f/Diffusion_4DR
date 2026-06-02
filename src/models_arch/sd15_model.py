import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from ..encoders.pointnet_plusplus import PhysicsAwareRadarEncoder
from ..encoders.pointpillars import PointPillarsEncoder

class StableDiffusionRadarConditioned(nn.Module):
    """
    4D Radar-Conditioned SD1.5 Model using U-Net.
    Supports PointNet++ or PointPillars encoders.
    """
    def __init__(self, 
                 encoder_type: str = "pointnet", 
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 unet_frozen: bool = True,
                 vae_frozen: bool = True,
                 train_cross_attention_only: bool = True):
        super().__init__()
        
        # Load pre-trained components
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Select Encoder
        if encoder_type == "pointnet":
            self.radar_encoder = PhysicsAwareRadarEncoder()
        else:
            self.radar_encoder = PointPillarsEncoder()

        # Apply Freezing Strategy based on config
        if vae_frozen:
            self.freeze_vae()
        
        if unet_frozen:
            if train_cross_attention_only:
                self.freeze_unet_except_cross_attention()
            else:
                self.freeze_unet_entirely()

    def freeze_vae(self):
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def freeze_unet_entirely(self):
        self.unet.eval()
        for param in self.unet.parameters():
            param.requires_grad = False

    def freeze_unet_except_cross_attention(self):
        self.unet.eval()
        for param in self.unet.parameters():
            param.requires_grad = False
            
        for name, module in self.unet.named_modules():
            if "attn2" in name or "cross_attention" in name.lower():
                for param in module.parameters():
                    param.requires_grad = True
        self.unet.train()

    def get_trainable_parameters(self) -> list:
        trainable_params = list(self.radar_encoder.parameters())
        trainable_params += [p for p in self.unet.parameters() if p.requires_grad]
        return trainable_params

    @torch.no_grad()
    def generate(self, radar_condition: torch.Tensor, num_inference_steps: int = 50):
        """
        Inference method for visual validation.
        """
        device = radar_condition.device
        encoder_hidden_states = self.radar_encoder(radar_condition)
        
        # Initial random noise
        latents = torch.randn((radar_condition.shape[0], 4, 64, 64), device=device)
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            # Predict noise residual
            noise_pred = self.unet(latents, t, encoder_hidden_states).sample
            # Compute previous noisy sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            
        # Decode latents
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, pixel_values: torch.Tensor, radar_condition: torch.Tensor) -> dict:
        """
        Standard MSE Denoising Objective.
        """
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.radar_encoder(radar_condition)

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        return {"loss": loss}
