from .models_arch.sd15_model import StableDiffusionRadarConditioned
from .models_arch.sd3_model import StableDiffusion3RadarConditioned

def get_model(backbone: str, encoder: str, config: dict):
    """
    Factory function to retrieve the specific configuration for performance comparison.
    backbone: 'sd15' or 'sd3'
    encoder: 'pointnet' or 'pointpillars'
    """
    model_kwargs = {
        "encoder_type": encoder,
        "unet_frozen": config["model"].get("unet_frozen", True),
        "vae_frozen": config["model"].get("vae_frozen", True),
        "train_cross_attention_only": config["model"].get("train_cross_attention_only", True)
    }
    
    if backbone == "sd15":
        return StableDiffusionRadarConditioned(**model_kwargs)
    elif backbone == "sd3":
        return StableDiffusion3RadarConditioned(**model_kwargs)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
