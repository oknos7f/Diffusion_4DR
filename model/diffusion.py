import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

# Custom Modules
from tokenizer import RadarPointNetPlusPlus
from KRadar_dataset import KRadarDataset


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # ==========================================
    # 0. 설정 (Configuration)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config_path = r'C:\Users\jdmdj\Desktop\Diffusion_4DR\config\config.yaml'
    
    try:
        config = load_config(config_path).get('training', {})
    except FileNotFoundError:
        print("⚠️ Config file not found. Using default settings.")
        config = {}
    
    # 하이퍼파라미터
    BATCH_SIZE = config.get('batch_size', 2)
    ACCUMULATION_STEPS = config.get('accumulation_steps', 4)
    LEARNING_RATE = float(config.get('learning_rate', 1e-5))
    NUM_EPOCHS = config.get('num_epochs', 10)
    NUM_WORKERS = config.get('num_workers', 4)
    
    print(f"🚀 Training Start on {device}")
    
    # ==========================================
    # 1. 모델 로드 및 설정
    # ==========================================
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # A. Pre-trained Models
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # B. Custom Radar Encoder
    radar_encoder = RadarPointNetPlusPlus(sd_hidden_dim=768)
    
    # C. 모델 이동 및 모드 설정
    # VAE는 학습하지 않으므로 eval() 및 gradient 비활성화
    vae.requires_grad_(False)
    vae.to(device, dtype=torch.float32)
    vae.eval()
    
    unet.train()
    unet.to(device)
    
    radar_encoder.train()
    radar_encoder.to(device)
    
    # Gradient Checkpointing (메모리 절약)
    unet.enable_gradient_checkpointing()
    
    # ==========================================
    # 2. 데이터셋 및 전처리
    # ==========================================
    dataset = KRadarDataset(config_path=config_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        list(unet.parameters()) + list(radar_encoder.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-2
    )
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # ==========================================
    # 3. Training Loop
    # ==========================================
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        
        for step, batch in enumerate(progress_bar):
            # ---------------------------------------------------------
            # 1. 데이터 로드
            # ---------------------------------------------------------
            # 이미지와 조건 데이터
            normalized_images = batch['image'].to(device)
            radar_condition = batch['condition'].to(device)
            
            # ---------------------------------------------------------
            # 2. VAE Encoding (중요: Autocast 제외)
            # ---------------------------------------------------------
            with torch.no_grad():
                latents = vae.encode(normalized_images.to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * 0.18215
            
            # ---------------------------------------------------------
            # 3. Forward Pass (Mixed Precision)
            # ---------------------------------------------------------
            with torch.amp.autocast('cuda'):
                # B. Add Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # C. Radar Encoding
                encoder_hidden_states = radar_encoder(radar_condition) # (B, 128, 768)

                # D. UNet Prediction
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                
                # E. Loss Calculation
                loss = F.mse_loss(model_pred, target, reduction="mean")
                loss = loss / ACCUMULATION_STEPS
            
            # ---------------------------------------------------------
            # 4. Backward & Optimizer Step
            # ---------------------------------------------------------
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0:
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(radar_encoder.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                global_step += 1
            
            # Logging
            current_loss = loss.item() * ACCUMULATION_STEPS
            epoch_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        # Epoch 종료 후 저장
        if (epoch + 1) % 1 == 0:
            save_path = f"./checkpoints/checkpoint_epoch_{epoch + 1}.pt"
            os.makedirs("./checkpoints", exist_ok=True)
            
            torch.save({
                'unet_state_dict': unet.state_dict(),
                'radar_encoder_state_dict': radar_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'config': config
            }, save_path)
            print(f"💾 Model saved: {save_path}")
    
    print("🎉 Training Finished!")


if __name__ == "__main__":
    main()