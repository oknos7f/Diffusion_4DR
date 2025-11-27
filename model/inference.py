import torch
import yaml
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

# Custom Modules (작성하신 파일들이 같은 경로에 있어야 합니다)
from tokenizer import RadarPointNetPlusPlus
from KRadar_dataset import KRadarDataset
# utils가 존재한다고 가정합니다 (학습 코드와 동일 환경)
import utils.data_processing as dp


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class RadarDiffusionInference:
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        self.device = device
        self.config = load_config(config_path)
        
        print(f"🚀 Loading models from {checkpoint_path}...")
        
        # 1. 모델 초기화 (Pre-trained weights 불러오기)
        model_id = "runwayml/stable-diffusion-v1-5"
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # 2. Custom Radar Encoder 초기화
        # 학습 코드의 hidden_dim과 일치해야 합니다 (기본 768)
        self.radar_encoder = RadarPointNetPlusPlus(sd_hidden_dim=768).to(self.device)
        
        # 3. 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # UNet 가중치 로드
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        # Radar Encoder 가중치 로드
        self.radar_encoder.load_state_dict(checkpoint['radar_encoder_state_dict'])
        
        # 4. Evaluation 모드 설정
        self.vae.eval()
        self.unet.eval()
        self.radar_encoder.eval()
        
        print("✅ Models loaded successfully.")
    
    @torch.no_grad()
    def generate(self, radar_condition_tensor, num_inference_steps=50, guidance_scale=1.0):
        """
        radar_condition_tensor: (1, N, 4) 형태의 텐서 (Batch size 1 가정)
        """
        # 이미지 크기 설정 (Config에서 가져오거나 기본값)
        height = self.config.get('dataset', {}).get('target_height', 384)
        width = self.config.get('dataset', {}).get('target_width', 680)
        
        # Latent 크기는 이미지의 1/8
        latent_height = height // 8
        latent_width = width // 8
        
        batch_size = radar_condition_tensor.shape[0]
        
        # 1. Radar Encoding (Conditioning)
        radar_condition_tensor = radar_condition_tensor.to(self.device)
        # (B, N, 4) -> (B, 128, 768)
        encoder_hidden_states = self.radar_encoder(radar_condition_tensor)
        
        # 2. 초기 노이즈 생성 (Latents)
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, latent_height, latent_width),
            device=self.device,
            dtype=torch.float32
        )
        
        # Scheduler 초기화
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 3. Denoising Loop
        print("🎨 Generating image...")
        for t in tqdm(self.scheduler.timesteps):
            # 모델 입력 스케일링 (DDPM은 보통 그대로지만, 스케줄러에 따라 다를 수 있음)
            latent_model_input = latents
            
            # Noise 예측
            # Unconditional guidance(CFG)를 쓴다면 여기서 noise_pred_uncond도 계산해야 하지만,
            # 현재 코드는 Radar Condition만 사용하는 구조이므로 생략합니다.
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # 이전 스텝의 Latent 계산 (x_t -> x_t-1)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 4. Decoding (Latent -> Image)
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        
        # 5. Post-processing ([-1, 1] -> [0, 1] -> PIL)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        if batch_size == 1:
            image = (image[0] * 255).astype(np.uint8)
            return Image.fromarray(image)
        else:
            images = (image * 255).astype(np.uint8)
            return [Image.fromarray(img) for img in images]


def preprocess_single_radar_file(npy_path, config):
    """
    KRadarDataset의 __getitem__ 로직을 참고하여 단일 파일 전처리
    (utils.data_processing이 있다고 가정)
    """
    dataset_cfg = config['dataset']
    threshold = dataset_cfg['condition_threshold']
    max_points = dataset_cfg['max_points']
    
    polar_matrix = np.load(npy_path).astype(np.float32)
    
    # Polar -> Cartesian -> Voxelize
    cartesian_matrix = dp.polar_to_cartesian(polar_matrix, threshold=threshold, coord_normalize=True)
    voxel_points = dp.voxelize(cartesian_matrix, agg='max')
    num_points = voxel_points.shape[0]
    
    # Sampling logic (Dataset과 동일하게)
    if num_points == 0:
        raise Exception("유효한 포인트가 없습니다.")
    elif num_points >= max_points:
        choice_idx = np.random.choice(num_points, max_points, replace=False)
        fixed_points = voxel_points[choice_idx, :]
    else:
        choice_idx = np.random.choice(num_points, max_points, replace=True)
        fixed_points = voxel_points[choice_idx, :]
    
    tensor = torch.from_numpy(fixed_points).float()  # (max_points, 4)
    return tensor.unsqueeze(0)  # (1, max_points, 4) -> Batch 차원 추가


def main():
    # 경로 설정
    config_path = r'C:\Users\jdmdj\Desktop\Diffusion_4DR\config\config.yaml'
    checkpoint_path = './checkpoints/checkpoint_epoch_10.pt'  # 사용하려는 체크포인트 경로
    
    # Inference 객체 생성
    inferencer = RadarDiffusionInference(checkpoint_path, config_path)
    
    # 테스트 방법 1: 데이터셋에서 하나 가져오기 (가장 쉬운 방법)
    print("\n🧪 Testing with a sample from KRadarDataset...")
    dataset = KRadarDataset(config_path)
    sample = dataset[0]  # 첫 번째 데이터
    
    radar_condition = sample['condition'].unsqueeze(0)  # (1, N, 4)
    ground_truth_img = dp.tensor_to_pil(sample['image'])  # (가정) 확인용
    
    # 이미지 생성
    generated_img = inferencer.generate(radar_condition, num_inference_steps=50)
    
    # 저장
    os.makedirs("results", exist_ok=True)
    generated_img.save("results/inference_result.png")
    print(f"💾 Generated image saved to results/inference_result.png")
    
    # 만약 원본(Ground Truth)과 비교하고 싶다면 원본도 저장
    # ground_truth_img.save("results/ground_truth.png")


if __name__ == "__main__":
    main()