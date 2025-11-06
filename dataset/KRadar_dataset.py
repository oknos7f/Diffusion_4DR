import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import yaml
from pathlib import Path

from utils.data_processing import polar_to_cartesian, crop_image_half


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class KRadarDataset(Dataset):
    def __init__(self, config_path: str):
        self.config = load_config(config_path)['dataset']
        
        # 워크스페이스 루트 폴더를 기준으로 경로 설정
        self.data_root = Path(self.config['data_root'])
        self.image_dir = self.data_root / self.config['image_dir']
        self.condition_dir = self.data_root / self.config['condition_dir']
        self.target_image_size = self.config.get('target_image_size', 512)
        
        # glob을 사용하여 .png 파일 경로 리스트를 가져온 후, stem만 추출합니다.
        image_paths = list(self.image_dir.glob('*.png'))
        self.file_stems = [p.stem for p in image_paths]
        
        if not self.file_stems:
            raise FileNotFoundError(f"이미지 파일이 {self.image_dir} 경로에 없습니다.")
        
        # LDM 학습을 위한 최종 이미지 Transform 정의
        self.ldm_transform = transforms.Compose([
            transforms.Resize(self.target_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.target_image_size),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.file_stems)
    
    def __getitem__(self, idx):
        stem = self.file_stems[idx]
        
        # A. 이미지 로드 및 전처리
        image_path = self.image_dir / f"{stem}.png"
        image = Image.open(image_path).convert('RGB')
        
        # **1단계 전처리: crop_image_half() 적용 (1280, 720)**
        cropped_image = crop_image_half(image)
        
        # **2단계 LDM 표준 전처리 적용 (512x512, [-1, 1])**
        image_tensor = self.ldm_transform(cropped_image)  # (3, 512, 512)
        
        # B. 조건 행렬 로드 및 전처리 (클라우드 포인트)
        condition_path = self.condition_dir / f"{stem}.npy"
        # numpy 로드: shape (256, 107, 37)의 polar float 행렬
        polar_matrix = np.load(condition_path).astype(np.float32)
        
        # **전처리: polar_to_cartesian() 적용**
        # (N, 4) np.ndarray 형태 (N은 포인트 개수, 4는 x, y, z, value 등)
        cartesian_matrix = polar_to_cartesian(polar_matrix)
        
        # (N, 4) numpy 행렬을 PyTorch 텐서로 변환
        condition_tensor = torch.from_numpy(cartesian_matrix)
        
        # 참고: 이 (N, 4) 텐서는 3D-CNN이 아닌 PointNet 계열 모델 또는
        # 토큰화(Tokenization)를 위한 입력으로 사용됩니다.
        # (만약 3D-CNN을 계속 사용한다면, 이 단계 후 3D 그리드로 재구성 필요)
        
        return {
            'image': image_tensor,  # (3, 512, 512)
            'condition': condition_tensor,  # (N, 4)
            'file_stem': stem
        }