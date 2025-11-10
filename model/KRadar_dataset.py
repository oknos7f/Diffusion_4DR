from pathlib import Path
import yaml
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils.data_processing as dp


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class KRadarDataset(Dataset):
    """
    Dataset that loads RGB images (.png) and condition numpy (.npy) files.
    Images are preprocessed with dp.crop_image_half and standard LDM transforms.
    Conditions are loaded with numpy and preprocessed with dp.polar_to_cartesian.
    """
    def __init__(self, config_path: str):
        self.config = load_config(config_path)['dataset']

        self.data_root = Path(self.config['data_root'])
        self.image_dir = self.data_root / self.config['image_dir']
        self.condition_dir = self.data_root / self.config['condition_dir']

        self.target_width = int(self.config.get('target_width', 1280))
        self.target_height = int(self.config.get('target_height', 720))

        image_paths = list(self.image_dir.glob('*.png'))
        self.file_stems = [p.stem for p in image_paths]

        if not self.file_stems:
            raise FileNotFoundError(f"이미지 파일이 {self.image_dir} 경로에 없습니다.")

        # 직사각형 해상도 (H, W)에 맞게 transforms를 조정합니다.
        self.ldm_transform = transforms.Compose([
            transforms.Resize((self.target_height, self.target_width),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0,1], shape (3, H, W)
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1]
        ])

    def __len__(self):
        return len(self.file_stems)

    def __getitem__(self, idx):
        stem = self.file_stems[idx]

        # --- Image load & preprocess ---
        image_path = self.image_dir / f"{stem}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
        image = Image.open(image_path).convert('RGB')

        # crop half (module function)
        cropped_image = dp.crop_image_half(image)

        image_tensor = self.ldm_transform(cropped_image)  # (3, H, W)

        # --- Condition load & preprocess ---
        condition_path = self.condition_dir / f"{stem}.npy"
        if not condition_path.exists():
            raise FileNotFoundError(f"조건 파일이 없습니다: {condition_path}")

        polar_matrix = np.load(condition_path).astype(np.float32)

        # convert polar grid -> cartesian points (N,4)
        cartesian_matrix = dp.polar_to_cartesian(polar_matrix, threshold=self.config['condition_threshold'])
        voxelized_matrix = dp.voxelize(cartesian_matrix, agg='max')
        
        # to torch tensor (float)
        condition_tensor = torch.from_numpy(voxelized_matrix).float()

        return {
            'image': image_tensor,          # (3, H, W)
            'condition': condition_tensor,  # (N, 4)
            'file_stem': stem
        }