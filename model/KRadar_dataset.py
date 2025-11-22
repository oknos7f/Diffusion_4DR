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
    A PyTorch Dataset for loading perfectly aligned RGB images (.png) and
    corresponding condition data stored as NumPy arrays (.npy).
    
    The **RGB images** are preprocessed using `dp.crop_image_half` followed by
    standard Latent Diffusion Model (LDM) transformations.
    
    The **condition data** is loaded via NumPy and subsequently processed with
    `dp.voxelize (N, 4)`.
    """
    def __init__(self, config_path: str):
        self.config = load_config(config_path)['dataset']

        self.data_root = Path(self.config['data_root'])
        self.image_dir = self.data_root / self.config['image_dir']
        self.condition_dir = self.data_root / self.config['condition_dir']
        
        self.threshold = self.config['condition_threshold']
        self.max_points = self.config['max_points']

        self.target_width = self.config.get('target_width', 680)
        self.target_height = self.config.get('target_height', 384)

        image_paths = list(self.image_dir.glob('*.png'))
        self.file_stems = [p.stem for p in image_paths]

        if not self.file_stems:
            raise FileNotFoundError(f"이미지 파일이 {self.image_dir} 경로에 없습니다.")

        self.ldm_transform = transforms.Compose([
            transforms.Resize((self.target_height, self.target_width),
                              interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
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

        # convert polar grid -> voxel points (M, 4), N은 약 1400~9800 가변
        cartesian_matrix = dp.polar_to_cartesian(polar_matrix, threshold=self.threshold, coord_normalize=True)
        voxel_points = dp.voxelize(cartesian_matrix, agg='max')
        num_points = voxel_points.shape[0]
        
        if num_points == 0:
            raise Exception(f"유효한 데이터가 아닙니다: {condition_path}")
        elif num_points >= self.max_points:
            # random sampling if too many points
            # consider slice by value(Intensity) later maybe
            choice_idx = np.random.choice(num_points, self.max_points, replace=False)
            fixed_points = voxel_points[choice_idx, :]
        else:
            # duplicate points if too few points
            choice_idx = np.random.choice(num_points, self.max_points, replace=True)
            fixed_points = voxel_points[choice_idx, :]
            
            # consider Zero padding later
            # fixed_points = np.zeros((self.max_points, 4), dtype=np.float32)
            # fixed_points[:num_points, :] = voxel_points
        
        
        condition_tensor = torch.from_numpy(fixed_points).float()

        return {
            'image': image_tensor,          # (3, 720, 1280)
            'condition': condition_tensor,  # (max_points, 4)
            'file_stem': stem
        }