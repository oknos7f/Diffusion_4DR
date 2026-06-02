import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Sequence

def polar_to_cartesian(data: np.ndarray,
                       num_points: int = 4096,
                       coord_normalize: bool = True) -> np.ndarray:
    """
    Polar (Range, Azimuth, Elevation) 3D grid data -> Cartesian (N, 5) [x, y, z, power, doppler].
    Extracts top num_points by power.
    """
    if data.ndim != 4 or data.shape[0] != 2:
        raise ValueError(f"Expected data shape (2, D, T, H), got {data.shape}")
    
    # 1. Power Data Preparation (data[0] is power, data[1] is doppler)
    raw_power = data[0]
    raw_doppler = data[1]
    
    epsilon = 1e-6
    power_db = 10 * np.log10(np.clip(raw_power, epsilon, None))
    
    d, t, h = power_db.shape
    flat_power = power_db.ravel()
    total_elements = flat_power.size
    
    target_k = min(num_points, total_elements)
    
    if target_k <= 0:
        return np.zeros((num_points, 5), dtype=np.float32)
    
    # 2. Extract top K points by power
    top_indices_flat = np.argpartition(flat_power, -target_k)[-target_k:]
    sorted_args = np.argsort(flat_power[top_indices_flat])[::-1]
    top_indices_flat = top_indices_flat[sorted_args]
    
    idx_d, idx_t, idx_h = np.unravel_index(top_indices_flat, power_db.shape)
    
    # Extract values
    valid_power = flat_power[top_indices_flat].astype(np.float32)
    valid_doppler = raw_doppler[idx_d, idx_t, idx_h].astype(np.float32)
    
    # 3. Coordinate Transformation
    # Range bins: 256, Azimuth bins: 107, Elevation bins: 37
    range_val, azimuth_val, elevation_val = 256, 107, 37
    
    rho_1d = np.linspace(0.0, range_val - 1.0, range_val)
    r_vec = rho_1d[idx_d]
    
    azimuth_max_deg = (azimuth_val - 1) / 2.0
    azimuth_1d = np.deg2rad(np.linspace(-azimuth_max_deg, azimuth_max_deg, azimuth_val))
    a_vec = azimuth_1d[idx_t]
    
    phi_1d = np.deg2rad(np.linspace(0.0, elevation_val - 1.0, elevation_val))
    p_vec = phi_1d[idx_h]
    
    r_proj = r_vec * np.cos(p_vec)
    x = r_proj * np.cos(a_vec)
    y = r_proj * np.sin(a_vec)
    z = r_vec * np.sin(p_vec)
    
    # 4. Normalization
    if coord_normalize:
        # User defined ranges
        x_min, x_max = 0.0, 255.0
        y_min, y_max = -203.7, 203.7
        z_min, z_max = 0.0, 149.9
        
        # Map to [-1, 1] for models
        x = (x - x_min) / (x_max - x_min) * 2.0 - 1.0
        y = (y - y_min) / (y_max - y_min) * 2.0 - 1.0
        z = (z - z_min) / (z_max - z_min) * 2.0 - 1.0

    # Power Normalization [0, 1]
    p_min, p_max = valid_power.min(), valid_power.max()
    if p_max - p_min > 1e-6:
        valid_power = (valid_power - p_min) / (p_max - p_min)
    else:
        valid_power = np.zeros_like(valid_power)
        
    # Doppler Normalization [-1, 1] (Assume range [-20, 20])
    valid_doppler = np.clip(valid_doppler / 20.0, -1.0, 1.0)
    
    cartesian_points = np.stack([x, y, z, valid_power, valid_doppler], axis=-1)
    
    # 5. Padding if needed
    if cartesian_points.shape[0] < num_points:
        padding = np.zeros((num_points - cartesian_points.shape[0], 5), dtype=np.float32)
        cartesian_points = np.concatenate([cartesian_points, padding], axis=0)
        
    return cartesian_points.astype(np.float32)

class RadarCameraDataset(Dataset):
    """
    Dataset class handling paired 4D Radar Tensors and Monocular RGB frames.
    Converts 4D Radar to Cartesian Point Clouds online.
    """
    def __init__(self, data_dir: str, num_points: int = 4096, target_res: int = 512, split: str = "train"):
        super().__init__()
        self.data_dir = data_dir
        self.num_points = num_points
        self.target_res = target_res
        
        # Discovery
        rgb_files = sorted([f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".png")])
        rpc_files = sorted([f for f in os.listdir(os.path.join(data_dir, "conditions")) if f.endswith(".npy")])
        
        rgb_names = {os.path.splitext(f)[0] for f in rgb_files}
        rpc_names = {os.path.splitext(f)[0] for f in rpc_files}
        common_names = sorted(list(rgb_names.intersection(rpc_names)))
        
        if len(common_names) == 0:
            raise RuntimeError(f"No paired data found in {data_dir}.")

        # Split
        np.random.seed(42)
        indices = np.arange(len(common_names))
        np.random.shuffle(indices)
        
        split_idx = int(len(common_names) * 8 / 9)
        self.file_names = [common_names[i] for i in indices[:split_idx]] if split == "train" else [common_names[i] for i in indices[split_idx:]]

        print(f"[{split}] dataset initialized with {len(self.file_names)} samples.")

        self.rgb_transforms = transforms.Compose([
            transforms.Resize((self.target_res, self.target_res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict:
        base_name = self.file_names[idx]
        
        # 1. Process RGB
        img_path = os.path.join(self.data_dir, "images", f"{base_name}.png")
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        # Left crop
        left_half = img.crop((0, 0, w // 2, h))
        rgb_tensor = self.rgb_transforms(left_half)

        # 2. Process Radar
        rpc_path = os.path.join(self.data_dir, "conditions", f"{base_name}.npy")
        rpc_4d = np.load(rpc_path) # [2, 256, 107, 37]
        
        # Convert Polar to Cartesian Point Cloud
        rpc_points = polar_to_cartesian(rpc_4d, num_points=self.num_points)
        
        return {
            "pixel_values": rgb_tensor,
            "radar_condition": torch.from_numpy(rpc_points)
        }
