import os
from torch.utils.data import DataLoader

from model.KRadar_dataset import KRadarDataset


config_file_path = 'config/config.yaml'
train_dataset = KRadarDataset(config_path=config_file_path)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=os.cpu_count() // 2 # 코어 수의 절반 정도 사용
)

for batch in train_dataloader:
    images = batch['image']  # (B, 3, 512, 512)
    conditions = batch['condition']  # (B, C, 256, 107, 37)
    
    print(f"Image Tensor Shape: {images.shape}, Dtype: {images.dtype}")
    print(f"Condition Tensor Shape: {conditions.shape}, Dtype: {conditions.dtype}")
    
    