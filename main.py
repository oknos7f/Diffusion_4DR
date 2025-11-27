import torch
import os
from pathlib import Path
import yaml
import model.diffusion as diffusion


print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")

config = Path(__file__).resolve().parent / "config/config.yaml"

print(f"Loading configuration from {config}")
if os.path.exists(config):
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)
    print("Configuration loaded successfully")
else:
    FileNotFoundError(f"Configuration file not found at {config}")
    
dataset = config_data["dataset"]
data_root = dataset["data_root"]

if not os.path.exists(data_root):
    os.makedirs(data_root)
    os.makedirs(os.path.join(data_root, dataset["image_dir"]))
    os.makedirs(os.path.join(data_root, dataset["condition_dir"]))
    
    FileNotFoundError(f"Created data root directory at {data_root}")
    
diffusion.main(config)