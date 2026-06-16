# Diffusion_4DR: 4D Radar Conditioned Diffusion Models

Diffusion_4DR is a research project exploring the use of Diffusion Models conditioned on 4D Radar data for high-fidelity sensor data generation and enhancement. By leveraging the sparse yet robust nature of radar point clouds, this project generates corresponding visual representations using state-of-the-art diffusion backbones.

## Samples
<img width="2058" height="1030" alt="sample_6000" src="https://github.com/user-attachments/assets/ca41a6a9-eac9-4052-b4cd-c00aa2456165" />
<img width="2058" height="1030" alt="image" src="https://github.com/user-attachments/assets/532c2aba-2174-45b1-b2a8-21d0ab325097" />


## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Diffusion_4DR.git
cd Diffusion_4DR

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Preparation

The project expects a `dataset/` directory with the following structure:

```text
dataset/
├── images/              # Paired RGB frames (.png)
└── conditions/          # 4D Radar tensors (.npy) [2, Range, Azimuth, Elevation]
```

- **Images**: The model processes the left half of the provided images by default.
- **Conditions**: Raw polar data is converted to Cartesian point clouds `(N, 5)` [x, y, z, power, doppler] during loading.

## 🏋️ Training

To start training with the default configuration:

```bash
python -m src.trainer --config config.yaml
```

## 🔍 Hyperparameter Optimization (HPO)

To run the automated HPO pipeline using Optuna:

```bash
python hpo_runner.py
```

## ⚙️ Configuration

Modify `config.yaml` to adjust:
- **Hardware**: Device, mixed precision, 8-bit optimizer.
- **Model**: Backbone (`sd15`, `sd3`), Encoder (`pointnet`, `pointpillars`), and frozen layers.
- **Training**: Learning rate, batch size, max steps.

---
*Developed for research in multi-modal sensor fusion and generative models.*
