# Diffusion_4DR: 4D Radar Conditioned Diffusion Models

Diffusion_4DR is a research project exploring the use of Diffusion Models conditioned on 4D Radar data for high-fidelity sensor data generation and enhancement. By leveraging the sparse yet robust nature of radar point clouds, this project generates corresponding visual representations using state-of-the-art diffusion backbones.

## 🚀 Key Features

- **Multi-Backbone Support**: Seamlessly switch between **Stable Diffusion 1.5** and **Stable Diffusion 3**.
- **Advanced Radar Encoders**:
  - **PointNet++**: Hierarchical feature learning for point clouds.
  - **PointPillars**: Efficient pillar-based encoding for faster inference and training.
- **Robust Training Pipeline**:
  - Powered by Hugging Face `diffusers` and `accelerate`.
  - Supports **Mixed Precision (bf16)** and **8-bit AdamW** for memory efficiency.
  - EMA (Exponential Moving Average) model weights for stable generation.
- **Automated Hyperparameter Optimization (HPO)**:
  - Integrated **Optuna** for automated searching of optimal learning rates, batch sizes, and architectural parameters.
  - Optimization objective: **FID (Fréchet Inception Distance)**.
- **Evaluation Metrics**: Automated calculation of **FID** and **Inception Score (IS)**.

## 📁 Project Structure

```text
Diffusion_4DR/
├── config.yaml          # Main configuration for hardware, training, and model
├── hpo_runner.py        # Entry point for Hyperparameter Optimization
├── src/
│   ├── dataset.py       # Data loading and Polar-to-Cartesian transformation
│   ├── models.py        # Model factory (SD1.5 vs SD3)
│   ├── trainer.py       # Core training and evaluation logic
│   ├── encoders/        # Radar feature extraction modules (PointNet++, PointPillars)
│   └── models_arch/     # Radar-conditioned UNet/Transformer architectures
├── dataset/             # (Not included) Expected data directory
└── requirements.txt     # Python dependencies
```

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

The HPO script will:
1. Pre-process polar radar data into Cartesian points (cached for speed).
2. Generate static data splits.
3. Pre-calculate Inception features for the real dataset.
4. Run multiple trials searching for the best backbone, encoder, and training parameters.
5. Report the best FID and calculate the final Inception Score.

## ⚙️ Configuration

Modify `config.yaml` to adjust:
- **Hardware**: Device, mixed precision, 8-bit optimizer.
- **Model**: Backbone (`sd15`, `sd3`), Encoder (`pointnet`, `pointpillars`), and frozen layers.
- **Training**: Learning rate, batch size, max steps.

---
*Developed for research in multi-modal sensor fusion and generative models.*
