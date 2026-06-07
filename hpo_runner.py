import os
import optuna
import yaml
import numpy as np
import torch
import shutil
from src.trainer import train

def pre_process_cartesian(data_dir):
    """
    Pre-process all polar radar data into Cartesian point clouds to save HPO time.
    """
    polar_dir = os.path.join(data_dir, "conditions")
    cartesian_dir = os.path.join(data_dir, "conditions_cartesian")
    os.makedirs(cartesian_dir, exist_ok=True)
    
    rpc_files = [f for f in os.listdir(polar_dir) if f.endswith(".npy")]
    print(f"Pre-processing {len(rpc_files)} radar files...")
    
    from src.dataset import polar_to_cartesian
    from tqdm import tqdm
    
    for f in tqdm(rpc_files):
        target_path = os.path.join(cartesian_dir, f)
        if os.path.exists(target_path):
            continue
            
        rpc_4d = np.load(os.path.join(polar_dir, f))
        rpc_points = polar_to_cartesian(rpc_4d)
        np.save(target_path, rpc_points)
    
    print("Pre-processing completed.")

def setup_static_splits(data_dir, output_dir):
    """
    Generate static data split indices if they don't exist.
    """
    rgb_files = sorted([f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".png")])
    rpc_files = sorted([f for f in os.listdir(os.path.join(data_dir, "conditions")) if f.endswith(".npy")])
    
    rgb_names = {os.path.splitext(f)[0] for f in rgb_files}
    rpc_names = {os.path.splitext(f)[0] for f in rpc_files}
    common_names = sorted(list(rgb_names.intersection(rpc_names)))
    
    if len(common_names) == 0:
        raise RuntimeError(f"No paired data found in {data_dir}.")

    indices = np.arange(len(common_names))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(len(common_names) * 8 / 9)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_idx.npy")
    val_path = os.path.join(output_dir, "val_idx.npy")
    
    if not os.path.exists(train_path):
        np.save(train_path, train_indices)
    if not os.path.exists(val_path):
        np.save(val_path, val_indices)
        
    return train_path, val_path

def objective(trial, real_features_cache):
    # 1. Sample Training Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    micro_batch_size = trial.suggest_categorical("micro_batch_size", [2, 4])
    
    # 2. Sample Architectural Hyperparameters
    backbone = trial.suggest_categorical("backbone", ["sd15", "sd3"])
    encoder = trial.suggest_categorical("encoder", ["pointnet", "pointpillars"])
    
    if encoder == "pointnet":
        sa1_mlp_str = trial.suggest_categorical("sa1_mlp", ["32-32-64", "64-64-128"])
        sa2_mlp_str = trial.suggest_categorical("sa2_mlp", ["128-128-256", "256-256-512"])
        physics_dim = trial.suggest_int("physics_dim", 8, 32, step=8)
        encoder_config = {
            "sa1_mlp": [int(x) for x in sa1_mlp_str.split("-")],
            "sa2_mlp": [int(x) for x in sa2_mlp_str.split("-")],
            "physics_dim": physics_dim
        }
    else: # pointpillars
        feat_channels = trial.suggest_categorical("feat_channels", [32, 64, 128])
        grid_size = trial.suggest_categorical("grid_size", [16, 32, 48])
        encoder_config = {
            "feat_channels": feat_channels,
            "grid_size": grid_size
        }
    
    # 3. Prepare Config Override
    config_override = {
        "training": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
        },
        "model": {
            "backbone": backbone,
            "encoder": encoder,
            "encoder_config": encoder_config
        },
        "data": {
            "train_idx_path": "dataset/splits/train_idx.npy",
            "val_idx_path": "dataset/splits/val_idx.npy",
        }
    }
    
    # 4. Run Training
    try:
        score = train(config_path="config.yaml", config_override=config_override, trial=trial, real_features_cache=real_features_cache)
        return score
    except torch.cuda.OutOfMemoryError:
        print(f"Trial {trial.number} failed due to CUDA OOM.")
        return float("inf")
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float("inf")

def main():
    # 1. Pre-process Cartesian data (Caching)
    pre_process_cartesian("dataset")
    
    # 2. Setup static data splits
    train_path, val_path = setup_static_splits("dataset", "dataset/splits")
    
    # 3. Pre-calculate Real Features for FID (Global Cache)
    # This is done once to save time across all trials
    print("Pre-calculating Real Features for FID...")
    from src.trainer import get_inception_features_for_dataset
    real_features_cache = get_inception_features_for_dataset(
        data_dir="dataset",
        val_idx_path=val_path,
        num_samples=1000 # Max needed for final eval
    )
    
    # 4. Create Optuna Study
    storage_url = "sqlite:///hpo_experiment.db"
    study = optuna.create_study(
        study_name="ldm_hpo",
        storage=storage_url,
        load_if_exists=True,
        direction="minimize", # Minimize FID
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2000, interval_steps=1000)
    )
    
    # 5. Optimization loop
    study.optimize(lambda t: objective(t, real_features_cache), n_trials=10)
    
    print("Optimization finished.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (FID): {trial.value}")
    
    # 6. Final Calculation: IS (Inception Score) for Best Trial
    # Only done once at the very end as requested
    print("Calculating Final Inception Score (IS) for the Best Trial...")
    from src.trainer import calculate_is_for_best_model
    final_is = calculate_is_for_best_model(trial, "config.yaml")
    print(f"Final Best Trial Inception Score (IS): {final_is}")
    
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
