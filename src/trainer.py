import os
import yaml
import torch
import shutil
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from tqdm.auto import tqdm
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import bitsandbytes as bnb
from torchvision.utils import save_image
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3

from .dataset import RadarCameraDataset
from .models import get_model

def load_config(config_path: str) -> dict:
    if config_path is None:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_is(features, num_splits=10):
    probs = torch.nn.functional.softmax(torch.from_numpy(features), dim=1).numpy()
    scores = []
    for i in range(num_splits):
        part = probs[i * (len(probs) // num_splits) : (i + 1) * (len(probs) // num_splits), :]
        if len(part) == 0: continue
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean([np.sum(p * np.log(np.clip(p / py, 1e-10, None))) for p in part])))
    return np.mean(scores) if scores else 0.0

def get_inception_features(dataloader_or_tensor, device, num_samples=100):
    """
    Refactored to take device directly instead of creating Accelerator.
    """
    print(f"Loading InceptionV3 for feature extraction on {device}...")
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    features = []
    
    with torch.inference_mode():
        if isinstance(dataloader_or_tensor, torch.Tensor):
            for i in range(0, len(dataloader_or_tensor), 8):
                imgs = dataloader_or_tensor[i:i+8].to(device)
                imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
                feat = inception(imgs)
                features.append(feat.cpu().numpy())
        else:
            count = 0
            pbar = tqdm(total=num_samples, desc="Extracting Inception features", leave=False)
            for batch in dataloader_or_tensor:
                if count >= num_samples:
                    break
                imgs = batch["pixel_values"].to(device)
                imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
                feat = inception(imgs)
                features.append(feat.cpu().numpy())
                count += imgs.shape[0]
                pbar.update(imgs.shape[0])
            pbar.close()
            
    return np.concatenate(features, axis=0)[:num_samples]

def get_inception_features_for_dataset(data_dir, val_idx_path, num_samples=1000):
    dataset = RadarCameraDataset(data_dir=data_dir, split="val", indices_path=val_idx_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Feature extraction requires a GPU.")
    device = torch.device("cuda")
    return get_inception_features(dataloader, device, num_samples=num_samples)

def calculate_is_for_best_model(trial, config_path):
    config = load_config(config_path)
    for k, v in trial.params.items():
        if k in ["learning_rate", "batch_size", "micro_batch_size"]:
            config["training"][k] = v
        elif k in ["backbone", "encoder"]:
            config["model"][k] = v
        else:
            if "encoder_config" not in config["model"]: config["model"]["encoder_config"] = {}
            config["model"]["encoder_config"][k] = v
            
    model_name = f"{config['model']['backbone']}_{config['model']['encoder']}"
    best_path = os.path.join("outputs", model_name, f"trial_{trial.number}", "checkpoint-best")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config["model"]["backbone"], config["model"]["encoder"], config)
    
    ema_path = os.path.join(best_path, "ema_model.bin")
    if os.path.exists(ema_path):
        state_dict = torch.load(ema_path, map_location="cpu")
        backbone = model.unet if hasattr(model, "unet") else model.transformer
        backbone.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    val_dataset = RadarCameraDataset(data_dir="dataset", split="val", indices_path="dataset/splits/val_idx.npy")
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    fake_images = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Generating for IS")):
            if i * 4 >= 1000: break
            imgs = model.generate(batch["radar_condition"].to(device))
            fake_images.append(imgs.cpu())
            
    fake_images_tensor = torch.cat(fake_images, dim=0)
    features = get_inception_features(fake_images_tensor, device, num_samples=1000)
    return calculate_is(features)

def sanitize_config(config):
    """
    Sanitize config for trackers (e.g., TensorBoard) which only support simple types.
    """
    sanitized = {}
    for k, v in config.items():
        if isinstance(v, dict):
            sanitized[k] = sanitize_config(v)
        elif isinstance(v, (list, tuple)):
            sanitized[k] = str(v)
        else:
            sanitized[k] = v
    return sanitized

def train(config_path: str = None, config_override: dict = None, trial = None, real_features_cache = None):
    config = load_config(config_path)
    if config_override:
        for k, v in config_override.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
    
    output_dir = config.get("output_dir", "outputs")
    model_name = f"{config['model']['backbone']}_{config['model']['encoder']}"
    output_dir = os.path.join(output_dir, model_name)
    if trial is not None:
        output_dir = os.path.join(output_dir, f"trial_{trial.number}")
        
    logging_dir = os.path.join(output_dir, "logs")
    project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config["hardware"]["mixed_precision"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
        log_with="tensorboard",
        project_config=project_config
    )

    if accelerator.device.type == "cpu":
        raise RuntimeError("CUDA is not available or could not be initialized. Training on CPU is disabled.")

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        tracker_name = "diffusion_4dr_hpo" if trial else "diffusion_4dr"
        # Sanitize config for trackers to avoid "value should be one of int, float, str, bool, or torch.Tensor" errors
        accelerator.init_trackers(tracker_name, config=sanitize_config(config))

    train_dataset = RadarCameraDataset(
        data_dir="dataset",
        num_points=config["data"]["rpc_num_points"],
        target_res=config["data"]["rgb_target_resolution"],
        split="train",
        indices_path=config["data"].get("train_idx_path")
    )
    val_dataset = RadarCameraDataset(
        data_dir="dataset",
        num_points=config["data"]["rpc_num_points"],
        target_res=config["data"]["rgb_target_resolution"],
        split="val",
        indices_path=config["data"].get("val_idx_path")
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["micro_batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["training"]["micro_batch_size"], shuffle=False)

    model = get_model(config["model"]["backbone"], config["model"]["encoder"], config)
    
    # Move model to device before creating optimizer for bitsandbytes compatibility
    model.to(accelerator.device)

    # Respect use_8bit_adam and handle CPU fallback (bitsandbytes requires GPU)
    use_8bit = config["hardware"].get("use_8bit_adam", False) and accelerator.device.type == "cuda"
    if use_8bit:
        optimizer = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(config["training"].get("learning_rate", 2.0e-5))
        )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(config["training"].get("learning_rate", 2.0e-5))
        )

    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=500 * config["training"]["gradient_accumulation_steps"],
        num_training_steps=config["training"]["max_train_steps"] * config["training"]["gradient_accumulation_steps"],
    )

    backbone_model = model.unet if hasattr(model, "unet") else model.transformer
    ema_model = EMAModel(backbone_model.parameters(), model_cls=type(backbone_model), model_config=backbone_model.config)

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    ema_model.to(accelerator.device)

    # Fixed Samples
    fixed_sample_batch = []
    samples_dir = os.path.join("dataset", "samples")
    if os.path.exists(samples_dir):
        from .dataset import polar_to_cartesian
        sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith(".npy")])[:8]
        for f in sample_files:
            raw_data = np.load(os.path.join(samples_dir, f))
            pts = polar_to_cartesian(raw_data, num_points=config["data"]["rpc_num_points"]) if raw_data.ndim == 4 else raw_data
            fixed_sample_batch.append(torch.from_numpy(pts))
        fixed_sample_batch = torch.stack(fixed_sample_batch).to(accelerator.device)
    
    global_step = 0
    best_fid = float("inf")
    last_checkpoint_path = None
    progress_bar = tqdm(total=config["training"]["max_train_steps"], disable=not accelerator.is_local_main_process)

    while global_step < config["training"]["max_train_steps"]:
        for batch in train_dataloader:
            model.train()
            with accelerator.accumulate(model):
                try:
                    loss = model(batch["pixel_values"], batch["radar_condition"])["loss"]
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else: raise e

            if accelerator.sync_gradients:
                unwrapped = accelerator.unwrap_model(model)
                backbone = unwrapped.unet if hasattr(unwrapped, "unet") else unwrapped.transformer
                ema_model.step(backbone.parameters())
                progress_bar.update(1)
                global_step += 1
                
                if global_step % 500 == 0:
                    model.eval()
                    with ema_model.average_parameters():
                        with torch.inference_mode():
                            val_loss = sum(model(b["pixel_values"], b["radar_condition"])["loss"].item() for b in val_dataloader) / len(val_dataloader)
                            if len(fixed_sample_batch) > 0:
                                imgs = unwrapped.generate(fixed_sample_batch)
                                if accelerator.is_main_process:
                                    save_image(imgs, os.path.join(output_dir, f"sample_{global_step}.png"), nrow=4)
                                    accelerator.log({"val_loss": val_loss}, step=global_step)
                            if trial and global_step < 2000:
                                trial.report(float(val_loss), global_step) # Ensure float
                                if trial.should_prune():
                                    if accelerator.is_main_process: shutil.rmtree(output_dir, ignore_errors=True)
                                    raise optuna.exceptions.TrialPruned()

                if global_step >= 2000 and global_step % 1000 == 0:
                    model.eval()
                    with ema_model.average_parameters():
                        num_fid = 1000 if global_step == config["training"]["max_train_steps"] else 100
                        fake_imgs = []
                        with torch.inference_mode():
                            for i, b in enumerate(val_dataloader):
                                if i * config["training"]["micro_batch_size"] >= num_fid: break
                                cond = b["radar_condition"]
                                gen = torch.cat([unwrapped.generate(c) for c in torch.split(cond, 4)], dim=0) if cond.shape[0] > 4 else unwrapped.generate(cond)
                                fake_imgs.append(gen)
                        fake_feats = get_inception_features(torch.cat(fake_imgs, dim=0), accelerator.device, num_samples=num_fid)
                        fid = calculate_fid(real_features_cache[:num_fid], fake_feats)
                        accelerator.log({"fid": fid}, step=global_step)
                        if trial:
                            trial.report(float(fid), global_step) # Ensure float
                            if trial.should_prune():
                                if accelerator.is_main_process: shutil.rmtree(output_dir, ignore_errors=True)
                                raise optuna.exceptions.TrialPruned()

                        ckpt_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(ckpt_path)
                        if accelerator.is_main_process:
                            torch.save(ema_model.state_dict(), os.path.join(ckpt_path, "ema_model.bin"))
                            if fid < best_fid:
                                best_fid = fid
                                b_path = os.path.join(output_dir, "checkpoint-best")
                                if os.path.exists(b_path): shutil.rmtree(b_path)
                                shutil.copytree(ckpt_path, b_path)
                            if last_checkpoint_path and os.path.exists(last_checkpoint_path) and "best" not in last_checkpoint_path:
                                shutil.rmtree(last_checkpoint_path)
                            last_checkpoint_path = ckpt_path
            if global_step >= config["training"]["max_train_steps"]: break
    if accelerator.is_main_process: accelerator.end_training()
    return best_fid if best_fid != float("inf") else val_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    train(parser.parse_args().config)
