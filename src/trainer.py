import os
import yaml
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from tqdm.auto import tqdm
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import bitsandbytes as bnb
from torchvision.utils import save_image

from .dataset import RadarCameraDataset
from .models import get_model

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(config_path: str):
    """
    Enhanced training loop with logging, checkpointing, and visual validation.
    """
    config = load_config(config_path)
    
    # Project Setup
    output_dir = config.get("output_dir", "outputs")
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

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.init_trackers("diffusion_4dr", config=config)

    # Initialize Dataset and DataLoader
    train_dataset = RadarCameraDataset(
        data_dir="dataset",
        num_points=config["data"]["rpc_num_points"],
        target_res=config["data"]["rgb_target_resolution"],
        split="train"
    )
    val_dataset = RadarCameraDataset(
        data_dir="dataset",
        num_points=config["data"]["rpc_num_points"],
        target_res=config["data"]["rgb_target_resolution"],
        split="val"
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["micro_batch_size"], 
        shuffle=True, 
        num_workers=config["hardware"]["dataloader_num_workers"]
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["training"]["micro_batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["dataloader_num_workers"]
    )

    # Initialize Model with Config
    model = get_model(config["model"]["backbone"], config["model"]["encoder"], config)
    
    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=float(config["training"].get("learning_rate", 2.0e-5)),
        weight_decay=float(config["training"].get("weight_decay", 1e-2))
    )

    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=500 * config["training"]["gradient_accumulation_steps"],
        num_training_steps=config["training"]["max_train_steps"] * config["training"]["gradient_accumulation_steps"],
    )

    # EMA Setup
    backbone_model = model.unet if hasattr(model, "unet") else model.transformer
    ema_model = EMAModel(backbone_model.parameters(), model_cls=type(backbone_model), model_config=backbone_model.config)

    # Prepare for DDP and AMP
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    ema_model.to(accelerator.device)

    # Train loop
    global_step = 0
    max_train_steps = config["training"]["max_train_steps"]
    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_local_main_process)
    
    model.train()
    
    while global_step < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                pixel_values = batch["pixel_values"]
                radar_condition = batch["radar_condition"]
                
                loss_dict = model(pixel_values, radar_condition)
                loss = loss_dict["loss"]
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                unwrapped_model = accelerator.unwrap_model(model)
                backbone_model = unwrapped_model.unet if hasattr(unwrapped_model, "unet") else unwrapped_model.transformer
                ema_model.step(backbone_model.parameters())
                
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                if global_step % 10 == 0:
                    accelerator.log({"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                
                # Evaluation & Visualization
                if global_step % 500 == 0:
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            v_pixel = val_batch["pixel_values"]
                            v_radar = val_batch["radar_condition"]
                            v_loss = model(v_pixel, v_radar)["loss"]
                            val_loss += v_loss.item()
                        
                        # Generate Samples
                        sample_radar = next(iter(val_dataloader))["radar_condition"][:4]
                        images = unwrapped_model.generate(sample_radar)
                        
                        if accelerator.is_main_process:
                            save_path = os.path.join(output_dir, f"sample_{global_step}.png")
                            save_image(images, save_path, nrow=2)
                            accelerator.log({"val_loss": val_loss / len(val_dataloader)}, step=global_step)
                    
                    accelerator.print(f"Step {global_step}: Val Loss = {val_loss / len(val_dataloader):.4f}")
                    model.train()

                # Checkpointing
                if global_step % 2000 == 0 or global_step == max_train_steps:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        ema_save_path = os.path.join(save_path, "ema_model.bin")
                        torch.save(ema_model.state_dict(), ema_save_path)

            if global_step >= max_train_steps:
                break
                
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.end_training()
        print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    train(args.config)
