#!/usr/bin/env python3
"""
Multi-Model Training Script

Extended version of train_single_episode.py that supports multiple architectures:
- ACT (proven working)
- Diffusion Policy (working)
- VQBet (working)

Usage:
    # Train ACT (your baseline)
    python train_multi_model.py --model act --episode 0 --steps 1000
    
    # Train Diffusion Policy
    python train_multi_model.py --model diffusion --episode 0 --steps 1000
    
    # Train VQBet  
    python train_multi_model.py --model vqbet --episode 0 --steps 1000
    
    # Compare all models with same episode/steps
    python train_multi_model.py --model act --episode 0 --steps 500 --output-dir ./models/act_ep0_500
    python train_multi_model.py --model diffusion --episode 0 --steps 500 --output-dir ./models/diffusion_ep0_500  
    python train_multi_model.py --model vqbet --episode 0 --steps 500 --output-dir ./models/vqbet_ep0_500
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler
import time
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

# Auto-detect and suppress warnings for cloud environments
def setup_environment(cloud_mode=False):
    """Setup environment based on cloud vs local."""
    if cloud_mode or torch.cuda.is_available():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtbufsize;0"
        return "cloud"
    else:
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        return "local"

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

# Import all model types
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def get_single_episode_indices(dataset: LeRobotDataset, episode_idx: int) -> list:
    """Get all data indices for a single episode."""
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    return list(range(from_idx, to_idx))


def create_single_episode_dataset(dataset_name: str, episode_idx: int, video_backend: str = "pyav"):
    """Create dataset containing only one episode."""
    print(f"📊 Loading dataset: {dataset_name}")
    
    # Load metadata first
    metadata = LeRobotDatasetMetadata(dataset_name)
    print(f"   Total episodes: {metadata.total_episodes}")
    print(f"   Total frames: {metadata.total_frames}")
    
    # Load full dataset
    full_dataset = LeRobotDataset(dataset_name, video_backend=video_backend)
    
    # Get indices for the specific episode
    episode_indices = get_single_episode_indices(full_dataset, episode_idx)
    episode_length = len(episode_indices)
    
    print(f"🎯 Episode {episode_idx}:")
    print(f"   Frame indices: {episode_indices[0]} to {episode_indices[-1]}")
    print(f"   Episode length: {episode_length} steps")
    
    # Create subset dataset with only this episode
    single_episode_dataset = Subset(full_dataset, episode_indices)
    
    return full_dataset, single_episode_dataset, metadata, episode_length


def setup_act_policy(input_features, output_features, metadata):
    """Setup ACT policy with proven configuration."""
    print(f"🤖 Setting up ACT policy...")
    
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # Proven ACT settings
        chunk_size=100,
        n_action_steps=100,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        dropout=0.1,
        kl_weight=10.0,
        optimizer_lr=1e-5,
        optimizer_weight_decay=1e-4,
        optimizer_lr_backbone=1e-5,
    )
    
    policy = ACTPolicy(config, dataset_stats=metadata.stats)
    
    print(f"   ✅ ACT configured")
    print(f"   Chunk size: {config.chunk_size}")
    print(f"   Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    return policy, config


def setup_diffusion_policy(input_features, output_features, metadata):
    """Setup Diffusion Policy with working configuration."""
    print(f"🌊 Setting up Diffusion Policy...")
    
    config = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # Working Diffusion settings
        horizon=16,
        n_action_steps=8,
        num_inference_steps=10,
    )
    
    policy = DiffusionPolicy(config, dataset_stats=metadata.stats)
    
    print(f"   ✅ Diffusion configured")
    print(f"   Horizon: {config.horizon}")
    print(f"   Action steps: {config.n_action_steps}")
    print(f"   Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    return policy, config


def setup_vqbet_policy(input_features, output_features, metadata):
    """Setup VQBet policy with working configuration."""
    print(f"🎰 Setting up VQBet policy...")
    
    config = VQBeTConfig(
        input_features=input_features,
        output_features=output_features,
    )
    
    policy = VQBeTPolicy(config, dataset_stats=metadata.stats)
    
    print(f"   ✅ VQBet configured")
    print(f"   Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    return policy, config


def setup_policy_and_config(model_type: str, metadata):
    """Setup policy based on model type."""
    print(f"\n🧠 Setting up {model_type.upper()} policy...")
    
    # Convert dataset features to policy features
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print(f"   Input features: {list(input_features.keys())}")
    print(f"   Output features: {list(output_features.keys())}")
    
    # Select policy based on model type
    if model_type == "act":
        policy, config = setup_act_policy(input_features, output_features, metadata)
    elif model_type == "diffusion":
        policy, config = setup_diffusion_policy(input_features, output_features, metadata)
    elif model_type == "vqbet":
        policy, config = setup_vqbet_policy(input_features, output_features, metadata)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Resolve delta timestamps for action chunking
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    
    return policy, config, delta_timestamps


def create_dataloader(full_dataset, single_episode_indices, delta_timestamps, batch_size=8, 
                     num_workers=None, video_backend="pyav"):
    """Create dataloader for single episode training."""
    print(f"\n📦 Creating single-episode dataloader...")
    
    # Auto-adjust num_workers based on environment
    if num_workers is None:
        if torch.cuda.is_available():
            num_workers = 8
        else:
            num_workers = 2
    
    # Create dataset with delta timestamps and transforms
    dataset_with_config = LeRobotDataset(
        full_dataset.repo_id,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend
    )
    
    # Create subset for our episode
    single_episode_dataset = Subset(dataset_with_config, single_episode_indices)
    
    # Create dataloader
    dataloader = DataLoader(
        single_episode_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print(f"   Episode length: {len(single_episode_dataset)} steps")
    print(f"   Batch size: {batch_size}")
    print(f"   Batches per epoch: {len(dataloader)}")
    print(f"   Workers: {num_workers}")
    
    return dataloader


def get_optimizer_config(model_type: str, config):
    """Get optimizer configuration for different model types."""
    if model_type == "act":
        # ACT has specific optimizer settings
        return {
            'lr': config.optimizer_lr,
            'weight_decay': config.optimizer_weight_decay,
        }
    elif model_type == "diffusion":
        # Diffusion typically uses different learning rates
        return {
            'lr': 1e-4,  # Diffusion often uses higher LR
            'weight_decay': 1e-4,
        }
    elif model_type == "vqbet":
        # VQBet settings
        return {
            'lr': 1e-4,
            'weight_decay': 1e-4,
        }
    else:
        # Default settings
        return {
            'lr': 1e-4,
            'weight_decay': 1e-4,
        }


def train_single_episode(policy, dataloader, num_steps, device, model_type, config, 
                        cloud_mode=False, accumulation_steps=1):
    """Train policy on single episode with model-specific optimizations."""
    print(f"\n🚀 Starting {model_type.upper()} training...")
    
    # Get optimizer config for this model type
    opt_config = get_optimizer_config(model_type, config)
    
    print(f"   Training steps: {num_steps}")
    print(f"   Learning rate: {opt_config['lr']}")
    print(f"   Weight decay: {opt_config['weight_decay']}")
    print(f"   Accumulation steps: {accumulation_steps}")
    print(f"   Device: {device}")
    print(f"   Mode: {'Cloud' if cloud_mode else 'Local'}")
    
    # Setup training
    policy.to(device)
    policy.train()
    
    # Optimizer - use AdamW for all model types
    optimizer = torch.optim.AdamW(
        policy.parameters(), 
        lr=opt_config['lr'],
        weight_decay=opt_config['weight_decay']
    )
    
    # Mixed precision training
    grad_scaler = GradScaler(device.type, enabled=device.type == "cuda")
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    # Cycle through dataloader efficiently
    dl_iter = cycle(dataloader)
    
    # Training metrics
    losses = []
    step_times = []
    
    # CUDA optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
    
    # Training loop
    start_time = time.time()
    log_every = 100 if cloud_mode or num_steps > 500 else 10
    
    for step in range(num_steps):
        step_start = time.time()
        accumulated_loss = 0
        
        # Gradient accumulation
        for acc_step in range(accumulation_steps):
            # Get batch
            batch = next(dl_iter)
            
            # Move to device efficiently
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, output_dict = policy.forward(batch)
                loss = loss / accumulation_steps
                accumulated_loss += loss.item()
            
            # Scaled backward pass
            grad_scaler.scale(loss).backward()
        
        # Gradient clipping and optimizer step
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), 
            max_norm=1.0,
            error_if_nonfinite=False
        )
        
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()
        
        # Record metrics
        loss_value = accumulated_loss
        losses.append(loss_value)
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Logging
        if step % log_every == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            eta_min = (num_steps - step - 1) / steps_per_sec / 60 if steps_per_sec > 0 else 0
            current_lr = optimizer.param_groups[0]["lr"]
            
            if cloud_mode:
                avg_loss = np.mean(losses[-min(50, len(losses)):])
                print(f"   Step {step:4d}/{num_steps} | "
                      f"Loss: {loss_value:.4f} | "
                      f"Avg: {avg_loss:.4f} | "
                      f"{steps_per_sec:.1f} steps/s | "
                      f"ETA: {eta_min:.1f}min")
            else:
                avg_loss = np.mean(losses[-log_every:])
                avg_time = np.mean(step_times[-log_every:])
                print(f"Step {step:4d}/{num_steps} | "
                      f"Loss: {loss_value:.6f} | "
                      f"Avg Loss: {avg_loss:.6f} | "
                      f"Time/step: {avg_time:.3f}s | "
                      f"LR: {current_lr:.1e} | "
                      f"ETA: {eta_min:.1f}min")
    
    total_time = time.time() - start_time
    final_steps_per_sec = num_steps / total_time if total_time > 0 else 0
    
    print(f"\n✅ {model_type.upper()} Training completed!")
    print(f"   Total time: {total_time/60:.2f} minutes")
    print(f"   Speed: {final_steps_per_sec:.1f} steps/second")
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Average loss (last 100 steps): {np.mean(losses[-100:]):.6f}")
    
    return losses


def save_model(policy, save_dir, model_type, episode_idx, final_loss):
    """Save the trained model with model type info."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving {model_type.upper()} model to: {save_path}")
    
    # Save using LeRobot's method
    policy.save_pretrained(save_path)
    
    # Save training info
    info = {
        "model_type": model_type,
        "episode_index": episode_idx,
        "final_loss": final_loss,
        "training_completed": True,
        "parameters": sum(p.numel() for p in policy.parameters())
    }
    
    import json
    with open(save_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"   ✅ {model_type.upper()} model saved successfully!")
    return save_path


def plot_training_progress(losses, model_type, save_path=None, show_plot=True):
    """Plot training loss curve with model info."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f"Training Loss - {model_type.upper()} (Single Episode)")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    # Add model info to plot
    params = len(losses)
    final_loss = losses[-1] if losses else 0
    plt.text(0.02, 0.98, f'Model: {model_type.upper()}\nFinal Loss: {final_loss:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path / f"training_curve_{model_type}.png", dpi=150, bbox_inches='tight')
        print(f"   📊 Training curve saved to: {save_path}/training_curve_{model_type}.png")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train multiple model architectures on single episode")
    parser.add_argument("--model", choices=["act", "diffusion", "vqbet"], default="act",
                       help="Model architecture to train")
    parser.add_argument("--dataset", default="bearlover365/red_cube_always_in_same_place", 
                       help="Dataset name/path")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to train on")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if not set)")
    parser.add_argument("--output-dir", default=None, help="Save directory (auto if not set)")
    parser.add_argument("--video-backend", choices=["pyav"], default="pyav", 
                       help="Video backend")
    
    # Cloud/advanced options
    parser.add_argument("--cloud", action="store_true", help="Enable cloud optimizations")
    parser.add_argument("--upload-model", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--no-plot", action="store_true", help="Don't show plots (save only)")
    
    args = parser.parse_args()
    
    # Auto-configure output directory
    if args.output_dir is None:
        args.output_dir = f"./models/{args.model}_episode_{args.episode}_{args.steps}steps"
    
    # Setup environment
    env_mode = setup_environment(args.cloud)
    
    # Auto-configure batch size
    if args.batch_size is None:
        if args.cloud or torch.cuda.is_available():
            # Different models have different memory requirements
            if args.model == "diffusion":
                args.batch_size = 16  # Diffusion is larger
                accumulation_steps = 8
            elif args.model == "vqbet":
                args.batch_size = 24  # VQBet is smaller
                accumulation_steps = 4
            else:  # ACT
                args.batch_size = 32
                accumulation_steps = 4
        else:
            args.batch_size = 8
            accumulation_steps = 1
    else:
        accumulation_steps = max(1, 128 // args.batch_size) if (args.cloud or torch.cuda.is_available()) else 1
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🤖 MULTI-MODEL TRAINING")
    print("=" * 50)
    print(f"Model: {args.model.upper()}")
    print(f"Environment: {env_mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Episode: {args.episode}")
    print(f"Training steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print()
    
    # Initialize wandb if requested
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project="multi-model-training",
            config={
                "model": args.model,
                "dataset": args.dataset,
                "episode": args.episode,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "environment": env_mode
            }
        )
        print("📊 W&B logging enabled")
    
    try:
        # 1. Create single episode dataset
        full_dataset, single_ep_dataset, metadata, episode_length = create_single_episode_dataset(
            args.dataset, args.episode, args.video_backend
        )
        
        # Get episode indices for dataloader
        episode_indices = get_single_episode_indices(full_dataset, args.episode)
        
        # 2. Setup policy based on model type
        policy, config, delta_timestamps = setup_policy_and_config(args.model, metadata)
        
        # 3. Create dataloader
        dataloader = create_dataloader(
            full_dataset, episode_indices, delta_timestamps, 
            args.batch_size, video_backend=args.video_backend
        )
        
        # 4. Train
        losses = train_single_episode(
            policy, dataloader, args.steps, device, args.model, config,
            cloud_mode=args.cloud, accumulation_steps=accumulation_steps
        )
        
        # Log to wandb
        if args.wandb and WANDB_AVAILABLE:
            for i, loss in enumerate(losses):
                wandb.log({"loss": loss, "step": i})
        
        # 5. Save model
        save_path = save_model(policy, args.output_dir, args.model, args.episode, losses[-1])
        
        # 6. Plot results
        plot_training_progress(losses, args.model, save_path, show_plot=not args.no_plot)
        
        print(f"\n🎉 {args.model.upper()} TRAINING COMPLETE!")
        print(f"   Episode {args.episode} trained for {args.steps} steps")
        print(f"   Final loss: {losses[-1]:.6f}")
        print(f"   Model saved to: {save_path}")
        
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({"final_loss": losses[-1]})
            wandb.finish()
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 