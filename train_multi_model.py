#!/usr/bin/env python3
"""
Multi-Model Training Script with Episode Slicing Support

Extended version of train_single_episode.py that supports multiple architectures and multi-episode training:
- ACT (proven working)
- Diffusion Policy (working)  
- VQBet (working)

Episode Selection Examples:
    # Single episode
    python train_multi_model.py --model act --episodes 0 --steps 1000
    
    # Multiple specific episodes
    python train_multi_model.py --model diffusion --episodes 1,4,7 --steps 1000
    
    # Episode range (slice notation)
    python train_multi_model.py --model vqbet --episodes 0:5 --steps 1000  # First 5 episodes
    python train_multi_model.py --model act --episodes 2: --steps 1000     # From episode 2 to end
    python train_multi_model.py --model diffusion --episodes :3 --steps 1000  # First 3 episodes
    
    # All episodes
    python train_multi_model.py --model act --episodes all --steps 1000
    
Model Comparison Examples:
    # Compare all models on same data
    python train_multi_model.py --model act --episodes 0:5 --steps 500 --output-dir ./models/act_first5_500
    python train_multi_model.py --model diffusion --episodes 0:5 --steps 500 --output-dir ./models/diffusion_first5_500  
    python train_multi_model.py --model vqbet --episodes 0:5 --steps 500 --output-dir ./models/vqbet_first5_500
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
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps

# Import all model types
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

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


def parse_episodes(episodes_str: str, total_episodes: int) -> list:
    """Parse episode specification into list of episode indices.
    
    Supports:
    - Single episode: "3"
    - Multiple episodes: "1,4,7"
    - Slice notation: "0:5", "1:", ":3"
    - All episodes: "all"
    
    Returns:
        List of episode indices
    """
    episodes_str = episodes_str.strip()
    
    # Handle "all" keyword
    if episodes_str.lower() == "all":
        return list(range(total_episodes))
    
    # Handle slice notation (e.g., "0:5", "1:", ":3")
    if ":" in episodes_str:
        parts = episodes_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid slice notation: {episodes_str}")
        
        start_str, end_str = parts
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else total_episodes
        
        # Validate bounds
        start = max(0, min(start, total_episodes))
        end = max(0, min(end, total_episodes))
        
        if start >= end:
            raise ValueError(f"Invalid slice: start ({start}) >= end ({end})")
            
        return list(range(start, end))
    
    # Handle comma-separated episodes (e.g., "1,4,7")
    if "," in episodes_str:
        try:
            episodes = [int(x.strip()) for x in episodes_str.split(",")]
            # Validate all episodes are in bounds
            for ep in episodes:
                if ep < 0 or ep >= total_episodes:
                    raise ValueError(f"Episode {ep} out of bounds (0-{total_episodes-1})")
            return sorted(list(set(episodes)))  # Remove duplicates and sort
        except ValueError as e:
            raise ValueError(f"Invalid episode list: {episodes_str}") from e
    
    # Handle single episode
    try:
        episode_idx = int(episodes_str)
        if episode_idx < 0 or episode_idx >= total_episodes:
            raise ValueError(f"Episode {episode_idx} out of bounds (0-{total_episodes-1})")
        return [episode_idx]
    except ValueError as e:
        raise ValueError(f"Invalid episode specification: {episodes_str}") from e


def get_multi_episode_indices(dataset: LeRobotDataset, episode_indices: list) -> list:
    """Get all data indices for multiple episodes."""
    all_indices = []
    for episode_idx in episode_indices:
        episode_data = get_single_episode_indices(dataset, episode_idx)
        all_indices.extend(episode_data)
    return all_indices


def create_multi_episode_dataset(dataset_name: str, episodes_spec: str, video_backend: str = "pyav"):
    """Create dataset containing specified episodes."""
    print(f"📊 Loading dataset: {dataset_name}")
    
    # Load metadata first
    metadata = LeRobotDatasetMetadata(dataset_name)
    print(f"   Total episodes: {metadata.total_episodes}")
    print(f"   Total frames: {metadata.total_frames}")
    
    # Parse episode specification
    episode_indices = parse_episodes(episodes_spec, metadata.total_episodes)
    print(f"🎯 Selected episodes: {episode_indices}")
    
    # Load full dataset
    full_dataset = LeRobotDataset(dataset_name, video_backend=video_backend)
    
    # Get indices for all specified episodes
    all_data_indices = get_multi_episode_indices(full_dataset, episode_indices)
    total_length = len(all_data_indices)
    
    print(f"📈 Training Data Summary:")
    print(f"   Episodes: {len(episode_indices)} episodes")
    print(f"   Frame indices: {all_data_indices[0]} to {all_data_indices[-1]}")
    print(f"   Total training steps: {total_length}")
    
    # Show per-episode breakdown
    if len(episode_indices) <= 10:  # Only show details for <= 10 episodes
        for ep_idx in episode_indices:
            ep_data = get_single_episode_indices(full_dataset, ep_idx)
            print(f"   Episode {ep_idx}: {len(ep_data)} steps")
    else:
        avg_length = total_length // len(episode_indices)
        print(f"   Average episode length: ~{avg_length} steps")
    
    # Create subset dataset with selected episodes
    multi_episode_dataset = Subset(full_dataset, all_data_indices)
    
    return full_dataset, multi_episode_dataset, metadata, total_length, episode_indices


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


def create_dataloader(full_dataset, multi_episode_indices, delta_timestamps, batch_size=8, 
                     num_workers=None, video_backend="pyav"):
    """Create dataloader for multi-episode training."""
    print(f"\n📦 Creating multi-episode dataloader...")
    
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
    
    # Create subset for our episodes
    multi_episode_dataset = Subset(dataset_with_config, multi_episode_indices)
    
    # Create dataloader
    dataloader = DataLoader(
        multi_episode_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle across all episodes for better training
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print(f"   Total training steps: {len(multi_episode_dataset)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Batches per epoch: {len(dataloader)}")
    print(f"   Workers: {num_workers}")
    print(f"   Data shuffling: ✅ Enabled (mixed episode training)")
    
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


def train_multi_episode(policy, dataloader, num_steps, device, model_type, config, 
                       episodes_info, cloud_mode=False, accumulation_steps=1):
    """Train policy on multiple episodes with model-specific optimizations."""
    episode_count = len(episodes_info) if isinstance(episodes_info, list) else 1
    print(f"\n🚀 Starting {model_type.upper()} training on {episode_count} episodes...")
    
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


def save_model(policy, save_dir, model_type, episode_indices, final_loss):
    """Save the trained model with model type info."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving {model_type.upper()} model to: {save_path}")
    
    # Save using LeRobot's method
    policy.save_pretrained(save_path)
    
    # Save training info
    info = {
        "model_type": model_type,
        "episode_indices": episode_indices,
        "num_episodes": len(episode_indices),
        "final_loss": final_loss,
        "training_completed": True,
        "parameters": sum(p.numel() for p in policy.parameters())
    }
    
    import json
    with open(save_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"   ✅ {model_type.upper()} model saved successfully!")
    print(f"   Trained on {len(episode_indices)} episodes: {episode_indices}")
    return save_path


def plot_training_progress(losses, model_type, save_path=None, show_plot=True, episode_info=None):
    """Plot training loss curve with model info."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    
    # Create title with episode information
    if episode_info and len(episode_info) > 1:
        if len(episode_info) <= 5:
            episode_str = f"Episodes {episode_info}"
        else:
            episode_str = f"{len(episode_info)} Episodes ({episode_info[0]}-{episode_info[-1]})"
    elif episode_info:
        episode_str = f"Episode {episode_info[0]}"
    else:
        episode_str = "Multi-Episode"
    
    plt.title(f"Training Loss - {model_type.upper()} ({episode_str})")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    # Add model info to plot
    final_loss = losses[-1] if losses else 0
    info_text = f'Model: {model_type.upper()}\nFinal Loss: {final_loss:.4f}'
    if episode_info:
        info_text += f'\nEpisodes: {len(episode_info)}'
    
    plt.text(0.02, 0.98, info_text, 
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
    parser.add_argument("--episodes", type=str, default="0", 
                       help="Episodes to train on. Examples: '0' (single), '0:5' (slice), '1,4,7' (specific), 'all' (all episodes)")
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
        # Create descriptive output directory name
        episodes_desc = args.episodes.replace(":", "_").replace(",", "_")
        if episodes_desc == "all":
            episodes_desc = "all_episodes"
        args.output_dir = f"./models/{args.model}_episodes_{episodes_desc}_{args.steps}steps"
    
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
    print(f"Episodes: {args.episodes}")
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
                "episodes": args.episodes,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "environment": env_mode
            }
        )
        print("📊 W&B logging enabled")
    
    try:
        # 1. Create multi-episode dataset
        full_dataset, multi_ep_dataset, metadata, total_length, episode_indices = create_multi_episode_dataset(
            args.dataset, args.episodes, args.video_backend
        )
        
        # Get data indices for dataloader
        all_data_indices = get_multi_episode_indices(full_dataset, episode_indices)
        
        # 2. Setup policy based on model type
        policy, config, delta_timestamps = setup_policy_and_config(args.model, metadata)
        
        # 3. Create dataloader
        dataloader = create_dataloader(
            full_dataset, all_data_indices, delta_timestamps, 
            args.batch_size, video_backend=args.video_backend
        )
        
        # 4. Train
        losses = train_multi_episode(
            policy, dataloader, args.steps, device, args.model, config, episode_indices,
            cloud_mode=args.cloud, accumulation_steps=accumulation_steps
        )
        
        # Log to wandb
        if args.wandb and WANDB_AVAILABLE:
            for i, loss in enumerate(losses):
                wandb.log({"loss": loss, "step": i})
            # Log additional metadata
            wandb.log({
                "num_episodes": len(episode_indices),
                "total_training_steps": total_length,
                "episode_indices": episode_indices
            })
        
        # 5. Save model
        save_path = save_model(policy, args.output_dir, args.model, episode_indices, losses[-1])
        
        # 6. Plot results
        plot_training_progress(losses, args.model, save_path, show_plot=not args.no_plot, episode_info=episode_indices)
        
        print(f"\n🎉 {args.model.upper()} TRAINING COMPLETE!")
        print(f"   Episodes {episode_indices} trained for {args.steps} steps")
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