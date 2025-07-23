#!/usr/bin/env python3
"""
Single Episode Training Script - Unified Version

Trains ACT on exactly 1 episode with full transparency.
Auto-detects cloud vs local environment and adjusts settings.

Usage:
    # Local quick test
    python train_single_episode.py --episode 0 --steps 50
    
    # Cloud training with all features
    python train_single_episode.py --episode 0 --steps 2000 --cloud --upload-model --wandb
    
    # Custom configuration
    python train_single_episode.py --episode 0 --steps 1000 --batch-size 16 --video-backend opencv
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import time
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Auto-detect and suppress warnings for cloud environments
def setup_environment(cloud_mode=False):
    """Setup environment based on cloud vs local."""
    if cloud_mode or torch.cuda.is_available():
        # Cloud/GPU environment - suppress video warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtbufsize;0"
        return "cloud"
    else:
        # Local environment - allow warnings but don't spam
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        return "local"

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

# Optional imports with graceful fallback
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


def setup_policy_and_config(metadata):
    """Setup ACT policy with transparent configuration."""
    print(f"\n🧠 Setting up ACT policy...")
    
    # Convert dataset features to policy features
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # Remove wrist camera if present (using front camera only)
    if "observation.wrist_image" in input_features:
        input_features.pop("observation.wrist_image")
    
    print(f"   Input features: {list(input_features.keys())}")
    print(f"   Output features: {list(output_features.keys())}")
    
    # Create ACT configuration
    cfg = ACTConfig(
        input_features=input_features, 
        output_features=output_features, 
        chunk_size=10,
        n_action_steps=10
    )
    
    print(f"   Chunk size: {cfg.chunk_size} (predicts {cfg.chunk_size} future actions)")
    print(f"   Hidden dim: {cfg.dim_model}")
    print(f"   Encoder layers: {cfg.n_encoder_layers}")
    print(f"   Decoder layers: {cfg.n_decoder_layers}")
    
    # Resolve delta timestamps for action chunking
    delta_timestamps = resolve_delta_timestamps(cfg, metadata)
    
    # Create policy
    policy = ACTPolicy(cfg, dataset_stats=metadata.stats)
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    return policy, cfg, delta_timestamps


def create_dataloader(full_dataset, single_episode_indices, delta_timestamps, batch_size=8, 
                     num_workers=None, video_backend="pyav"):
    """Create dataloader for single episode training."""
    print(f"\n📦 Creating single-episode dataloader...")
    
    # Auto-adjust num_workers based on environment
    if num_workers is None:
        if torch.cuda.is_available():
            num_workers = 4  # Cloud/GPU
        else:
            num_workers = 2  # Local/CPU
    
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


def train_single_episode(policy, dataloader, num_steps, device, learning_rate=1e-4, 
                        log_every=None, cloud_mode=False):
    """Train policy on single episode with environment-appropriate logging."""
    print(f"\n🚀 Starting training on single episode...")
    print(f"   Training steps: {num_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {device}")
    print(f"   Mode: {'Cloud' if cloud_mode else 'Local'}")
    
    # Auto-adjust logging frequency
    if log_every is None:
        if cloud_mode or num_steps > 500:
            log_every = 100  # Less frequent for cloud/long runs
        else:
            log_every = 50   # More frequent for local/short runs
    
    # Setup training
    policy.to(device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Training metrics
    losses = []
    step_times = []
    
    # Training loop
    start_time = time.time()
    dataloader_iter = iter(dataloader)
    
    for step in range(num_steps):
        step_start = time.time()
        
        # Get batch (cycle through episode data)
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=torch.cuda.is_available())
        
        # Forward pass
        loss, output_dict = policy.forward(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        
        # Update
        optimizer.step()
        
        # Record metrics
        loss_value = loss.item()
        losses.append(loss_value)
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Logging
        if step % log_every == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            eta_min = (num_steps - step - 1) / steps_per_sec / 60 if steps_per_sec > 0 else 0
            
            if cloud_mode:
                # Compact logging for cloud
                avg_loss = np.mean(losses[-min(50, len(losses)):])
                print(f"   Step {step:4d}/{num_steps} | "
                      f"Loss: {loss_value:.4f} | "
                      f"Avg: {avg_loss:.4f} | "
                      f"{steps_per_sec:.1f} steps/s | "
                      f"ETA: {eta_min:.1f}min")
            else:
                # Detailed logging for local
                avg_loss = np.mean(losses[-log_every:])
                avg_time = np.mean(step_times[-log_every:])
                print(f"Step {step:4d}/{num_steps} | "
                      f"Loss: {loss_value:.6f} | "
                      f"Avg Loss: {avg_loss:.6f} | "
                      f"Time/step: {avg_time:.3f}s | "
                      f"ETA: {eta_min:.1f}min")
    
    total_time = time.time() - start_time
    final_steps_per_sec = num_steps / total_time if total_time > 0 else 0
    
    print(f"\n✅ Training completed!")
    print(f"   Total time: {total_time/60:.2f} minutes")
    print(f"   Speed: {final_steps_per_sec:.1f} steps/second")
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Average loss (last 100 steps): {np.mean(losses[-100:]):.6f}")
    
    return losses


def evaluate_policy(policy, dataloader, device, quick_mode=False):
    """Evaluate policy on the training episode."""
    print(f"\n🔬 Evaluating policy{'(quick mode)' if quick_mode else ''}...")
    
    policy.eval()
    policy.reset()
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if quick_mode and i >= 10:  # Quick mode: only first 10 batches
                break
                
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get prediction
            predicted_action = policy.select_action(batch)
            ground_truth_action = batch["action"][:, 0, :]  # First action in chunk
            
            all_predictions.append(predicted_action)
            all_ground_truth.append(ground_truth_action)
    
    if all_predictions:
        # Concatenate all predictions
        predictions = torch.cat(all_predictions, dim=0)
        ground_truth = torch.cat(all_ground_truth, dim=0)
        
        # Handle size mismatch (common with batching)
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]
        
        # Calculate errors
        mae = torch.mean(torch.abs(predictions - ground_truth)).item()
        mse = torch.mean((predictions - ground_truth) ** 2).item()
        max_error = torch.max(torch.abs(predictions - ground_truth)).item()
        
        print(f"   Mean Absolute Error: {mae:.6f}")
        print(f"   Mean Squared Error: {mse:.6f}")
        print(f"   Max Absolute Error: {max_error:.6f}")
        
        if mae < 0.01:
            print("   🎉 Excellent! Model memorized the episode very well")
        elif mae < 0.1:
            print("   ✅ Good! Model learned the general trajectory")
        else:
            print("   ⚠️  High error - might need more training")
        
        return mae, mse, predictions, ground_truth
    else:
        print("   ❌ No predictions generated")
        return None, None, None, None


def save_model(policy, save_dir, episode_idx, final_loss, mae):
    """Save the trained model."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving model to: {save_path}")
    
    # Save using LeRobot's method
    policy.save_pretrained(save_path)
    
    # Save training info
    info = {
        "episode_index": episode_idx,
        "final_loss": final_loss,
        "mae": mae if mae is not None else "N/A",
        "training_completed": True
    }
    
    import json
    with open(save_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"   ✅ Model saved successfully!")
    return save_path


def upload_model(save_path, episode_idx):
    """Upload model to HuggingFace."""
    if not HF_AVAILABLE:
        print("   ❌ HuggingFace Hub not available - skipping upload")
        return None
        
    print(f"🚀 Uploading to HuggingFace...")
    try:
        api = HfApi()
        repo_name = f"single_episode_{episode_idx}_model"
        api.create_repo(repo_name, exist_ok=True)
        api.upload_folder(folder_path=save_path, repo_id=repo_name)
        print(f"   ✅ Uploaded to: {repo_name}")
        return repo_name
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return None


def plot_training_progress(losses, save_path=None, show_plot=True):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss - Single Episode")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path / "training_curve.png", dpi=150, bbox_inches='tight')
        print(f"   📊 Training curve saved to: {save_path}/training_curve.png")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train ACT on single episode")
    parser.add_argument("--dataset", default="bearlover365/red_cube_always_in_same_place", 
                       help="Dataset name/path")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to train on")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if not set)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", default="./single_episode_model", help="Save directory")
    parser.add_argument("--video-backend", choices=["pyav", "opencv"], default=None, 
                       help="Video backend (auto if not set)")
    
    # Cloud/advanced options
    parser.add_argument("--cloud", action="store_true", help="Enable cloud optimizations")
    parser.add_argument("--upload-model", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--quick-eval", action="store_true", help="Quick evaluation mode")
    parser.add_argument("--no-plot", action="store_true", help="Don't show plots (save only)")
    
    args = parser.parse_args()
    
    # Setup environment
    env_mode = setup_environment(args.cloud)
    
    # Auto-configure based on environment
    if args.batch_size is None:
        args.batch_size = 32 if (args.cloud or torch.cuda.is_available()) else 8
    
    if args.video_backend is None:
        args.video_backend = "opencv" if (args.cloud or torch.cuda.is_available()) else "pyav"
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎯 SINGLE EPISODE TRAINING")
    print("=" * 50)
    print(f"Environment: {env_mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Episode: {args.episode}")
    print(f"Training steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Video backend: {args.video_backend}")
    print(f"Device: {device}")
    print()
    
    # Initialize wandb if requested
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project="single-episode-training",
            config={
                "dataset": args.dataset,
                "episode": args.episode,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
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
        
        # 2. Setup policy
        policy, cfg, delta_timestamps = setup_policy_and_config(metadata)
        
        # 3. Create dataloader
        dataloader = create_dataloader(
            full_dataset, episode_indices, delta_timestamps, 
            args.batch_size, video_backend=args.video_backend
        )
        
        # 4. Train
        losses = train_single_episode(
            policy, dataloader, args.steps, device, args.lr, 
            cloud_mode=args.cloud
        )
        
        # Log to wandb
        if args.wandb and WANDB_AVAILABLE:
            for i, loss in enumerate(losses):
                wandb.log({"loss": loss, "step": i})
        
        # 5. Evaluate
        mae, mse, predictions, ground_truth = evaluate_policy(
            policy, dataloader, device, quick_mode=args.quick_eval
        )
        
        # 6. Save model
        save_path = save_model(policy, args.output_dir, args.episode, losses[-1], mae)
        
        # 7. Plot results
        plot_training_progress(losses, save_path, show_plot=not args.no_plot)
        
        # 8. Upload if requested
        uploaded_repo = None
        if args.upload_model:
            uploaded_repo = upload_model(save_path, args.episode)
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"   Episode {args.episode} trained for {args.steps} steps")
        print(f"   Final loss: {losses[-1]:.6f}")
        if mae is not None:
            print(f"   Mean Absolute Error: {mae:.6f}")
        print(f"   Model saved to: {save_path}")
        if uploaded_repo:
            print(f"   Uploaded to: {uploaded_repo}")
        
        if args.wandb and WANDB_AVAILABLE:
            if mae is not None:
                wandb.log({"final_mae": mae, "final_loss": losses[-1]})
            wandb.finish()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 