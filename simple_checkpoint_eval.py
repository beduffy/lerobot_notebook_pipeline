#!/usr/bin/env python3
"""
Simple checkpoint evaluation - loads model state directly.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Import SmolVLA for direct loading
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    SMOLVLA_AVAILABLE = True
except ImportError:
    SMOLVLA_AVAILABLE = False


def load_smolvla_from_checkpoint(checkpoint_path):
    """Load SmolVLA model directly from checkpoint."""
    print(f"🧠 Loading SmolVLA from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"   ✅ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    
    # Create SmolVLA model with default config
    if not SMOLVLA_AVAILABLE:
        raise RuntimeError("SmolVLA not available")
    
    # Create a minimal config for SmolVLA
    config = SmolVLAConfig()
    
    # Create policy with minimal dataset stats (we'll override later)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    dummy_metadata = LeRobotDatasetMetadata("bearlover365/red_cube_always_in_same_place")
    
    policy = SmolVLAPolicy(config, dataset_stats=dummy_metadata.stats)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        # Use strict=False to ignore unexpected keys like normalization buffers
        missing_keys, unexpected_keys = policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   ✅ Loaded state dict successfully")
        if missing_keys:
            print(f"   ⚠️  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"   ⚠️  Ignored unexpected keys: {unexpected_keys}")
    else:
        print("   ⚠️  No model_state_dict found in checkpoint")
    
    return policy


def evaluate_smolvla_on_episode(checkpoint_path, dataset_name, episode_idx, device, camera_remap=None):
    """Evaluate SmolVLA model on a specific episode."""
    print(f"🔬 Evaluating SmolVLA model on episode {episode_idx}...")
    
    # Load model from checkpoint
    policy = load_smolvla_from_checkpoint(checkpoint_path)
    policy.to(device)
    policy.eval()
    policy.reset()
    
    # Load dataset  
    dataset = LeRobotDataset(dataset_name, video_backend="pyav")
    
    # Apply camera remapping if specified
    if camera_remap:
        print(f"   📷 Applying camera remapping: {camera_remap}")
        
        class CameraRemapDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, camera_remap):
                self.dataset = dataset
                self.camera_remap = camera_remap
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                batch = self.dataset[idx]
                remapped_batch = {}
                
                for key, value in batch.items():
                    if key in self.camera_remap:
                        # Remap camera keys
                        new_key = self.camera_remap[key]
                        remapped_batch[new_key] = value
                    else:
                        remapped_batch[key] = value
                
                return remapped_batch
        
        dataset = CameraRemapDataset(dataset, camera_remap)
    
    # Get episode indices
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    episode_indices = list(range(from_idx, to_idx))
    
    print(f"   Episode {episode_idx}: {len(episode_indices)} steps")
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i, idx in enumerate(episode_indices):
            try:
                sample = dataset[idx]
                
                # Prepare input - ONLY OBSERVATIONS FOR INFERENCE
                batch = {}
                for key, value in sample.items():
                    if key.startswith("observation.") and isinstance(value, torch.Tensor):
                        batch[key] = value.unsqueeze(0).to(device)
                
                # Add language task for SmolVLA
                batch["task"] = "grab red cube and put to left"
                
                # Get prediction
                pred_action_chunk = policy.select_action(batch)
                
                # Handle both chunked and non-chunked outputs
                if pred_action_chunk.dim() == 3:  # [batch, chunk_size, action_dim]
                    pred_action = pred_action_chunk[0, 0, :]  # Take first action from chunk
                else:  # [batch, action_dim]
                    pred_action = pred_action_chunk[0, :]  # Take first action
                
                gt_action = sample["action"]  # Ground truth action
                
                # Ensure both tensors are 1D before appending
                if pred_action.dim() > 1:
                    pred_action = pred_action.squeeze()
                if gt_action.dim() > 1:
                    gt_action = gt_action.squeeze()
                
                predictions.append(pred_action.cpu())
                ground_truths.append(gt_action.cpu())
                
                # Show progress for long episodes
                if i % 50 == 0:
                    print(f"   Processed {i}/{len(episode_indices)} steps...")
                    
            except Exception as e:
                print(f"   Warning: Skipped step {idx}: {e}")
                continue
    
    if predictions:
        predictions = torch.stack(predictions, dim=0)  # Stack to create [num_steps, action_dim]
        ground_truths = torch.stack(ground_truths, dim=0)  # Stack to create [num_steps, action_dim]
        
        # Calculate metrics
        mae = torch.mean(torch.abs(predictions - ground_truths)).item()
        mse = torch.mean((predictions - ground_truths) ** 2).item()
        max_error = torch.max(torch.abs(predictions - ground_truths)).item()
        
        print(f"   📊 Results:")
        print(f"      Mean Absolute Error: {mae:.6f}")
        print(f"      Mean Squared Error: {mse:.6f}")
        print(f"      Max Absolute Error: {max_error:.6f}")
        
        if mae < 0.01:
            print("      🎉 Excellent performance!")
        elif mae < 0.1:
            print("      ✅ Good performance")
        elif mae < 1.0:
            print("      ⚠️  Moderate performance")
        else:
            print("      ❌ Poor performance")
        
        # Per-joint analysis
        action_dim = predictions.shape[1]
        joint_names = [f'Joint {i+1}' for i in range(action_dim-1)] + ['Gripper'] if action_dim == 7 else [f'Action {i+1}' for i in range(action_dim)]
        print(f"\n🔧 Per-joint errors:")
        for i, joint_name in enumerate(joint_names):
            joint_error = torch.mean(torch.abs(predictions[:, i] - ground_truths[:, i])).item()
            print(f"      {joint_name}: {joint_error:.6f}")
            
        return mae, mse, max_error, predictions, ground_truths, joint_names
    else:
        print("   ❌ No valid predictions generated")
        return None, None, None, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Simple SmolVLA checkpoint evaluation")
    parser.add_argument("checkpoint_path", help="Path to checkpoint file")
    parser.add_argument("--dataset", default="bearlover365/red_cube_always_in_same_place")
    parser.add_argument("--episode", type=int, default=0, help="Episode to evaluate on")
    parser.add_argument("--camera-remap", type=str, default=None,
                       help="Camera remapping (e.g., 'observation.images.front:observation.images.wrist')")
    
    args = parser.parse_args()
    
    # Parse camera remapping
    camera_remap = None
    if args.camera_remap:
        camera_remap = {}
        for mapping in args.camera_remap.split(','):
            if ':' in mapping:
                old_key, new_key = mapping.strip().split(':')
                camera_remap[old_key.strip()] = new_key.strip()
        print(f"📷 Camera remapping: {camera_remap}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📂 Checkpoint: {args.checkpoint_path}")
    print(f"📊 Dataset: {args.dataset}")
    
    if not Path(args.checkpoint_path).exists():
        print(f"❌ Checkpoint path does not exist: {args.checkpoint_path}")
        return 1
    
    try:
        # Single episode evaluation
        mae, mse, max_err, predictions, ground_truths, joint_names = evaluate_smolvla_on_episode(
            args.checkpoint_path, args.dataset, args.episode, device, 
            camera_remap=camera_remap
        )
        
        if mae is not None:
            print(f"\n✅ Evaluation completed!")
            print(f"   Episode: {args.episode}")
            print(f"   MAE: {mae:.6f}")
            print(f"   MSE: {mse:.6f}")
            print(f"   Max Error: {max_err:.6f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 