#!/usr/bin/env python3
"""
Evaluate model from checkpoint file.

This script can evaluate a model that's still being trained by loading from a checkpoint.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset

# Import all model types
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

# Import foundation models (VLAs)
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    SMOLVLA_AVAILABLE = True
except ImportError:
    SMOLVLA_AVAILABLE = False

try:
    from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
    from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
    PI0FAST_AVAILABLE = True
except ImportError:
    PI0FAST_AVAILABLE = False


def load_policy_from_checkpoint(checkpoint_path, model_type):
    """Load policy from checkpoint file."""
    print(f"🧠 Loading {model_type.upper()} policy from checkpoint {checkpoint_path}...")
    
    # Load checkpoint with weights_only=False to allow loading custom classes
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("   ✅ Loaded checkpoint with weights_only=False")
    except Exception as e:
        print(f"   ⚠️  Failed to load with weights_only=False: {e}")
        # Try with safe globals for SmolVLA
        if model_type == 'smolvla' and SMOLVLA_AVAILABLE:
            try:
                import torch.serialization
                torch.serialization.add_safe_globals([SmolVLAConfig])
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                print("   ✅ Loaded with safe globals")
            except Exception as e2:
                print(f"   ❌ Failed to load with safe globals: {e2}")
                raise e
        else:
            raise e
    
    # Create policy based on model type
    if model_type == 'act':
        policy = ACTPolicy.from_pretrained("lerobot/act_pusht")  # Use base config
    elif model_type == 'diffusion':
        policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")  # Use base config
    elif model_type == 'vqbet':
        policy = VQBeTPolicy.from_pretrained("lerobot/vqbet_pusht")  # Use base config
    elif model_type == 'smolvla':
        if not SMOLVLA_AVAILABLE:
            raise RuntimeError("SmolVLA not available. Install with updated LeRobot version.")
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_pusht")  # Use base config
    elif model_type == 'pi0fast':
        if not PI0FAST_AVAILABLE:
            raise RuntimeError("π0-FAST not available. Install with updated LeRobot version.")
        policy = PI0FASTPolicy.from_pretrained("lerobot/pi0fast_pusht")  # Use base config
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Loaded state dict from step {checkpoint.get('step', 'unknown')}")
    else:
        print("   ⚠️  No model_state_dict found in checkpoint")
    
    return policy


def evaluate_model_on_episode(checkpoint_path, model_type, dataset_name, episode_idx, device, use_dataloader=True, camera_remap=None):
    """Evaluate model on a specific episode and return predictions/ground truth for plotting."""
    print(f"🔬 Evaluating {model_type.upper()} model on episode {episode_idx}...")
    
    # Load model from checkpoint
    policy = load_policy_from_checkpoint(checkpoint_path, model_type)
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
                
                # Add language task for VLA models (SmolVLA, π0-FAST)
                if model_type in ["smolvla", "pi0fast"]:
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
    parser = argparse.ArgumentParser(description="Evaluate model from checkpoint")
    parser.add_argument("checkpoint_path", help="Path to checkpoint file")
    parser.add_argument("--model-type", type=str, required=True,
                       help="Model type (act, diffusion, vqbet, smolvla, pi0fast)")
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
    print(f"🧠 Model type: {args.model_type.upper()}")
    print(f"📊 Dataset: {args.dataset}")
    
    if not Path(args.checkpoint_path).exists():
        print(f"❌ Checkpoint path does not exist: {args.checkpoint_path}")
        return 1
    
    try:
        # Single episode evaluation
        mae, mse, max_err, predictions, ground_truths, joint_names = evaluate_model_on_episode(
            args.checkpoint_path, args.model_type, args.dataset, args.episode, device, 
            camera_remap=camera_remap
        )
        
        if mae is not None:
            print(f"\n✅ Evaluation completed!")
            print(f"   Model type: {args.model_type.upper()}")
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