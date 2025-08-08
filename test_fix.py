#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the memory fix for diffusion models.
"""

import torch
import torch.nn.functional as F
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.policies import FeatureType


def test_memory_fix():
    """Test if the memory fix works."""
    
    print("🧪 Testing memory fix for diffusion model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Initial GPU Memory: {initial_memory:.2f}GB / {total_memory:.2f}GB")
    
    try:
        # Load dataset and metadata
        dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place")
        metadata = LeRobotDatasetMetadata("bearlover365/red_cube_always_in_same_place")
        
        # Convert dataset features to policy features
        features = dataset_to_policy_features(metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        # Create memory-efficient config
        config = DiffusionConfig(
            input_features=input_features,
            output_features=output_features,
            n_obs_steps=1,
            horizon=8,
            n_action_steps=4,
            vision_backbone="resnet18",
            crop_shape=(64, 64),
            num_train_timesteps=50,
        )
        
        # Create policy
        policy = DiffusionPolicy(config, dataset_stats=metadata.stats)
        
        # Enable gradient checkpointing
        if hasattr(policy, 'gradient_checkpointing_enable'):
            policy.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        
        # Set memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            print("✅ GPU memory fraction set to 80%")
        
        # Move to device
        policy.to(device)
        policy.train()
        
        if device.type == "cuda":
            memory_after_model = torch.cuda.memory_allocated() / 1024**3
            print(f"Memory after model: {memory_after_model:.2f}GB")
        
        # Get actual sample from dataset
        sample = dataset[0]
        batch_size = 1
        dummy_batch = {}
        
        # Create batch with proper shapes
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                # Add batch dimension
                dummy_batch[key] = value.unsqueeze(0).to(device, non_blocking=True)
        
        # Reshape action to match horizon (8 steps)
        if "action" in dummy_batch:
            action = dummy_batch["action"]  # Shape: [1, 6]
            # Repeat action to match horizon
            action = action.unsqueeze(1).expand(1, 8, 6)  # Shape: [1, 8, 6]
            dummy_batch["action"] = action
        
        # Reshape observation state to match n_obs_steps (1 step)
        if "observation.state" in dummy_batch:
            obs_state = dummy_batch["observation.state"]  # Shape: [1, 6]
            # Repeat observation to match n_obs_steps
            obs_state = obs_state.unsqueeze(1).expand(1, 1, 6)  # Shape: [1, 1, 6]
            dummy_batch["observation.state"] = obs_state
        
        # Add required action_is_pad field for diffusion model
        dummy_batch["action_is_pad"] = torch.zeros(1, 8, dtype=torch.bool, device=device)
        
        # Resize image inputs to match config.crop_shape and add time dim [B, T, C, H, W]
        if "observation.images.front" in dummy_batch:
            img = dummy_batch["observation.images.front"]  # [1, 3, H, W]
            img = F.interpolate(img, size=(64, 64), mode="bilinear", align_corners=False)
            dummy_batch["observation.images.front"] = img.unsqueeze(1)  # [1, 1, 3, 64, 64]

        print(f"   Batch keys: {list(dummy_batch.keys())}")
        print(f"   Batch shapes: {[(k, v.shape) for k, v in dummy_batch.items()]}")
        
        # Test forward pass
        print(f"\n🚀 Testing forward pass...")
        if device.type == "cuda":
            memory_before = torch.cuda.memory_allocated() / 1024**3
            print(f"Memory before forward: {memory_before:.2f}GB")
        
        loss, output_dict = policy.forward(dummy_batch)
        
        if device.type == "cuda":
            memory_after = torch.cuda.memory_allocated() / 1024**3
            print(f"Memory after forward: {memory_after:.2f}GB")
            print(f"Memory used: {memory_after - memory_before:.2f}GB")
            
            if memory_after < 15:  # Should be much less than 15GB
                print(f"✅ Memory usage looks good!")
            else:
                print(f"⚠️  Still using a lot of memory: {memory_after:.2f}GB")
        
        print(f"✅ Forward pass successful!")
        print(f"Loss: {loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise e

if __name__ == "__main__":
    test_memory_fix() 