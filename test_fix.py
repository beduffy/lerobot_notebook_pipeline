#!/usr/bin/env python3
"""
Test the memory fix for diffusion models.
"""

import torch
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
        # Load dataset
        dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place")
        
        # Create memory-efficient config
        config = DiffusionConfig(
            input_features=["observation.state", "observation.images.front"],
            output_features=["action"],
            n_obs_steps=1,
            horizon=8,
            n_action_steps=4,
            vision_backbone="resnet18",
            crop_shape=(64, 64),
            num_train_timesteps=50,
        )
        
        # Create policy
        policy = DiffusionPolicy(config, dataset_stats=dataset.stats)
        
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
        
        # Create minimal batch
        batch_size = 1
        dummy_batch = {
            "observation.state": torch.randn(batch_size, 1, 7),
            "observation.images.front": torch.randn(batch_size, 1, 3, 64, 64),
            "action": torch.randn(batch_size, 4, 7),
        }
        
        # Move to device
        for key in dummy_batch:
            dummy_batch[key] = dummy_batch[key].to(device, non_blocking=True)
        
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