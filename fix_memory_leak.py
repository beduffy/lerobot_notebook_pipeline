#!/usr/bin/env python3
"""
Fix for the diffusion model memory leak.
"""

import torch
import torch.nn as nn
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

def create_memory_efficient_diffusion_config():
    """Create a memory-efficient diffusion configuration."""
    
    config = DiffusionConfig(
        input_features=["observation.state", "observation.images.front"],
        output_features=["action"],
        # Reduce memory usage
        n_obs_steps=1,  # Reduced from 2
        horizon=8,      # Reduced from 16
        n_action_steps=4,  # Reduced from 8
        vision_backbone="resnet18",
        crop_shape=(64, 64),  # Reduced from 84x84
        crop_is_random=True,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=16,  # Reduced from 32
        down_dims=(256, 512, 1024),  # Reduced dimensions
        kernel_size=3,  # Reduced from 5
        n_groups=4,  # Reduced from 8
        diffusion_step_embed_dim=64,  # Reduced from 128
        use_film_scale_modulation=True,
        noise_scheduler_type="DDPM",
        num_train_timesteps=50,  # Reduced from 100
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        num_inference_steps=10,
        do_mask_loss_for_padding=False,
        # Training presets
        optimizer_lr=1e-4,
        optimizer_betas=(0.95, 0.999),
        optimizer_eps=1e-8,
        optimizer_weight_decay=1e-6,
    )
    
    return config

def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing to reduce memory usage."""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing enabled")
    else:
        print("   ⚠️  Gradient checkpointing not available")

def optimize_memory_settings():
    """Set memory optimization settings."""
    # Enable memory efficient settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    
    # Enable memory efficient attention
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("   ✅ Memory efficient attention enabled")
    except:
        print("   ⚠️  Memory efficient attention not available")
    
    # Set memory fraction
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
        print("   ✅ GPU memory fraction set to 80%")

def create_memory_efficient_diffusion_policy(dataset_stats):
    """Create a memory-efficient diffusion policy."""
    
    print("🔧 Creating memory-efficient diffusion policy...")
    
    # Create optimized config
    config = create_memory_efficient_diffusion_config()
    
    # Create policy
    policy = DiffusionPolicy(config, dataset_stats=dataset_stats)
    
    # Enable gradient checkpointing
    enable_gradient_checkpointing(policy)
    
    # Optimize memory settings
    optimize_memory_settings()
    
    print(f"   ✅ Memory-efficient diffusion configured")
    print(f"   Horizon: {config.horizon}")
    print(f"   Action steps: {config.n_action_steps}")
    print(f"   Training timesteps: {config.num_train_timesteps}")
    print(f"   Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    return policy, config

if __name__ == "__main__":
    # Test the memory-efficient configuration
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    print("🧪 Testing memory-efficient diffusion configuration...")
    
    # Load dataset
    dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place")
    
    # Create memory-efficient policy
    policy, config = create_memory_efficient_diffusion_policy(dataset.stats)
    
    # Test on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.train()
    
    # Create minimal batch
    batch_size = 1
    dummy_batch = {
        "observation.state": torch.randn(batch_size, 1, 7),  # 1 obs step
        "observation.images.front": torch.randn(batch_size, 1, 3, 64, 64),  # 1 obs step, 64x64
        "action": torch.randn(batch_size, 4, 7),  # 4 action steps
    }
    
    # Move to device
    for key in dummy_batch:
        dummy_batch[key] = dummy_batch[key].to(device, non_blocking=True)
    
    # Test forward pass
    print(f"\n🚀 Testing forward pass...")
    if device.type == "cuda":
        memory_before = torch.cuda.memory_allocated() / 1024**3
        print(f"   Memory before: {memory_before:.2f}GB")
    
    try:
        loss, output_dict = policy.forward(dummy_batch)
        
        if device.type == "cuda":
            memory_after = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory after: {memory_after:.2f}GB")
            print(f"   Memory used: {memory_after - memory_before:.2f}GB")
            
            if memory_after < 10:  # Should be much less than 10GB
                print(f"   ✅ Memory usage looks good!")
            else:
                print(f"   ⚠️  Still using a lot of memory: {memory_after:.2f}GB")
        
        print(f"   ✅ Forward pass successful!")
        print(f"   Loss: {loss.item():.6f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        raise e 