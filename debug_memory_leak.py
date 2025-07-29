#!/usr/bin/env python3
"""
Minimal test to debug the memory leak in diffusion models.
"""

import torch
import gc
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset

def test_minimal_diffusion():
    """Test diffusion model with minimal setup to find memory leak."""
    
    print("🔍 MINIMAL DIFFUSION MEMORY TEST")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Initial GPU Memory: {initial_memory:.2f}GB / {total_memory:.2f}GB")
    
    try:
        # Step 1: Load dataset
        print(f"\n📊 Step 1: Loading dataset...")
        dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place")
        
        if device.type == "cuda":
            memory_after_dataset = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory after dataset: {memory_after_dataset:.2f}GB")
        
        # Step 2: Create model with minimal config
        print(f"\n🧠 Step 2: Creating diffusion model...")
        config = DiffusionConfig(
            input_features=["observation.state", "observation.images.front"],
            output_features=["action"],
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            vision_backbone="resnet18",
            crop_shape=(84, 84),
            num_train_timesteps=100,
        )
        
        policy = DiffusionPolicy(config, dataset_stats=dataset.stats)
        
        if device.type == "cuda":
            memory_after_model = torch.cuda.memory_allocated() / 1024**3
            model_memory = memory_after_model - memory_after_dataset
            print(f"   Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
            print(f"   Model memory: {model_memory:.2f}GB")
        
        # Step 3: Move to device
        print(f"\n🚀 Step 3: Moving model to GPU...")
        policy.to(device)
        policy.train()
        
        if device.type == "cuda":
            memory_after_to_device = torch.cuda.memory_allocated() / 1024**3
            device_memory = memory_after_to_device - memory_after_model
            print(f"   Memory after to_device: {memory_after_to_device:.2f}GB")
            print(f"   Device transfer overhead: {device_memory:.2f}GB")
        
        # Step 4: Create tiny batch
        print(f"\n📦 Step 4: Creating tiny batch...")
        # Create minimal batch manually to avoid dataset issues
        batch_size = 1  # Start with batch size 1!
        
        # Create dummy data with correct shapes
        dummy_batch = {
            "observation.state": torch.randn(batch_size, 2, 7),  # 2 obs steps, 7 state dims
            "observation.images.front": torch.randn(batch_size, 2, 3, 84, 84),  # 2 obs steps, RGB, 84x84
            "action": torch.randn(batch_size, 8, 7),  # 8 action steps, 7 action dims
        }
        
        if device.type == "cuda":
            memory_after_batch = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory after batch creation: {memory_after_batch:.2f}GB")
        
        # Step 5: Move batch to device
        print(f"\n🔄 Step 5: Moving batch to GPU...")
        for key in dummy_batch:
            dummy_batch[key] = dummy_batch[key].to(device, non_blocking=True)
        
        if device.type == "cuda":
            memory_after_batch_to_device = torch.cuda.memory_allocated() / 1024**3
            batch_device_memory = memory_after_batch_to_device - memory_after_batch
            print(f"   Memory after batch to device: {memory_after_batch_to_device:.2f}GB")
            print(f"   Batch device transfer: {batch_device_memory:.2f}GB")
        
        # Step 6: Forward pass
        print(f"\n⚡ Step 6: Forward pass with batch size 1...")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            memory_before_forward = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory before forward: {memory_before_forward:.2f}GB")
        
        # Forward pass
        with torch.no_grad():  # Use no_grad to avoid storing gradients
            loss, output_dict = policy.forward(dummy_batch)
        
        if device.type == "cuda":
            memory_after_forward = torch.cuda.memory_allocated() / 1024**3
            forward_memory = memory_after_forward - memory_before_forward
            print(f"   Memory after forward: {memory_after_forward:.2f}GB")
            print(f"   Forward pass memory: {forward_memory:.2f}GB")
            print(f"   Peak memory during forward: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
        
        # Step 7: Test with gradients
        print(f"\n🔄 Step 7: Forward pass WITH gradients...")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            memory_before_grad = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory before grad forward: {memory_before_grad:.2f}GB")
        
        # Forward pass with gradients
        loss, output_dict = policy.forward(dummy_batch)
        
        if device.type == "cuda":
            memory_after_grad_forward = torch.cuda.memory_allocated() / 1024**3
            grad_forward_memory = memory_after_grad_forward - memory_before_grad
            print(f"   Memory after grad forward: {memory_after_grad_forward:.2f}GB")
            print(f"   Grad forward memory: {grad_forward_memory:.2f}GB")
        
        # Step 8: Backward pass
        print(f"\n🔄 Step 8: Backward pass...")
        
        if device.type == "cuda":
            memory_before_backward = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory before backward: {memory_before_backward:.2f}GB")
        
        loss.backward()
        
        if device.type == "cuda":
            memory_after_backward = torch.cuda.memory_allocated() / 1024**3
            backward_memory = memory_after_backward - memory_before_backward
            print(f"   Memory after backward: {memory_after_backward:.2f}GB")
            print(f"   Backward pass memory: {backward_memory:.2f}GB")
        
        # Step 9: Optimizer step
        print(f"\n🎯 Step 9: Optimizer step...")
        
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        
        if device.type == "cuda":
            memory_before_optimizer = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory before optimizer: {memory_before_optimizer:.2f}GB")
        
        optimizer.step()
        optimizer.zero_grad()
        
        if device.type == "cuda":
            memory_after_optimizer = torch.cuda.memory_allocated() / 1024**3
            optimizer_memory = memory_after_optimizer - memory_before_optimizer
            print(f"   Memory after optimizer: {memory_after_optimizer:.2f}GB")
            print(f"   Optimizer memory: {optimizer_memory:.2f}GB")
        
        # Final summary
        print(f"\n📊 FINAL MEMORY BREAKDOWN")
        print("=" * 40)
        
        if device.type == "cuda":
            print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
            print(f"Parameter memory: ~{sum(p.numel() for p in policy.parameters()) * 4 / 1024**3:.2f}GB")
            print(f"Total GPU memory used: {memory_after_optimizer:.2f}GB")
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            print(f"Memory efficiency: {sum(p.numel() for p in policy.parameters()) * 4 / 1024**3 / memory_after_optimizer * 100:.1f}%")
            
            if memory_after_optimizer > 20:  # More than 20GB for a 263M parameter model
                print(f"\n🚨 ALARM: This is WAY too much memory!")
                print(f"   Expected: ~5-10GB for 263M parameters")
                print(f"   Actual: {memory_after_optimizer:.2f}GB")
                print(f"   This indicates a memory leak!")
        
        print(f"\n✅ Test completed!")
        print(f"   Loss: {loss.item():.6f}")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ CUDA Out of Memory Error!")
        if device.type == "cuda":
            print(f"   GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            print(f"   Peak Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        raise e
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise e

if __name__ == "__main__":
    test_minimal_diffusion() 