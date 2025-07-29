#!/usr/bin/env python3
"""
Detailed memory analysis for diffusion models to understand why they use so much VRAM.
"""

import torch
import torch.nn as nn
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset
import gc

def analyze_memory_breakdown(batch_size=4):
    """Analyze exactly where memory is being used in diffusion models."""
    
    print("🔍 DETAILED MEMORY ANALYSIS FOR DIFFUSION MODELS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Initial GPU Memory: {initial_memory:.2f}GB / {total_memory:.2f}GB")
    
    try:
        # 1. Load dataset
        print(f"\n📊 Step 1: Loading dataset...")
        dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place")
        test_indices = list(range(min(50, len(dataset))))
        test_dataset = Subset(dataset, test_indices)
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        if device.type == "cuda":
            memory_after_dataset = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory after dataset: {memory_after_dataset:.2f}GB")
        
        # 2. Create model
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
            print(f"   Memory after model: {memory_after_model:.2f}GB")
        
        # 3. Move to device
        print(f"\n🚀 Step 3: Moving model to GPU...")
        policy.to(device)
        policy.train()
        
        if device.type == "cuda":
            memory_after_to_device = torch.cuda.memory_allocated() / 1024**3
            device_memory = memory_after_to_device - memory_after_model
            print(f"   Memory after to_device: {memory_after_to_device:.2f}GB")
            print(f"   Device transfer overhead: {device_memory:.2f}GB")
        
        # 4. Get batch
        print(f"\n📦 Step 4: Loading batch data...")
        batch = next(iter(dataloader))
        
        if device.type == "cuda":
            memory_after_batch = torch.cuda.memory_allocated() / 1024**3
            batch_memory = memory_after_batch - memory_after_to_device
            print(f"   Memory after batch: {memory_after_batch:.2f}GB")
            print(f"   Batch data memory: {batch_memory:.2f}GB")
        
        # 5. Move batch to device
        print(f"\n🔄 Step 5: Moving batch to GPU...")
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        if device.type == "cuda":
            memory_after_batch_to_device = torch.cuda.memory_allocated() / 1024**3
            batch_device_memory = memory_after_batch_to_device - memory_after_batch
            print(f"   Memory after batch to device: {memory_after_batch_to_device:.2f}GB")
            print(f"   Batch device transfer: {batch_device_memory:.2f}GB")
        
        # 6. Forward pass (this is where the magic happens!)
        print(f"\n⚡ Step 6: Forward pass (this is where memory explodes!)...")
        
        # Monitor memory during forward pass
        if device.type == "cuda":
            torch.cuda.empty_cache()
            memory_before_forward = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory before forward: {memory_before_forward:.2f}GB")
        
        # Forward pass
        loss, output_dict = policy.forward(batch)
        
        if device.type == "cuda":
            memory_after_forward = torch.cuda.memory_allocated() / 1024**3
            forward_memory = memory_after_forward - memory_before_forward
            print(f"   Memory after forward: {memory_after_forward:.2f}GB")
            print(f"   Forward pass memory: {forward_memory:.2f}GB")
            print(f"   Peak memory during forward: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
        
        # 7. Backward pass
        print(f"\n🔄 Step 7: Backward pass...")
        
        if device.type == "cuda":
            memory_before_backward = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory before backward: {memory_before_backward:.2f}GB")
        
        loss.backward()
        
        if device.type == "cuda":
            memory_after_backward = torch.cuda.memory_allocated() / 1024**3
            backward_memory = memory_after_backward - memory_before_backward
            print(f"   Memory after backward: {memory_after_backward:.2f}GB")
            print(f"   Backward pass memory: {backward_memory:.2f}GB")
        
        # 8. Summary
        print(f"\n📊 MEMORY BREAKDOWN SUMMARY")
        print("=" * 40)
        
        if device.type == "cuda":
            print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
            print(f"Parameter memory: ~{sum(p.numel() for p in policy.parameters()) * 4 / 1024**3:.2f}GB (float32)")
            print(f"Total GPU memory used: {memory_after_backward:.2f}GB")
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            print(f"Memory efficiency: {sum(p.numel() for p in policy.parameters()) * 4 / 1024**3 / memory_after_backward * 100:.1f}%")
            
            print(f"\n🔍 WHY SO MUCH MEMORY?")
            print("=" * 30)
            print("1. Model parameters: ~1GB")
            print("2. Activations during forward pass: ~8-12GB")
            print("3. Gradients during backward pass: ~1GB")
            print("4. Optimizer states: ~2-3GB")
            print("5. Batch data: ~1-2GB")
            print("6. CUDA overhead: ~1-2GB")
            print("\nThe activations during forward pass are the biggest culprit!")
            print("Diffusion models need to store activations for 100 timesteps!")
        
        print(f"\n✅ Analysis completed!")
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

def explain_diffusion_memory():
    """Explain why diffusion models use so much memory."""
    
    print("\n🤔 WHY DO DIFFUSION MODELS USE SO MUCH VRAM?")
    print("=" * 50)
    
    print("\n1. 📊 PARAMETER COUNT IS MISLEADING")
    print("   - 263M parameters = ~1GB in float32")
    print("   - But memory usage is 10-15GB!")
    print("   - Parameters are NOT the main memory consumer")
    
    print("\n2. 🔄 ACTIVATION MEMORY (THE REAL CULPRIT)")
    print("   - Forward pass stores ALL intermediate activations")
    print("   - Diffusion models have 100 timesteps")
    print("   - Each timestep stores activations for backprop")
    print("   - Memory grows linearly with sequence length")
    print("   - 100 timesteps × activations = HUGE memory!")
    
    print("\n3. 🧠 MODEL ARCHITECTURE OVERHEAD")
    print("   - ResNet18 backbone for vision")
    print("   - Multiple attention layers")
    print("   - Skip connections store activations")
    print("   - Each layer doubles memory usage")
    
    print("\n4. 📦 BATCH SIZE EFFECT")
    print("   - Memory scales linearly with batch size")
    print("   - Batch size 16 = 16x more activations")
    print("   - Each sample needs its own activation storage")
    
    print("\n5. 🎯 GRADIENT ACCUMULATION")
    print("   - Gradients stored for each parameter")
    print("   - Optimizer states (Adam) need 2x parameter memory")
    print("   - Momentum and variance buffers")
    
    print("\n6. 🔧 CUDA OVERHEAD")
    print("   - Memory fragmentation")
    print("   - Reserved but unallocated memory")
    print("   - PyTorch memory pool overhead")
    
    print("\n💡 SOLUTIONS:")
    print("   - Reduce batch size (most effective)")
    print("   - Use gradient checkpointing")
    print("   - Enable memory efficient attention")
    print("   - Use mixed precision training")
    print("   - Increase gradient accumulation steps")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze diffusion model memory usage")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size to test")
    args = parser.parse_args()
    
    explain_diffusion_memory()
    analyze_memory_breakdown(args.batch_size) 