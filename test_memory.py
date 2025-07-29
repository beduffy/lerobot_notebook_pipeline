#!/usr/bin/env python3
"""
Test GPU memory usage with different batch sizes for diffusion model.
"""

import torch
import argparse
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset

def test_memory_usage(batch_size=8, model_type="diffusion"):
    """Test GPU memory usage with given batch size."""
    
    print(f"🧪 Testing {model_type.upper()} model with batch size {batch_size}")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        # Clear memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Initial GPU Memory: {initial_memory:.2f}GB / {total_memory:.2f}GB")
    
    try:
        # Load dataset
        print("\n📊 Loading dataset...")
        dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place")
        
        # Create small subset for testing
        test_indices = list(range(min(100, len(dataset))))
        test_dataset = Subset(dataset, test_indices)
        
        # Create dataloader
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Setup model with proper configuration
        print(f"\n🧠 Setting up {model_type.upper()} model...")
        if model_type == "diffusion":
            # Configure diffusion model with proper input features
            config = DiffusionConfig()
            config.input_features = ["observation.state", "observation.images.front"]
            config.output_features = ["action"]
            policy = DiffusionPolicy(config)
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
        policy.to(device)
        policy.train()
        
        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"   Parameters: {total_params:,}")
        
        # Test forward pass
        print(f"\n🚀 Testing forward pass with batch size {batch_size}...")
        batch = next(iter(dataloader))
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        # Measure memory before forward pass
        if device.type == "cuda":
            memory_before = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory before forward pass: {memory_before:.2f}GB")
        
        # Forward pass
        with torch.no_grad():
            loss, output_dict = policy.forward(batch)
        
        # Measure memory after forward pass
        if device.type == "cuda":
            memory_after = torch.cuda.memory_allocated() / 1024**3
            memory_used = memory_after - memory_before
            memory_free = total_memory - memory_after
            print(f"   Memory after forward pass: {memory_after:.2f}GB")
            print(f"   Memory used by forward pass: {memory_used:.2f}GB")
            print(f"   Memory free: {memory_free:.2f}GB")
            print(f"   Memory utilization: {memory_after/total_memory*100:.1f}%")
            
            if memory_free < 1.0:  # Less than 1GB free
                print(f"   ⚠️  WARNING: Low memory available ({memory_free:.2f}GB free)")
            else:
                print(f"   ✅ Memory usage looks good")
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Loss: {loss.item():.6f}")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ CUDA Out of Memory Error!")
        if device.type == "cuda":
            print(f"   GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            print(f"   Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
        print(f"   Try reducing batch size from {batch_size}")
        raise e
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Test GPU memory usage")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size to test")
    parser.add_argument("--model", choices=["diffusion"], default="diffusion", help="Model type")
    
    args = parser.parse_args()
    
    test_memory_usage(args.batch_size, args.model)

if __name__ == "__main__":
    main() 