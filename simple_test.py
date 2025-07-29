#!/usr/bin/env python3
"""
Very simple test to isolate the memory leak.
"""

import torch
import gc

def test_basic_memory():
    """Test basic memory usage without any models."""
    
    print("🔍 BASIC MEMORY TEST")
    print("=" * 30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Initial GPU Memory: {initial_memory:.2f}GB / {total_memory:.2f}GB")
        
        # Test 1: Simple tensor operations
        print(f"\n📊 Test 1: Simple tensor operations...")
        x = torch.randn(1000, 1000).to(device)
        memory_after_tensor = torch.cuda.memory_allocated() / 1024**3
        print(f"   Memory after tensor: {memory_after_tensor:.2f}GB")
        
        # Test 2: Simple model
        print(f"\n🧠 Test 2: Simple linear model...")
        model = torch.nn.Linear(1000, 1000).to(device)
        memory_after_model = torch.cuda.memory_allocated() / 1024**3
        print(f"   Memory after model: {memory_after_model:.2f}GB")
        
        # Test 3: Forward pass
        print(f"\n⚡ Test 3: Forward pass...")
        output = model(x)
        memory_after_forward = torch.cuda.memory_allocated() / 1024**3
        print(f"   Memory after forward: {memory_after_forward:.2f}GB")
        
        # Test 4: Backward pass
        print(f"\n🔄 Test 4: Backward pass...")
        loss = output.sum()
        loss.backward()
        memory_after_backward = torch.cuda.memory_allocated() / 1024**3
        print(f"   Memory after backward: {memory_after_backward:.2f}GB")
        
        # Test 5: Optimizer
        print(f"\n🎯 Test 5: Optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()
        memory_after_optimizer = torch.cuda.memory_allocated() / 1024**3
        print(f"   Memory after optimizer: {memory_after_optimizer:.2f}GB")
        
        print(f"\n✅ Basic test completed!")
        print(f"   Final memory: {memory_after_optimizer:.2f}GB")

if __name__ == "__main__":
    test_basic_memory() 