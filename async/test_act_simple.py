#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple ACT Test with Real Dataset Observations

Test ACT async inference using real observations from the dataset.

Usage:
    python3 async/test_act_simple.py
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

# Add async folder to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from async_inference_server import AsyncInferenceEngine
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ASYNC_AVAILABLE = True
except ImportError as e:
    print(f"Async inference not available: {e}")
    ASYNC_AVAILABLE = False


def test_act_with_real_data():
    """Test ACT model with real dataset observations."""
    print("Testing ACT with real dataset observations...")
    
    if not ASYNC_AVAILABLE:
        print("Async inference not available")
        return False
    
    try:
        # Load dataset and get real observations
        print("Loading dataset...")
        dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place", video_backend="pyav")
        
        # Get sample observation
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        
        # Extract observation data
        obs_dict = {}
        for key, value in sample.items():
            if key.startswith("observation.") and isinstance(value, torch.Tensor):
                obs_dict[key] = value.unsqueeze(0)  # Add batch dimension
                print(f"  {key}: {value.shape} -> {obs_dict[key].shape}")
        
        print(f"Observation keys: {list(obs_dict.keys())}")
        
        # Create inference engine with ACT
        print("Creating ACT inference engine...")
        engine = AsyncInferenceEngine(model_type="act")
        
        # Submit inference request
        print("Submitting inference request...")
        request_id = engine.submit_request(
            observations=obs_dict,
            task_description=None  # ACT doesn't need task description
        )
        
        print(f"Request ID: {request_id}")
        
        # Wait for response
        print("Waiting for response...")
        response = engine.get_response(request_id, timeout=30.0)
        
        if response and response.status == "success":
            print("ACT inference successful!")
            print(f"Action shape: {response.actions.shape}")
            print(f"Inference time: {response.inference_time:.3f}s")
            print(f"Model type: {response.model_type}")
            return True
        else:
            print(f"ACT inference failed: {response.error if response else 'Timeout'}")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'engine' in locals():
            engine.shutdown()


def main():
    """Main test function."""
    print("ACT Async Inference Test with Real Data")
    print("=" * 50)
    print("Testing ACT with real dataset observations...")
    print()
    
    # Test ACT with real data
    success = test_act_with_real_data()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"ACT inference: {'PASSED' if success else 'FAILED'}")
    
    if success:
        print("\nACT async inference is working!")
        print("\nNext steps:")
        print("1. Start server with ACT:")
        print("   python3 async/async_inference_server.py --model act --port 8000")
        print()
        print("2. Test with robot:")
        print("   python3 async/async_inference_client.py --server http://localhost:8000 --robot-control")
    else:
        print("\nTest failed. Check the output above for details.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 