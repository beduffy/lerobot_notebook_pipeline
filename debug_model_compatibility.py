#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Model Compatibility Issues

This script helps diagnose why the trained model is failing with tensor dimension errors.
"""

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
import json


def debug_model_compatibility():
    """Debug the model compatibility issue."""
    
    print("🔍 Model Compatibility Debugger")
    print("=" * 40)
    
    # Load model
    print("1. Loading model...")
    try:
        policy = ACTPolicy.from_pretrained("./single_episode_model")
        print("✅ Model loaded successfully")
        # Check device by looking at model parameters
        device = next(policy.parameters()).device
        print(f"   Device: {device}")
        print(f"   Config keys: {list(policy.config.__dict__.keys())}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Load dataset
    print("\n2. Loading dataset...")
    try:
        dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place", video_backend="pyav")
        print("✅ Dataset loaded successfully")
        print(f"   Length: {len(dataset)} samples")
        print(f"   Episodes: {len(dataset.meta.episodes)}")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # Analyze sample structure
    print("\n3. Analyzing sample structure...")
    sample = dataset[0]
    print("Sample keys and shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {type(value)} = {value}")
    
    # Check model config vs sample
    print("\n4. Model configuration:")
    config_dict = policy.config.__dict__
    print(f"   Input features: {config_dict.get('input_features', 'Not found')}")
    print(f"   Output features: {config_dict.get('output_features', 'Not found')}")
    print(f"   n_obs_steps: {config_dict.get('n_obs_steps', 'Not found')}")
    print(f"   chunk_size: {config_dict.get('chunk_size', 'Not found')}")
    
    # Try minimal prediction
    print("\n5. Testing minimal prediction...")
    try:
        # Prepare batch - ONLY PASS OBSERVATION KEYS, NOT ACTION OR METADATA!
        batch = {}
        for key, value in sample.items():
            # Only include observation keys for inference
            if key.startswith("observation.") and isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)  # Add batch dimension
        
        print("   Batch prepared (observations only):")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
        
        # Try prediction
        with torch.no_grad():
            prediction = policy.select_action(batch)
            print(f"✅ Prediction successful! Shape: {prediction.shape}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("\n   Detailed error:")
        import traceback
        traceback.print_exc()
        
        # Additional debugging
        print("\n   Trying to identify the issue...")
        
        # Check if this is a VAE encoder issue
        if "vae_encoder_input" in str(e):
            print("   🔍 Issue is in VAE encoder - checking observation processing...")
            
            # Check observation keys
            obs_keys = [k for k in sample.keys() if k.startswith('observation.')]
            print(f"   Observation keys: {obs_keys}")
            
            for key in obs_keys:
                if isinstance(sample[key], torch.Tensor):
                    print(f"     {key}: shape={sample[key].shape}, dtype={sample[key].dtype}")
    
    # Check training info
    print("\n6. Training information:")
    try:
        with open("./single_episode_model/training_info.json", 'r') as f:
            training_info = json.load(f)
            print(f"   Training info: {training_info}")
    except Exception as e:
        print(f"   No training info available: {e}")
    
    print("\n💡 Recommendations:")
    print("   1. Check if lerobot version matches training environment")
    print("   2. Verify model was trained on same dataset format")
    print("   3. Consider retraining with current lerobot version")
    print("   4. Try loading model with different device (CPU vs GPU)")


if __name__ == "__main__":
    debug_model_compatibility() 