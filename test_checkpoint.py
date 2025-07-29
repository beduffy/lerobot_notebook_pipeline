#!/usr/bin/env python3
"""Test checkpoint loading."""

import torch
import sys

try:
    checkpoint_path = "models/smolvla_episodes_0_10_40000steps/checkpoint_step_1000.pt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Successfully loaded checkpoint!")
    print("Keys:", list(checkpoint.keys()))
    
    if 'model_type' in checkpoint:
        print("Model type:", checkpoint['model_type'])
    
    if 'step' in checkpoint:
        print("Step:", checkpoint['step'])
    
    if 'final_loss' in checkpoint:
        print("Final loss:", checkpoint['final_loss'])
    
    if 'model_state_dict' in checkpoint:
        print("Model state dict keys:", list(checkpoint['model_state_dict'].keys())[:5])
    
    print("Checkpoint loaded successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 