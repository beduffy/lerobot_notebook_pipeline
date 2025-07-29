#!/usr/bin/env python3
"""Check checkpoint contents."""

import torch
import json

checkpoint_path = "models/smolvla_episodes_0_10_40000steps/checkpoint_step_1000.pt"

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint keys:", list(checkpoint.keys()))
    
    if 'model_type' in checkpoint:
        print("Model type:", checkpoint['model_type'])
    
    if 'config' in checkpoint:
        print("Config keys:", list(checkpoint['config'].keys()))
    
    if 'model_state_dict' in checkpoint:
        print("Model state dict keys:", list(checkpoint['model_state_dict'].keys())[:10])
    
    print("Step:", checkpoint.get('step', 'Not found'))
    print("Final loss:", checkpoint.get('final_loss', 'Not found'))
    
except Exception as e:
    print(f"Error loading checkpoint: {e}") 