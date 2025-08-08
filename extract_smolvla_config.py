#!/usr/bin/env python3
"""
Extract SmolVLA configuration from checkpoint file and create proper model directory structure.
"""

import torch
import json
import shutil
from pathlib import Path


def extract_smolvla_config(checkpoint_path, output_dir):
    """Extract SmolVLA configuration from checkpoint and create model directory."""
    
    print(f"📁 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract and save config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"📝 Found config type: {type(config)}")
        
        # Convert config to dict format
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__.copy()
        elif isinstance(config, str):
            # Parse the string representation
            config_dict = {}
            # Extract key-value pairs from the string
            config_str = config.replace("SmolVLAConfig(", "").replace(")", "")
            for item in config_str.split(", "):
                if "=" in item:
                    key, value = item.split("=", 1)
                    config_dict[key] = value
        else:
            config_dict = {}
        
        # Add the required 'type' field for SmolVLA
        config_dict['type'] = 'smolvla'
        
        # Save config as JSON
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"💾 Saved config to: {config_path}")
    
    # Save training info
    training_info = {
        'model_type': checkpoint.get('model_type', 'smolvla'),
        'step': checkpoint.get('step', 0),
        'final_loss': checkpoint.get('final_loss', None),
        'training_time': checkpoint.get('training_time', None),
        'episodes_info': checkpoint.get('episodes_info', {})
    }
    
    training_info_path = output_path / "training_info.json"
    with open(training_info_path, 'w') as f:
        json.dump(training_info, f, indent=2, default=str)
    print(f"💾 Saved training info to: {training_info_path}")
    
    # Copy checkpoint file
    checkpoint_name = Path(checkpoint_path).name
    new_checkpoint_path = output_path / checkpoint_name
    shutil.copy2(checkpoint_path, new_checkpoint_path)
    print(f"💾 Copied checkpoint to: {new_checkpoint_path}")
    
    print(f"✅ Model directory created at: {output_path}")
    return output_path


if __name__ == "__main__":
    checkpoint_path = "models/smolvla_episodes_0_10_40000steps/checkpoint_step_1000.pt"
    output_dir = "models/smolvla_episodes_0_10_40000steps/model"
    
    extract_smolvla_config(checkpoint_path, output_dir) 