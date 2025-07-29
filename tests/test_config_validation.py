#!/usr/bin/env python3
"""
Tests for configuration validation and error handling - ensuring robustness.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
import torch

def test_act_config_validation():
    """Test ACT configuration validation."""
    from lerobot.policies.act.configuration_act import ACTConfig
    
    # Test default config creation
    config = ACTConfig()
    assert hasattr(config, 'n_obs_steps')
    assert hasattr(config, 'n_action_steps')
    assert hasattr(config, 'chunk_size')
    print("✅ ACT config default creation works")
    
    # Test config modification
    config.n_obs_steps = 5
    config.n_action_steps = 10
    assert config.n_obs_steps == 5
    assert config.n_action_steps == 10
    print("✅ ACT config modification works")

def test_invalid_dataset_handling():
    """Test handling of invalid dataset configurations."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    # Test with non-existent dataset
    with pytest.raises((ValueError, FileNotFoundError, Exception)):
        dataset = LeRobotDataset("invalid/nonexistent_dataset")
    
    print("✅ Invalid dataset handling works")

def test_model_input_validation():
    """Test model input shape validation."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    import torch
    
    config = ACTConfig()
    config.input_shapes = {
        "observation.state": [6],
        "observation.images.front": [3, 96, 96]
    }
    config.output_shapes = {"action": [6]}
    
    policy = ACTPolicy(config)
    
    # Test with correct input shapes
    correct_batch = {
        "observation.state": torch.randn(1, 1, 6),
        "observation.images.front": torch.randn(1, 1, 3, 96, 96),
        "action": torch.randn(1, 1, 6)
    }
    
    # This should work
    with torch.no_grad():
        loss, _ = policy(correct_batch)
        assert isinstance(loss, torch.Tensor)
    
    print("✅ Model input validation works")

def test_file_path_validation():
    """Test file path validation and error handling."""
    from pathlib import Path
    
    # Test non-existent file handling
    non_existent_file = Path("this_file_does_not_exist.json")
    assert not non_existent_file.exists()
    
    # Test directory creation
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = Path(tmp_dir) / "nested" / "directory"
        test_dir.mkdir(parents=True, exist_ok=True)
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    print("✅ File path validation works")

def test_json_config_validation():
    """Test JSON configuration file validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test valid JSON config
        valid_config = {
            "model_type": "act",
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 100
            },
            "dataset": {
                "name": "test_dataset",
                "train_split": 0.8
            }
        }
        
        config_file = Path(tmp_dir) / "config.json"
        with open(config_file, 'w') as f:
            json.dump(valid_config, f)
        
        # Test loading valid config
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["model_type"] == "act"
        assert loaded_config["training"]["batch_size"] == 32
        
        # Test invalid JSON
        invalid_json_file = Path(tmp_dir) / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write('{"invalid": json, syntax}')
        
        with pytest.raises(json.JSONDecodeError):
            with open(invalid_json_file, 'r') as f:
                json.load(f)
        
        print("✅ JSON config validation works")

def test_device_configuration():
    """Test device configuration and fallback."""
    import torch
    
    # Test device selection logic
    def get_device(prefer_cuda=True):
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    device = get_device()
    assert device.type in ["cuda", "mps", "cpu"]
    
    # Test tensor operations on selected device
    test_tensor = torch.randn(2, 3).to(device)
    assert test_tensor.device.type == device.type
    
    print(f"✅ Device configuration works: {device}")

def test_training_parameter_validation():
    """Test training parameter validation."""
    def validate_training_params(params):
        """Validate training parameters."""
        errors = []
        
        if params.get("batch_size", 0) <= 0:
            errors.append("batch_size must be positive")
        
        if params.get("learning_rate", 0) <= 0:
            errors.append("learning_rate must be positive")
        
        if params.get("num_epochs", 0) <= 0:
            errors.append("num_epochs must be positive")
        
        if params.get("training_steps", 0) <= 0:
            errors.append("training_steps must be positive")
        
        return errors
    
    # Test valid parameters
    valid_params = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "training_steps": 10000
    }
    errors = validate_training_params(valid_params)
    assert len(errors) == 0
    
    # Test invalid parameters
    invalid_params = {
        "batch_size": -1,
        "learning_rate": 0,
        "num_epochs": 0,
        "training_steps": -100
    }
    errors = validate_training_params(invalid_params)
    assert len(errors) > 0
    assert "batch_size must be positive" in errors
    
    print("✅ Training parameter validation works")

def test_model_checkpoint_validation():
    """Test model checkpoint validation."""
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create and save a model
        config = ACTConfig()
        config.input_shapes = {
            "observation.state": [6],
            "observation.images.front": [3, 96, 96]
        }
        config.output_shapes = {"action": [6]}
        
        policy = ACTPolicy(config)
        
        checkpoint_file = Path(tmp_dir) / "model.pt"
        torch.save({
            'model_state_dict': policy.state_dict(),
            'config': config.__dict__,
            'training_step': 1000,
            'loss': 0.5
        }, checkpoint_file)
        
        # Test loading checkpoint
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'config']
        for key in required_keys:
            assert key in checkpoint, f"Checkpoint missing key: {key}"
        
        # Test loading model from checkpoint
        loaded_policy = ACTPolicy(config)
        loaded_policy.load_state_dict(checkpoint['model_state_dict'])
        
        print("✅ Model checkpoint validation works")

def test_dataset_format_validation():
    """Test dataset format validation."""
    import torch
    import numpy as np
    
    def validate_episode_format(episode_data):
        """Validate episode data format."""
        required_keys = ["action", "observation.state", "observation.images.front"]
        errors = []
        
        for key in required_keys:
            if key not in episode_data:
                errors.append(f"Missing required key: {key}")
        
        # Validate shapes
        if "action" in episode_data:
            action = episode_data["action"]
            if not isinstance(action, (torch.Tensor, np.ndarray)):
                errors.append("action must be tensor or array")
            elif len(action.shape) != 1:
                errors.append("action must be 1D")
        
        if "observation.state" in episode_data:
            state = episode_data["observation.state"]
            if not isinstance(state, (torch.Tensor, np.ndarray)):
                errors.append("observation.state must be tensor or array")
        
        return errors
    
    # Test valid episode
    valid_episode = {
        "action": torch.randn(6),
        "observation.state": torch.randn(6),
        "observation.images.front": torch.randint(0, 255, (3, 96, 96), dtype=torch.uint8)
    }
    errors = validate_episode_format(valid_episode)
    assert len(errors) == 0
    
    # Test invalid episode
    invalid_episode = {
        "action": torch.randn(6, 6),  # Wrong shape
        "observation.state": "not_a_tensor"  # Wrong type
        # Missing observation.images.front
    }
    errors = validate_episode_format(invalid_episode)
    assert len(errors) > 0
    
    print("✅ Dataset format validation works")

def test_memory_usage_validation():
    """Test memory usage validation."""
    import psutil
    import torch
    
    def check_memory_usage():
        """Check current memory usage."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / (1024**3),  # GB
            "available": memory.available / (1024**3),  # GB
            "used": memory.used / (1024**3),  # GB
            "percent": memory.percent
        }
    
    def estimate_model_memory(config):
        """Estimate model memory usage."""
        # Rough estimation based on parameter count
        param_count = 0
        
        # This is a very rough estimate
        if hasattr(config, 'input_shapes') and hasattr(config, 'output_shapes'):
            input_size = sum(np.prod(shape) for shape in config.input_shapes.values())
            output_size = sum(np.prod(shape) for shape in config.output_shapes.values())
            param_count = input_size * output_size * 4  # Rough estimate
        
        # 4 bytes per float32 parameter, convert to GB
        memory_gb = (param_count * 4) / (1024**3)
        return memory_gb
    
    memory_info = check_memory_usage()
    assert memory_info["available"] > 0
    
    # Test with ACT config
    from lerobot.policies.act.configuration_act import ACTConfig
    config = ACTConfig()
    config.input_shapes = {"obs": [6]}
    config.output_shapes = {"action": [6]}
    
    estimated_memory = estimate_model_memory(config)
    assert estimated_memory >= 0
    
    print(f"✅ Memory validation works (available: {memory_info['available']:.1f}GB)")

def test_hyperparameter_ranges():
    """Test hyperparameter range validation."""
    def validate_hyperparameters(params):
        """Validate hyperparameter ranges."""
        errors = []
        
        # Learning rate validation
        lr = params.get("learning_rate", 1e-4)
        if not (1e-6 <= lr <= 1e-1):
            errors.append(f"learning_rate {lr} outside recommended range [1e-6, 1e-1]")
        
        # Batch size validation
        batch_size = params.get("batch_size", 32)
        if not (1 <= batch_size <= 1024):
            errors.append(f"batch_size {batch_size} outside recommended range [1, 1024]")
        
        # Chunk size validation
        chunk_size = params.get("chunk_size", 10)
        if not (1 <= chunk_size <= 100):
            errors.append(f"chunk_size {chunk_size} outside recommended range [1, 100]")
        
        return errors
    
    # Test valid hyperparameters
    valid_params = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "chunk_size": 10
    }
    errors = validate_hyperparameters(valid_params)
    assert len(errors) == 0
    
    # Test invalid hyperparameters
    invalid_params = {
        "learning_rate": 1.0,  # Too high
        "batch_size": 2000,    # Too high
        "chunk_size": 0        # Too low
    }
    errors = validate_hyperparameters(invalid_params)
    assert len(errors) > 0
    
    print("✅ Hyperparameter range validation works")

def test_error_logging():
    """Test error logging and reporting."""
    import logging
    import sys
    from io import StringIO
    
    # Create a string buffer to capture log output
    log_buffer = StringIO()
    
    # Set up logging to capture errors
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.ERROR)
    handler = logging.StreamHandler(log_buffer)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except ValueError as e:
        logger.error(f"Caught error: {e}")
    
    # Check that error was logged
    log_output = log_buffer.getvalue()
    assert "ERROR:" in log_output
    assert "Test error for logging" in log_output
    
    print("✅ Error logging works")

def test_graceful_degradation():
    """Test graceful degradation when features are unavailable."""
    import torch
    
    # Test CUDA fallback
    def get_device_with_fallback():
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                # Test that device actually works
                test_tensor = torch.randn(2, 2).to(device)
                return device
            except Exception:
                pass
        
        return torch.device("cpu")
    
    device = get_device_with_fallback()
    assert device.type in ["cuda", "cpu"]
    
    # Test optional import fallback
    def import_with_fallback():
        try:
            import some_optional_package  # This will fail
            return True
        except ImportError:
            return False
    
    # Should return False gracefully
    assert not import_with_fallback()
    
    print("✅ Graceful degradation works")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 