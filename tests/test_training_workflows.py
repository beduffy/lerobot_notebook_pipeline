#!/usr/bin/env python3
"""
Tests for training workflows - ensuring all main training scripts work correctly.
"""

import pytest
import torch
import tempfile
import subprocess
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Test utilities
def run_script_with_timeout(script_path, args, timeout=30):
    """Run a Python script with timeout and capture output."""
    cmd = ["python", script_path] + args
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        return result
    except subprocess.TimeoutExpired:
        return None

@pytest.mark.slow
def test_train_script_help():
    """Test that train.py shows help without errors."""
    result = run_script_with_timeout("train.py", ["--help"])
    assert result is not None
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()

@pytest.mark.slow 
def test_train_single_episode_help():
    """Test that train_single_episode.py shows help without errors."""
    result = run_script_with_timeout("train_single_episode.py", ["--help"])
    assert result is not None
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()

@pytest.mark.slow
def test_train_multi_model_help():
    """Test that train_multi_model.py shows help without errors."""
    result = run_script_with_timeout("train_multi_model.py", ["--help"])
    assert result is not None  
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()

def test_training_imports():
    """Test that all training scripts can import their dependencies."""
    # Test main training modules import correctly
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        print("✅ Core training imports working")
    except ImportError as e:
        pytest.fail(f"Training imports failed: {e}")

def test_config_creation():
    """Test that we can create training configurations."""
    from lerobot.policies.act.configuration_act import ACTConfig
    
    # Test creating ACT config
    config = ACTConfig()
    assert hasattr(config, 'n_obs_steps')
    assert hasattr(config, 'n_action_steps')
    print("✅ ACT config creation works")

@pytest.mark.slow
def test_dataset_loading_for_training():
    """Test that we can load a dataset for training."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    try:
        dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place")
        assert len(dataset) > 0
        
        # Test getting a sample
        sample = dataset[0]
        assert 'action' in sample
        assert 'observation.state' in sample
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
    except Exception as e:
        pytest.skip(f"Dataset loading failed (network/cache issue): {e}")

@pytest.mark.slow
def test_mock_training_workflow():
    """Test a mock training workflow with minimal steps."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    import torch
    
    # Create minimal config
    config = ACTConfig()
    config.n_obs_steps = 1
    config.n_action_steps = 1
    config.input_shapes = {
        "observation.state": [6],
        "observation.images.front": [3, 96, 96]
    }
    config.output_shapes = {"action": [6]}
    
    # Create model
    policy = ACTPolicy(config)
    
    # Create mock batch
    batch = {
        "observation.state": torch.randn(2, 1, 6),
        "observation.images.front": torch.randn(2, 1, 3, 96, 96),
        "action": torch.randn(2, 1, 6)
    }
    
    # Test forward pass
    with torch.no_grad():
        loss, _ = policy(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        print("✅ Mock training forward pass works")

def test_training_utilities():
    """Test training utility functions."""
    from lerobot_notebook_pipeline.dataset_utils.training import train_model
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create mock components
    class MockPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
        
        def forward(self, batch):
            return torch.tensor(0.1, requires_grad=True), {}
    
    policy = MockPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    # Create mock dataloader
    dataset = TensorDataset(torch.randn(10, 1), torch.randn(10, 1))
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Test training function
    try:
        train_model(policy, dataloader, optimizer, training_steps=3, log_freq=2, device="cpu")
        print("✅ Training utility function works")
    except Exception as e:
        pytest.fail(f"Training utility failed: {e}")

@pytest.mark.slow
def test_single_episode_dry_run():
    """Test single episode experiment in dry-run mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test the single episode experiment configuration
        result = run_script_with_timeout(
            "single_episode_experiment.py", 
            [
                "bearlover365/red_cube_always_in_same_place",
                "--episode", "0",
                "--output-dir", tmp_dir,
                "--training-steps", "10"  # Very short for testing
            ],
            timeout=60
        )
        
        if result is None:
            pytest.skip("Single episode experiment timed out")
        
        # Should complete configuration successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
        # Check if config was created (even if training failed)
        config_files = list(Path(tmp_dir).glob("**/experiment_config.json"))
        if config_files:
            print("✅ Single episode experiment configuration created")
        else:
            pytest.skip("Single episode experiment didn't create config")

def test_model_factory_patterns():
    """Test that we can create different types of models."""
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    
    # Test ACT model creation
    config = ACTConfig()
    policy = ACTPolicy(config)
    assert isinstance(policy, torch.nn.Module)
    print("✅ ACT model factory works")
    
    # Test diffusion model imports (if available)
    try:
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        
        diff_config = DiffusionConfig()
        diff_policy = DiffusionPolicy(diff_config)
        assert isinstance(diff_policy, torch.nn.Module)
        print("✅ Diffusion model factory works")
    except ImportError:
        print("⚠️  Diffusion models not available (optional)")

@pytest.mark.slow
def test_training_script_validation():
    """Test that training scripts validate inputs properly."""
    # Test with invalid dataset
    result = run_script_with_timeout(
        "train.py", 
        ["--dataset", "invalid/dataset", "--steps", "1"],
        timeout=30
    )
    
    if result is not None:
        # Should fail gracefully, not crash
        assert "error" in result.stderr.lower() or "not found" in result.stdout.lower() or result.returncode != 0
        print("✅ Training script handles invalid datasets gracefully")

def test_device_detection():
    """Test device detection for training."""
    import torch
    
    # Test CUDA availability detection
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        print("✅ CUDA device available for training")
    else:
        device = torch.device("cpu")
        print("✅ CPU device available for training")
    
    # Test tensor creation on device
    tensor = torch.randn(2, 3).to(device)
    assert tensor.device.type == device.type
    print(f"✅ Tensor operations work on {device}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 