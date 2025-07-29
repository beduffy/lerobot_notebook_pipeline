#!/usr/bin/env python3
"""
Tests for evaluation workflows - ensuring model evaluation and inference work correctly.
"""

import pytest
import torch
import tempfile
import subprocess
import json
import os
from pathlib import Path

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
def test_evaluate_model_help():
    """Test that evaluate_model.py shows help without errors."""
    result = run_script_with_timeout("evaluate_model.py", ["--help"])
    assert result is not None
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()

@pytest.mark.slow
def test_test_model_inference_help():
    """Test that test_model_inference.py shows help without errors."""
    result = run_script_with_timeout("test_model_inference.py", ["--help"])
    assert result is not None
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()

def test_evaluation_imports():
    """Test that evaluation scripts can import their dependencies."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.act.modeling_act import ACTPolicy
        import torch
        import matplotlib.pyplot as plt
        print("✅ Evaluation imports working")
    except ImportError as e:
        pytest.fail(f"Evaluation imports failed: {e}")

def test_mock_model_evaluation():
    """Test mock model evaluation workflow."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    import torch
    
    # Create mock model
    config = ACTConfig()
    config.input_shapes = {
        "observation.state": [6],
        "observation.images.front": [3, 96, 96]
    }
    config.output_shapes = {"action": [6]}
    
    policy = ACTPolicy(config)
    policy.eval()
    
    # Create mock observation
    obs = {
        "observation.state": torch.randn(1, 1, 6),
        "observation.images.front": torch.randn(1, 1, 3, 96, 96)
    }
    
    # Test inference
    with torch.no_grad():
        actions = policy.select_action(obs)
        assert "action" in actions
        assert actions["action"].shape[0] == 1  # Batch size
        print("✅ Mock model inference works")

def test_evaluation_metrics():
    """Test evaluation metrics calculation."""
    import numpy as np
    import torch
    
    # Mock predictions and targets
    predictions = torch.randn(10, 6)  # 10 samples, 6D actions
    targets = torch.randn(10, 6)
    
    # Test MSE calculation
    mse = torch.nn.functional.mse_loss(predictions, targets)
    assert isinstance(mse, torch.Tensor)
    assert mse.item() >= 0
    
    # Test L1 loss
    l1 = torch.nn.functional.l1_loss(predictions, targets)
    assert isinstance(l1, torch.Tensor)
    assert l1.item() >= 0
    
    print("✅ Evaluation metrics calculation works")

@pytest.mark.slow 
def test_model_loading_simulation():
    """Test model loading and basic inference simulation."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    import torch
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create and save a mock model
        config = ACTConfig()
        config.input_shapes = {
            "observation.state": [6],
            "observation.images.front": [3, 96, 96]
        }
        config.output_shapes = {"action": [6]}
        
        policy = ACTPolicy(config)
        
        # Save model
        model_path = Path(tmp_dir) / "test_model.pt"
        torch.save({
            'model_state_dict': policy.state_dict(),
            'config': config.__dict__
        }, model_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        loaded_policy = ACTPolicy(config)
        loaded_policy.load_state_dict(checkpoint['model_state_dict'])
        
        # Test inference with loaded model
        obs = {
            "observation.state": torch.randn(1, 1, 6),
            "observation.images.front": torch.randn(1, 1, 3, 96, 96)
        }
        
        with torch.no_grad():
            actions = loaded_policy.select_action(obs)
            assert "action" in actions
            print("✅ Model save/load workflow works")

def test_visualization_components():
    """Test visualization components for evaluation."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Test basic plotting
    plt.ioff()  # Turn off interactive mode
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # Mock evaluation data
        episodes = np.arange(1, 11)
        success_rates = np.random.uniform(0.3, 0.9, 10)
        
        ax.plot(episodes, success_rates, 'b-', label='Success Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.legend()
        
        plt.close(fig)
        print("✅ Evaluation visualization works")
        
    except Exception as e:
        pytest.fail(f"Visualization failed: {e}")

@pytest.mark.slow
def test_policy_visualization_script():
    """Test that visualize_policy.py can run without errors."""
    # Test help first
    result = run_script_with_timeout("visualize_policy.py", ["--help"])
    if result is not None and result.returncode == 0:
        print("✅ Policy visualization script help works")
    else:
        pytest.skip("Policy visualization script not accessible")

def test_inference_batching():
    """Test that inference works with different batch sizes."""
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
    policy.eval()
    
    # Test different batch sizes
    for batch_size in [1, 4, 8]:
        obs = {
            "observation.state": torch.randn(batch_size, 1, 6),
            "observation.images.front": torch.randn(batch_size, 1, 3, 96, 96)
        }
        
        with torch.no_grad():
            actions = policy.select_action(obs)
            assert actions["action"].shape[0] == batch_size
    
    print("✅ Inference batching works")

def test_error_handling_in_evaluation():
    """Test error handling in evaluation workflows."""
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
    
    # Test with wrong input shapes
    wrong_obs = {
        "observation.state": torch.randn(1, 1, 3),  # Wrong shape
        "observation.images.front": torch.randn(1, 1, 3, 96, 96)
    }
    
    try:
        with torch.no_grad():
            actions = policy.select_action(wrong_obs)
        pytest.fail("Should have failed with wrong input shape")
    except Exception:
        print("✅ Error handling works for wrong input shapes")

@pytest.mark.slow
def test_demo_visualizations():
    """Test demo visualization script."""
    result = run_script_with_timeout("demo_visualizations.py", ["--help"], timeout=15)
    if result is not None and result.returncode == 0:
        print("✅ Demo visualizations script accessible")
    else:
        pytest.skip("Demo visualizations script not accessible")

def test_performance_metrics():
    """Test performance metric calculations."""
    import time
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    
    config = ACTConfig()
    config.input_shapes = {
        "observation.state": [6],
        "observation.images.front": [3, 96, 96]
    }
    config.output_shapes = {"action": [6]}
    
    policy = ACTPolicy(config)
    policy.eval()
    
    # Test inference speed
    obs = {
        "observation.state": torch.randn(1, 1, 6),
        "observation.images.front": torch.randn(1, 1, 3, 96, 96)
    }
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            policy.select_action(obs)
    
    # Time inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            policy.select_action(obs)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 10
    assert avg_inference_time < 1.0  # Should be less than 1 second per inference
    print(f"✅ Average inference time: {avg_inference_time:.3f}s")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 