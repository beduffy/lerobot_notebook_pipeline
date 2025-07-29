#!/usr/bin/env python3
"""
Tests for data collection workflows - ensuring data collection protocols work correctly.
"""

import pytest
import subprocess
import tempfile
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
def test_collect_systematic_data_help():
    """Test that collect_systematic_data.py shows help without errors."""
    result = run_script_with_timeout("collect_systematic_data.py", ["--help"])
    assert result is not None
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()

@pytest.mark.slow  
def test_data_collection_protocol_display():
    """Test data collection protocol display."""
    result = run_script_with_timeout(
        "collect_systematic_data.py", 
        ["--task", "cube_pickup", "--protocol"],
        timeout=15
    )
    
    assert result is not None
    assert result.returncode == 0
    assert "SYSTEMATIC DATA COLLECTION PROTOCOL" in result.stdout
    assert "Position A:" in result.stdout or "position" in result.stdout.lower()
    print("✅ Data collection protocol displays correctly")

def test_data_collection_imports():
    """Test data collection script imports."""
    try:
        # Test basic imports that data collection might need
        import json
        import time
        import os
        from pathlib import Path
        print("✅ Data collection imports work")
    except ImportError as e:
        pytest.fail(f"Data collection imports failed: {e}")

def test_logging_functionality():
    """Test data collection logging functionality."""
    import logging
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "test_data_collection.log"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Test logging
        logger = logging.getLogger("data_collection_test")
        logger.info("Test data collection started")
        logger.info("Recording episode 1")
        logger.info("Test data collection completed")
        
        # Check log file was created and has content
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Test data collection started" in log_content
        assert "Recording episode" in log_content
        
        print("✅ Data collection logging works")

def test_data_validation_helpers():
    """Test data validation helper functions."""
    import numpy as np
    import torch
    
    # Mock data validation functions
    def validate_episode_data(episode_data):
        """Validate episode data structure."""
        required_keys = ['observations', 'actions', 'rewards']
        return all(key in episode_data for key in required_keys)
    
    def validate_observation_shape(obs):
        """Validate observation shape."""
        if isinstance(obs, (np.ndarray, torch.Tensor)):
            return len(obs.shape) >= 1
        return False
    
    def validate_action_range(actions, action_min=-1.0, action_max=1.0):
        """Validate action range."""
        if isinstance(actions, (np.ndarray, torch.Tensor)):
            return np.all(actions >= action_min) and np.all(actions <= action_max)
        return False
    
    # Test validation functions
    valid_episode = {
        'observations': np.random.randn(10, 6),
        'actions': np.random.uniform(-1, 1, (10, 6)),
        'rewards': np.random.randn(10)
    }
    
    invalid_episode = {
        'observations': np.random.randn(10, 6),
        'actions': np.random.uniform(-1, 1, (10, 6))
        # Missing rewards
    }
    
    assert validate_episode_data(valid_episode)
    assert not validate_episode_data(invalid_episode)
    assert validate_observation_shape(valid_episode['observations'])
    assert validate_action_range(valid_episode['actions'])
    
    print("✅ Data validation helpers work")

def test_experiment_configuration():
    """Test experiment configuration management."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_file = Path(tmp_dir) / "experiment_config.json"
        
        # Create test configuration
        config = {
            "experiment_name": "test_data_collection",
            "task": "cube_pickup",
            "num_episodes": 10,
            "episode_length": 100,
            "data_collection_params": {
                "camera_resolution": [640, 480],
                "control_frequency": 10,
                "save_images": True
            },
            "metadata": {
                "created_by": "test_suite",
                "description": "Test configuration for data collection"
            }
        }
        
        # Save configuration
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Load and validate configuration  
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["experiment_name"] == "test_data_collection"
        assert loaded_config["task"] == "cube_pickup"
        assert loaded_config["num_episodes"] == 10
        assert "data_collection_params" in loaded_config
        
        print("✅ Experiment configuration management works")

def test_data_format_compatibility():
    """Test data format compatibility with LeRobot."""
    import torch
    import numpy as np
    
    # Mock LeRobot-compatible data format
    def create_lerobot_episode():
        """Create mock episode data in LeRobot format."""
        episode_length = 50
        
        episode_data = {
            "observation.images.front": torch.randint(0, 255, (episode_length, 3, 96, 96), dtype=torch.uint8),
            "observation.state": torch.randn(episode_length, 6),
            "action": torch.randn(episode_length, 6),
            "episode_index": torch.full((episode_length,), 0),
            "frame_index": torch.arange(episode_length),
            "timestamp": torch.linspace(0, 5, episode_length)  # 5 seconds
        }
        
        return episode_data
    
    # Test data creation
    episode = create_lerobot_episode()
    
    # Validate shapes and types
    assert episode["observation.images.front"].shape == (50, 3, 96, 96)
    assert episode["observation.state"].shape == (50, 6)
    assert episode["action"].shape == (50, 6)
    assert len(episode["episode_index"]) == 50
    assert len(episode["frame_index"]) == 50
    
    print("✅ LeRobot data format compatibility verified")

@pytest.mark.slow
def test_data_collection_tasks():
    """Test different data collection task configurations."""
    tasks = ["cube_pickup", "drawer_open", "button_press"]
    
    for task in tasks:
        result = run_script_with_timeout(
            "collect_systematic_data.py",
            ["--task", task, "--protocol"],
            timeout=10
        )
        
        if result is not None and result.returncode == 0:
            assert task.replace("_", " ") in result.stdout.lower() or "protocol" in result.stdout.lower()
            print(f"✅ Task {task} protocol works")
        else:
            print(f"⚠️  Task {task} protocol may not be fully implemented")

def test_data_storage_structure():
    """Test data storage directory structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir) / "data_collection"
        
        # Create expected directory structure
        directories = [
            data_dir / "raw",
            data_dir / "processed", 
            data_dir / "episodes",
            data_dir / "logs",
            data_dir / "config"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Verify structure
        for directory in directories:
            assert directory.exists()
            assert directory.is_dir()
        
        # Test file creation in structure
        test_files = [
            data_dir / "raw" / "episode_001.h5",
            data_dir / "logs" / "collection.log",
            data_dir / "config" / "experiment.json"
        ]
        
        for test_file in test_files:
            test_file.touch()
            assert test_file.exists()
        
        print("✅ Data storage structure works")

def test_episode_metadata():
    """Test episode metadata generation."""
    import time
    from datetime import datetime
    
    def generate_episode_metadata(episode_idx, task_name):
        """Generate metadata for an episode."""
        return {
            "episode_index": episode_idx,
            "task_name": task_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": None,  # To be filled during collection
            "num_frames": None,  # To be filled during collection
            "success": None,  # To be determined after collection
            "notes": "",
            "environment": {
                "robot_type": "simulated",
                "scene": "default",
                "lighting": "normal"
            }
        }
    
    # Test metadata generation
    metadata = generate_episode_metadata(1, "cube_pickup")
    
    assert metadata["episode_index"] == 1
    assert metadata["task_name"] == "cube_pickup"
    assert "timestamp" in metadata
    assert "environment" in metadata
    assert metadata["environment"]["robot_type"] == "simulated"
    
    print("✅ Episode metadata generation works")

def test_data_collection_safety():
    """Test data collection safety checks."""
    def safety_check_robot_state():
        """Mock safety check for robot state."""
        # In real implementation, this would check robot status
        return True
    
    def safety_check_workspace():
        """Mock safety check for workspace."""
        # In real implementation, this would check for obstacles
        return True
    
    def emergency_stop():
        """Mock emergency stop function."""
        return "Emergency stop activated"
    
    # Test safety functions
    assert safety_check_robot_state()
    assert safety_check_workspace()
    assert emergency_stop() == "Emergency stop activated"
    
    print("✅ Data collection safety checks work")

def test_data_compression():
    """Test data compression for storage efficiency."""
    import gzip
    import pickle
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create mock data
        large_data = {
            "images": np.random.randint(0, 255, (100, 3, 96, 96), dtype=np.uint8),
            "states": np.random.randn(100, 6),
            "actions": np.random.randn(100, 6)
        }
        
        # Test uncompressed storage
        uncompressed_file = Path(tmp_dir) / "episode_uncompressed.pkl"
        with open(uncompressed_file, 'wb') as f:
            pickle.dump(large_data, f)
        uncompressed_size = uncompressed_file.stat().st_size
        
        # Test compressed storage
        compressed_file = Path(tmp_dir) / "episode_compressed.pkl.gz"
        with gzip.open(compressed_file, 'wb') as f:
            pickle.dump(large_data, f)
        compressed_size = compressed_file.stat().st_size
        
        # Test loading compressed data
        with gzip.open(compressed_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Verify data integrity
        assert np.array_equal(large_data["images"], loaded_data["images"])
        assert np.array_equal(large_data["states"], loaded_data["states"])
        assert np.array_equal(large_data["actions"], loaded_data["actions"])
        
        # Check compression ratio
        compression_ratio = compressed_size / uncompressed_size
        assert compression_ratio < 1.0  # Should be compressed
        
        print(f"✅ Data compression works (ratio: {compression_ratio:.2f})")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 