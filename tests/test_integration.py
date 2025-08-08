#!/usr/bin/env python3
"""
Integration tests to ensure the full pipeline works correctly.
"""

import pytest
import os
from pathlib import Path
import tempfile
import json

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_notebook_pipeline.dataset_utils.analysis import get_dataset_stats, analyze_episodes
from lerobot_notebook_pipeline.dataset_utils.visualization import plot_all_action_histograms


# working_dataset fixture is now defined in conftest.py


def test_dataset_loading(working_dataset):
    """Test that dataset loads correctly."""
    assert len(working_dataset) > 0
    assert working_dataset.num_episodes > 0
    print(f"✅ Dataset loaded: {len(working_dataset)} samples, {working_dataset.num_episodes} episodes")


def test_dataset_structure(working_dataset):
    """Test that dataset has expected structure."""
    sample = working_dataset[0]
    
    # Check required keys exist
    required_keys = ['action', 'observation.state', 'observation.images.front']
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    
    # Check shapes
    assert sample['action'].shape == (6,), f"Wrong action shape: {sample['action'].shape}"
    assert sample['observation.state'].shape == (6,), f"Wrong state shape: {sample['observation.state'].shape}"
    assert len(sample['observation.images.front'].shape) == 3, "Image should be 3D"
    
    print(f"✅ Dataset structure validated")



@pytest.mark.slow
def test_analysis_functions(working_dataset):
    """Test that analysis functions work without errors."""
    # Test basic stats
    stats = get_dataset_stats(working_dataset)
    assert isinstance(stats, dict)
    assert 'num_steps' in stats
    assert 'num_episodes' in stats
    
    # Test episode analysis (returns dict with episode indices as keys)
    episode_stats = analyze_episodes(working_dataset)
    assert len(episode_stats) == working_dataset.num_episodes
    assert all(isinstance(ep_data, dict) and 'length' in ep_data for ep_data in episode_stats.values())
    
    print(f"✅ Analysis functions work correctly")


@pytest.mark.slow
def test_visualization_functions(working_dataset, tmp_path):
    """Test that visualization functions create plots."""
    # Test action histograms
    save_path = tmp_path / "test_histograms.png"
    plot_all_action_histograms(
        working_dataset, 
        sample_ratio=0.1,  # Fast test
        save_path=str(save_path)
    )
    
    assert save_path.exists(), "Plot file should be created"
    assert save_path.stat().st_size > 1000, "Plot file should not be empty"
    
    print(f"✅ Visualization functions work correctly")


@pytest.mark.slow
def test_single_episode_experiment_config():
    """Test that single episode experiment creates proper configs."""
    from single_episode_experiment import run_single_episode_experiment
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        
        # Run experiment setup (not actual training)
        result = run_single_episode_experiment(
            "bearlover365/red_cube_always_in_same_place",
            episode_idx=0,
            augmentation_level="medium", 
            output_dir=output_dir,
            training_steps=1000
        )
        
        assert result['status'] == 'configured'
        assert result['episode_index'] == 0
        
        # Check config file was created
        config_file = output_dir / "episode_0_aug_medium" / "experiment_config.json"
        assert config_file.exists()
        
        with open(config_file) as f:
            config = json.load(f)
        assert config['episode_index'] == 0
        assert config['augmentation_level'] == 'medium'
        
        print(f"✅ Single episode experiment configuration works")


def test_data_collection_protocol():
    """Test that data collection protocol runs without errors."""
    import subprocess
    
    # Test protocol display
    result = subprocess.run([
        'python', 'collect_systematic_data.py', 
        '--task', 'cube_pickup', 
        '--protocol'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Protocol script failed: {result.stderr}"
    assert "SYSTEMATIC DATA COLLECTION PROTOCOL" in result.stdout
    assert "Position A: Front-left" in result.stdout
    
    print(f"✅ Data collection protocol works")


@pytest.mark.slow
def test_analysis_script_integration():
    """Test that the main analysis script works end-to-end."""
    import subprocess
    
    # Test analysis script with fast mode
    result = subprocess.run([
        'python', 'analyse_dataset.py', '.', '--fast'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Analysis script failed: {result.stderr}"
    assert "Dataset loaded successfully" in result.stdout
    assert "Analysis complete" in result.stdout
    
    print(f"✅ Full analysis script integration works")


def test_performance_requirements(working_dataset):
    """Test that operations complete within reasonable time limits."""
    import time
    
    # Test dataset loading speed
    start = time.time()
    sample = working_dataset[0]
    load_time = time.time() - start
    assert load_time < 1.0, f"Dataset loading too slow: {load_time:.2f}s"
    
    # Test analysis speed
    start = time.time()
    stats = get_dataset_stats(working_dataset)
    analysis_time = time.time() - start
    assert analysis_time < 2.0, f"Analysis too slow: {analysis_time:.2f}s"
    
    print(f"✅ Performance requirements met")


def test_data_consistency(working_dataset):
    """Test that data is consistent across multiple loads."""
    # Load same sample multiple times
    sample1 = working_dataset[0]
    sample2 = working_dataset[0] 
    sample3 = working_dataset[0]
    
    # Should be identical
    assert (sample1['action'] == sample2['action']).all()
    assert (sample2['action'] == sample3['action']).all()
    assert (sample1['observation.state'] == sample2['observation.state']).all()
    
    print(f"✅ Data consistency verified")


def test_augmentation_safety():
    """Test that augmentations don't break data."""
    from lerobot_notebook_pipeline.dataset_utils.visualization import AddGaussianNoise
    import torch
    
    # Create test image
    test_image = torch.rand(3, 100, 100)  # Random image
    
    # Test different noise levels
    for noise_level in [0.01, 0.02, 0.05]:
        augment = AddGaussianNoise(mean=0., std=noise_level)
        augmented = augment(test_image)
        
        # Should remain same shape
        assert augmented.shape == test_image.shape
        
        # Should be close to original (not pure noise)
        mse = ((augmented - test_image) ** 2).mean()
        assert mse < 0.1, f"Too much noise added: MSE={mse:.4f}"
    
    print(f"✅ Augmentation safety verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 