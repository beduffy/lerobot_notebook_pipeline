"""
Pytest configuration for headless testing.

This configuration ensures that:
1. Datasets use the pyav video backend instead of torchcodec (which requires FFmpeg)
2. Matplotlib uses a non-interactive backend
3. All GUI-related functionality is disabled
"""

import pytest
import os
import matplotlib
import tempfile
from pathlib import Path

# Configure matplotlib for headless operation
matplotlib.use('Agg')  # Non-interactive backend

# Set environment variables for headless operation
os.environ['DISPLAY'] = ''  # Ensure no display
os.environ['MPLBACKEND'] = 'Agg'  # Force matplotlib backend


@pytest.fixture(scope="session", autouse=True)
def setup_headless_environment():
    """Set up environment for headless testing."""
    # Configure matplotlib
    matplotlib.use('Agg')
    
    # Create temporary directories for test outputs
    test_output_dir = Path(tempfile.mkdtemp(prefix="lerobot_test_"))
    os.environ['LEROBOT_TEST_OUTPUT_DIR'] = str(test_output_dir)
    
    yield test_output_dir
    
    # Cleanup is handled by tempfile


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for test outputs."""
    return Path(os.environ.get('LEROBOT_TEST_OUTPUT_DIR', tempfile.mkdtemp()))


@pytest.fixture
def headless_dataset_kwargs():
    """Dataset kwargs that work in headless environments."""
    return {
        'video_backend': 'pyav',  # Use pyav instead of torchcodec
    }


@pytest.fixture
def simple_dataset(headless_dataset_kwargs):
    """Load a simple dataset for testing with headless configuration."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    # Use a small, well-known dataset with headless backend
    dataset = LeRobotDataset(
        "bearlover365/red_cube_always_in_same_place",
        **headless_dataset_kwargs
    )
    return dataset


@pytest.fixture  
def working_dataset(headless_dataset_kwargs):
    """Load a working dataset for integration tests."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    dataset = LeRobotDataset(
        "bearlover365/red_cube_always_in_same_place", 
        **headless_dataset_kwargs
    )
    return dataset 