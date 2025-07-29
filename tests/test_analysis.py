import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_notebook_pipeline.dataset_utils.analysis import get_dataset_stats, visualize_sample
import matplotlib.pyplot as plt
import torch

@pytest.fixture
def simple_dataset():
    """A pytest fixture to provide a simple LeRobotDataset for testing."""
    # Using a real but small dataset from the hub for realistic testing
    return LeRobotDataset("bearlover365/red_cube_always_in_same_place")

def test_get_dataset_stats(simple_dataset):
    """
    Tests the get_dataset_stats function.
    """
    stats = get_dataset_stats(simple_dataset)

    # Check that the stats dictionary is not empty
    assert stats

    # Check for expected keys
    assert "num_steps" in stats
    assert "num_episodes" in stats
    assert "observation.images.front_shape" in stats
    assert "action_shape" in stats
    assert "dataset_stats" in stats

    # Check that the number of steps is correct
    assert stats["num_steps"] == len(simple_dataset)

    # Check that the number of episodes is correct
    assert stats["num_episodes"] == len(simple_dataset.meta.episodes)

    # Check that the observation image shape is correct
    sample = simple_dataset[0]
    
    image_key = None
    for key in sample.keys():
        if "image" in key and isinstance(sample[key], torch.Tensor):
            image_key = key
            break
    
    assert f"{image_key}_shape" in stats
    assert stats[f"{image_key}_shape"] == tuple(sample[image_key].shape)
    
    # Check that the action shape is correct
    assert stats["action_shape"] == tuple(sample["action"].shape)

    # Check that the dataset stats are correct
    assert stats["dataset_stats"] == simple_dataset.meta.stats


def test_visualize_sample(simple_dataset):
    """
    Tests the visualize_sample function.
    """
    # Use interactive mode to avoid blocking the test
    plt.ion()
    try:
        visualize_sample(simple_dataset, 0)
    except Exception as e:
        pytest.fail(f"visualize_sample raised an exception: {e}")
    finally:
        # Turn off interactive mode
        plt.ioff()
        plt.close() 