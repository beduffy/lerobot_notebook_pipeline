import pytest
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot_notebook_pipeline.dataset_utils.visualization import AddGaussianNoise, visualize_augmentations, plot_action_histogram

@pytest.fixture
def sample_image():
    """A pytest fixture to provide a sample image for testing."""
    return torch.rand(3, 128, 128)


@pytest.fixture
def simple_dataset():
    """A pytest fixture to provide a simple LeRobotDataset for testing."""
    # Using a real but small dataset from the hub for realistic testing
    return LeRobotDataset("bearlover365/red_cube_always_in_same_place")

def test_visualize_augmentations(sample_image):
    """
    Tests the visualize_augmentations function.
    """
    transform = transforms.Compose([
        AddGaussianNoise(mean=0., std=0.02),
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ])

    # Use interactive mode to avoid blocking the test
    plt.ion()
    try:
        visualize_augmentations(sample_image, transform)
    except Exception as e:
        pytest.fail(f"visualize_augmentations raised an exception: {e}")
    finally:
        # Turn off interactive mode
        plt.ioff()
        plt.close()


def test_plot_action_histogram(simple_dataset):
    """
    Tests the plot_action_histogram function.
    """
    # Use interactive mode to avoid blocking the test
    plt.ion()
    try:
        plot_action_histogram(simple_dataset, 0)
    except Exception as e:
        pytest.fail(f"plot_action_histogram raised an exception: {e}")
    finally:
        # Turn off interactive mode
        plt.ioff()
        plt.close() 