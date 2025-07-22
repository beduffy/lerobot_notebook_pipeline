import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt

def get_dataset_stats(dataset: LeRobotDataset):
    """
    Analyzes a LeRobotDataset and returns a dictionary of summary statistics.

    Args:
        dataset: A LeRobotDataset instance.

    Returns:
        A dictionary containing summary statistics.
    """
    stats = {}
    stats["num_steps"] = len(dataset)
    stats["num_episodes"] = len(dataset.meta.episodes)

    # Get observation and action space dimensions
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            stats[f"{key}_shape"] = tuple(value.shape)
            stats[f"{key}_dtype"] = value.dtype

    # Get dataset statistics
    stats["dataset_stats"] = dataset.meta.stats

    return stats


def visualize_sample(dataset: LeRobotDataset, index: int):
    """
    Visualizes a single sample from a LeRobotDataset.

    Args:
        dataset: A LeRobotDataset instance.
        index: The index of the sample to visualize.
    """
    sample = dataset[index]

    # Visualize the observation image
    image_key = None
    for key in sample.keys():
        if "image" in key and isinstance(sample[key], torch.Tensor):
            image_key = key
            break

    if image_key:
        image = sample[image_key]
        if isinstance(image, torch.Tensor):
            # Convert to numpy and transpose if necessary
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = image.permute(1, 2, 0).numpy()
            else:
                image = image.numpy()

            plt.imshow(image)
            plt.title(f"Observation Image ({image_key}) at Index {index}")
            plt.show()

    # Print the action
    if "action" in sample:
        action = sample["action"]
        print(f"Action at Index {index}: {action}") 