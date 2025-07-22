import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class AddGaussianNoise:
    """
    Adds Gaussian noise to a tensor.
    """
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Adds noise: tensor remains a tensor.
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

def visualize_augmentations(sample_image: torch.Tensor, transform: transforms.Compose):
    """
    Visualizes the effect of a transform on a sample image.

    Args:
        sample_image: A torch.Tensor representing the original image.
        transform: A torchvision.transforms.Compose object.
    """
    augmented_image = transform(sample_image)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(sample_image.permute(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Augmented image 
    axes[1].imshow(augmented_image.permute(1, 2, 0))
    axes[1].set_title("Augmented Image")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"🔍 Image shape: {sample_image.shape}")
    print(f"📊 Original image stats:")
    print(f"   Mean: {sample_image.mean():.3f}")
    print(f"   Std:  {sample_image.std():.3f}")
    print(f"   Min:  {sample_image.min():.3f}")
    print(f"   Max:  {sample_image.max():.3f}")
    
    print(f"📊 Augmented image stats:")
    print(f"   Mean: {augmented_image.mean():.3f}")
    print(f"   Std:  {augmented_image.std():.3f}")
    print(f"   Min:  {augmented_image.min():.3f}")
    print(f"   Max:  {augmented_image.max():.3f}") 


def plot_action_histogram(dataset: LeRobotDataset, action_dim: int):
    """
    Plots a histogram of the action values for a specific dimension.

    Args:
        dataset: A LeRobotDataset instance.
        action_dim: The index of the action dimension to plot.
    """
    actions = [sample["action"][action_dim].item() for sample in dataset]

    plt.hist(actions, bins=50)
    plt.title(f"Action Dimension {action_dim} Distribution")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.show() 