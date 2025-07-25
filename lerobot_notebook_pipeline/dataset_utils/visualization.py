import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class AddGaussianNoise:
    """
    Adds Gaussian noise to a tensor.
    """
    def __init__(self, mean=0., std=0.005):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Adds noise: tensor remains a tensor.
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def visualize_augmentations(sample_image: torch.Tensor, transform: transforms.Compose, save_path: str = None):
    """
    Visualizes the effect of a transform on a sample image.

    Args:
        sample_image: A torch.Tensor representing the original image.
        transform: A torchvision.transforms.Compose object.
        save_path: Path to save the plot. If None, uses default naming.
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
    
    # Save plot instead of showing
    if save_path is None:
        save_path = "data/plots/augmentation_comparison.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved augmentation comparison to {save_path}")

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


def plot_action_histogram(dataset: LeRobotDataset, action_dim: int, save_path: str = None):
    """
    Plots a histogram of the action values for a specific dimension.

    Args:
        dataset: A LeRobotDataset instance.
        action_dim: The index of the action dimension to plot.
        save_path: Path to save the plot. If None, uses default naming.
    """
    actions = [sample["action"][action_dim].item() for sample in dataset]

    plt.hist(actions, bins=50)
    plt.title(f"Action Dimension {action_dim} Distribution")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    
    # Save plot instead of showing
    if save_path is None:
        save_path = f"data/plots/action_histogram_dim_{action_dim}.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved action histogram to {save_path}")


def plot_all_action_histograms(dataset: LeRobotDataset, sample_ratio: float = 1.0, save_path: str = None):
    """
    Plots histograms for all action dimensions.
    
    Args:
        dataset: A LeRobotDataset instance.
        sample_ratio: Fraction of data to sample for faster plotting (0.1 = 10% of data)
        save_path: Path to save the plot. If None, uses default naming.
    """
    start_time = time.time()
    print("📊 Creating action histograms...")
    
    # Get action information
    print("   🔍 Analyzing action dimensions...")
    sample_start = time.time()
    sample = dataset[0]
    action_dim = sample["action"].shape[0]
    joint_names = [f'Joint {i+1}' for i in range(action_dim-1)] + ['Gripper'] if action_dim == 7 else [f'Dim {i}' for i in range(action_dim)]
    print(f"      Action space: {action_dim} dimensions ({time.time() - sample_start:.3f}s)")
    
    # Determine sampling strategy
    dataset_len = len(dataset)
    if sample_ratio < 1.0:
        sample_size = int(dataset_len * sample_ratio)
        indices = np.random.choice(dataset_len, sample_size, replace=False)
        indices = sorted(indices)
        print(f"   🎲 Sampling {sample_size}/{dataset_len} data points ({sample_ratio*100:.1f}%) for faster plotting")
    else:
        indices = range(dataset_len)
        sample_size = dataset_len
        print(f"   📊 Plotting full dataset ({dataset_len} data points)")
    
    # Collect all actions
    print("   📊 Loading action data...")
    load_start = time.time()
    all_actions = []
    
    for i, idx in enumerate(indices):
        if i % max(1, len(indices) // 10) == 0:  # Progress every 10%
            print(f"      Loading: {i}/{len(indices)} ({i/len(indices)*100:.1f}%)")
        all_actions.append(dataset[int(idx)]["action"].numpy())
    
    all_actions = np.array(all_actions)
    load_time = time.time() - load_start
    print(f"      Action data loaded: {all_actions.shape} ({load_time:.2f}s)")
    
    # Create subplots
    print("   🎨 Setting up plot structure...")
    setup_start = time.time()
    rows = (action_dim + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    setup_time = time.time() - setup_start
    print(f"      Plot structure ready: {rows}x2 grid ({setup_time:.3f}s)")
    
    # Create histograms
    print("   📈 Creating histograms...")
    plot_start = time.time()
    
    for i in range(action_dim):
        if i % max(1, action_dim // 4) == 0:  # Progress every 25%
            print(f"      Plotting dimension {i+1}/{action_dim}...")
        
        row, col = i // 2, i % 2
        axes[row, col].hist(all_actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'{joint_names[i]} - Action Distribution')
        axes[row, col].set_xlabel('Action Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(all_actions[:, i])
        std_val = np.std(all_actions[:, i])
        axes[row, col].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        axes[row, col].legend()
    
    # Hide empty subplot if odd number of actions
    if action_dim % 2 == 1:
        axes[rows-1, 1].axis('off')
    
    plot_time = time.time() - plot_start
    print(f"      Histograms created ({plot_time:.2f}s)")
    
    # Finalize plot
    print("   🎨 Finalizing plot...")
    finalize_start = time.time()
    plt.tight_layout()
    plt.suptitle('📊 All Action Dimension Distributions', fontsize=16, y=1.02)
    finalize_time = time.time() - finalize_start
    
    # Save plot instead of showing
    print("   💾 Saving plot...")
    save_start = time.time()
    if save_path is None:
        save_path = "data/plots/all_action_histograms.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    save_time = time.time() - save_start
    print(f"💾 Saved action histograms to {save_path}")
    
    total_time = time.time() - start_time
    print(f"⏱️  Action histogram timing:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Data loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
    print(f"   Plot setup: {setup_time:.3f}s ({setup_time/total_time*100:.1f}%)")
    print(f"   Histogram creation: {plot_time:.2f}s ({plot_time/total_time*100:.1f}%)")
    print(f"   Finalization: {finalize_time:.3f}s ({finalize_time/total_time*100:.1f}%)")
    print(f"   Saving: {save_time:.3f}s ({save_time/total_time*100:.1f}%)")


def visualize_episode_trajectory(dataset: LeRobotDataset, episode_idx: int = 0, save_path: str = None):
    """
    Visualizes the action trajectory for a specific episode.
    
    Args:
        dataset: A LeRobotDataset instance
        episode_idx: Index of the episode to visualize
        save_path: Path to save the plot. If None, uses default naming.
    """
    start_time = time.time()
    print(f"📈 Visualizing episode {episode_idx} trajectory...")
    
    # Get episode data indices
    print("   📊 Loading episode data...")
    load_start = time.time()
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    episode_length = to_idx - from_idx
    print(f"      Episode range: {from_idx} to {to_idx} ({episode_length} steps)")
    
    # Collect episode data
    episode_actions = []
    episode_states = []
    
    for i in range(from_idx, to_idx):
        if (i - from_idx) % max(1, episode_length // 10) == 0:  # Progress every 10%
            progress = (i - from_idx) / episode_length * 100
            print(f"      Loading step {i - from_idx + 1}/{episode_length} ({progress:.1f}%)")
        
        sample = dataset[i]
        episode_actions.append(sample['action'].numpy())
        if 'observation.state' in sample:
            episode_states.append(sample['observation.state'].numpy())
    
    episode_actions = np.array(episode_actions)
    episode_states = np.array(episode_states) if episode_states else None
    load_time = time.time() - load_start
    print(f"      Episode data loaded ({load_time:.2f}s)")
    
    action_dim = episode_actions.shape[1]
    joint_names = [f'Joint {i+1}' for i in range(action_dim-1)] + ['Gripper'] if action_dim == 7 else [f'Dim {i}' for i in range(action_dim)]
    
    # Create plots
    print("   🎨 Setting up trajectory plots...")
    setup_start = time.time()
    fig, axes = plt.subplots(action_dim, 1, figsize=(15, 3*action_dim))
    if action_dim == 1:
        axes = [axes]
    
    time_steps = range(len(episode_actions))
    setup_time = time.time() - setup_start
    print(f"      Plot structure ready: {action_dim} subplots ({setup_time:.3f}s)")
    
    # Plot trajectories
    print("   📈 Creating trajectory plots...")
    plot_start = time.time()
    
    for i in range(action_dim):
        if i % max(1, action_dim // 4) == 0:  # Progress every 25%
            print(f"      Plotting {joint_names[i]} ({i+1}/{action_dim})...")
        
        axes[i].plot(time_steps, episode_actions[:, i], 'b-', linewidth=2, label='Action', alpha=0.8)
        
        if episode_states is not None and i < episode_states.shape[1]:
            axes[i].plot(time_steps, episode_states[:, i], 'r--', linewidth=2, label='State', alpha=0.8)
        
        axes[i].set_title(f'{joint_names[i]} - Episode {episode_idx} Trajectory')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add some statistics
        action_range = episode_actions[:, i].max() - episode_actions[:, i].min()
        axes[i].text(0.02, 0.98, f'Range: {action_range:.3f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plot_time = time.time() - plot_start
    print(f"      Trajectory plots created ({plot_time:.2f}s)")
    
    # Finalize plot
    print("   🎨 Finalizing plot...")
    finalize_start = time.time()
    plt.tight_layout()
    plt.suptitle(f'🎯 Episode {episode_idx} Action/State Trajectory', fontsize=16, y=1.02)
    finalize_time = time.time() - finalize_start
    
    # Save plot instead of showing
    print("   💾 Saving plot...")
    save_start = time.time()
    if save_path is None:
        save_path = f"data/plots/episode_{episode_idx}_trajectory.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    save_time = time.time() - save_start
    print(f"💾 Saved episode trajectory to {save_path}")
    
    total_time = time.time() - start_time
    
    print(f"   Episode length: {len(episode_actions)} steps")
    print(f"   Action range: [{episode_actions.min():.3f}, {episode_actions.max():.3f}]")
    if episode_states is not None:
        print(f"   State range: [{episode_states.min():.3f}, {episode_states.max():.3f}]")
    
    print(f"⏱️  Episode trajectory timing:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Data loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
    print(f"   Plot setup: {setup_time:.3f}s ({setup_time/total_time*100:.1f}%)")
    print(f"   Trajectory plotting: {plot_time:.2f}s ({plot_time/total_time*100:.1f}%)")
    print(f"   Finalization: {finalize_time:.3f}s ({finalize_time/total_time*100:.1f}%)")
    print(f"   Saving: {save_time:.3f}s ({save_time/total_time*100:.1f}%)")


def create_training_animation(dataset: LeRobotDataset, episode_idx: int = 0, max_frames: int = 20, 
                             frame_skip: int = 5, resize_factor: float = 0.25, save_path: str = None):
    """
    Creates an animation showing the robot's view during an episode.
    Optimized for speed with image resizing and frame skipping.
    
    Args:
        dataset: A LeRobotDataset instance
        episode_idx: Episode to animate
        max_frames: Maximum number of frames to include (default: 20, reduced from 100)
        frame_skip: Take every Nth frame (default: 5, skip 4 frames between each used frame)
        resize_factor: Factor to resize images (default: 0.25 = 1/4 size)
        save_path: Path to save the plot. If None, uses default naming.
    """
    start_time = time.time()
    print(f"🎬 Creating optimized animation for episode {episode_idx}...")
    print(f"   🚀 Speed optimizations: max_frames={max_frames}, frame_skip={frame_skip}, resize_factor={resize_factor}")
    
    # Get episode data indices  
    print("   📊 Getting episode information...")
    indices_start = time.time()
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    episode_length = to_idx - from_idx
    
    # Apply frame skipping first, then limit to max_frames
    available_indices = list(range(from_idx, to_idx, frame_skip))
    step_size = max(1, len(available_indices) // max_frames)
    selected_indices = available_indices[::step_size][:max_frames]
    actual_frames = len(selected_indices)
    
    print(f"      Episode: {episode_length} steps")
    print(f"      After frame skip ({frame_skip}): {len(available_indices)} frames") 
    print(f"      After max limit ({max_frames}): {actual_frames} frames")
    print(f"      Selection completed ({time.time() - indices_start:.3f}s)")
    
    # Collect frames and actions
    print("   🖼️  Loading and resizing frames...")
    load_start = time.time()
    frames = []
    actions = []
    
    for i, step_idx in enumerate(selected_indices):
        if i % max(1, actual_frames // 5) == 0:  # Progress every 20%
            print(f"      Loading frame {i+1}/{actual_frames} ({i/actual_frames*100:.1f}%)")
        
        sample = dataset[step_idx]
        
        # Get image
        image_key = None
        for key in sample.keys():
            if "image" in key and isinstance(sample[key], torch.Tensor):
                image_key = key
                break
        
        if image_key:
            image = sample[image_key]
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = image.permute(1, 2, 0).numpy()
            else:
                image = image.numpy()
            
            # Resize image for speed
            if resize_factor != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                if len(image.shape) == 3:
                    # Use simple numpy slicing for downsampling (faster than cv2 for our case)
                    step_h, step_w = max(1, h // new_h), max(1, w // new_w)
                    image = image[::step_h, ::step_w]
                
            frames.append(image)
            actions.append(sample['action'].numpy())
    
    load_time = time.time() - load_start
    
    if not frames:
        print("❌ No images found in episode")
        return
    
    print(f"      Loaded and resized {len(frames)} frames ({load_time:.2f}s)")
    if frames:
        print(f"      Frame size: {frames[0].shape} (resize factor: {resize_factor})")
    
    # Create static visualization instead of animation for better compatibility
    print("   🎨 Creating frame visualization...")
    viz_start = time.time()
    num_display = min(6, len(frames))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # Smaller figure size
    axes = axes.flatten()
    
    for i in range(num_display):
        frame_idx = i * len(frames) // num_display
        axes[i].imshow(frames[frame_idx])
        axes[i].set_title(f'Frame {frame_idx + 1}', fontsize=10)
        axes[i].axis('off')
        
        # Add action info with smaller font
        action_text = f'Action: [{", ".join([f"{x:.2f}" for x in actions[frame_idx][:3]])}...]'
        axes[i].text(0.02, 0.02, action_text, transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=7)
    
    # Hide empty subplots
    for i in range(num_display, 6):
        axes[i].axis('off')
    
    viz_time = time.time() - viz_start
    print(f"      Frame grid created ({viz_time:.2f}s)")
    
    # Finalize and save
    print("   🎨 Finalizing animation...")
    finalize_start = time.time()
    plt.tight_layout()
    plt.suptitle(f'🎬 Episode {episode_idx} - Key Frames (Optimized)', fontsize=14, y=1.02)
    finalize_time = time.time() - finalize_start
    
    print("   💾 Saving animation...")
    save_start = time.time()
    if save_path is None:
        save_path = f"data/plots/episode_{episode_idx}_animation_optimized.png"
    plt.savefig(save_path, dpi=80, bbox_inches='tight')  # Lower DPI for speed
    plt.close()
    save_time = time.time() - save_start
    print(f"💾 Saved optimized animation to {save_path}")
    
    total_time = time.time() - start_time
    print(f"⏱️  Optimized animation timing:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Frame loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
    print(f"   Visualization: {viz_time:.2f}s ({viz_time/total_time*100:.1f}%)")
    print(f"   Finalization: {finalize_time:.3f}s ({finalize_time/total_time*100:.1f}%)")
    print(f"   Saving: {save_time:.3f}s ({save_time/total_time*100:.1f}%)")


def compare_augmentation_effects(dataset: LeRobotDataset, transforms_list: list, transform_names: list, save_path: str = None):
    """
    Compare the effects of different augmentation transforms.
    
    Args:
        dataset: A LeRobotDataset instance
        transforms_list: List of transform objects to compare
        transform_names: List of names for the transforms
        save_path: Path to save the plot. If None, uses default naming.
    """
    print("🎨 Comparing augmentation effects...")
    
    # Get a sample image
    sample = dataset[0]
    image_key = None
    for key in sample.keys():
        if "image" in key and isinstance(sample[key], torch.Tensor):
            image_key = key
            break
    
    if not image_key:
        print("❌ No image found in dataset")
        return
    
    original_image = sample[image_key]
    
    # Create comparison plot
    num_transforms = len(transforms_list)
    fig, axes = plt.subplots(1, num_transforms + 1, figsize=(5 * (num_transforms + 1), 5))
    
    # Original image
    axes[0].imshow(original_image.permute(1, 2, 0))
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Apply each transform
    for i, (transform, name) in enumerate(zip(transforms_list, transform_names)):
        augmented = transform(original_image)
        axes[i + 1].imshow(augmented.permute(1, 2, 0))
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')
        
        # Print stats
        print(f"{name}: Mean={augmented.mean():.3f}, Std={augmented.std():.3f}")
    
    plt.tight_layout()
    plt.suptitle('🎨 Augmentation Comparison', fontsize=16, y=1.02)
    
    # Save plot instead of showing
    if save_path is None:
        save_path = "data/plots/augmentation_effects_comparison.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved augmentation comparison to {save_path}") 