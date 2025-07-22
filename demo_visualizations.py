#!/usr/bin/env python3
"""
Visualization Demo Script

This script demonstrates all the visualization capabilities of the lerobot_notebook_pipeline
without any training code. Perfect for exploring and understanding your dataset.

Usage:
    python demo_visualizations.py --dataset "bearlover365/red_cube_always_in_same_place"
    python demo_visualizations.py --dataset "my_dataset" --root "./data" --episodes 0 1 2
"""

import argparse
import sys
import torch
from torchvision import transforms
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Import all our visualization functions
from lerobot_notebook_pipeline.dataset_utils.analysis import (
    get_dataset_stats, visualize_sample
)
from lerobot_notebook_pipeline.dataset_utils.visualization import (
    AddGaussianNoise, visualize_augmentations, plot_action_histogram,
    plot_all_action_histograms, visualize_episode_trajectory, 
    create_training_animation, compare_augmentation_effects
)


def demo_basic_visualizations(dataset, sample_idx=0):
    """Demonstrate basic dataset visualizations."""
    print("🔍 BASIC DATASET VISUALIZATIONS")
    print("=" * 50)
    
    # Dataset statistics
    print("📊 Dataset Statistics:")
    stats = get_dataset_stats(dataset)
    for key, value in stats.items():
        if key != "dataset_stats":
            print(f"  {key}: {value}")
    print()
    
    # Sample visualization
    print(f"🖼️  Sample Visualization (Index {sample_idx}):")
    visualize_sample(dataset, sample_idx)
    print()
    
    # Action histograms
    print("📊 Action Distribution Analysis:")
    plot_all_action_histograms(dataset)
    print()


def demo_episode_visualizations(dataset, episode_indices):
    """Demonstrate episode-specific visualizations."""
    print("🎬 EPISODE VISUALIZATIONS")
    print("=" * 50)
    
    available_episodes = list(range(len(dataset.meta.episodes)))
    valid_episodes = [ep for ep in episode_indices if ep in available_episodes]
    
    if not valid_episodes:
        print(f"❌ No valid episodes found. Available: {available_episodes}")
        return
    
    # Individual episode trajectories
    for ep_idx in valid_episodes[:3]:  # Limit to first 3 to avoid too many plots
        print(f"📈 Episode {ep_idx} Trajectory:")
        visualize_episode_trajectory(dataset, ep_idx)
        print()
    
    # Episode animations (static frame views)
    for ep_idx in valid_episodes[:2]:  # Limit to first 2
        print(f"🎬 Episode {ep_idx} Animation (Key Frames):")
        create_training_animation(dataset, ep_idx, max_frames=50)
        print()


def demo_augmentation_visualizations(dataset):
    """Demonstrate data augmentation visualizations."""
    print("🎨 DATA AUGMENTATION VISUALIZATIONS")
    print("=" * 50)
    
    # Get sample image
    sample = dataset[0]
    image_key = None
    for key in sample.keys():
        if "image" in key and isinstance(sample[key], torch.Tensor):
            image_key = key
            break
    
    if not image_key:
        print("❌ No image found in dataset for augmentation demo")
        return
    
    original_image = sample[image_key]
    
    # Single augmentation demo
    print("🔍 Single Augmentation Effect:")
    transform = transforms.Compose([
        AddGaussianNoise(mean=0., std=0.02),
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ])
    
    try:
        visualize_augmentations(original_image, transform)
        print()
    except Exception as e:
        print(f"⚠️  Augmentation visualization skipped: {e}")
        return
    
    # Comparison of multiple augmentation strategies
    print("🎨 Augmentation Strategy Comparison:")
    
    augmentations = [
        transforms.Compose([AddGaussianNoise(mean=0., std=0.01)]),
        transforms.Compose([AddGaussianNoise(mean=0., std=0.03)]),
        transforms.Compose([AddGaussianNoise(mean=0., std=0.05)]),
    ]
    
    augmentation_names = [
        "Light Noise (σ=0.01)",
        "Medium Noise (σ=0.03)", 
        "Heavy Noise (σ=0.05)"
    ]
    
    try:
        compare_augmentation_effects(dataset, augmentations, augmentation_names)
        print()
    except Exception as e:
        print(f"⚠️  Augmentation comparison skipped: {e}")


def demo_action_analysis(dataset):
    """Demonstrate action-specific analysis and visualization."""
    print("🎯 ACTION ANALYSIS & VISUALIZATION")
    print("=" * 50)
    
    # Get action information
    sample = dataset[0]
    action_dim = sample["action"].shape[0]
    print(f"📊 Action space: {action_dim} dimensions")
    
    # Individual action dimension histograms
    print("📈 Individual Action Dimension Analysis:")
    for i in range(min(3, action_dim)):  # Show first 3 dimensions
        print(f"  Dimension {i}:")
        plot_action_histogram(dataset, i)
    
    print()


def demo_comprehensive_visualization(dataset):
    """Run a comprehensive visualization demo."""
    print("🚀 COMPREHENSIVE VISUALIZATION DEMO")
    print("=" * 60)
    
    num_episodes = len(dataset.meta.episodes)
    num_steps = len(dataset)
    
    print(f"Dataset: {num_episodes} episodes, {num_steps} total steps")
    print(f"Average episode length: {num_steps/num_episodes:.1f} steps")
    print()
    
    # All visualizations
    demo_basic_visualizations(dataset, sample_idx=0)
    demo_episode_visualizations(dataset, episode_indices=[0, 1, 2])
    demo_augmentation_visualizations(dataset)
    demo_action_analysis(dataset)
    
    print("🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("💡 Visualization Summary:")
    print("   ✅ Basic dataset statistics and sample visualization")
    print("   ✅ Episode trajectory analysis")
    print("   ✅ Animation key frames")
    print("   ✅ Data augmentation effects")
    print("   ✅ Action distribution analysis")
    print()
    print("🔧 For deeper analysis, use:")
    print("   python analyse_dataset.py --dataset [your_dataset]")
    print("   python visualize_policy.py --policy-path [trained_model] --dataset [your_dataset]")


def main():
    parser = argparse.ArgumentParser(description='Demo all visualization capabilities')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name or path')
    parser.add_argument('--root', type=str, default=None,
                       help='Root directory for local datasets')
    parser.add_argument('--episodes', type=int, nargs='+', default=[0, 1, 2],
                       help='Episode indices to visualize')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--video-backend', type=str, default='pyav',
                       help='Video backend to use (pyav or cv2)')
    parser.add_argument('--demo-type', type=str, 
                       choices=['all', 'basic', 'episodes', 'augmentation', 'actions'],
                       default='all',
                       help='Type of demo to run')
    
    args = parser.parse_args()
    
    print("🎨 LeRobot Visualization Demo")
    print("=" * 40)
    print(f"Dataset: {args.dataset}")
    if args.root:
        print(f"Root: {args.root}")
    print(f"Demo type: {args.demo_type}")
    print()
    
    try:
        # Load dataset
        print("📦 Loading dataset...")
        if args.root:
            dataset = LeRobotDataset(args.dataset, root=args.root, video_backend=args.video_backend)
        else:
            dataset = LeRobotDataset(args.dataset, video_backend=args.video_backend)
        print("✅ Dataset loaded successfully!")
        print()
        
        # Run selected demo
        if args.demo_type == 'all':
            demo_comprehensive_visualization(dataset)
        elif args.demo_type == 'basic':
            demo_basic_visualizations(dataset, args.sample_idx)
        elif args.demo_type == 'episodes':
            demo_episode_visualizations(dataset, args.episodes)
        elif args.demo_type == 'augmentation':
            demo_augmentation_visualizations(dataset)
        elif args.demo_type == 'actions':
            demo_action_analysis(dataset)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 