#!/usr/bin/env python3
"""
Demonstration of LeRobot dataset visualization capabilities.

This script showcases all available visualization functions including:
- Basic dataset visualization (samples, statistics)
- Episode trajectory analysis
- Action distribution plotting
- Data augmentation effects
- Animation creation (key frames)

Usage:
    python demo_visualizations.py <dataset_path> [options]
    python demo_visualizations.py --help

Examples:
    # All visualizations with default settings
    python demo_visualizations.py /path/to/dataset
    
    # Just basic visualizations
    python demo_visualizations.py /path/to/dataset --demo basic
    
    # Save plots to custom directory
    python demo_visualizations.py /path/to/dataset --output-dir ./demo_results
"""

import argparse
import sys
import torch
from pathlib import Path

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_notebook_pipeline.dataset_utils.analysis import visualize_sample
    from lerobot_notebook_pipeline.dataset_utils.visualization import (
        plot_action_histogram, plot_all_action_histograms, 
        visualize_episode_trajectory, create_training_animation,
        visualize_augmentations, compare_augmentation_effects, AddGaussianNoise
    )
    from lerobot_notebook_pipeline.dataset_utils.training import train_model
    from torchvision import transforms
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the lerobot package and this project package.")
    sys.exit(1)


def demo_basic_visualizations(dataset, sample_idx=0, output_dir="."):
    """Demonstrate basic dataset visualizations."""
    print("🔍 BASIC DATASET VISUALIZATIONS")
    print("=" * 50)
    
    output_path = Path(output_dir)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of episodes: {len(dataset.meta.episodes)}")
    print()
    
    print(f"📖 Sample {sample_idx} visualization:")
    visualize_sample(dataset, sample_idx, 
                    save_path=str(output_path / f"demo_sample_{sample_idx}.png"))
    print()
    
    print("📊 All action distributions:")
    plot_all_action_histograms(dataset, sample_ratio=0.2,  # Use 20% for demo speed
                              save_path=str(output_path / "demo_all_action_histograms.png"))
    print()


def demo_episode_visualizations(dataset, episode_indices=[0, 1, 2], output_dir="."):
    """Demonstrate episode-specific visualizations."""
    print("📈 EPISODE VISUALIZATIONS")
    print("=" * 50)
    
    output_path = Path(output_dir)
    available_episodes = list(range(len(dataset.meta.episodes)))
    valid_episodes = [ep for ep in episode_indices if ep in available_episodes]
    
    if not valid_episodes:
        print(f"❌ No valid episodes found. Available: {available_episodes}")
        return
    
    # Individual episode trajectories
    for ep_idx in valid_episodes[:3]:  # Limit to first 3 to avoid too many plots
        print(f"📈 Episode {ep_idx} Trajectory:")
        visualize_episode_trajectory(dataset, ep_idx,
                                   save_path=str(output_path / f"demo_episode_{ep_idx}_trajectory.png"))
        print()
    
    # Episode animations (static frame views)
    for ep_idx in valid_episodes[:2]:  # Limit to first 2
        print(f"🎬 Episode {ep_idx} Animation (Key Frames):")
        create_training_animation(dataset, ep_idx, max_frames=12, frame_skip=10, resize_factor=0.5,
                                save_path=str(output_path / f"demo_episode_{ep_idx}_animation.png"))
        print()


def demo_augmentation_visualizations(dataset, output_dir="."):
    """Demonstrate data augmentation visualizations."""
    print("🎨 DATA AUGMENTATION VISUALIZATIONS")
    print("=" * 50)
    
    output_path = Path(output_dir)
    
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
        visualize_augmentations(original_image, transform,
                              save_path=str(output_path / "demo_single_augmentation.png"))
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
        compare_augmentation_effects(dataset, augmentations, augmentation_names,
                                   save_path=str(output_path / "demo_augmentation_comparison.png"))
        print()
    except Exception as e:
        print(f"⚠️  Augmentation comparison skipped: {e}")


def demo_action_analysis(dataset, output_dir="."):
    """Demonstrate action-specific analysis and visualization."""
    print("🎯 ACTION ANALYSIS & VISUALIZATION")
    print("=" * 50)
    
    output_path = Path(output_dir)
    
    # Get action information
    sample = dataset[0]
    action_dim = sample["action"].shape[0]
    print(f"📊 Action space: {action_dim} dimensions")
    
    # Individual action dimension histograms
    print("📈 Individual Action Dimension Analysis:")
    for i in range(min(3, action_dim)):  # Show first 3 dimensions
        print(f"  Dimension {i}:")
        plot_action_histogram(dataset, i,
                            save_path=str(output_path / f"demo_action_dim_{i}_histogram.png"))
    
    print()


def demo_comprehensive_visualization(dataset, output_dir="."):
    """Run a comprehensive visualization demo."""
    print("🚀 COMPREHENSIVE VISUALIZATION DEMO")
    print("=" * 60)
    
    num_episodes = len(dataset.meta.episodes)
    num_steps = len(dataset)
    
    print(f"Dataset: {num_episodes} episodes, {num_steps} total steps")
    print(f"Average episode length: {num_steps/num_episodes:.1f} steps")
    print()
    
    # All visualizations
    demo_basic_visualizations(dataset, sample_idx=0, output_dir=output_dir)
    demo_episode_visualizations(dataset, episode_indices=[0, 1, 2], output_dir=output_dir)
    demo_augmentation_visualizations(dataset, output_dir=output_dir)
    demo_action_analysis(dataset, output_dir=output_dir)
    
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
    print("   python analyse_dataset.py <dataset_path> --fast")
    print()


def main():
    parser = argparse.ArgumentParser(description="Demo LeRobot dataset visualization capabilities")
    parser.add_argument("dataset_path", help="Path to the dataset")
    parser.add_argument("--root", default=".", help="Root directory for datasets")
    parser.add_argument("--demo", choices=["basic", "episodes", "augmentation", "actions", "comprehensive"], 
                       default="comprehensive", help="Type of demo to run")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2], 
                       help="Episode indices for episode demos")
    parser.add_argument("--video-backend", default="pyav", help="Video backend for dataset loading")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Directory to save demo plots (default: current directory)")
    
    args = parser.parse_args()
    
    # Set output_dir to be from root to data/plots
    output_dir = Path(args.root) / "data" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("🎨 LeRobot Visualization Demo")
        print("=" * 50)
        print(f"📁 Dataset: {args.dataset_path}")
        print(f"📁 Root: {args.root}")
        print(f"💾 Output directory: {output_dir.absolute()}")
        print(f"🎬 Video backend: {args.video_backend}")
        print(f"🎯 Demo type: {args.demo}")
        print()

        # Load dataset
        print("📊 Loading dataset...")
        if args.dataset_path in [".", "local", "local_dataset"]:
            # Use the working method for local datasets
            dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place")
        else:
            dataset = LeRobotDataset(args.dataset_path, root=args.root, video_backend=args.video_backend)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        print()

        # Run selected demo
        if args.demo == "basic":
            demo_basic_visualizations(dataset, args.sample_idx, output_dir)
        elif args.demo == "episodes":
            demo_episode_visualizations(dataset, args.episodes, output_dir)
        elif args.demo == "augmentation":
            demo_augmentation_visualizations(dataset, output_dir)
        elif args.demo == "actions":
            demo_action_analysis(dataset, output_dir)
        elif args.demo == "comprehensive":
            demo_comprehensive_visualization(dataset, output_dir)
        
        print(f"🎉 Demo complete! All plots saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 