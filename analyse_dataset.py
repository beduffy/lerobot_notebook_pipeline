#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive dataset analysis script.

This script provides detailed analysis of robot demonstration datasets including:
- Basic dataset statistics
- Episode-by-episode analysis
- Action pattern analysis with timing breakdowns
- Visualization capabilities
- Overfitting risk assessment

Usage:
    python analyse_dataset.py <dataset_path> [options]
    python analyse_dataset.py --help

Performance Options:
    --fast              Use 10%% data sampling for faster analysis
    --sample-ratio 0.2  Use 20%% data sampling (custom ratio)
    --skip-animation    Skip animation creation for faster execution
    --output-dir        Directory to save plots (default: current directory)

Examples:
    # Quick analysis with 10%% sampling
    python analyse_dataset.py /path/to/dataset --fast
    
    # Custom sampling ratio
    python analyse_dataset.py /path/to/dataset --sample-ratio 0.15
    
    # Full analysis with custom output directory  
    python analyse_dataset.py /path/to/dataset --output-dir ./analysis_results
"""

import argparse
import sys
import os
from pathlib import Path

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_notebook_pipeline.dataset_utils.analysis import (
        get_dataset_stats, analyze_episodes, analyze_overfitting_risk, 
        analyze_action_patterns, visualize_sample
    )
    from lerobot_notebook_pipeline.dataset_utils.visualization import (
        plot_all_action_histograms, visualize_episode_trajectory, 
        create_training_animation
    )
    from lerobot_notebook_pipeline.dataset_utils.analysis import compare_episodes
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the lerobot package and this project package.")
    print("Run: pip install -e . (from the project root)")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis of robot demonstration datasets")
    parser.add_argument("dataset_path", help="Path to the dataset")
    parser.add_argument("--root", default=".", help="Root directory for datasets (default: current dir)")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2], 
                       help="Episode indices to analyze in detail (default: 0 1 2)")
    parser.add_argument("--sample-idx", type=int, default=0, 
                       help="Sample index for detailed visualization (default: 0)")
    parser.add_argument("--video-backend", default="pyav", 
                       help="Video backend for dataset loading (default: pyav)")
    parser.add_argument("--fast", action="store_true",
                       help="Enable fast mode with 10%% data sampling for quicker analysis")
    parser.add_argument("--sample-ratio", type=float, default=1.0,
                       help="Fraction of data to sample for faster analysis (0.1 = 10%%, default: 1.0)")
    parser.add_argument("--skip-animation", action="store_true",
                       help="Skip animation creation for faster execution")
    parser.add_argument("--output-dir", type=str, default="data/plots",
                       help="Directory to save analysis plots (default: current directory)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("🤖 LeRobot Dataset Analysis Tool")
        print("=" * 50)
        print(f"📁 Dataset: {args.dataset_path}")
        print(f"📁 Root: {args.root}")
        print(f"💾 Output directory: {output_dir.absolute()}")
        print(f"🎬 Video backend: {args.video_backend}")
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

        # 1. Basic Statistics
        print("📈 BASIC DATASET STATISTICS")
        print("-" * 30)
        stats = get_dataset_stats(dataset)
        for key, stat_dict in stats.items():
            if isinstance(stat_dict, dict) and 'mean' in stat_dict:
                print(f"📊 {key}:")
                print(f"    mean: {stat_dict['mean']}")
                print(f"    std:  {stat_dict['std']}")
                print(f"    min:  {stat_dict['min']}")
                print(f"    max:  {stat_dict['max']}")
            print()
        
        # 2. Episode Analysis
        print("🔍 EPISODE-BY-EPISODE ANALYSIS")
        print("-" * 30)
        episode_analysis = analyze_episodes(dataset)
        print()
        
        # 3. Overfitting Risk Assessment
        print("⚠️  OVERFITTING RISK ASSESSMENT")
        print("-" * 30)
        analyze_overfitting_risk(dataset)
        print()
        
        # Determine sampling ratio
        sample_ratio = args.sample_ratio
        if args.fast:
            sample_ratio = 0.1
            print("🚀 Fast mode enabled - using 10% data sampling for quicker analysis")
        elif sample_ratio < 1.0:
            print(f"🎲 Using {sample_ratio*100:.1f}% data sampling")
        
        # 4. Action Pattern Analysis
        print("🎯 ACTION PATTERN ANALYSIS")
        print("-" * 30)
        analyze_action_patterns(dataset, sample_ratio=sample_ratio, 
                              save_path_prefix=str(output_dir / "action_patterns"))
        print()
        
        # 5. Action Histograms
        print("📊 ACTION DISTRIBUTIONS")
        print("-" * 30)
        plot_all_action_histograms(dataset, sample_ratio=sample_ratio,
                                  save_path=str(output_dir / "all_action_histograms.png"))
        print()
        
        # 6. Episode Comparison
        available_episodes = list(range(len(dataset.meta.episodes)))
        episodes_to_compare = [ep for ep in args.episodes if ep in available_episodes]
        
        if len(episodes_to_compare) > 1:
            print(f"📈 EPISODE COMPARISON ({episodes_to_compare})")
            print("-" * 30)
            compare_episodes(dataset, episodes_to_compare, 
                           save_path=str(output_dir / "episode_comparison.png"))
            print()
        
        # 7. Detailed Episode Visualization
        for ep_idx in episodes_to_compare[:2]:  # Only first 2 to avoid too many plots
            print(f"🎯 EPISODE {ep_idx} DETAILED ANALYSIS")
            print("-" * 30)
            visualize_episode_trajectory(dataset, ep_idx,
                                       save_path=str(output_dir / f"episode_{ep_idx}_trajectory.png"))
            print()
            
            if not args.skip_animation:
                create_training_animation(dataset, ep_idx,
                                        save_path=str(output_dir / f"episode_{ep_idx}_animation.png"))
                print()
        
        # 8. Sample Visualization
        print(f"🖼️  SAMPLE VISUALIZATION (Index {args.sample_idx})")
        print("-" * 30)
        visualize_sample(dataset, args.sample_idx,
                        save_path=str(output_dir / f"sample_{args.sample_idx}_visualization.png"))
        print()
        
        # 9. Summary and Recommendations
        print("💡 ANALYSIS SUMMARY & RECOMMENDATIONS")
        print("-" * 30)
        
        num_episodes = len(dataset.meta.episodes)
        num_steps = len(dataset)
        avg_episode_length = num_steps / num_episodes
        
        print(f"✅ Dataset loaded successfully with {num_episodes} episodes and {num_steps} steps")
        print(f"📏 Average episode length: {avg_episode_length:.1f} steps")
        
        # Training recommendations
        print(f"\n🎯 Training Recommendations:")
        if num_episodes < 5:
            print("   ⚠️  Consider collecting more episodes for better generalization")
        if num_episodes >= 10:
            print("   ✅ Good number of episodes for training")
        
        if avg_episode_length < 50:
            print("   ⚠️  Short episodes - may need longer demonstrations")
        elif avg_episode_length > 1000:
            print("   ⚠️  Very long episodes - consider chunking or subsampling")
        
        print("   💡 Monitor training/validation split performance")
        print("   💡 Use data augmentation to increase effective dataset size")
        print("   💡 Consider early stopping to prevent overfitting")
        
        print(f"\n🎉 Analysis complete! All plots saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 