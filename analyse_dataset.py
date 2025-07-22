#!/usr/bin/env python3
"""
Dataset Analysis Script

This script provides comprehensive analysis of LeRobot datasets including:
- Basic statistics and metadata
- Episode-by-episode analysis  
- Action pattern analysis
- Overfitting risk assessment
- Visualization of trajectories and distributions

Usage:
    python analyse_dataset.py --dataset "bearlover365/red_cube_always_in_same_place"
    python analyse_dataset.py --dataset "my_dataset" --root "./data"
"""

import argparse
import sys
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Import our utility functions
from lerobot_notebook_pipeline.dataset_utils.analysis import (
    get_dataset_stats, analyze_episodes, compare_episodes, 
    analyze_action_patterns, analyze_overfitting_risk, visualize_sample
)
from lerobot_notebook_pipeline.dataset_utils.visualization import (
    plot_all_action_histograms, visualize_episode_trajectory, 
    create_training_animation
)


def main():
    parser = argparse.ArgumentParser(description='Analyze LeRobot dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name or path')
    parser.add_argument('--root', type=str, default=None,
                       help='Root directory for local datasets')
    parser.add_argument('--episodes', type=int, nargs='+', default=[0, 1, 2],
                       help='Episode indices to analyze in detail')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--skip-animation', action='store_true',
                       help='Skip creating episode animations')
    parser.add_argument('--video-backend', type=str, default='pyav',
                       help='Video backend to use (pyav or cv2)')
    
    args = parser.parse_args()
    
    print("🔍 LeRobot Dataset Analysis Tool")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    if args.root:
        print(f"Root: {args.root}")
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
        
        # 1. Basic Statistics
        print("📊 BASIC DATASET STATISTICS")
        print("-" * 30)
        stats = get_dataset_stats(dataset)
        for key, value in stats.items():
            if key != "dataset_stats":
                print(f"  {key}: {value}")
        print()
        
        # 2. Normalization Statistics
        print("📈 NORMALIZATION STATISTICS")
        print("-" * 30)
        for key, stat_dict in stats["dataset_stats"].items():
            print(f"  {key}:")
            if 'mean' in stat_dict:
                print(f"    mean: {stat_dict['mean']}")
                print(f"    std:  {stat_dict['std']}")
            if 'min' in stat_dict:
                print(f"    min:  {stat_dict['min']}")
                print(f"    max:  {stat_dict['max']}")
            print()
        
        # 3. Episode Analysis
        print("🔍 EPISODE-BY-EPISODE ANALYSIS")
        print("-" * 30)
        episode_analysis = analyze_episodes(dataset)
        print()
        
        # 4. Overfitting Risk Assessment
        print("⚠️  OVERFITTING RISK ASSESSMENT")
        print("-" * 30)
        analyze_overfitting_risk(dataset)
        print()
        
        # 5. Action Pattern Analysis
        print("🎯 ACTION PATTERN ANALYSIS")
        print("-" * 30)
        analyze_action_patterns(dataset)
        print()
        
        # 6. Action Histograms
        print("📊 ACTION DISTRIBUTIONS")
        print("-" * 30)
        plot_all_action_histograms(dataset)
        print()
        
        # 7. Episode Comparison
        available_episodes = list(range(len(dataset.meta.episodes)))
        episodes_to_compare = [ep for ep in args.episodes if ep in available_episodes]
        
        if len(episodes_to_compare) > 1:
            print(f"📈 EPISODE COMPARISON ({episodes_to_compare})")
            print("-" * 30)
            compare_episodes(dataset, episodes_to_compare)
            print()
        
        # 8. Detailed Episode Visualization
        for ep_idx in episodes_to_compare[:2]:  # Only first 2 to avoid too many plots
            print(f"🎯 EPISODE {ep_idx} DETAILED ANALYSIS")
            print("-" * 30)
            visualize_episode_trajectory(dataset, ep_idx)
            print()
            
            if not args.skip_animation:
                create_training_animation(dataset, ep_idx)
                print()
        
        # 9. Sample Visualization
        print(f"🖼️  SAMPLE VISUALIZATION (Index {args.sample_idx})")
        print("-" * 30)
        visualize_sample(dataset, args.sample_idx)
        print()
        
        # 10. Summary and Recommendations
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
        
        print(f"\n🎉 Analysis complete! Check the plots above for detailed insights.")
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 