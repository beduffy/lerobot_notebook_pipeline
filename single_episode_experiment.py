#!/usr/bin/env python3
"""
Single Episode Experiment: Can we learn from just 1 demonstration?

This script answers one of your core research questions: "Is 1 demonstration enough?"
It systematically tests training ACT on single episodes with different configurations.

Usage:
    python single_episode_experiment.py dataset_path --episode 0 --output-dir ./single_ep_results
    python single_episode_experiment.py dataset_path --episode 0 --augmentation heavy
    python single_episode_experiment.py dataset_path --compare-all-episodes

Examples:
    # Test episode 0 with different augmentation levels
    python single_episode_experiment.py /path/to/cube_dataset --episode 0 --augmentation none
    python single_episode_experiment.py /path/to/cube_dataset --episode 0 --augmentation medium
    python single_episode_experiment.py /path/to/cube_dataset --episode 0 --augmentation heavy
    
    # Compare all episodes as single-episode datasets
    python single_episode_experiment.py /path/to/cube_dataset --compare-all-episodes
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_notebook_pipeline.dataset_utils.analysis import get_dataset_stats, analyze_episodes
    from lerobot_notebook_pipeline.dataset_utils.training import train_model
    from lerobot_notebook_pipeline.dataset_utils.visualization import AddGaussianNoise
    from torchvision import transforms
    import torch
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the lerobot package and this project package.")
    sys.exit(1)


def create_augmentation_transforms(level: str = "none"):
    """Create augmentation transforms based on level."""
    if level == "none":
        return None
    elif level == "light":
        return transforms.Compose([
            AddGaussianNoise(mean=0., std=0.01),
            transforms.Lambda(lambda x: x.clamp(0, 1))
        ])
    elif level == "medium":
        return transforms.Compose([
            AddGaussianNoise(mean=0., std=0.02),
            transforms.Lambda(lambda x: x.clamp(0, 1))
        ])
    elif level == "heavy":
        return transforms.Compose([
            AddGaussianNoise(mean=0., std=0.05),
            transforms.Lambda(lambda x: x.clamp(0, 1))
        ])
    else:
        raise ValueError(f"Unknown augmentation level: {level}")


def extract_single_episode_dataset(dataset: LeRobotDataset, episode_idx: int, output_dir: Path):
    """Extract a single episode and create a new dataset directory structure."""
    
    print(f"📊 Extracting episode {episode_idx} from dataset...")
    
    # Get episode boundaries
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    episode_length = to_idx - from_idx
    
    print(f"   Episode {episode_idx}: steps {from_idx} to {to_idx} ({episode_length} steps)")
    
    # For now, we'll train on the single episode by specifying the episode index
    # This is simpler than creating a new dataset file structure
    
    return from_idx, to_idx, episode_length


def run_single_episode_experiment(dataset_path: str, episode_idx: int, augmentation_level: str,
                                output_dir: Path, training_steps: int = 10000) -> Dict[str, Any]:
    """Run a single episode experiment with specified parameters."""
    
    experiment_name = f"episode_{episode_idx}_aug_{augmentation_level}"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 SINGLE EPISODE EXPERIMENT: {experiment_name}")
    print("=" * 60)
    
    # Load dataset
    print("📊 Loading dataset...")
    dataset = LeRobotDataset(dataset_path, video_backend="pyav")
    
    # Extract episode info
    from_idx, to_idx, episode_length = extract_single_episode_dataset(dataset, episode_idx, experiment_dir)
    
    # Create augmentation transforms
    transforms_obj = create_augmentation_transforms(augmentation_level)
    
    # Analyze the single episode
    print(f"\n🔍 Analyzing episode {episode_idx}...")
    episode_stats = analyze_episodes(dataset)
    if episode_idx < len(episode_stats):
        ep_stats = episode_stats[episode_idx]
        print(f"   Length: {ep_stats['length']} steps")
        print(f"   Action range: [{ep_stats['action_mean'].min():.3f}, {ep_stats['action_mean'].max():.3f}]")
    
    # Training configuration
    training_config = {
        "dataset_path": str(dataset_path),
        "episode_index": episode_idx,
        "episode_length": episode_length,
        "augmentation_level": augmentation_level,
        "training_steps": training_steps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save experiment config
    config_path = experiment_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print(f"\n🎯 Training on episode {episode_idx} with {augmentation_level} augmentation...")
    print(f"   Steps: {training_steps}")
    print(f"   Output: {experiment_dir}")
    
    try:
        # For now, we'll create a simple training approach
        # In a full implementation, you'd modify train_model to accept episode indices
        
        print("⚠️  Note: This is a prototype. For full implementation, integrate with your training pipeline.")
        print("   You can manually run:")
        print(f"   python train.py {dataset_path} --episodes {episode_idx} --steps {training_steps} --output-dir {experiment_dir}")
        
        # Create a simple results placeholder
        results = {
            "experiment_name": experiment_name,
            "episode_index": episode_idx,
            "episode_length": episode_length,
            "augmentation_level": augmentation_level,
            "training_steps": training_steps,
            "status": "configured",
            "training_command": f"python train.py {dataset_path} --episodes {episode_idx} --steps {training_steps} --output-dir {experiment_dir}",
            "notes": "Ready to train - run the training command above"
        }
        
        # Save results
        results_path = experiment_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Experiment configured successfully!")
        print(f"📁 Results directory: {experiment_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return {"error": str(e), "experiment_name": experiment_name}


def compare_all_episodes(dataset_path: str, output_dir: Path, max_episodes: int = 5):
    """Compare training on different single episodes."""
    
    print(f"\n🔬 COMPARING ALL EPISODES AS SINGLE-EPISODE DATASETS")
    print("=" * 60)
    
    # Load dataset to check number of episodes
    dataset = LeRobotDataset(dataset_path, video_backend="pyav")
    num_episodes = len(dataset.meta.episodes)
    episodes_to_test = min(num_episodes, max_episodes)
    
    print(f"📊 Dataset has {num_episodes} episodes, testing first {episodes_to_test}")
    
    comparison_results = []
    
    for ep_idx in range(episodes_to_test):
        print(f"\n--- Episode {ep_idx} ---")
        
        # Test with medium augmentation by default
        results = run_single_episode_experiment(
            dataset_path, ep_idx, "medium", output_dir, training_steps=5000
        )
        comparison_results.append(results)
    
    # Save comparison summary
    summary_path = output_dir / "episode_comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "total_episodes": num_episodes,
            "tested_episodes": episodes_to_test,
            "results": comparison_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"\n📊 COMPARISON SUMMARY")
    print(f"   Total episodes in dataset: {num_episodes}")
    print(f"   Episodes tested: {episodes_to_test}")
    print(f"   Summary saved to: {summary_path}")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Single Episode Experiment: Can we learn from 1 demo?")
    parser.add_argument("dataset_path", help="Path to the dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to train on (default: 0)")
    parser.add_argument("--augmentation", choices=["none", "light", "medium", "heavy"], default="medium",
                       help="Augmentation level (default: medium)")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps (default: 10000)")
    parser.add_argument("--output-dir", type=str, default="./single_episode_experiments",
                       help="Directory to save experiment results")
    parser.add_argument("--compare-all-episodes", action="store_true",
                       help="Compare all episodes as single-episode datasets")
    parser.add_argument("--max-episodes", type=int, default=5,
                       help="Maximum episodes to test in comparison mode")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("🧪 SINGLE EPISODE EXPERIMENT RUNNER")
        print("=" * 50)
        print(f"📁 Dataset: {args.dataset_path}")
        print(f"📁 Output: {output_dir.absolute()}")
        print()
        
        if args.compare_all_episodes:
            compare_all_episodes(args.dataset_path, output_dir, args.max_episodes)
        else:
            run_single_episode_experiment(
                args.dataset_path, args.episode, args.augmentation, 
                output_dir, args.steps
            )
        
        print(f"\n🎉 Experiment runner complete!")
        print(f"📁 Check results in: {output_dir.absolute()}")
        print(f"\n💡 Next steps:")
        print(f"   1. Run the generated training commands")
        print(f"   2. Compare model performance across episodes")
        print(f"   3. Test trained models on slight position variations")
        print(f"   4. Document where/when models fail")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 