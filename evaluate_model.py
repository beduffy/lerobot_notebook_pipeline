#!/usr/bin/env python3
"""
Simple Model Evaluation Script

Evaluates a trained ACT model on dataset episodes.

Usage:
    python evaluate_model.py ./single_episode_model --episode 0
    python evaluate_model.py path/to/model --episode 1 --compare-episodes 0,1,2
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader, Subset


def get_episode_indices(dataset, episode_idx):
    """Get indices for a specific episode."""
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    return list(range(from_idx, to_idx))


def evaluate_model_on_episode(model_path, dataset_name, episode_idx, device):
    """Evaluate model on a specific episode."""
    print(f"🔬 Evaluating model on episode {episode_idx}...")
    
    # Load model
    policy = ACTPolicy.from_pretrained(model_path)
    policy.to(device)
    policy.eval()
    policy.reset()
    
    # Load dataset  
    dataset = LeRobotDataset(dataset_name, video_backend="opencv")
    episode_indices = get_episode_indices(dataset, episode_idx)
    
    print(f"   Episode {episode_idx}: {len(episode_indices)} steps")
    
    # Evaluate step by step
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for idx in episode_indices:
            try:
                sample = dataset[idx]
                
                # Prepare input
                batch = {}
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.unsqueeze(0).to(device)
                    else:
                        batch[key] = value
                
                # Get prediction
                pred_action = policy.select_action(batch)
                gt_action = sample["action"].to(device)
                
                predictions.append(pred_action.cpu())
                ground_truths.append(gt_action.cpu())
                
            except Exception as e:
                print(f"   Warning: Skipped step {idx}: {e}")
                continue
    
    if predictions:
        predictions = torch.cat(predictions, dim=0)
        ground_truths = torch.cat(ground_truths, dim=0)
        
        # Calculate metrics
        mae = torch.mean(torch.abs(predictions - ground_truths)).item()
        mse = torch.mean((predictions - ground_truths) ** 2).item()
        max_error = torch.max(torch.abs(predictions - ground_truths)).item()
        
        print(f"   📊 Results:")
        print(f"      Mean Absolute Error: {mae:.6f}")
        print(f"      Mean Squared Error: {mse:.6f}")
        print(f"      Max Absolute Error: {max_error:.6f}")
        
        if mae < 0.01:
            print("      🎉 Excellent performance!")
        elif mae < 0.1:
            print("      ✅ Good performance")
        elif mae < 1.0:
            print("      ⚠️  Moderate performance")
        else:
            print("      ❌ Poor performance")
            
        return mae, mse, max_error
    else:
        print("   ❌ No valid predictions generated")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ACT model")
    parser.add_argument("model_path", help="Path to trained model directory")
    parser.add_argument("--dataset", default="bearlover365/red_cube_always_in_same_place")
    parser.add_argument("--episode", type=int, default=0, help="Episode to evaluate on")
    parser.add_argument("--compare-episodes", help="Comma-separated episode indices to compare")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📂 Model: {args.model_path}")
    print(f"📊 Dataset: {args.dataset}")
    
    if not Path(args.model_path).exists():
        print(f"❌ Model path does not exist: {args.model_path}")
        return 1
    
    try:
        if args.compare_episodes:
            # Compare performance across multiple episodes
            episodes = [int(x.strip()) for x in args.compare_episodes.split(",")]
            print(f"\n🔄 Comparing performance across episodes: {episodes}")
            
            results = {}
            for ep in episodes:
                mae, mse, max_err = evaluate_model_on_episode(
                    args.model_path, args.dataset, ep, device
                )
                if mae is not None:
                    results[ep] = {"mae": mae, "mse": mse, "max_error": max_err}
            
            # Summary
            print(f"\n📈 Summary:")
            print(f"{'Episode':<8} {'MAE':<12} {'MSE':<12} {'Max Error':<12}")
            print("-" * 48)
            for ep, metrics in results.items():
                print(f"{ep:<8} {metrics['mae']:<12.6f} {metrics['mse']:<12.6f} {metrics['max_error']:<12.6f}")
            
            if results:
                best_ep = min(results.keys(), key=lambda x: results[x]["mae"])
                worst_ep = max(results.keys(), key=lambda x: results[x]["mae"])
                print(f"\n🏆 Best episode: {best_ep} (MAE: {results[best_ep]['mae']:.6f})")
                print(f"🤔 Worst episode: {worst_ep} (MAE: {results[worst_ep]['mae']:.6f})")
                
                if best_ep == args.episode:
                    print("✅ Model performs best on its training episode (expected)")
                else:
                    print("🤯 Model generalizes! Performs better on different episode")
        
        else:
            # Single episode evaluation
            evaluate_model_on_episode(args.model_path, args.dataset, args.episode, device)
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 