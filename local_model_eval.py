#!/usr/bin/env python3
"""
Simple Local Model Evaluation Script

Evaluates ACT models on dataset episodes using local model paths (no HF uploads required).

Usage:
    # Basic evaluation
    python local_model_eval.py ./models/red_cube_experiments/red_cube_40k_steps_10_episodes

    # Compare multiple models
    python local_model_eval.py ./single_episode_model ./models/red_cube_experiments/red_cube_40k_steps_10_episodes --compare

    # With plots and save results
    python local_model_eval.py ./models/red_cube_experiments/red_cube_40k_steps_10_episodes --plot --save-results
"""

import argparse
import torch
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import json

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy


class LocalModelEvaluator:
    """Local model evaluation without HF dependencies."""
    
    def __init__(self, model_path, dataset_name="bearlover365/red_cube_always_in_same_place", device="auto"):
        self.model_path = Path(model_path)
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        print(f"🔧 Loading: {self.model_path.name}")
        
        # Load model and dataset
        self._load_model()
        self._load_dataset()
        
    def _load_model(self):
        """Load the ACT policy from local path."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        
        try:
            self.policy = ACTPolicy.from_pretrained(str(self.model_path))
        except Exception as e:
            print(f"   Failed to load model: {e}")
            raise
            
        self.policy.to(self.device)
        self.policy.eval()
        
    def _load_dataset(self):
        """Load the dataset."""
        self.dataset = LeRobotDataset(self.dataset_name, video_backend="pyav")
        
    def evaluate_on_episode(self, episode_idx=0, max_steps=None):
        """Evaluate model on a specific dataset episode."""
        # Get episode boundaries
        from_idx = self.dataset.episode_data_index["from"][episode_idx].item()
        to_idx = self.dataset.episode_data_index["to"][episode_idx].item()
        episode_length = to_idx - from_idx
        
        if max_steps:
            episode_length = min(episode_length, max_steps)
            to_idx = from_idx + episode_length
            
        print(f"   Episode {episode_idx}: {episode_length} steps")
        
        predictions = []
        ground_truths = []
        prediction_times = []
        
        self.policy.reset()
        
        with torch.no_grad():
            for step, idx in enumerate(range(from_idx, to_idx)):
                try:
                    sample = self.dataset[idx]
                    
                    # Prepare input - ONLY OBSERVATIONS
                    batch = {}
                    for key, value in sample.items():
                        if key.startswith("observation.") and isinstance(value, torch.Tensor):
                            batch[key] = value.unsqueeze(0).to(self.device)
                    
                    # Get prediction with timing
                    start_time = time.time()
                    pred_action_chunk = self.policy.select_action(batch)
                    pred_time = time.time() - start_time
                    
                    # Handle chunked output
                    if pred_action_chunk.dim() == 3:  # [batch, chunk_size, action_dim]
                        pred_action = pred_action_chunk[0, 0, :]  # Take first action from chunk
                    else:  # [batch, action_dim]
                        pred_action = pred_action_chunk[0, :]
                        
                    gt_action = sample["action"]
                    
                    # Store results
                    predictions.append(pred_action.cpu())
                    ground_truths.append(gt_action)
                    prediction_times.append(pred_time)
                    
                    # Progress update
                    if step % 50 == 0:
                        print(f"   Processed {step}/{episode_length} steps...")
                        
                except Exception as e:
                    print(f"   Warning: Skipped step {step}: {e}")
                    continue
        
        if not predictions:
            print("   ❌ No valid predictions generated")
            return None
            
        # Calculate metrics
        predictions = torch.stack(predictions, dim=0)
        ground_truths = torch.stack(ground_truths, dim=0)
        
        mae = torch.mean(torch.abs(predictions - ground_truths)).item()
        mse = torch.mean((predictions - ground_truths) ** 2).item()
        max_error = torch.max(torch.abs(predictions - ground_truths)).item()
        avg_pred_time = np.mean(prediction_times)
        
        results = {
            'model_name': self.model_path.name,
            'episode': episode_idx,
            'steps': len(predictions),
            'mae': mae,
            'mse': mse,
            'max_error': max_error,
            'avg_prediction_time': avg_pred_time,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
        
        # Print results
        print(f"   MAE: {mae:.6f}")
        print(f"   MSE: {mse:.6f}")
        print(f"   Max Error: {max_error:.6f}")
        print(f"   Prediction Time: {avg_pred_time*1000:.1f}ms")
        
        if mae < 0.01:
            print("   🎉 Excellent!")
        elif mae < 0.1:
            print("   ✅ Good")
        elif mae < 1.0:
            print("   ⚠️  Moderate")
        else:
            print("   ❌ Poor")
            
        return results


def plot_comparison(results_list, save_path=None):
    """Plot comparison between multiple models."""
    if not results_list or len(results_list) < 2:
        return
        
    model_names = [r['model_name'] for r in results_list if r]
    maes = [r['mae'] for r in results_list if r]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE comparison
    bars = ax1.bar(model_names, maes, alpha=0.7, color='skyblue')
    ax1.set_title('📊 Model Comparison: Mean Absolute Error')
    ax1.set_ylabel('MAE')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.6f}', ha='center', va='bottom')
    
    # Performance summary
    ax2.axis('off')
    
    summary_text = "🏆 Performance Ranking\n\n"
    sorted_results = sorted(results_list, key=lambda x: x['mae'] if x else float('inf'))
    
    for i, result in enumerate(sorted_results):
        if result:
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            summary_text += f"{rank_emoji} {result['model_name']}\n"
            summary_text += f"   MAE: {result['mae']:.6f}\n"
            summary_text += f"   Steps: {result['steps']}\n"
            summary_text += f"   Pred Time: {result['avg_prediction_time']*1000:.1f}ms\n\n"
    
    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Simple Local Model Evaluation (No HF Required!)')
    parser.add_argument('model_paths', nargs='+', help='Path(s) to trained model directories')
    parser.add_argument('--dataset', default='bearlover365/red_cube_always_in_same_place', help='Dataset name')
    parser.add_argument('--episode', type=int, default=0, help='Episode to evaluate on')
    parser.add_argument('--steps', type=int, default=None, help='Max steps for evaluation (default: full episode)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--save-results', action='store_true', help='Save results to JSON')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    print("🚀 Simple Local Model Evaluation")
    print("=" * 40)
    print("✅ No HuggingFace uploads required!")
    print("✅ Works with local model directories!")
    print()
    
    # Validate model paths
    for model_path in args.model_paths:
        if not Path(model_path).exists():
            print(f"❌ Model path does not exist: {model_path}")
            return 1
    
    results = []
    
    try:
        for i, model_path in enumerate(args.model_paths):
            print(f"\n[{i+1}/{len(args.model_paths)}] Evaluating: {Path(model_path).name}")
            print("-" * 40)
            
            evaluator = LocalModelEvaluator(model_path, args.dataset, args.device)
            result = evaluator.evaluate_on_episode(args.episode, args.steps)
            results.append(result)
        
        # Comparison summary
        if args.compare and len(args.model_paths) > 1:
            print(f"\n🏁 COMPARISON SUMMARY")
            print("=" * 40)
            
            valid_results = [r for r in results if r]
            if valid_results:
                # Sort by MAE (best first)
                sorted_results = sorted(valid_results, key=lambda x: x['mae'])
                
                print(f"{'Rank':<6} {'Model':<30} {'MAE':<12} {'Steps':<8}")
                print("-" * 60)
                
                for i, result in enumerate(sorted_results):
                    rank = f"#{i+1}"
                    name = result['model_name'][:28]  # Truncate long names
                    mae = f"{result['mae']:.6f}"
                    steps = result['steps']
                    
                    print(f"{rank:<6} {name:<30} {mae:<12} {steps:<8}")
                
                # Winner announcement
                winner = sorted_results[0]
                print(f"\n🏆 Winner: {winner['model_name']} (MAE: {winner['mae']:.6f})")
                
                if len(sorted_results) > 1:
                    improvement = (sorted_results[-1]['mae'] - winner['mae']) / sorted_results[-1]['mae'] * 100
                    print(f"   {improvement:.1f}% better than worst model")
        
        # Generate plots
        if args.plot and len(args.model_paths) > 1:
            valid_results = [r for r in results if r]
            if valid_results:
                save_path = "model_comparison.png" if args.save_results else None
                plot_comparison(valid_results, save_path)
        
        # Save results
        if args.save_results:
            output_data = {
                'evaluation_config': {
                    'dataset': args.dataset,
                    'episode': args.episode,
                    'steps': args.steps,
                    'device': str(args.device)
                },
                'results': []
            }
            
            for result in results:
                if result:
                    # Convert tensors to lists for JSON serialization
                    result_copy = result.copy()
                    if 'predictions' in result_copy:
                        result_copy['predictions'] = result_copy['predictions'].tolist()
                    if 'ground_truths' in result_copy:
                        result_copy['ground_truths'] = result_copy['ground_truths'].tolist()
                    output_data['results'].append(result_copy)
            
            output_file = "evaluation_results.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
        
        print(f"\n🎉 Evaluation complete!")
        print(f"\n💡 For robot simulation, use: python simulate_robot_control.py --policy-path <model_path>")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 