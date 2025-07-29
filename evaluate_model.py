#!/usr/bin/env python3
"""
Simple Model Evaluation Script

Evaluates a trained ACT model on dataset episodes with visualization.

Usage:
    python evaluate_model.py ./single_episode_model --episode 0
    python evaluate_model.py path/to/model --episode 1 --compare-episodes 0,1,2
    python evaluate_model.py ./single_episode_model --episode 0 --plot --save-plots
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader, Subset


class EpisodeSampler(torch.utils.data.Sampler):
    """Custom sampler to iterate through a specific episode."""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def get_episode_indices(dataset, episode_idx):
    """Get indices for a specific episode."""
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    return list(range(from_idx, to_idx))


def evaluate_model_on_episode(model_path, dataset_name, episode_idx, device, use_dataloader=True):
    """Evaluate model on a specific episode and return predictions/ground truth for plotting."""
    print(f"🔬 Evaluating model on episode {episode_idx}...")
    
    # Load model
    policy = ACTPolicy.from_pretrained(model_path)
    policy.to(device)
    policy.eval()
    policy.reset()
    
    # Load dataset  
    dataset = LeRobotDataset(dataset_name, video_backend="pyav")
    
    if use_dataloader:
        # Use the more robust EpisodeSampler approach
        episode_sampler = EpisodeSampler(dataset, episode_idx)
        test_dataloader = DataLoader(
            dataset,
            num_workers=4,
            batch_size=1,
            shuffle=False,
            pin_memory=device.type != "cpu",
            sampler=episode_sampler,
        )
        
        print(f"   Episode {episode_idx}: {len(episode_sampler)} steps")
        
        predictions = []
        ground_truths = []
        images = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                try:
                    # Prepare input - ONLY OBSERVATIONS FOR INFERENCE
                    inp_batch = {}
                    for key, value in batch.items():
                        if key.startswith("observation.") and isinstance(value, torch.Tensor):
                            inp_batch[key] = value.to(device)
                    
                    # Get prediction
                    pred_action_chunk = policy.select_action(inp_batch)
                    
                    # Handle both chunked and non-chunked outputs
                    if pred_action_chunk.dim() == 3:  # [batch, chunk_size, action_dim]
                        pred_action = pred_action_chunk[0, 0, :]  # Take first action from chunk
                    else:  # [batch, action_dim]
                        pred_action = pred_action_chunk[0, :]  # Take first action
                    
                    gt_action = batch["action"][0]  # Ground truth action from batch
                    
                    # Ensure both tensors are 1D before appending
                    if pred_action.dim() > 1:
                        pred_action = pred_action.squeeze()
                    if gt_action.dim() > 1:
                        gt_action = gt_action.squeeze()
                    
                    predictions.append(pred_action.cpu())
                    ground_truths.append(gt_action.cpu())
                    
                    # Show progress for long episodes
                    if i % 50 == 0:
                        print(f"   Processed {i}/{len(episode_sampler)} steps...")
                        
                except Exception as e:
                    print(f"   Warning: Skipped step {i}: {e}")
                    continue
        
    else:
        # Fallback to original approach
        episode_indices = get_episode_indices(dataset, episode_idx)
        print(f"   Episode {episode_idx}: {len(episode_indices)} steps")
        
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for idx in episode_indices:
                try:
                    sample = dataset[idx]
                    
                    # Prepare input - ONLY OBSERVATIONS FOR INFERENCE
                    batch = {}
                    for key, value in sample.items():
                        if key.startswith("observation.") and isinstance(value, torch.Tensor):
                            batch[key] = value.unsqueeze(0).to(device)
                    
                    # Get prediction
                    pred_action_chunk = policy.select_action(batch)
                    
                    # Handle both chunked and non-chunked outputs
                    if pred_action_chunk.dim() == 3:  # [batch, chunk_size, action_dim]
                        pred_action = pred_action_chunk[0, 0, :]  # Take first action from chunk
                    else:  # [batch, action_dim]
                        pred_action = pred_action_chunk[0, :]  # Take first action
                    
                    gt_action = sample["action"]  # Ground truth action
                    
                    # Ensure both tensors are 1D before appending
                    if pred_action.dim() > 1:
                        pred_action = pred_action.squeeze()
                    if gt_action.dim() > 1:
                        gt_action = gt_action.squeeze()
                    
                    predictions.append(pred_action.cpu())
                    ground_truths.append(gt_action.cpu())
                    
                except Exception as e:
                    print(f"   Warning: Skipped step {idx}: {e}")
                    continue
    
    if predictions:
        predictions = torch.stack(predictions, dim=0)  # Stack to create [num_steps, action_dim]
        ground_truths = torch.stack(ground_truths, dim=0)  # Stack to create [num_steps, action_dim]
        
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
        
        # Per-joint analysis
        action_dim = predictions.shape[1]
        joint_names = [f'Joint {i+1}' for i in range(action_dim-1)] + ['Gripper'] if action_dim == 7 else [f'Action {i+1}' for i in range(action_dim)]
        print(f"\n🔧 Per-joint errors:")
        for i, joint_name in enumerate(joint_names):
            joint_error = torch.mean(torch.abs(predictions[:, i] - ground_truths[:, i])).item()
            print(f"      {joint_name}: {joint_error:.6f}")
            
        return mae, mse, max_error, predictions, ground_truths, joint_names
    else:
        print("   ❌ No valid predictions generated")
        return None, None, None, None, None, None


def plot_predictions_vs_ground_truth(predictions, ground_truths, joint_names, episode_idx, save_plots=False, output_dir="./plots"):
    """Create detailed plots comparing predictions vs ground truth."""
    print(f"📈 Creating plots for episode {episode_idx}...")
    
    action_dim = predictions.shape[1]
    
    # Create output directory if saving plots
    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)
    
    # Create subplots for each joint
    fig, axes = plt.subplots(action_dim, 1, figsize=(15, 3*action_dim))
    if action_dim == 1:
        axes = [axes]

    for i in range(action_dim):
        gt_values = ground_truths[:, i].cpu().numpy()
        pred_values = predictions[:, i].cpu().numpy()
        
        axes[i].plot(gt_values, label='Ground Truth', linewidth=2, alpha=0.8, color='blue')
        axes[i].plot(pred_values, label='Predicted', linewidth=2, alpha=0.8, linestyle='--', color='red')
        axes[i].set_title(f'{joint_names[i]} - Predicted vs Ground Truth')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Action Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Calculate and show error for this joint
        joint_error = torch.mean(torch.abs(predictions[:, i] - ground_truths[:, i])).item()
        axes[i].text(0.02, 0.98, f'MAE: {joint_error:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.suptitle(f'🎯 Model Performance: Predicted vs Ground Truth Actions (Episode {episode_idx})', 
                 fontsize=16, y=1.02)
    
    if save_plots:
        plot_path = Path(output_dir) / f"episode_{episode_idx}_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved plot to: {plot_path}")
    
    plt.show()
    
    # Create a summary error plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate per-timestep error
    per_step_error = torch.mean(torch.abs(predictions - ground_truths), dim=1).cpu().numpy()
    
    ax.plot(per_step_error, linewidth=2, color='orange')
    ax.set_title(f'Per-Timestep Mean Absolute Error (Episode {episode_idx})')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean Absolute Error')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line for overall MAE
    overall_mae = np.mean(per_step_error)
    ax.axhline(y=overall_mae, color='red', linestyle='--', alpha=0.7, label=f'Overall MAE: {overall_mae:.4f}')
    ax.legend()
    
    if save_plots:
        error_plot_path = Path(output_dir) / f"episode_{episode_idx}_error_over_time.png"
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved error plot to: {error_plot_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"📊 Summary Statistics for Episode {episode_idx}:")
    print(f"   Dataset contains {len(predictions)} action steps")
    print(f"   Overall Mean Absolute Error: {overall_mae:.4f}")
    best_joint_idx = torch.argmin(torch.mean(torch.abs(predictions - ground_truths), dim=0))
    worst_joint_idx = torch.argmax(torch.mean(torch.abs(predictions - ground_truths), dim=0))
    print(f"   Best performing joint: {joint_names[best_joint_idx]}")
    print(f"   Worst performing joint: {joint_names[worst_joint_idx]}")

    if overall_mae < 0.01:
        print(f"\n🎉 Excellent! The model has learned the demonstration very well.")
        print(f"   Next steps: Collect more diverse demonstrations for better generalization!")
    elif overall_mae < 0.1:
        print(f"\n✅ Good performance! The model learned the general trajectory well.")
    else:
        print(f"\n🔧 Consider: More training steps, different learning rate, or data quality issues.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ACT model")
    parser.add_argument("model_path", help="Path to trained model directory")
    parser.add_argument("--dataset", default="bearlover365/red_cube_always_in_same_place")
    parser.add_argument("--episode", type=int, default=0, help="Episode to evaluate on")
    parser.add_argument("--compare-episodes", help="Comma-separated episode indices to compare")
    parser.add_argument("--plot", action="store_true", help="Generate plots showing predictions vs ground truth")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to disk")
    parser.add_argument("--output-dir", default="./plots", help="Directory to save plots")
    parser.add_argument("--use-simple-loader", action="store_true", help="Use simple data loading instead of DataLoader")
    
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
                mae, mse, max_err, predictions, ground_truths, joint_names = evaluate_model_on_episode(
                    args.model_path, args.dataset, ep, device, use_dataloader=not args.use_simple_loader
                )
                if mae is not None:
                    results[ep] = {"mae": mae, "mse": mse, "max_error": max_err}
                    
                    # Plot if requested
                    if args.plot and predictions is not None:
                        plot_predictions_vs_ground_truth(
                            predictions, ground_truths, joint_names, ep, 
                            save_plots=args.save_plots, output_dir=args.output_dir
                        )
            
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
            mae, mse, max_err, predictions, ground_truths, joint_names = evaluate_model_on_episode(
                args.model_path, args.dataset, args.episode, device, use_dataloader=not args.use_simple_loader
            )
            
            # Plot if requested and we have data
            if args.plot and predictions is not None:
                plot_predictions_vs_ground_truth(
                    predictions, ground_truths, joint_names, args.episode,
                    save_plots=args.save_plots, output_dir=args.output_dir
                )
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 