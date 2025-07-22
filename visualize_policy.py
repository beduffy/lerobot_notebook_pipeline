#!/usr/bin/env python3
"""
Policy Visualization Script

This script loads a trained policy and visualizes its performance including:
- Predicted vs ground truth action comparison
- Episode rollout visualization  
- Policy inference analysis
- Error metrics and joint-specific performance

Usage:
    python visualize_policy.py --policy-path "./ckpt/act_y" --dataset "bearlover365/red_cube_always_in_same_place"
    python visualize_policy.py --policy-path "./ckpt/act_y" --dataset "my_dataset" --root "./data" --episode 0
"""

import argparse
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot_notebook_pipeline.dataset_utils.visualization import create_training_animation


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def load_policy(policy_path: str, device: torch.device):
    """Load a trained policy from disk."""
    print(f"🤖 Loading policy from {policy_path}...")
    try:
        policy = ACTPolicy.from_pretrained(policy_path)
        policy.eval()
        policy.to(device)
        print("✅ Policy loaded successfully!")
        return policy
    except Exception as e:
        print(f"❌ Error loading policy: {e}")
        raise


def evaluate_policy_on_episode(policy, dataset, episode_idx, device):
    """Evaluate policy performance on a specific episode."""
    print(f"🔬 Evaluating policy on episode {episode_idx}...")
    
    episode_sampler = EpisodeSampler(dataset, episode_idx)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type != "cpu",
        sampler=episode_sampler,
    )
    
    print(f"📏 Episode length: {len(episode_sampler)} steps")
    
    policy.reset()
    actions = []
    gt_actions = []
    images = []
    
    for i, batch in enumerate(test_dataloader):
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        action = policy.select_action(inp_batch)
        actions.append(action)
        gt_actions.append(inp_batch["action"][:,0,:])
        
        # Store first image for each step
        image_key = None
        for key in inp_batch.keys():
            if "image" in key:
                image_key = key
                break
        if image_key:
            images.append(inp_batch[image_key])
        
        # Show progress for long episodes
        if i % 50 == 0:
            print(f"   Processed {i}/{len(episode_sampler)} steps...")
    
    actions = torch.cat(actions, dim=0)
    gt_actions = torch.cat(gt_actions, dim=0)
    
    return actions, gt_actions, images


def analyze_policy_performance(actions, gt_actions):
    """Analyze policy performance metrics."""
    print("\n📊 Policy Performance Analysis:")
    
    # Calculate detailed error metrics
    abs_errors = torch.abs(actions - gt_actions)
    mean_abs_error = torch.mean(abs_errors).item()
    max_abs_error = torch.max(abs_errors).item()
    mse = torch.mean((actions - gt_actions)**2).item()
    rmse = torch.sqrt(torch.mean((actions - gt_actions)**2)).item()
    
    print(f"   Mean Absolute Error: {mean_abs_error:.4f}")
    print(f"   Root Mean Square Error: {rmse:.4f}")
    print(f"   Max Absolute Error: {max_abs_error:.4f}")
    
    # Per-joint analysis
    action_dim = actions.shape[1]
    joint_names = [f'Joint {i+1}' for i in range(action_dim-1)] + ['Gripper'] if action_dim == 7 else [f'Dim {i}' for i in range(action_dim)]
    
    print(f"\n🔧 Per-joint Performance:")
    joint_errors = []
    for i, joint_name in enumerate(joint_names):
        if i < actions.shape[1]:
            joint_error = torch.mean(abs_errors[:, i]).item()
            joint_errors.append(joint_error)
            print(f"   {joint_name}: MAE = {joint_error:.4f}")
    
    # Performance assessment
    print(f"\n💡 Performance Assessment:")
    if mean_abs_error < 0.01:
        print("   ✅ Excellent! Very low error - model has learned the demonstration well")
        print("   ⚠️  However, this may indicate overfitting to the training data")
    elif mean_abs_error < 0.1:
        print("   ✅ Good performance - model learned the general trajectory")
    elif mean_abs_error < 0.5:
        print("   ⚠️  Moderate performance - some deviation from demonstrations")
    else:
        print("   ❌ Poor performance - significant deviation from demonstrations")
    
    # Identify best and worst performing joints
    if len(joint_errors) > 1:
        best_joint_idx = np.argmin(joint_errors)
        worst_joint_idx = np.argmax(joint_errors)
        print(f"   🏆 Best performing joint: {joint_names[best_joint_idx]} (MAE: {joint_errors[best_joint_idx]:.4f})")
        print(f"   🔧 Worst performing joint: {joint_names[worst_joint_idx]} (MAE: {joint_errors[worst_joint_idx]:.4f})")
    
    return {
        'mae': mean_abs_error,
        'rmse': rmse,
        'max_error': max_abs_error,
        'joint_errors': joint_errors,
        'joint_names': joint_names
    }


def visualize_predictions(actions, gt_actions, metrics):
    """Create comprehensive visualization of policy predictions."""
    print("\n📈 Creating prediction visualizations...")
    
    action_dim = actions.shape[1]
    joint_names = metrics['joint_names']
    
    # Create subplots for each joint
    fig, axes = plt.subplots(action_dim, 1, figsize=(15, 3*action_dim))
    if action_dim == 1:
        axes = [axes]
    
    for i in range(action_dim):
        # Plot ground truth and predictions
        axes[i].plot(gt_actions[:, i].cpu().numpy(), label='Ground Truth', 
                    linewidth=2, alpha=0.8, color='blue')
        axes[i].plot(actions[:, i].cpu().numpy(), label='Predicted', 
                    linewidth=2, alpha=0.8, linestyle='--', color='red')
        
        axes[i].set_title(f'{joint_names[i]} - Predicted vs Ground Truth')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Action Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add error information
        joint_mae = metrics['joint_errors'][i]
        axes[i].text(0.02, 0.98, f'MAE: {joint_mae:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Highlight areas of high error
        errors = torch.abs(actions[:, i] - gt_actions[:, i]).cpu().numpy()
        high_error_threshold = np.percentile(errors, 90)  # Top 10% errors
        high_error_indices = np.where(errors > high_error_threshold)[0]
        
        if len(high_error_indices) > 0:
            axes[i].scatter(high_error_indices, actions[high_error_indices, i].cpu().numpy(), 
                          color='orange', s=50, alpha=0.7, marker='x', 
                          label=f'High Error (>{high_error_threshold:.3f})')
            axes[i].legend()
    
    plt.tight_layout()
    plt.suptitle('🎯 Policy Performance: Predicted vs Ground Truth Actions', 
                 fontsize=16, y=1.02)
    plt.show()
    
    # Error distribution plot
    print("\n📊 Error Distribution Analysis:")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall error distribution
    all_errors = torch.abs(actions - gt_actions).flatten().cpu().numpy()
    ax1.hist(all_errors, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_title('Overall Error Distribution')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.mean(all_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_errors):.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Per-joint error comparison
    joint_errors = [torch.abs(actions[:, i] - gt_actions[:, i]).mean().item() 
                   for i in range(action_dim)]
    bars = ax2.bar(range(action_dim), joint_errors, alpha=0.7)
    ax2.set_title('Per-Joint Error Comparison')
    ax2.set_xlabel('Joint')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_xticks(range(action_dim))
    ax2.set_xticklabels([name[:8] for name in joint_names], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Color bars by performance
    for i, (bar, error) in enumerate(zip(bars, joint_errors)):
        if error < np.mean(joint_errors):
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize trained policy performance')
    parser.add_argument('--policy-path', type=str, required=True,
                       help='Path to the trained policy directory')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name or path')
    parser.add_argument('--root', type=str, default=None,
                       help='Root directory for local datasets')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index to evaluate on')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--video-backend', type=str, default='pyav',
                       help='Video backend to use (pyav or cv2)')
    parser.add_argument('--show-episode-animation', action='store_true',
                       help='Show episode animation with actions')
    
    args = parser.parse_args()
    
    print("🤖 Policy Visualization Tool")
    print("=" * 40)
    print(f"Policy: {args.policy_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Episode: {args.episode}")
    if args.root:
        print(f"Root: {args.root}")
    print()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🔧 Using device: {device}")
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
        
        # Validate episode index
        num_episodes = len(dataset.meta.episodes)
        if args.episode >= num_episodes:
            print(f"❌ Episode {args.episode} not found. Dataset has {num_episodes} episodes (0-{num_episodes-1})")
            sys.exit(1)
        
        # Load policy
        policy = load_policy(args.policy_path, device)
        print()
        
        # Evaluate policy on episode
        actions, gt_actions, images = evaluate_policy_on_episode(
            policy, dataset, args.episode, device
        )
        
        # Analyze performance
        metrics = analyze_policy_performance(actions, gt_actions)
        
        # Create visualizations
        visualize_predictions(actions, gt_actions, metrics)
        
        # Show episode animation if requested
        if args.show_episode_animation:
            print("\n🎬 Creating episode animation...")
            create_training_animation(dataset, args.episode)
        
        print(f"\n🎉 Policy evaluation complete!")
        print(f"   Overall MAE: {metrics['mae']:.4f}")
        print(f"   Overall RMSE: {metrics['rmse']:.4f}")
        print(f"   Check the plots above for detailed analysis.")
        
    except Exception as e:
        print(f"❌ Error during policy visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 