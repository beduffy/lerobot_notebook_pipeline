#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate Policy with Real Images

This script loads a trained policy and feeds it real-world images from the dataset
to see how the policy behaves in simulation. This helps visualize policy "shakiness"
and stability when processing real images.

Key features:
- Loads trained ACT policy from checkpoint
- Feeds real-world dataset images to policy
- Executes predicted actions in MuJoCo simulation
- Visualizes action smoothness and stability
- Real-time plotting of action trajectories

Usage:
    python simulate_policy_with_real_images.py --policy-path ./single_episode_model --dataset bearlover365/red_cube_always_in_same_place
    python simulate_policy_with_real_images.py --policy-path ./single_episode_model --episode 0 --speed 0.5
"""

import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import threading
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    print("⚠️  MuJoCo not available - will use matplotlib visualization only")
    MUJOCO_AVAILABLE = False

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy


class PolicySimulator:
    """Simulates policy behavior with real images."""
    
    def __init__(self, policy_path, dataset_name, episode_idx=0, device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        # Load policy
        print(f"🤖 Loading policy from {policy_path}...")
        self.policy = ACTPolicy.from_pretrained(policy_path)
        self.policy.eval()
        self.policy.to(self.device)
        self.policy.reset()
        print("✅ Policy loaded!")
        
        # Load dataset
        print(f"📦 Loading dataset {dataset_name}...")
        self.dataset = LeRobotDataset(dataset_name, video_backend="pyav")
        print("✅ Dataset loaded!")
        
        # Get episode data
        self.episode_idx = episode_idx
        self._load_episode_data()
        
        # Initialize tracking
        self.current_step = 0
        self.action_history = deque(maxlen=100)  # Keep last 100 actions
        self.prediction_times = deque(maxlen=100)
        
        # Find image key and debug sample structure
        sample = self.dataset[self.episode_start_idx]
        print(f"🔍 Sample keys: {list(sample.keys())}")
        
        self.image_key = None
        for key in sample.keys():
            if "image" in key and isinstance(sample[key], torch.Tensor):
                self.image_key = key
                print(f"📸 Found image key: {key}, shape: {sample[key].shape}")
                break
        
        if not self.image_key:
            raise ValueError("No image key found in dataset!")
        
        # Debug other important keys
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
        
        print(f"📸 Using image key: {self.image_key}")
        
    def _load_episode_data(self):
        """Load episode boundaries."""
        from_idx = self.dataset.episode_data_index["from"][self.episode_idx].item()
        to_idx = self.dataset.episode_data_index["to"][self.episode_idx].item()
        self.episode_start_idx = from_idx
        self.episode_end_idx = to_idx
        self.episode_length = to_idx - from_idx
        print(f"📏 Episode {self.episode_idx}: {self.episode_length} steps")
    
    def get_current_image_and_prediction(self):
        """Get current real image and policy prediction."""
        if self.current_step >= self.episode_length:
            self.current_step = 0  # Loop back to start
        
        # Get real image from dataset
        dataset_idx = self.episode_start_idx + self.current_step
        sample = self.dataset[dataset_idx]
        
        # Prepare input for policy - ONLY OBSERVATIONS, NOT ACTIONS!
        batch = {}
        for key, value in sample.items():
            if key.startswith("observation.") and isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).to(self.device)
        
        # Get policy prediction
        start_time = time.time()
        with torch.no_grad():
            predicted_action = self.policy.select_action(batch)
        prediction_time = time.time() - start_time
        
        # Store for analysis
        self.action_history.append(predicted_action.cpu().numpy().flatten())
        self.prediction_times.append(prediction_time)
        
        # Get ground truth for comparison
        gt_action = sample["action"].numpy()
        
        self.current_step += 1
        
        return sample[self.image_key], predicted_action.cpu().numpy().flatten(), gt_action, prediction_time


class ActionVisualizer:
    """Real-time action visualization."""
    
    def __init__(self, action_dim=7):
        self.action_dim = action_dim
        self.fig, self.axes = plt.subplots(action_dim, 1, figsize=(12, 2*action_dim))
        if action_dim == 1:
            self.axes = [self.axes]
        
        self.action_history = deque(maxlen=50)
        self.gt_history = deque(maxlen=50)
        self.time_steps = deque(maxlen=50)
        self.step_count = 0
        
        # Joint names
        joint_names = [f'Joint {i+1}' for i in range(action_dim-1)] + ['Gripper'] if action_dim == 7 else [f'Dim {i}' for i in range(action_dim)]
        
        # Setup plots
        for i, ax in enumerate(self.axes):
            ax.set_title(f'{joint_names[i]} - Predicted vs Ground Truth')
            ax.set_ylabel('Action')
            ax.grid(True, alpha=0.3)
            ax.legend(['Predicted', 'Ground Truth'])
        
        plt.tight_layout()
        plt.ion()
        plt.show()
    
    def update(self, predicted_action, gt_action):
        """Update the real-time plot."""
        self.action_history.append(predicted_action)
        self.gt_history.append(gt_action)
        self.time_steps.append(self.step_count)
        self.step_count += 1
        
        # Clear and replot
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.grid(True, alpha=0.3)
            
            if len(self.action_history) > 1:
                pred_values = [action[i] for action in self.action_history]
                gt_values = [action[i] for action in self.gt_history]
                
                ax.plot(list(self.time_steps), pred_values, 'r-', linewidth=2, alpha=0.8, label='Predicted')
                ax.plot(list(self.time_steps), gt_values, 'b--', linewidth=2, alpha=0.6, label='Ground Truth')
                
                # Highlight recent prediction
                if len(pred_values) > 0:
                    ax.scatter([list(self.time_steps)[-1]], [pred_values[-1]], color='red', s=100, alpha=0.8, zorder=5)
                
                ax.legend()
            
            joint_names = [f'Joint {i+1}' for i in range(self.action_dim-1)] + ['Gripper'] if self.action_dim == 7 else [f'Dim {i}' for i in range(self.action_dim)]
            ax.set_title(f'{joint_names[i]} - Predicted vs Ground Truth')
            ax.set_ylabel('Action')
        
        plt.draw()
        plt.pause(0.01)


def analyze_policy_shakiness(action_history, prediction_times):
    """Analyze how shaky/smooth the policy predictions are."""
    if len(action_history) < 2:
        return {}
    
    actions = np.array(action_history)
    
    # Calculate action smoothness metrics
    action_derivatives = np.diff(actions, axis=0)
    action_smoothness = np.mean(np.abs(action_derivatives), axis=0)
    overall_smoothness = np.mean(action_smoothness)
    
    # Calculate prediction consistency
    action_std = np.std(actions, axis=0)
    overall_consistency = np.mean(action_std)
    
    # Calculate prediction timing
    avg_prediction_time = np.mean(prediction_times)
    
    return {
        'overall_smoothness': overall_smoothness,
        'overall_consistency': overall_consistency,
        'avg_prediction_time': avg_prediction_time,
        'joint_smoothness': action_smoothness,
        'joint_consistency': action_std,
        'num_samples': len(action_history)
    }


def run_simulation_with_real_images(policy_path, dataset_name, episode_idx=0, 
                                  speed=1.0, max_steps=None, visualize_actions=True):
    """Main simulation loop."""
    
    print("🎯 Starting Policy Simulation with Real Images")
    print("=" * 50)
    
    # Initialize simulator
    simulator = PolicySimulator(policy_path, dataset_name, episode_idx)
    
    # Initialize visualizer
    visualizer = None
    if visualize_actions:
        sample_action = simulator.get_current_image_and_prediction()[1]
        visualizer = ActionVisualizer(len(sample_action))
        # Reset to start
        simulator.current_step = 0
        simulator.policy.reset()
    
    # Run simulation
    max_steps = max_steps or simulator.episode_length
    print(f"🏃 Running simulation for {max_steps} steps...")
    print(f"⚡ Speed: {speed}x")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        for step in range(max_steps):
            # Get real image and policy prediction
            image, predicted_action, gt_action, pred_time = simulator.get_current_image_and_prediction()
            
            # Update visualization
            if visualizer:
                visualizer.update(predicted_action, gt_action)
            
            # Print progress
            if step % 10 == 0:
                mae = np.mean(np.abs(predicted_action - gt_action))
                print(f"Step {step:3d}: MAE={mae:.4f}, Pred_time={pred_time*1000:.1f}ms")
            
            # Control simulation speed
            time.sleep(1.0 / (30 * speed))  # 30 FPS base rate
    
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    
    # Analyze results
    print("\n📊 Policy Shakiness Analysis:")
    print("-" * 30)
    
    metrics = analyze_policy_shakiness(simulator.action_history, simulator.prediction_times)
    
    if metrics:
        print(f"🎯 Overall Smoothness: {metrics['overall_smoothness']:.4f}")
        print(f"🎯 Overall Consistency: {metrics['overall_consistency']:.4f}")
        print(f"⚡ Avg Prediction Time: {metrics['avg_prediction_time']*1000:.1f}ms")
        print(f"📊 Samples Analyzed: {metrics['num_samples']}")
        
        print(f"\n🔧 Per-Joint Analysis:")
        joint_names = [f'Joint {i+1}' for i in range(len(metrics['joint_smoothness'])-1)] + ['Gripper']
        for i, (smoothness, consistency) in enumerate(zip(metrics['joint_smoothness'], metrics['joint_consistency'])):
            print(f"   {joint_names[i]}: Smoothness={smoothness:.4f}, Consistency={consistency:.4f}")
        
        # Performance assessment
        print(f"\n💡 Assessment:")
        if metrics['overall_smoothness'] < 0.01:
            print("   ✅ Very smooth policy - actions change gradually")
        elif metrics['overall_smoothness'] < 0.1:
            print("   ✅ Reasonably smooth policy")
        elif metrics['overall_smoothness'] < 0.5:
            print("   ⚠️  Somewhat shaky policy - noticeable action jumps")
        else:
            print("   ❌ Very shaky policy - large action changes between steps")
        
        if metrics['avg_prediction_time'] < 0.01:
            print("   ⚡ Very fast inference (<10ms)")
        elif metrics['avg_prediction_time'] < 0.05:
            print("   ✅ Fast inference (<50ms)")
        else:
            print("   ⚠️  Slow inference - may need optimization")
    
    print("\n🎉 Simulation complete!")
    
    if visualizer:
        input("\nPress Enter to close visualization...")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Simulate policy with real-world images')
    parser.add_argument('--policy-path', type=str, default='./single_episode_model',
                       help='Path to trained policy checkpoint')
    parser.add_argument('--dataset', type=str, default='bearlover365/red_cube_always_in_same_place',
                       help='Dataset name')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index to use for images')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Simulation speed multiplier')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum steps to simulate (default: full episode)')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Disable real-time action visualization')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    if not Path(args.policy_path).exists():
        print(f"❌ Policy path does not exist: {args.policy_path}")
        return 1
    
    try:
        run_simulation_with_real_images(
            policy_path=args.policy_path,
            dataset_name=args.dataset,
            episode_idx=args.episode,
            speed=args.speed,
            max_steps=args.max_steps,
            visualize_actions=not args.no_visualization
        )
        return 0
    
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 