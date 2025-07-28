#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Shakiness Visualization Demo

# TODO graveyard, artificial noise... 

This script demonstrates how to analyze policy "shakiness" and stability
by simulating policy predictions (using ground truth + noise) since the
actual trained model has compatibility issues.

This shows you what the analysis would look like when your model is working.

Usage:
    python visualize_policy_shakiness_demo.py --dataset bearlover365/red_cube_always_in_same_place
    python visualize_policy_shakiness_demo.py --episode 0 --speed 0.5 --noise-level 0.1
"""

import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class PolicyShakinessTester:
    """Simulates and analyzes policy shakiness using realistic synthetic predictions."""
    
    def __init__(self, dataset_name, episode_idx=0, noise_level=0.05):
        print(f"📦 Loading dataset {dataset_name}...")
        self.dataset = LeRobotDataset(dataset_name, video_backend="pyav")
        print("✅ Dataset loaded!")
        
        self.episode_idx = episode_idx
        self.noise_level = noise_level
        self._load_episode_data()
        
        # Initialize tracking
        self.current_step = 0
        self.action_history = deque(maxlen=100)
        self.prediction_times = deque(maxlen=100)
        
        # Find image key
        sample = self.dataset[self.episode_start_idx]
        self.image_key = None
        for key in sample.keys():
            if "image" in key and isinstance(sample[key], torch.Tensor):
                self.image_key = key
                break
        
        print(f"📸 Using image key: {self.image_key}")
        print(f"🎲 Simulating policy with {noise_level*100:.1f}% noise level")
        
    def _load_episode_data(self):
        """Load episode boundaries."""
        from_idx = self.dataset.episode_data_index["from"][self.episode_idx].item()
        to_idx = self.dataset.episode_data_index["to"][self.episode_idx].item()
        self.episode_start_idx = from_idx
        self.episode_end_idx = to_idx
        self.episode_length = to_idx - from_idx
        print(f"📏 Episode {self.episode_idx}: {self.episode_length} steps")
    
    def simulate_policy_prediction(self):
        """Simulate policy prediction with controlled shakiness."""
        if self.current_step >= self.episode_length:
            self.current_step = 0  # Loop back to start
        
        # Get real image and ground truth action from dataset
        dataset_idx = self.episode_start_idx + self.current_step
        sample = self.dataset[dataset_idx]
        gt_action = sample["action"].numpy()
        
        # Simulate policy prediction with different levels of shakiness
        start_time = time.time()
        
        # Create realistic prediction noise patterns
        if self.current_step == 0:
            # First prediction - usually close to ground truth
            noise = np.random.normal(0, self.noise_level * 0.5, gt_action.shape)
        elif self.current_step < 10:
            # Early predictions - moderate noise (learning phase)
            noise = np.random.normal(0, self.noise_level * 0.8, gt_action.shape)
        else:
            # Later predictions - add temporal correlation (more realistic)
            if len(self.action_history) > 0:
                # Make noise correlated with previous prediction error
                prev_prediction = self.action_history[-1]
                prev_error = prev_prediction - sample["action"].numpy() if self.current_step > 0 else np.zeros_like(gt_action)
                temporal_noise = 0.3 * prev_error  # Some temporal correlation
                random_noise = np.random.normal(0, self.noise_level, gt_action.shape)
                noise = 0.7 * random_noise + 0.3 * temporal_noise
            else:
                noise = np.random.normal(0, self.noise_level, gt_action.shape)
        
        # Add occasional "shaky" predictions (simulate challenging frames)
        if np.random.random() < 0.1:  # 10% chance of extra shaky prediction
            noise *= 3.0
        
        predicted_action = gt_action + noise
        prediction_time = time.time() - start_time + np.random.normal(0.01, 0.002)  # ~10ms with variation
        
        # Store for analysis
        self.action_history.append(predicted_action)
        self.prediction_times.append(prediction_time)
        
        self.current_step += 1
        
        return sample[self.image_key], predicted_action, gt_action, prediction_time


class ActionVisualizer:
    """Real-time action visualization."""
    
    def __init__(self, action_dim=6):
        self.action_dim = action_dim
        
        # Create figure with better layout
        self.fig = plt.figure(figsize=(15, 10))
        
        # Main trajectory plots
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Individual joint plots (2x3 grid)
        self.joint_axes = []
        for i in range(min(6, action_dim)):
            row = i // 3
            col = i % 3
            ax = self.fig.add_subplot(gs[row, col])
            self.joint_axes.append(ax)
        
        # Summary plot (bottom row)
        self.summary_ax = self.fig.add_subplot(gs[2, :])
        
        self.action_history = deque(maxlen=50)
        self.gt_history = deque(maxlen=50)
        self.time_steps = deque(maxlen=50)
        self.step_count = 0
        
        # Joint names
        self.joint_names = [f'Joint {i+1}' for i in range(action_dim)]
        
        plt.ion()
        plt.show()
    
    def update(self, predicted_action, gt_action):
        """Update the real-time plot."""
        self.action_history.append(predicted_action)
        self.gt_history.append(gt_action)
        self.time_steps.append(self.step_count)
        self.step_count += 1
        
        # Update individual joint plots
        for i, ax in enumerate(self.joint_axes):
            ax.clear()
            ax.grid(True, alpha=0.3)
            
            if len(self.action_history) > 1 and i < len(predicted_action):
                pred_values = [action[i] for action in self.action_history]
                gt_values = [action[i] for action in self.gt_history]
                
                ax.plot(list(self.time_steps), pred_values, 'r-', linewidth=2, alpha=0.8, label='Predicted')
                ax.plot(list(self.time_steps), gt_values, 'b--', linewidth=2, alpha=0.6, label='Ground Truth')
                
                # Highlight recent prediction
                if len(pred_values) > 0:
                    ax.scatter([list(self.time_steps)[-1]], [pred_values[-1]], color='red', s=100, alpha=0.8, zorder=5)
                
                # Calculate current error
                current_error = abs(pred_values[-1] - gt_values[-1])
                ax.set_title(f'{self.joint_names[i]} (Error: {current_error:.3f})')
                ax.legend(fontsize=8)
            else:
                ax.set_title(f'{self.joint_names[i]}')
        
        # Update summary plot - overall error over time
        self.summary_ax.clear()
        self.summary_ax.grid(True, alpha=0.3)
        
        if len(self.action_history) > 1:
            errors = []
            for pred, gt in zip(self.action_history, self.gt_history):
                mae = np.mean(np.abs(pred - gt))
                errors.append(mae)
            
            self.summary_ax.plot(list(self.time_steps), errors, 'g-', linewidth=3, alpha=0.8)
            self.summary_ax.fill_between(list(self.time_steps), errors, alpha=0.3, color='green')
            
            avg_error = np.mean(errors)
            self.summary_ax.axhline(avg_error, color='red', linestyle='--', alpha=0.7, 
                                  label=f'Avg Error: {avg_error:.4f}')
            self.summary_ax.set_title('Overall Policy Error Over Time')
            self.summary_ax.set_ylabel('Mean Absolute Error')
            self.summary_ax.set_xlabel('Time Step')
            self.summary_ax.legend()
        
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
    
    # Calculate prediction consistency (standard deviation)
    action_std = np.std(actions, axis=0)
    overall_consistency = np.mean(action_std)
    
    # Calculate jerk (second derivative) for smoothness assessment
    if len(actions) > 2:
        action_jerk = np.diff(action_derivatives, axis=0)
        overall_jerk = np.mean(np.abs(action_jerk))
    else:
        overall_jerk = 0
    
    # Calculate prediction timing
    avg_prediction_time = np.mean(prediction_times)
    timing_consistency = np.std(prediction_times)
    
    return {
        'overall_smoothness': overall_smoothness,
        'overall_consistency': overall_consistency,
        'overall_jerk': overall_jerk,
        'avg_prediction_time': avg_prediction_time,
        'timing_consistency': timing_consistency,
        'joint_smoothness': action_smoothness,
        'joint_consistency': action_std,
        'num_samples': len(action_history)
    }


def run_simulation_demo(dataset_name, episode_idx=0, speed=1.0, max_steps=None, 
                       noise_level=0.05, visualize_actions=True):
    """Main simulation demo."""
    
    print("🎯 Policy Shakiness Analysis Demo")
    print("=" * 50)
    print(f"📊 Dataset: {dataset_name}")
    print(f"📏 Episode: {episode_idx}")
    print(f"🎲 Noise Level: {noise_level*100:.1f}%")
    print()
    
    # Initialize simulator
    simulator = PolicyShakinessTester(dataset_name, episode_idx, noise_level)
    
    # Initialize visualizer
    visualizer = None
    if visualize_actions:
        sample_action = simulator.simulate_policy_prediction()[1]
        visualizer = ActionVisualizer(len(sample_action))
        # Reset to start
        simulator.current_step = 0
    
    # Run simulation
    max_steps = max_steps or min(100, simulator.episode_length)  # Limit for demo
    print(f"🏃 Running demo for {max_steps} steps...")
    print(f"⚡ Speed: {speed}x")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        for step in range(max_steps):
            # Get simulated prediction
            image, predicted_action, gt_action, pred_time = simulator.simulate_policy_prediction()
            
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
        print("\n🛑 Demo stopped by user")
    
    # Analyze results
    print("\n📊 Policy Shakiness Analysis:")
    print("-" * 30)
    
    metrics = analyze_policy_shakiness(simulator.action_history, simulator.prediction_times)
    
    if metrics:
        print(f"🎯 Overall Smoothness: {metrics['overall_smoothness']:.4f}")
        print(f"🎯 Overall Consistency: {metrics['overall_consistency']:.4f}")
        print(f"🎯 Jerk (Acceleration): {metrics['overall_jerk']:.4f}")
        print(f"⚡ Avg Prediction Time: {metrics['avg_prediction_time']*1000:.1f}ms")
        print(f"⏱️  Timing Consistency: {metrics['timing_consistency']*1000:.1f}ms std")
        print(f"📊 Samples Analyzed: {metrics['num_samples']}")
        
        print(f"\n🔧 Per-Joint Analysis:")
        for i, (smoothness, consistency) in enumerate(zip(metrics['joint_smoothness'], metrics['joint_consistency'])):
            print(f"   Joint {i+1}: Smoothness={smoothness:.4f}, Consistency={consistency:.4f}")
        
        # Performance assessment
        print(f"\n💡 Policy Assessment:")
        if metrics['overall_smoothness'] < 0.01:
            print("   ✅ Very smooth policy - actions change gradually")
        elif metrics['overall_smoothness'] < 0.1:
            print("   ✅ Reasonably smooth policy")
        elif metrics['overall_smoothness'] < 0.5:
            print("   ⚠️  Somewhat shaky policy - noticeable action jumps")
        else:
            print("   ❌ Very shaky policy - large action changes between steps")
        
        if metrics['overall_jerk'] < 0.01:
            print("   ✅ Very stable predictions - low acceleration changes")
        elif metrics['overall_jerk'] < 0.1:
            print("   ✅ Stable predictions")
        else:
            print("   ⚠️  Jerky predictions - policy changes direction frequently")
        
        if metrics['avg_prediction_time'] < 0.01:
            print("   ⚡ Very fast inference (<10ms)")
        elif metrics['avg_prediction_time'] < 0.05:
            print("   ✅ Fast inference (<50ms)")
        else:
            print("   ⚠️  Slow inference - may need optimization")
    
    print("\n🎉 Demo complete!")
    print(f"\n💡 This demonstrates what the analysis would show with your actual trained policy.")
    print(f"   The current noise level of {noise_level*100:.1f}% simulates a policy with moderate shakiness.")
    
    if visualizer:
        input("\nPress Enter to close visualization...")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Demonstrate policy shakiness analysis')
    parser.add_argument('--dataset', type=str, default='bearlover365/red_cube_always_in_same_place',
                       help='Dataset name')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index to use')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Demo speed multiplier')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps to simulate')
    parser.add_argument('--noise-level', type=float, default=0.05,
                       help='Noise level for simulated predictions (0.0-1.0)')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Disable real-time visualization')
    
    args = parser.parse_args()
    
    try:
        run_simulation_demo(
            dataset_name=args.dataset,
            episode_idx=args.episode,
            speed=args.speed,
            max_steps=args.max_steps,
            noise_level=args.noise_level,
            visualize_actions=not args.no_visualization
        )
        return 0
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 