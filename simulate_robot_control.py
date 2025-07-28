#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Control Simulation with Real Images

This script demonstrates a trained policy controlling a simulated robot
while being fed real-world images from the dataset. This shows policy
"shakiness" in actual robot control, not just prediction accuracy.

Key features:
- MuJoCo robot simulation
- Real-world images fed to policy  
- Policy controls simulated robot joints
- Visualizes control stability and smoothness
- Records robot movement for analysis

Usage:
    python simulate_robot_control.py --policy-path ./single_episode_model
    python simulate_robot_control.py --policy-path ./single_episode_model --episode 0 --steps 100
"""

import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    print("⚠️  MuJoCo not available - will use matplotlib-only visualization")
    MUJOCO_AVAILABLE = False

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy


class RobotControlSimulator:
    """Simulates robot control with real images and policy predictions."""
    
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
        
        # Initialize robot state and tracking
        self.current_step = 0
        self.robot_joint_positions = np.zeros(6)  # 6-DOF robot
        self.joint_velocities = np.zeros(6)
        
        # Control tracking
        self.action_history = deque(maxlen=200)
        self.position_history = deque(maxlen=200)
        self.velocity_history = deque(maxlen=200)
        self.prediction_times = deque(maxlen=200)
        
        # Find image key
        sample = self.dataset[self.episode_start_idx]
        self.image_key = None
        for key in sample.keys():
            if "image" in key and isinstance(sample[key], torch.Tensor):
                self.image_key = key
                break
        
        print(f"📸 Using image key: {self.image_key}")
        
        # Initialize MuJoCo if available
        self.mujoco_model = None
        self.mujoco_data = None
        if MUJOCO_AVAILABLE:
            self._setup_mujoco()
        
    def _load_episode_data(self):
        """Load episode boundaries."""
        from_idx = self.dataset.episode_data_index["from"][self.episode_idx].item()
        to_idx = self.dataset.episode_data_index["to"][self.episode_idx].item()
        self.episode_start_idx = from_idx
        self.episode_end_idx = to_idx
        self.episode_length = to_idx - from_idx
        print(f"📏 Episode {self.episode_idx}: {self.episode_length} steps")
        
    def _setup_mujoco(self):
        """Setup MuJoCo simulation if available."""
        try:
            # Create a simple 6-DOF arm model
            xml_string = """
            <mujoco model="simple_arm">
                <option timestep="0.01"/>
                <worldbody>
                    <light pos="0 0 3" dir="0 0 -1"/>
                    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
                    
                    <body name="base" pos="0 0 0.1">
                        <geom type="cylinder" size="0.05 0.1" rgba="0.7 0.7 0.7 1"/>
                        <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
                        
                        <body name="link1" pos="0 0 0.2" euler="0 0 0">
                            <geom type="box" size="0.03 0.03 0.1" rgba="0.8 0.3 0.3 1"/>
                            <joint name="joint2" type="hinge" axis="0 1 0" range="-90 90"/>
                            
                            <body name="link2" pos="0 0 0.2" euler="0 0 0">
                                <geom type="box" size="0.03 0.03 0.1" rgba="0.3 0.8 0.3 1"/>
                                <joint name="joint3" type="hinge" axis="0 1 0" range="-90 90"/>
                                
                                <body name="link3" pos="0 0 0.2" euler="0 0 0">
                                    <geom type="box" size="0.03 0.03 0.1" rgba="0.3 0.3 0.8 1"/>
                                    <joint name="joint4" type="hinge" axis="1 0 0" range="-90 90"/>
                                    
                                    <body name="link4" pos="0 0 0.15" euler="0 0 0">
                                        <geom type="box" size="0.02 0.02 0.08" rgba="0.8 0.8 0.3 1"/>
                                        <joint name="joint5" type="hinge" axis="0 1 0" range="-90 90"/>
                                        
                                        <body name="end_effector" pos="0 0 0.1" euler="0 0 0">
                                            <geom type="box" size="0.02 0.02 0.05" rgba="0.8 0.3 0.8 1"/>
                                            <joint name="joint6" type="hinge" axis="0 0 1" range="-180 180"/>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </worldbody>
                
                <actuator>
                    <position joint="joint1" name="actuator1"/>
                    <position joint="joint2" name="actuator2"/>
                    <position joint="joint3" name="actuator3"/>
                    <position joint="joint4" name="actuator4"/>
                    <position joint="joint5" name="actuator5"/>
                    <position joint="joint6" name="actuator6"/>
                </actuator>
            </mujoco>
            """
            
            self.mujoco_model = mujoco.MjModel.from_xml_string(xml_string)
            self.mujoco_data = mujoco.MjData(self.mujoco_model)
            print("✅ MuJoCo robot simulation initialized!")
            
        except Exception as e:
            print(f"⚠️  MuJoCo setup failed: {e}")
            self.mujoco_model = None
            self.mujoco_data = None
    
    def get_policy_action_and_apply(self):
        """Get policy action from real image and apply to robot."""
        if self.current_step >= self.episode_length:
            self.current_step = 0  # Loop episode
        
        # Get real image from dataset
        dataset_idx = self.episode_start_idx + self.current_step
        sample = self.dataset[dataset_idx]
        
        # Prepare input for policy - ONLY OBSERVATIONS
        batch = {}
        for key, value in sample.items():
            if key.startswith("observation.") and isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).to(self.device)
        
        # Get policy prediction
        start_time = time.time()
        with torch.no_grad():
            predicted_action_chunk = self.policy.select_action(batch)
            
            # Handle action chunking - take first action
            if predicted_action_chunk.dim() == 3:  # [batch, chunk_size, action_dim]
                predicted_action = predicted_action_chunk[0, 0, :].cpu().numpy()
            else:  # [batch, action_dim]
                predicted_action = predicted_action_chunk[0, :].cpu().numpy()
                
        prediction_time = time.time() - start_time
        
        # Apply action to simulated robot
        self._apply_action_to_robot(predicted_action)
        
        # Store for analysis
        self.action_history.append(predicted_action.copy())
        self.position_history.append(self.robot_joint_positions.copy())
        self.velocity_history.append(self.joint_velocities.copy())
        self.prediction_times.append(prediction_time)
        
        # Get ground truth for comparison
        gt_action = sample["action"].numpy()
        
        self.current_step += 1
        
        return sample[self.image_key], predicted_action, gt_action, prediction_time
    
    def _apply_action_to_robot(self, action):
        """Apply predicted action to robot (both MuJoCo and internal state)."""
        # Clip actions to reasonable ranges
        action = np.clip(action, -np.pi, np.pi)
        
        # Apply to MuJoCo if available
        if self.mujoco_model is not None and self.mujoco_data is not None:
            # Set joint targets
            self.mujoco_data.ctrl[:] = action
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
            
            # Get actual joint positions and velocities
            self.robot_joint_positions = self.mujoco_data.qpos[:6].copy()
            self.joint_velocities = self.mujoco_data.qvel[:6].copy()
        else:
            # Simple kinematic simulation without MuJoCo
            dt = 0.01  # 100 Hz
            max_velocity = 2.0  # rad/s
            
            # Calculate desired velocity toward target
            position_error = action - self.robot_joint_positions
            desired_velocity = np.clip(position_error * 5.0, -max_velocity, max_velocity)
            
            # Update positions
            self.robot_joint_positions += desired_velocity * dt
            self.joint_velocities = desired_velocity


class RobotControlVisualizer:
    """Real-time visualization of robot control."""
    
    def __init__(self, joint_names=None):
        self.joint_names = joint_names or [f'Joint {i+1}' for i in range(6)]
        
        # Create comprehensive visualization
        self.fig = plt.figure(figsize=(16, 12))
        gs = self.fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Joint position plots (top 2 rows)
        self.joint_axes = []
        for i in range(6):
            row = i // 3
            col = i % 3
            ax = self.fig.add_subplot(gs[row, col])
            self.joint_axes.append(ax)
        
        # Control smoothness plot (bottom left)
        self.smoothness_ax = self.fig.add_subplot(gs[2, :2])
        
        # Velocity plot (bottom right)
        self.velocity_ax = self.fig.add_subplot(gs[2, 2:])
        
        # Overall assessment (bottom)
        self.assessment_ax = self.fig.add_subplot(gs[3, :])
        
        # Data storage
        self.position_history = deque(maxlen=100)
        self.velocity_history = deque(maxlen=100)
        self.action_history = deque(maxlen=100)
        self.time_steps = deque(maxlen=100)
        self.step_count = 0
        
        plt.ion()
        plt.show()
    
    def update(self, predicted_action, joint_positions, joint_velocities):
        """Update all visualizations."""
        self.position_history.append(joint_positions)
        self.velocity_history.append(joint_velocities)
        self.action_history.append(predicted_action)
        self.time_steps.append(self.step_count)
        self.step_count += 1
        
        if len(self.position_history) < 2:
            return
        
        # Update joint position plots
        for i, ax in enumerate(self.joint_axes):
            ax.clear()
            ax.grid(True, alpha=0.3)
            
            # Plot commanded vs actual positions
            commanded = [action[i] for action in self.action_history]
            actual = [pos[i] for pos in self.position_history]
            
            ax.plot(list(self.time_steps), commanded, 'r-', linewidth=2, alpha=0.8, label='Commanded')
            ax.plot(list(self.time_steps), actual, 'b-', linewidth=2, alpha=0.6, label='Actual')
            
            # Highlight current point
            if len(commanded) > 0:
                ax.scatter([list(self.time_steps)[-1]], [commanded[-1]], color='red', s=80, alpha=0.8, zorder=5)
                ax.scatter([list(self.time_steps)[-1]], [actual[-1]], color='blue', s=80, alpha=0.8, zorder=5)
            
            # Calculate tracking error
            if len(commanded) > 0:
                tracking_error = abs(commanded[-1] - actual[-1])
                ax.set_title(f'{self.joint_names[i]} (Error: {tracking_error:.3f})')
            else:
                ax.set_title(f'{self.joint_names[i]}')
            
            ax.legend(fontsize=8)
            ax.set_ylabel('Position (rad)')
        
        # Update control smoothness
        self.smoothness_ax.clear()
        self.smoothness_ax.grid(True, alpha=0.3)
        
        if len(self.action_history) > 1:
            # Calculate action smoothness (derivative)
            action_changes = []
            for i in range(1, len(self.action_history)):
                change = np.linalg.norm(np.array(self.action_history[i]) - np.array(self.action_history[i-1]))
                action_changes.append(change)
            
            time_steps_smooth = list(self.time_steps)[1:]
            self.smoothness_ax.plot(time_steps_smooth, action_changes, 'g-', linewidth=2, alpha=0.8)
            self.smoothness_ax.fill_between(time_steps_smooth, action_changes, alpha=0.3, color='green')
            
            avg_change = np.mean(action_changes)
            self.smoothness_ax.axhline(avg_change, color='red', linestyle='--', alpha=0.7, 
                                     label=f'Avg Change: {avg_change:.4f}')
            self.smoothness_ax.set_title('Control Smoothness (Action Changes)')
            self.smoothness_ax.set_ylabel('||Δaction||')
            self.smoothness_ax.set_xlabel('Time Step')
            self.smoothness_ax.legend()
        
        # Update velocity plot
        self.velocity_ax.clear()
        self.velocity_ax.grid(True, alpha=0.3)
        
        if len(self.velocity_history) > 0:
            # Plot joint velocities
            for i in range(6):
                velocities = [vel[i] for vel in self.velocity_history]
                self.velocity_ax.plot(list(self.time_steps), velocities, 
                                    linewidth=1.5, alpha=0.7, label=f'J{i+1}')
            
            self.velocity_ax.set_title('Joint Velocities')
            self.velocity_ax.set_ylabel('Velocity (rad/s)')
            self.velocity_ax.set_xlabel('Time Step')
            self.velocity_ax.legend(ncol=6, fontsize=8)
        
        # Update assessment
        self.assessment_ax.clear()
        self.assessment_ax.axis('off')
        
        if len(self.action_history) > 10:
            # Calculate metrics
            action_changes = []
            tracking_errors = []
            
            for i in range(1, len(self.action_history)):
                change = np.linalg.norm(np.array(self.action_history[i]) - np.array(self.action_history[i-1]))
                action_changes.append(change)
                
                error = np.linalg.norm(np.array(self.action_history[i]) - np.array(self.position_history[i]))
                tracking_errors.append(error)
            
            avg_smoothness = np.mean(action_changes)
            avg_tracking = np.mean(tracking_errors)
            
            # Assessment text
            assessment = "🤖 ROBOT CONTROL ASSESSMENT\n\n"
            
            if avg_smoothness < 0.1:
                assessment += "✅ Very smooth control - minimal action changes\n"
            elif avg_smoothness < 0.5:
                assessment += "✅ Reasonably smooth control\n"
            elif avg_smoothness < 1.0:
                assessment += "⚠️  Somewhat shaky control - noticeable action jumps\n"
            else:
                assessment += "❌ Very shaky control - large action changes\n"
            
            if avg_tracking < 0.1:
                assessment += "✅ Excellent tracking - robot follows commands well\n"
            elif avg_tracking < 0.5:
                assessment += "✅ Good tracking performance\n"
            else:
                assessment += "⚠️  Poor tracking - robot struggles to follow commands\n"
            
            assessment += f"\nMetrics:\n"
            assessment += f"• Average Control Smoothness: {avg_smoothness:.4f}\n"
            assessment += f"• Average Tracking Error: {avg_tracking:.4f}\n"
            assessment += f"• Steps Analyzed: {len(action_changes)}"
            
            self.assessment_ax.text(0.05, 0.95, assessment, transform=self.assessment_ax.transAxes,
                                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.draw()
        plt.pause(0.01)


def analyze_control_performance(action_history, position_history, velocity_history, prediction_times):
    """Analyze robot control performance and stability."""
    if len(action_history) < 2:
        return {}
    
    actions = np.array(action_history)
    positions = np.array(position_history)
    velocities = np.array(velocity_history)
    
    # Control smoothness (action derivatives)
    action_derivatives = np.diff(actions, axis=0)
    control_smoothness = np.mean(np.linalg.norm(action_derivatives, axis=1))
    
    # Tracking performance
    tracking_errors = np.linalg.norm(actions - positions, axis=1)
    avg_tracking_error = np.mean(tracking_errors)
    max_tracking_error = np.max(tracking_errors)
    
    # Velocity smoothness
    velocity_changes = np.diff(velocities, axis=0)
    velocity_smoothness = np.mean(np.linalg.norm(velocity_changes, axis=1))
    
    # Prediction timing
    avg_prediction_time = np.mean(prediction_times)
    
    return {
        'control_smoothness': control_smoothness,
        'avg_tracking_error': avg_tracking_error,
        'max_tracking_error': max_tracking_error,
        'velocity_smoothness': velocity_smoothness,
        'avg_prediction_time': avg_prediction_time,
        'num_samples': len(action_history)
    }


def run_robot_control_simulation(policy_path, dataset_name, episode_idx=0, 
                                max_steps=100, speed=1.0, visualize=True, use_mujoco=True):
    """Main robot control simulation loop."""
    
    print("🤖 Robot Control Simulation")
    print("=" * 40)
    print(f"🧠 Policy: {policy_path}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"📏 Episode: {episode_idx}")
    print(f"🎯 Max Steps: {max_steps}")
    print(f"⚡ Speed: {speed}x")
    print(f"🖥️  MuJoCo: {'Enabled' if use_mujoco and MUJOCO_AVAILABLE else 'Disabled'}")
    print()
    
    # Initialize simulator
    simulator = RobotControlSimulator(policy_path, dataset_name, episode_idx)
    
    if not use_mujoco:
        simulator.mujoco_model = None
        simulator.mujoco_data = None
    
    # Initialize visualizer
    visualizer = None
    mujoco_viewer = None
    
    if visualize:
        visualizer = RobotControlVisualizer()
    
    if simulator.mujoco_model is not None and MUJOCO_AVAILABLE:
        try:
            mujoco_viewer = mujoco.viewer.launch_passive(simulator.mujoco_model, simulator.mujoco_data)
            print("🎮 MuJoCo viewer launched! You can see the robot moving.")
        except Exception as e:
            print(f"⚠️  MuJoCo viewer failed: {e}")
            mujoco_viewer = None
    
    print(f"\n🏃 Starting robot control simulation...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        for step in range(max_steps):
            # Get policy action and apply to robot
            image, predicted_action, gt_action, pred_time = simulator.get_policy_action_and_apply()
            
            # Update visualizations
            if visualizer:
                visualizer.update(predicted_action, simulator.robot_joint_positions, simulator.joint_velocities)
            
            if mujoco_viewer is not None:
                mujoco_viewer.sync()
            
            # Print progress
            if step % 10 == 0:
                tracking_error = np.linalg.norm(predicted_action - simulator.robot_joint_positions)
                print(f"Step {step:3d}: Tracking_Error={tracking_error:.4f}, Pred_time={pred_time*1000:.1f}ms")
            
            # Control simulation speed
            time.sleep(1.0 / (30 * speed))  # 30 FPS base rate
    
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    
    finally:
        if mujoco_viewer is not None:
            mujoco_viewer.close()
    
    # Analyze results
    print("\n📊 Robot Control Analysis:")
    print("-" * 30)
    
    metrics = analyze_control_performance(
        simulator.action_history, 
        simulator.position_history, 
        simulator.velocity_history,
        simulator.prediction_times
    )
    
    if metrics:
        print(f"🎯 Control Smoothness: {metrics['control_smoothness']:.4f}")
        print(f"🎯 Avg Tracking Error: {metrics['avg_tracking_error']:.4f}")
        print(f"🎯 Max Tracking Error: {metrics['max_tracking_error']:.4f}")
        print(f"🎯 Velocity Smoothness: {metrics['velocity_smoothness']:.4f}")
        print(f"⚡ Avg Prediction Time: {metrics['avg_prediction_time']*1000:.1f}ms")
        print(f"📊 Samples Analyzed: {metrics['num_samples']}")
        
        # Performance assessment
        print(f"\n💡 Robot Control Assessment:")
        if metrics['control_smoothness'] < 0.1:
            print("   ✅ Very smooth control - robot moves fluidly")
        elif metrics['control_smoothness'] < 0.5:
            print("   ✅ Reasonably smooth control")
        elif metrics['control_smoothness'] < 1.0:
            print("   ⚠️  Somewhat jerky control - robot has noticeable jitters")
        else:
            print("   ❌ Very jerky control - robot moves erratically")
        
        if metrics['avg_tracking_error'] < 0.1:
            print("   ✅ Excellent tracking - robot follows commands precisely")
        elif metrics['avg_tracking_error'] < 0.5:
            print("   ✅ Good tracking performance")
        else:
            print("   ⚠️  Poor tracking - robot struggles to follow policy commands")
    
    print("\n🎉 Robot control simulation complete!")
    
    if visualizer:
        input("\nPress Enter to close visualization...")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Robot control simulation with real images')
    parser.add_argument('--policy-path', type=str, default='./single_episode_model',
                       help='Path to trained policy checkpoint')
    parser.add_argument('--dataset', type=str, default='bearlover365/red_cube_always_in_same_place',
                       help='Dataset name')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index to use for images')
    parser.add_argument('--steps', type=int, default=100,
                       help='Maximum steps to simulate')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Simulation speed multiplier')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Disable real-time visualization')
    parser.add_argument('--no-mujoco', action='store_true',
                       help='Disable MuJoCo physics simulation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    if not Path(args.policy_path).exists():
        print(f"❌ Policy path does not exist: {args.policy_path}")
        return 1
    
    try:
        run_robot_control_simulation(
            policy_path=args.policy_path,
            dataset_name=args.dataset,
            episode_idx=args.episode,
            max_steps=args.steps,
            speed=args.speed,
            visualize=not args.no_visualization,
            use_mujoco=not args.no_mujoco
        )
        return 0
    
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 