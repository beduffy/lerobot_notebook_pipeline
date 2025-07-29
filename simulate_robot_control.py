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

    worked since running matplotlib at same time causes LibGL errors, conflicts:
    python simulate_robot_control.py --steps 10 --speed 2.0 --no-visualization
"""

import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
import json

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    print("⚠️  MuJoCo not available - will use matplotlib-only visualization")
    MUJOCO_AVAILABLE = False

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Import all model types
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

# Import foundation models (VLAs)
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    SMOLVLA_AVAILABLE = True
except ImportError:
    # Fallback to old structure or not available
    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        SMOLVLA_AVAILABLE = True
    except ImportError:
        SMOLVLA_AVAILABLE = False

try:
    from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
    PI0FAST_AVAILABLE = True
except ImportError:
    # Fallback to old structure or not available
    try:
        from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
        PI0FAST_AVAILABLE = True
    except ImportError:
        PI0FAST_AVAILABLE = False


def detect_model_type(model_path):
    """Detect model type from model directory."""
    model_path = Path(model_path)
    
    # Check for training_info.json first
    info_path = model_path / "training_info.json"
    if info_path.exists():
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
                if 'model_type' in info:
                    return info['model_type']
        except:
            pass
    
    # Check for config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Check for model-specific identifiers
                if 'model_type' in config:
                    return config['model_type']
                elif 'architectures' in config:
                    arch = config['architectures'][0] if config['architectures'] else ""
                    if 'ACT' in arch:
                        return 'act'
                    elif 'Diffusion' in arch:
                        return 'diffusion'
                    elif 'VQBeT' in arch:
                        return 'vqbet'
                    elif 'SmolVLA' in arch:
                        return 'smolvla'
                    elif 'PI0FAST' in arch:
                        return 'pi0fast'
        except:
            pass
    
    # Fallback: try to load each model type and see which one works
    model_types = ['act', 'diffusion', 'vqbet']
    if SMOLVLA_AVAILABLE:
        model_types.append('smolvla')
    if PI0FAST_AVAILABLE:
        model_types.append('pi0fast')
    
    for model_type in model_types:
        try:
            if model_type == 'act':
                ACTPolicy.from_pretrained(model_path)
                return 'act'
            elif model_type == 'diffusion':
                DiffusionPolicy.from_pretrained(model_path)
                return 'diffusion'
            elif model_type == 'vqbet':
                VQBeTPolicy.from_pretrained(model_path)
                return 'vqbet'
            elif model_type == 'smolvla' and SMOLVLA_AVAILABLE:
                SmolVLAPolicy.from_pretrained(model_path)
                return 'smolvla'
            elif model_type == 'pi0fast' and PI0FAST_AVAILABLE:
                PI0FASTPolicy.from_pretrained(model_path)
                return 'pi0fast'
        except:
            continue
    
    # Default to ACT if nothing works
    print("⚠️  Could not detect model type, defaulting to ACT")
    return 'act'


def load_policy(model_path, model_type):
    """Load policy based on model type."""
    print(f"🧠 Loading {model_type.upper()} policy from {model_path}...")
    
    if model_type == 'act':
        policy = ACTPolicy.from_pretrained(model_path)
    elif model_type == 'diffusion':
        policy = DiffusionPolicy.from_pretrained(model_path)
    elif model_type == 'vqbet':
        policy = VQBeTPolicy.from_pretrained(model_path)
    elif model_type == 'smolvla':
        if not SMOLVLA_AVAILABLE:
            raise RuntimeError("SmolVLA not available. Install with updated LeRobot version.")
        policy = SmolVLAPolicy.from_pretrained(model_path)
    elif model_type == 'pi0fast':
        if not PI0FAST_AVAILABLE:
            raise RuntimeError("π0-FAST not available. Install with updated LeRobot version.")
        policy = PI0FASTPolicy.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return policy


class RobotControlSimulator:
    """Simulates robot control with real images and policy predictions."""
    
    def __init__(self, policy_path, dataset_name, episode_idx=0, start_step=0, device="auto", camera_remap=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        # Detect model type
        self.model_type = detect_model_type(policy_path)
        print(f"   Detected model type: {self.model_type.upper()}")
        
        # Load policy
        print(f"Loading policy from {policy_path}...")
        self.policy = load_policy(policy_path, self.model_type)
        self.policy.eval()
        self.policy.to(self.device)
        self.policy.reset()
        print("Policy loaded!")
        
        # Load dataset
        print(f"Loading dataset {dataset_name}...")
        self.dataset = LeRobotDataset(dataset_name, video_backend="pyav")
        print("Dataset loaded!")
        
        # Apply camera remapping if specified
        self.camera_remap = camera_remap
        if camera_remap:
            print(f"   📷 Applying camera remapping: {camera_remap}")
            
            class CameraRemapDataset(torch.utils.data.Dataset):
                def __init__(self, dataset, camera_remap):
                    self.dataset = dataset
                    self.camera_remap = camera_remap
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    batch = self.dataset[idx]
                    remapped_batch = {}
                    
                    for key, value in batch.items():
                        if key in self.camera_remap:
                            # Remap camera keys
                            new_key = self.camera_remap[key]
                            remapped_batch[new_key] = value
                        else:
                            remapped_batch[key] = value
                    
                    return remapped_batch
            
            self.dataset = CameraRemapDataset(self.dataset, camera_remap)
        
        # Get episode data
        self.episode_idx = episode_idx
        self._load_episode_data()
        
        # Initialize robot state and tracking
        self.current_step = start_step
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
        self.mujoco_joint_indices = None
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
        """Setup MuJoCo simulation using existing SO101 arm."""
        try:
            # Use the existing SO101 arm XML file
            # xml_path = "lerobot_some_original_code/standalone_scene.xml"
            xml_path = "lerobot_some_original_code/simple_scene.xml"
            
            self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mujoco_data = mujoco.MjData(self.mujoco_model)
            
            # Map Mujoco joint names (same as teleoperate_sim_aditya.py)
            mujoco_joint_names = [self.mujoco_model.joint(i).name for i in range(self.mujoco_model.njnt)]
            print(f"MuJoCo joint names: {mujoco_joint_names}")
            
            # Map joint indices for joints "1" through "6"
            try:
                self.mujoco_joint_indices = [mujoco_joint_names.index(str(i)) for i in range(1, 7)]
                print(f"Joint indices: {self.mujoco_joint_indices}")
            except ValueError as e:
                print(f"⚠️  Could not find all joints 1-6: {e}")
                # Fallback to first 6 joints
                self.mujoco_joint_indices = list(range(min(6, self.mujoco_model.njnt)))
                print(f"Using fallback indices: {self.mujoco_joint_indices}")
            
            print("MuJoCo SO101 arm simulation initialized!")
            
        except Exception as e:
            print(f"⚠️  MuJoCo setup failed: {e}")
            print(f"   Make sure {xml_path} exists")
            self.mujoco_model = None
            self.mujoco_data = None
            self.mujoco_joint_indices = None
    
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
        
        # Add language task for VLA models (SmolVLA, π0-FAST)
        if self.model_type in ["smolvla", "pi0fast"]:
            batch["task"] = "grab red cube and put to left"
        
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
        """Apply predicted action to robot (following teleoperate_sim_aditya.py approach)."""
        # Apply to MuJoCo if available
        if self.mujoco_model is not None and self.mujoco_data is not None and hasattr(self, 'mujoco_joint_indices'):
            # Dataset actions are in degrees, MuJoCo expects radians (following teleoperate_sim_aditya.py)
            joint_values_deg = action[:6]  # Policy outputs degrees (same as dataset)
            joint_values_rad = np.deg2rad(joint_values_deg)  # Convert to radians for MuJoCo
            
            # Set joint positions using the mapped indices
            for idx, val in zip(self.mujoco_joint_indices, joint_values_rad):
                if idx < len(self.mujoco_data.qpos):
                    self.mujoco_data.qpos[idx] = val
            
            # Step the simulation
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
            
            # Get actual joint positions and velocities
            actual_positions = []
            actual_velocities = []
            for idx in self.mujoco_joint_indices:
                if idx < len(self.mujoco_data.qpos):
                    actual_positions.append(self.mujoco_data.qpos[idx])
                    actual_velocities.append(self.mujoco_data.qvel[idx] if idx < len(self.mujoco_data.qvel) else 0.0)
                else:
                    actual_positions.append(0.0)
                    actual_velocities.append(0.0)
            
            self.robot_joint_positions = np.array(actual_positions)
            self.joint_velocities = np.array(actual_velocities)
        else:
            # Simple kinematic simulation without MuJoCo
            dt = 0.01  # 100 Hz
            max_velocity = 2.0  # rad/s
            
            # Calculate desired velocity toward target
            position_error = action[:6] - self.robot_joint_positions
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
            
            # Assessment text (avoiding emoji for font compatibility)
            assessment = "ROBOT CONTROL ASSESSMENT\n\n"
            
            if avg_smoothness < 0.1:
                assessment += "[EXCELLENT] Very smooth control - minimal action changes\n"
            elif avg_smoothness < 0.5:
                assessment += "[GOOD] Reasonably smooth control\n"
            elif avg_smoothness < 1.0:
                assessment += "[WARNING] Somewhat shaky control - noticeable action jumps\n"
            else:
                assessment += "[POOR] Very shaky control - large action changes\n"
            
            if avg_tracking < 0.1:
                assessment += "[EXCELLENT] Excellent tracking - robot follows commands well\n"
            elif avg_tracking < 0.5:
                assessment += "[GOOD] Good tracking performance\n"
            else:
                assessment += "[WARNING] Poor tracking - robot struggles to follow commands\n"
            
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
    
    # Tracking performance (convert positions from radians to degrees for comparison)
    positions_deg = np.rad2deg(positions)
    tracking_errors = np.linalg.norm(actions - positions_deg, axis=1)
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


def run_robot_control_simulation(policy_path, dataset_name, episode_idx=0, start_step=0,
                                max_steps=100, speed=1.0, visualize=True, use_mujoco=True, camera_remap=None):
    """Main robot control simulation loop."""
    
    print("🤖 Robot Control Simulation")
    print("=" * 40)
    print(f"🧠 Policy: {policy_path}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"📏 Episode: {episode_idx}")
    print(f"🏁 Start Step: {start_step}")
    print(f"🎯 Max Steps: {max_steps}")
    print(f"⚡ Speed: {speed}x")
    print(f"🖥️  MuJoCo: {'Enabled' if use_mujoco and MUJOCO_AVAILABLE else 'Disabled'}")
    print()
    
    # Initialize simulator
    simulator = RobotControlSimulator(policy_path, dataset_name, episode_idx, start_step, camera_remap=camera_remap)
    
    if not use_mujoco:
        simulator.mujoco_model = None
        simulator.mujoco_data = None
    
    # Initialize visualizer
    visualizer = None
    mujoco_viewer = None
    
    if visualize:
        visualizer = RobotControlVisualizer()
    
    print(f"\n🏃 Starting robot control simulation...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Use MuJoCo viewer context manager (same as teleoperate_sim_aditya.py)
        if simulator.mujoco_model is not None and MUJOCO_AVAILABLE:
            with mujoco.viewer.launch_passive(simulator.mujoco_model, simulator.mujoco_data) as viewer:
                print("MuJoCo viewer launched! You can see the robot moving.")
                
                for step in range(max_steps):
                    if not viewer.is_running():
                        break
                        
                    # Get policy action and apply to robot
                    image, predicted_action, gt_action, pred_time = simulator.get_policy_action_and_apply()
                    
                    # Update visualizations
                    if visualizer:
                        visualizer.update(predicted_action, simulator.robot_joint_positions, simulator.joint_velocities)
                    
                    # Sync MuJoCo viewer
                    viewer.sync()
                    
                    # Print progress and action values
                    if step % 5 == 0:
                        # Convert robot positions back to degrees for comparison
                        robot_pos_deg = np.rad2deg(simulator.robot_joint_positions)
                        tracking_error = np.linalg.norm(predicted_action - robot_pos_deg)
                        print(f"Step {step:3d}: Tracking_Error={tracking_error:.4f}°, Pred_time={pred_time*1000:.1f}ms")
                        print(f"  Policy Action (deg): {predicted_action}")
                        print(f"  GT Action (deg):     {gt_action}")
                        print(f"  Robot Pos (deg):     {robot_pos_deg}")
                        print(f"  Robot Pos (rad):     {simulator.robot_joint_positions}")
                        print(f"  Action Range:        [{predicted_action.min():.3f}, {predicted_action.max():.3f}]°")
                        print()
                    
                    # Control simulation speed
                    time.sleep(1.0 / (30 * speed))  # 30 FPS base rate
        else:
            # No MuJoCo viewer, just run simulation
            for step in range(max_steps):
                # Get policy action and apply to robot
                image, predicted_action, gt_action, pred_time = simulator.get_policy_action_and_apply()
                
                # Update visualizations
                if visualizer:
                    visualizer.update(predicted_action, simulator.robot_joint_positions, simulator.joint_velocities)
                
                # Print progress and action values
                if step % 5 == 0:
                    # Convert robot positions back to degrees for comparison
                    robot_pos_deg = np.rad2deg(simulator.robot_joint_positions)
                    tracking_error = np.linalg.norm(predicted_action - robot_pos_deg)
                    print(f"Step {step:3d}: Tracking_Error={tracking_error:.4f}°, Pred_time={pred_time*1000:.1f}ms")
                    print(f"  Policy Action (deg): {predicted_action}")
                    print(f"  GT Action (deg):     {gt_action}")
                    print(f"  Robot Pos (deg):     {robot_pos_deg}")
                    print(f"  Robot Pos (rad):     {simulator.robot_joint_positions}")
                    print(f"  Action Range:        [{predicted_action.min():.3f}, {predicted_action.max():.3f}]°")
                    print()
                
                # Control simulation speed
                time.sleep(1.0 / (30 * speed))  # 30 FPS base rate
    
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    
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
    parser.add_argument('--start-step', type=int, default=0,
                       help='Starting step within episode')
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
    parser.add_argument('--camera-remap', type=str, default=None,
                       help="Camera remapping (e.g., 'observation.images.front:observation.images.wrist')")
    parser.add_argument('--model-type', type=str, default=None, 
                       help="Force model type (act, diffusion, vqbet, smolvla, pi0fast). Auto-detected if not specified.")
    
    args = parser.parse_args()
    
    # Parse camera remapping
    camera_remap = None
    if args.camera_remap:
        camera_remap = {}
        for mapping in args.camera_remap.split(','):
            if ':' in mapping:
                old_key, new_key = mapping.strip().split(':')
                camera_remap[old_key.strip()] = new_key.strip()
        print(f"📷 Camera remapping: {camera_remap}")
    
    if not Path(args.policy_path).exists():
        print(f"❌ Policy path does not exist: {args.policy_path}")
        return 1
    
    try:
        run_robot_control_simulation(
            policy_path=args.policy_path,
            dataset_name=args.dataset,
            episode_idx=args.episode,
            start_step=args.start_step,
            max_steps=args.steps,
            speed=args.speed,
            visualize=not args.no_visualization,
            use_mujoco=not args.no_mujoco,
            camera_remap=camera_remap
        )
        return 0
    
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 