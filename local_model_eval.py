#!/usr/bin/env python3
"""
Local Model Evaluation Script

A completely local solution for evaluating ACT models without HF dependencies.
Combines dataset evaluation and robot simulation capabilities.

Usage:
    # Basic evaluation on dataset
    python local_model_eval.py ./models/red_cube_experiments/red_cube_40k_steps_10_episodes

    # With robot simulation 
    python local_model_eval.py ./models/red_cube_experiments/red_cube_40k_steps_10_episodes --simulate-robot --steps 50

    # Compare multiple models
    python local_model_eval.py ./single_episode_model ./models/red_cube_experiments/red_cube_40k_steps_10_episodes --compare

    # Full analysis with plots
    python local_model_eval.py ./models/red_cube_experiments/red_cube_40k_steps_10_episodes --plot --save-results
"""

import argparse
import torch
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque
import json

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    print("⚠️  MuJoCo not available - robot simulation disabled")
    MUJOCO_AVAILABLE = False

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy


class LocalModelEvaluator:
    """Local model evaluation without HF dependencies."""
    
    def __init__(self, model_path, dataset_name="bearlover365/red_cube_always_in_same_place", device="auto"):
        self.model_path = Path(model_path)
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        print(f"🔧 Local Model Evaluator")
        print(f"   Model: {self.model_path}")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Device: {self.device}")
        
        # Load model and dataset
        self._load_model()
        self._load_dataset()
        
    def _load_model(self):
        """Load the ACT policy from local path."""
        print(f"📖 Loading model from {self.model_path}...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        
        try:
            # Try loading as pretrained model
            self.policy = ACTPolicy.from_pretrained(str(self.model_path))
        except Exception as e:
            print(f"   Failed to load as pretrained: {e}")
            # Try alternative loading methods here if needed
            raise
            
        self.policy.to(self.device)
        self.policy.eval()
        print("   ✅ Model loaded successfully!")
        
    def _load_dataset(self):
        """Load the dataset."""
        print(f"📊 Loading dataset {self.dataset_name}...")
        self.dataset = LeRobotDataset(self.dataset_name, video_backend="pyav")
        print(f"   ✅ Dataset loaded: {len(self.dataset)} samples, {self.dataset.num_episodes} episodes")
        
    def evaluate_on_dataset(self, episode_idx=0, max_steps=None):
        """Evaluate model on a specific dataset episode."""
        print(f"\n🔬 Dataset Evaluation - Episode {episode_idx}")
        print("-" * 40)
        
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
        print(f"\n📊 Dataset Evaluation Results:")
        print(f"   Mean Absolute Error: {mae:.6f}")
        print(f"   Mean Squared Error: {mse:.6f}")
        print(f"   Max Absolute Error: {max_error:.6f}")
        print(f"   Avg Prediction Time: {avg_pred_time*1000:.1f}ms")
        
        if mae < 0.01:
            print("   🎉 Excellent performance!")
        elif mae < 0.1:
            print("   ✅ Good performance")
        elif mae < 1.0:
            print("   ⚠️  Moderate performance")
        else:
            print("   ❌ Poor performance")
            
        return results
    
    def simulate_robot_control(self, episode_idx=0, start_step=0, max_steps=100, use_mujoco=True):
        """Simulate robot control with policy predictions."""
        print(f"\n🤖 Robot Control Simulation - Episode {episode_idx}")
        print("-" * 40)
        
        if not MUJOCO_AVAILABLE and use_mujoco:
            print("   ⚠️  MuJoCo not available, using kinematic simulation")
            use_mujoco = False
            
        # Initialize robot simulation
        robot_positions = np.zeros(6)
        robot_velocities = np.zeros(6)
        mujoco_model = None
        mujoco_data = None
        mujoco_joint_indices = None
        
        # Setup MuJoCo if available
        if use_mujoco and MUJOCO_AVAILABLE:
            try:
                xml_path = "lerobot_some_original_code/standalone_scene.xml"
                mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
                mujoco_data = mujoco.MjData(mujoco_model)
                
                # Map joint indices
                joint_names = [mujoco_model.joint(i).name for i in range(mujoco_model.njnt)]
                mujoco_joint_indices = [joint_names.index(str(i)) for i in range(1, 7)]
                print(f"   🔧 MuJoCo simulation initialized")
            except Exception as e:
                print(f"   ⚠️  MuJoCo setup failed: {e}")
                use_mujoco = False
        
        # Get episode data
        from_idx = self.dataset.episode_data_index["from"][episode_idx].item()
        to_idx = self.dataset.episode_data_index["to"][episode_idx].item()
        episode_length = to_idx - from_idx
        
        if start_step >= episode_length:
            start_step = 0
            
        max_steps = min(max_steps, episode_length - start_step)
        print(f"   Episode {episode_idx}: steps {start_step} to {start_step + max_steps}")
        
        # Tracking data
        action_history = []
        position_history = []
        velocity_history = []
        prediction_times = []
        tracking_errors = []
        
        self.policy.reset()
        
        # Run simulation
        try:
            if use_mujoco and mujoco_model:
                # Use MuJoCo viewer context manager
                with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
                    print("   🖥️  MuJoCo viewer launched!")
                    
                    for step in range(max_steps):
                        if not viewer.is_running():
                            break
                            
                        # Get dataset sample
                        dataset_idx = from_idx + start_step + step
                        sample = self.dataset[dataset_idx]
                        
                        # Policy prediction
                        batch = {}
                        for key, value in sample.items():
                            if key.startswith("observation.") and isinstance(value, torch.Tensor):
                                batch[key] = value.unsqueeze(0).to(self.device)
                        
                        start_time = time.time()
                        with torch.no_grad():
                            pred_chunk = self.policy.select_action(batch)
                            if pred_chunk.dim() == 3:
                                predicted_action = pred_chunk[0, 0, :].cpu().numpy()
                            else:
                                predicted_action = pred_chunk[0, :].cpu().numpy()
                        pred_time = time.time() - start_time
                        
                        # Apply to MuJoCo robot
                        joint_values_deg = predicted_action[:6]
                        joint_values_rad = np.deg2rad(joint_values_deg)
                        
                        for idx, val in zip(mujoco_joint_indices, joint_values_rad):
                            if idx < len(mujoco_data.qpos):
                                mujoco_data.qpos[idx] = val
                        
                        mujoco.mj_step(mujoco_model, mujoco_data)
                        
                        # Get actual robot state
                        actual_positions = []
                        actual_velocities = []
                        for idx in mujoco_joint_indices:
                            if idx < len(mujoco_data.qpos):
                                actual_positions.append(mujoco_data.qpos[idx])
                                actual_velocities.append(mujoco_data.qvel[idx] if idx < len(mujoco_data.qvel) else 0.0)
                            else:
                                actual_positions.append(0.0)
                                actual_velocities.append(0.0)
                        
                        robot_positions = np.array(actual_positions)
                        robot_velocities = np.array(actual_velocities)
                        
                        # Track performance
                        robot_pos_deg = np.rad2deg(robot_positions)
                        tracking_error = np.linalg.norm(predicted_action - robot_pos_deg)
                        
                        action_history.append(predicted_action.copy())
                        position_history.append(robot_positions.copy())
                        velocity_history.append(robot_velocities.copy())
                        prediction_times.append(pred_time)
                        tracking_errors.append(tracking_error)
                        
                        # Update viewer
                        viewer.sync()
                        
                        # Progress
                        if step % 10 == 0:
                            print(f"   Step {step:3d}: Tracking_Error={tracking_error:.4f}°, Pred_time={pred_time*1000:.1f}ms")
                        
                        time.sleep(0.01)  # Small delay for visualization
            else:
                # Kinematic simulation without MuJoCo
                print("   🔧 Using kinematic simulation")
                
                for step in range(max_steps):
                    # Get dataset sample
                    dataset_idx = from_idx + start_step + step
                    sample = self.dataset[dataset_idx]
                    
                    # Policy prediction
                    batch = {}
                    for key, value in sample.items():
                        if key.startswith("observation.") and isinstance(value, torch.Tensor):
                            batch[key] = value.unsqueeze(0).to(self.device)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        pred_chunk = self.policy.select_action(batch)
                        if pred_chunk.dim() == 3:
                            predicted_action = pred_chunk[0, 0, :].cpu().numpy()
                        else:
                            predicted_action = pred_chunk[0, :].cpu().numpy()
                    pred_time = time.time() - start_time
                    
                    # Simple kinematic model
                    dt = 0.01
                    max_velocity = 2.0
                    target_rad = np.deg2rad(predicted_action[:6])
                    position_error = target_rad - robot_positions
                    desired_velocity = np.clip(position_error * 5.0, -max_velocity, max_velocity)
                    robot_positions += desired_velocity * dt
                    robot_velocities = desired_velocity
                    
                    # Track performance
                    robot_pos_deg = np.rad2deg(robot_positions)
                    tracking_error = np.linalg.norm(predicted_action - robot_pos_deg)
                    
                    action_history.append(predicted_action.copy())
                    position_history.append(robot_positions.copy())
                    velocity_history.append(robot_velocities.copy())
                    prediction_times.append(pred_time)
                    tracking_errors.append(tracking_error)
                    
                    # Progress
                    if step % 10 == 0:
                        print(f"   Step {step:3d}: Tracking_Error={tracking_error:.4f}°, Pred_time={pred_time*1000:.1f}ms")
                    
                    time.sleep(0.03)  # Visualization delay
                        
        except KeyboardInterrupt:
            print("\n   🛑 Simulation stopped by user")
        
        # Analyze results
        if action_history:
            actions = np.array(action_history)
            positions = np.array(position_history)
            velocities = np.array(velocity_history)
            
            # Control smoothness
            if len(actions) > 1:
                action_changes = np.linalg.norm(np.diff(actions, axis=0), axis=1)
                control_smoothness = np.mean(action_changes)
            else:
                control_smoothness = 0.0
            
            # Velocity smoothness
            if len(velocities) > 1:
                velocity_changes = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
                velocity_smoothness = np.mean(velocity_changes)
            else:
                velocity_smoothness = 0.0
            
            avg_tracking_error = np.mean(tracking_errors)
            max_tracking_error = np.max(tracking_errors)
            avg_pred_time = np.mean(prediction_times)
            
            results = {
                'episode': episode_idx,
                'start_step': start_step,
                'steps': len(action_history),
                'control_smoothness': control_smoothness,
                'velocity_smoothness': velocity_smoothness,
                'avg_tracking_error': avg_tracking_error,
                'max_tracking_error': max_tracking_error,
                'avg_prediction_time': avg_pred_time,
                'used_mujoco': use_mujoco
            }
            
            print(f"\n🎯 Robot Control Results:")
            print(f"   Control Smoothness: {control_smoothness:.4f}")
            print(f"   Velocity Smoothness: {velocity_smoothness:.4f}")
            print(f"   Avg Tracking Error: {avg_tracking_error:.4f}°")
            print(f"   Max Tracking Error: {max_tracking_error:.4f}°")
            print(f"   Avg Prediction Time: {avg_pred_time*1000:.1f}ms")
            print(f"   Steps Simulated: {len(action_history)}")
            
            # Performance assessment
            print(f"\n💡 Assessment:")
            if control_smoothness < 0.1:
                print("   ✅ Very smooth control - excellent!")
            elif control_smoothness < 0.5:
                print("   ✅ Good control smoothness")
            elif control_smoothness < 1.0:
                print("   ⚠️  Somewhat jerky control")
            else:
                print("   ❌ Very jerky control")
                
            if avg_tracking_error < 0.1:
                print("   ✅ Excellent tracking performance")
            elif avg_tracking_error < 0.5:
                print("   ✅ Good tracking performance")
            else:
                print("   ⚠️  Poor tracking performance")
                
            return results
        else:
            print("   ❌ No simulation data collected")
            return None


def plot_comparison(results_list, model_names, save_path=None):
    """Plot comparison between multiple models."""
    if not results_list:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔍 Model Comparison Results', fontsize=16)
    
    # Dataset metrics
    if all('mae' in r for r in results_list if r):
        maes = [r['mae'] for r in results_list if r and 'mae' in r]
        model_subset = [name for r, name in zip(results_list, model_names) if r and 'mae' in r]
        
        axes[0, 0].bar(model_subset, maes, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Dataset MAE')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
    
    # Robot control metrics
    if all(r and 'control_smoothness' in r for r in results_list):
        smoothness = [r['control_smoothness'] for r in results_list]
        tracking = [r['avg_tracking_error'] for r in results_list]
        
        axes[0, 1].bar(model_names, smoothness, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Control Smoothness')
        axes[0, 1].set_ylabel('Action Change Magnitude')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].bar(model_names, tracking, alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('Tracking Error')
        axes[1, 0].set_ylabel('Average Error (degrees)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Performance summary table
    axes[1, 1].axis('off')
    
    # Create summary text
    summary_text = "📊 Performance Summary\n\n"
    for i, (result, name) in enumerate(zip(results_list, model_names)):
        if result:
            summary_text += f"{name}:\n"
            if 'mae' in result:
                summary_text += f"  Dataset MAE: {result['mae']:.6f}\n"
            if 'control_smoothness' in result:
                summary_text += f"  Control Smoothness: {result['control_smoothness']:.4f}\n"
                summary_text += f"  Tracking Error: {result['avg_tracking_error']:.4f}°\n"
            summary_text += "\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Local Model Evaluation (No HF Required!)')
    parser.add_argument('model_paths', nargs='+', help='Path(s) to trained model directories')
    parser.add_argument('--dataset', default='bearlover365/red_cube_always_in_same_place', help='Dataset name')
    parser.add_argument('--episode', type=int, default=0, help='Episode to evaluate on')
    parser.add_argument('--start-step', type=int, default=0, help='Starting step for robot simulation')
    parser.add_argument('--steps', type=int, default=50, help='Max steps for evaluation/simulation')
    parser.add_argument('--simulate-robot', action='store_true', help='Run robot control simulation')
    parser.add_argument('--no-mujoco', action='store_true', help='Disable MuJoCo (use kinematic sim)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--save-results', action='store_true', help='Save results to JSON')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    print("🚀 Local Model Evaluation System")
    print("=" * 50)
    print("✅ No HuggingFace uploads required!")
    print("✅ Works with local model directories!")
    print("✅ Combines dataset + robot evaluation!")
    print()
    
    # Validate model paths
    for model_path in args.model_paths:
        if not Path(model_path).exists():
            print(f"❌ Model path does not exist: {model_path}")
            return 1
    
    results = []
    model_names = [Path(p).name for p in args.model_paths]
    
    try:
        for i, model_path in enumerate(args.model_paths):
            print(f"\n{'='*20} Model {i+1}/{len(args.model_paths)}: {model_names[i]} {'='*20}")
            
            evaluator = LocalModelEvaluator(model_path, args.dataset, args.device)
            
            result = {}
            
            # Dataset evaluation
            dataset_result = evaluator.evaluate_on_dataset(args.episode, args.steps)
            if dataset_result:
                result.update(dataset_result)
            
            # Robot simulation
            if args.simulate_robot:
                sim_result = evaluator.simulate_robot_control(
                    args.episode, args.start_step, args.steps, 
                    use_mujoco=not args.no_mujoco
                )
                if sim_result:
                    result.update(sim_result)
            
            results.append(result if result else None)
        
        # Comparison and plotting
        if args.compare and len(args.model_paths) > 1:
            print(f"\n{'='*20} Model Comparison {'='*20}")
            
            # Print comparison table
            print(f"\n{'Model':<30} {'Dataset MAE':<15} {'Control Smooth':<15} {'Tracking Error':<15}")
            print("-" * 75)
            
            for name, result in zip(model_names, results):
                if result:
                    mae = result.get('mae', 'N/A')
                    smoothness = result.get('control_smoothness', 'N/A')
                    tracking = result.get('avg_tracking_error', 'N/A')
                    
                    mae_str = f"{mae:.6f}" if isinstance(mae, float) else str(mae)
                    smooth_str = f"{smoothness:.4f}" if isinstance(smoothness, float) else str(smoothness)
                    track_str = f"{tracking:.4f}°" if isinstance(tracking, float) else str(tracking)
                    
                    print(f"{name:<30} {mae_str:<15} {smooth_str:<15} {track_str:<15}")
            
            # Best model analysis
            dataset_results = [r for r in results if r and 'mae' in r]
            sim_results = [r for r in results if r and 'control_smoothness' in r]
            
            if dataset_results:
                best_dataset_idx = min(range(len(dataset_results)), key=lambda i: dataset_results[i]['mae'])
                best_dataset_model = [name for name, r in zip(model_names, results) if r and 'mae' in r][best_dataset_idx]
                print(f"\n🏆 Best Dataset Performance: {best_dataset_model}")
            
            if sim_results:
                best_sim_idx = min(range(len(sim_results)), key=lambda i: sim_results[i]['control_smoothness'])
                best_sim_model = [name for name, r in zip(model_names, results) if r and 'control_smoothness' in r][best_sim_idx]
                print(f"🤖 Best Robot Control: {best_sim_model}")
        
        # Generate plots
        if args.plot and len(args.model_paths) > 1:
            save_path = "model_comparison.png" if args.save_results else None
            plot_comparison(results, model_names, save_path)
        
        # Save results
        if args.save_results:
            output_data = {
                'models': model_names,
                'results': results,
                'evaluation_config': {
                    'dataset': args.dataset,
                    'episode': args.episode,
                    'steps': args.steps,
                    'simulate_robot': args.simulate_robot,
                    'device': str(args.device)
                }
            }
            
            output_file = "evaluation_results.json"
            with open(output_file, 'w') as f:
                # Convert tensors to lists for JSON serialization
                def convert_tensors(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_tensors(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_tensors(v) for v in obj]
                    else:
                        return obj
                
                json.dump(convert_tensors(output_data), f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
        
        print(f"\n🎉 Local evaluation complete!")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 