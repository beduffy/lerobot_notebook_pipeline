#!/usr/bin/env python3
"""
Local Robot Evaluation Script

Live robot evaluation using local model paths (no HF uploads required).
Connects to real robot, gets live camera feed, runs policy inference, sends actions.

Equivalent to:
python -m lerobot.record --policy.path=HF_MODEL --robot.type=so101_follower ...

But with local model paths:
python local_robot_eval.py ./models/red_cube_experiments/red_cube_40k_steps_10_episodes

Usage:
    # Basic live evaluation 
    python local_robot_eval.py ./models/red_cube_experiments/red_cube_40k_steps_10_episodes \
        --robot-port=/dev/ttyACM0 \
        --robot-type=so101_follower \
        --robot-id=my_awesome_follower_arm \
        --robot-cameras='{"front": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}}' \
        --task="Grab red cube and put to left" \
        --episodes=1

    # With recording to local dataset
    python local_robot_eval.py ./single_episode_model_20k_steps \
        --robot-port=/dev/ttyACM0 \
        --robot-type=so101_follower \
        --robot-id=my_awesome_follower_arm \
        --robot-cameras='{"front": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}}' \
        --task="Grab red cube and put to left" \
        --episodes=3 \
        --record \
        --dataset-repo-id=local_eval_results \
        --dataset-root=./eval_datasets
"""

import argparse
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    print("⚠️  Rerun not available - visualization disabled")
    RERUN_AVAILABLE = False

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robots import make_robot_from_config, so101_follower
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.utils.control_utils import predict_action, init_keyboard_listener
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, log_say
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@dataclass  
class LocalRobotEvalConfig:
    """Configuration for local robot evaluation."""
    # Robot configuration
    robot_type: str = "so101_follower"
    robot_port: str = "/dev/ttyACM0" 
    robot_id: str = "my_awesome_follower_arm"
    robot_cameras: dict = None
    
    # Task configuration
    task: str = "Grab red cube and put to left"
    episodes: int = 1
    episode_time_s: float = 60.0
    reset_time_s: float = 7.0
    fps: int = 30
    
    # Recording configuration  
    record: bool = False
    dataset_repo_id: str = "local_eval_results"
    dataset_root: str = "./eval_datasets"
    
    # Display and control
    display_data: bool = True
    play_sounds: bool = True
    
    def __post_init__(self):
        if self.robot_cameras is None:
            # Default camera config
            self.robot_cameras = {
                "front": {
                    "type": "opencv", 
                    "index_or_path": 4, 
                    "width": 640, 
                    "height": 480, 
                    "fps": 30
                }
            }


class LocalRobotEvaluator:
    """Live robot evaluation with local policy."""
    
    def __init__(self, model_path: str, config: LocalRobotEvalConfig):
        self.model_path = Path(model_path)
        self.config = config
        # Use robust device detection - avoid "auto" which isn't supported
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"🤖 Local Robot Evaluator")
        print(f"   Model: {self.model_path}")
        print(f"   Robot: {config.robot_type} on {config.robot_port}")
        print(f"   Device: {self.device}")
        print(f"   Task: {config.task}")
        
        # Load policy
        self._load_policy()
        
        # Setup robot
        self._setup_robot()
        
        # Setup dataset if recording, but always create features for observation processing
        self.dataset = None
        self._setup_features()  # Always setup features for observation processing
        if config.record:
            self._setup_dataset()
    
    def _load_policy(self):
        """Load policy from local path."""
        print(f"📖 Loading policy from {self.model_path}...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        
        try:
            self.policy = ACTPolicy.from_pretrained(str(self.model_path))
            self.policy.to(self.device)
            self.policy.eval()
            print("   ✅ Policy loaded successfully!")
        except Exception as e:
            print(f"   ❌ Failed to load policy: {e}")
            raise
    
    def _setup_robot(self):
        """Setup robot connection."""
        print(f"🔧 Setting up robot...")
        print(f"   Robot type: {self.config.robot_type}")
        print(f"   Robot ID: {self.config.robot_id} (important for calibration)")
        print(f"   Robot port: {self.config.robot_port}")
        
        # Convert camera dict configs to proper CameraConfig objects
        camera_configs = {}
        for camera_name, camera_dict in self.config.robot_cameras.items():
            if camera_dict["type"] == "opencv":
                camera_configs[camera_name] = OpenCVCameraConfig(
                    index_or_path=camera_dict["index_or_path"],
                    width=camera_dict["width"],
                    height=camera_dict["height"],
                    fps=camera_dict["fps"]
                )
            else:
                raise ValueError(f"Unsupported camera type: {camera_dict['type']}")
        
        print(f"   Camera configs: {list(camera_configs.keys())}")
        
        # Create robot config (following lerobot record approach)
        if self.config.robot_type == "so101_follower":
            from lerobot.common.robots.so101_follower import SO101FollowerConfig
            
            robot_config = SO101FollowerConfig(
                port=self.config.robot_port,
                cameras=camera_configs,  # Use proper CameraConfig objects
                id=self.config.robot_id  # This ID is crucial for calibration data
            )
        else:
            raise ValueError(f"Unsupported robot type: {self.config.robot_type}")
        
        self.robot = make_robot_from_config(robot_config)
        print("   ✅ Robot configuration created!")
    
    def _setup_features(self):
        """Setup features dictionary for observation processing (always needed)."""
        action_features = hw_to_dataset_features(self.robot.action_features, "action", True)
        obs_features = hw_to_dataset_features(self.robot.observation_features, "observation", True) 
        self.features = {**action_features, **obs_features}
        print(f"   📋 Features configured: {list(self.features.keys())}")
    
    def _setup_dataset(self):
        """Setup dataset for recording evaluation results."""
        print(f"📊 Setting up dataset for recording...")
        
        # Create dataset using pre-configured features
        self.dataset = LeRobotDataset.create(
            self.config.dataset_repo_id,
            self.config.fps,
            root=self.config.dataset_root,
            robot_type=self.robot.name,
            features=self.features,  # Use pre-configured features
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * len(self.robot.cameras) if hasattr(self.robot, 'cameras') else 4,
        )
        print("   ✅ Dataset ready for recording!")
    
    def run_evaluation_episode(self, episode_idx: int):
        """Run a single evaluation episode with live robot."""
        print(f"\n🎯 Episode {episode_idx + 1}/{self.config.episodes}")
        print("-" * 40)
        
        # Reset policy
        self.policy.reset()
        
        # Episode metrics
        action_count = 0
        prediction_times = []
        start_time = time.perf_counter()
        
        print(f"   Starting {self.config.episode_time_s}s episode...")
        if self.config.play_sounds:
            log_say(f"Starting episode {episode_idx + 1}", True)
        
        # Main control loop (similar to record_loop in lerobot.record)
        timestamp = 0
        start_episode_t = time.perf_counter()
        
        while timestamp < self.config.episode_time_s:
            start_loop_t = time.perf_counter()
            
            # Get live observation from robot
            observation = self.robot.get_observation()
            
            # Prepare observation for policy
            observation_frame = build_dataset_frame(
                self.features,  # Always use self.features (setup in _setup_features)
                observation, 
                prefix="observation"
            )
            
            # Predict action using policy
            pred_start = time.perf_counter()
            action_values = predict_action(
                observation_frame,
                self.policy,
                self.device,
                self.policy.config.use_amp,
                task=self.config.task,
                robot_type=self.robot.robot_type,
            )
            pred_time = time.perf_counter() - pred_start
            prediction_times.append(pred_time)
            
            # Convert to robot action format
            action = {key: action_values[i].item() for i, key in enumerate(self.robot.action_features)}
            
            # Send action to robot (this returns the actual action sent after clipping)
            sent_action = self.robot.send_action(action)
            
            action_count += 1
            
                    # Record to dataset if enabled
        if self.dataset is not None:
            action_frame = build_dataset_frame(self.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            self.dataset.add_frame(frame, task=self.config.task)
            
            # Display data if enabled
            if self.config.display_data and RERUN_AVAILABLE:
                for obs_key, obs_val in observation.items():
                    if isinstance(obs_val, float):
                        rr.log(f"observation.{obs_key}", rr.Scalar(obs_val))
                    elif isinstance(obs_val, np.ndarray):
                        rr.log(f"observation.{obs_key}", rr.Image(obs_val), static=True)
                
                for act_key, act_val in action.items():
                    if isinstance(act_val, float):
                        rr.log(f"action.{act_key}", rr.Scalar(act_val))
            
            # Progress update
            if action_count % (self.config.fps * 5) == 0:  # Every 5 seconds
                elapsed = time.perf_counter() - start_time
                avg_pred_time = np.mean(prediction_times[-50:]) if prediction_times else 0
                print(f"   {elapsed:.1f}s: Actions={action_count}, Pred_time={avg_pred_time*1000:.1f}ms")
            
            # Maintain control frequency
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / self.config.fps - dt_s)
            
            timestamp = time.perf_counter() - start_episode_t
        
        # Episode completed
        elapsed_time = time.perf_counter() - start_time
        avg_pred_time = np.mean(prediction_times) if prediction_times else 0
        
        print(f"   ✅ Episode completed!")
        print(f"   Duration: {elapsed_time:.1f}s")
        print(f"   Actions sent: {action_count}")
        print(f"   Avg prediction time: {avg_pred_time*1000:.1f}ms")
        print(f"   Control frequency: {action_count/elapsed_time:.1f} Hz")
        
        # Save episode if recording
        if self.dataset is not None:
            self.dataset.save_episode()
            print(f"   💾 Episode saved to dataset")
        
        return {
            'episode': episode_idx,
            'duration_s': elapsed_time,
            'actions_sent': action_count,
            'avg_prediction_time_ms': avg_pred_time * 1000,
            'control_frequency_hz': action_count / elapsed_time
        }
    
    def run_reset_period(self, episode_idx: int):
        """Run reset period between episodes."""
        if episode_idx < self.config.episodes - 1:  # Don't reset after last episode
            print(f"\n⏸️  Reset period ({self.config.reset_time_s}s)")
            if self.config.play_sounds:
                log_say("Reset the environment", True)
            
            # Run robot without recording during reset
            start_reset_t = time.perf_counter()
            timestamp = 0
            
            while timestamp < self.config.reset_time_s:
                start_loop_t = time.perf_counter()
                
                # Just maintain robot connection, no actions
                observation = self.robot.get_observation()
                
                # Maintain timing
                dt_s = time.perf_counter() - start_loop_t  
                busy_wait(1 / self.config.fps - dt_s)
                
                timestamp = time.perf_counter() - start_reset_t
            
            print("   Ready for next episode!")
    
    def run_evaluation(self):
        """Run complete evaluation with all episodes."""
        print(f"\n🚀 Starting Live Robot Evaluation")
        print("=" * 50)
        print(f"Model: {self.model_path.name}")
        print(f"Episodes: {self.config.episodes}")
        print(f"Task: {self.config.task}")
        print(f"Recording: {'Yes' if self.config.record else 'No'}")
        print()
        
        # Initialize display
        if self.config.display_data and RERUN_AVAILABLE:
            _init_rerun(session_name="local_robot_eval")
        
        # Connect to robot
        print("🔌 Connecting to robot...")
        self.robot.connect()
        print("   ✅ Robot connected!")
        
        # Initialize keyboard listener for emergency stop
        listener, events = init_keyboard_listener()
        
        results = []
        
        try:
            for episode_idx in range(self.config.episodes):
                # Run evaluation episode
                episode_result = self.run_evaluation_episode(episode_idx)
                results.append(episode_result)
                
                # Check for early exit
                if events and events.get("stop_recording", False):
                    print("\n🛑 Early termination requested")
                    break
                
                # Reset period between episodes
                self.run_reset_period(episode_idx)
            
            # Summary
            print(f"\n🎉 Evaluation Complete!")
            print("=" * 30)
            
            total_actions = sum(r['actions_sent'] for r in results)
            total_time = sum(r['duration_s'] for r in results)
            avg_pred_time = np.mean([r['avg_prediction_time_ms'] for r in results])
            
            print(f"Episodes completed: {len(results)}")
            print(f"Total actions: {total_actions}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Average prediction time: {avg_pred_time:.1f}ms")
            print(f"Overall control frequency: {total_actions/total_time:.1f} Hz")
            
            if self.config.record and self.dataset:
                print(f"Dataset saved to: {self.config.dataset_root}/{self.config.dataset_repo_id}")
            
            return results
            
        except KeyboardInterrupt:
            print("\n🛑 Evaluation interrupted by user")
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            raise
            
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            try:
                self.robot.disconnect()
                print("   Robot disconnected")
            except:
                pass
                
            if listener:
                try:
                    listener.stop()
                    print("   Keyboard listener stopped")
                except:
                    pass
        
        return results


def parse_cameras_arg(cameras_str: str) -> dict:
    """Parse cameras argument string to dict."""
    try:
        # Simple eval of the string (be careful with untrusted input)
        import ast
        return ast.literal_eval(cameras_str)
    except:
        # Fallback to default
        return {
            "front": {
                "type": "opencv",
                "index_or_path": 4,
                "width": 640,
                "height": 480,
                "fps": 30
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Local Robot Evaluation (No HF Required!)')
    parser.add_argument('model_path', help='Path to local trained model directory')
    
    # Robot configuration (IMPORTANT: ID and type needed for calibration!)
    parser.add_argument('--robot-type', default='so101_follower', help='Robot type (important for calibration)')
    parser.add_argument('--robot-port', default='/dev/ttyACM0', help='Robot port')
    parser.add_argument('--robot-id', default='my_awesome_follower_arm', help='Robot ID (important for calibration)')
    parser.add_argument('--robot-cameras', type=str, 
                       default='{"front": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}}',
                       help='Camera configuration (JSON string)')
    
    # Task configuration
    parser.add_argument('--task', default='Grab red cube and put to left', help='Task description')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--episode-time', type=float, default=60.0, help='Episode duration (seconds)')
    parser.add_argument('--reset-time', type=float, default=7.0, help='Reset duration (seconds)')
    parser.add_argument('--fps', type=int, default=30, help='Control frequency')
    
    # Recording configuration
    parser.add_argument('--record', action='store_true', help='Record evaluation results to dataset')
    parser.add_argument('--dataset-repo-id', default='local_eval_results', help='Dataset repository ID')
    parser.add_argument('--dataset-root', default='./eval_datasets', help='Dataset root directory')
    
    # Display and control
    parser.add_argument('--no-display', action='store_true', help='Disable data visualization')
    parser.add_argument('--no-sounds', action='store_true', help='Disable sound notifications')
    
    args = parser.parse_args()
    
    print("🚀 Local Robot Evaluation System")
    print("=" * 40)
    print("✅ No HuggingFace uploads required!")
    print("✅ Works with local model directories!")
    print("✅ Live robot control and evaluation!")
    print()
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"❌ Model path does not exist: {args.model_path}")
        print(f"   Looking for: {Path(args.model_path).absolute()}")
        return 1
    
    try:
        # Parse camera configuration
        cameras = parse_cameras_arg(args.robot_cameras)
        
        # Create configuration
        config = LocalRobotEvalConfig(
            robot_type=args.robot_type,
            robot_port=args.robot_port, 
            robot_id=args.robot_id,
            robot_cameras=cameras,
            task=args.task,
            episodes=args.episodes,
            episode_time_s=args.episode_time,
            reset_time_s=args.reset_time,
            fps=args.fps,
            record=args.record,
            dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root,
            display_data=not args.no_display,
            play_sounds=not args.no_sounds
        )
        
        # Run evaluation
        evaluator = LocalRobotEvaluator(args.model_path, config)
        results = evaluator.run_evaluation()
        
        # Save results summary
        if results:
            results_file = Path(args.dataset.root) / f"{config.dataset_repo_id}_summary.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            summary = {
                'model_path': str(args.model_path),
                'config': config.__dict__,
                'results': results,
                'timestamp': time.time()
            }
            
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📊 Results summary saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 