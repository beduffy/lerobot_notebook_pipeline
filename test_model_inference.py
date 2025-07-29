#!/usr/bin/env python3
"""
Multi-Model Inference Test Script with FPS Profiling

Tests inference for all available lerobot model architectures before training.
This is like a "unit test" to ensure all model types work with your pipeline.
Now includes FPS profiling to measure real-time performance!

Available models to test:
- ACT (already working)
- Diffusion
- SmolVLA (your focus)  
- PI0 (pi zero)
- PI0FAST
- TDMPC
- VQBet

Usage:
    # Test all models with FPS profiling
    python test_model_inference.py
    
    # Test specific models only
    python test_model_inference.py --models act,diffusion,smolvla
    
    # Test with your dataset
    python test_model_inference.py --dataset bearlover365/red_cube_always_in_same_place
    
    # Test with pretrained models (for SmolVLA)
    python test_model_inference.py --models smolvla --use-pretrained
    
    # Quick test (fewer FPS samples)
    python test_model_inference.py --quick
    
    # Detailed profiling (more samples)
    python test_model_inference.py --detailed
"""

import argparse
import torch
import warnings
from pathlib import Path
import traceback
from typing import Dict, Any
import numpy as np
import time
import gc
import psutil
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# LeRobot imports - Updated for new structure
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.datasets.factory import resolve_delta_timestamps
    LEROBOT_NEW_STRUCTURE = True
except ImportError:
    # Fallback to old structure
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    LEROBOT_NEW_STRUCTURE = False


class ModelInferenceTester:
    """Test inference for different model architectures with FPS profiling."""
    
    def __init__(self, dataset_name: str, device: str = "auto", profile_samples: int = 100):
        self.dataset_name = dataset_name
        self.profile_samples = profile_samples
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🧪 Model Inference Tester with FPS Profiling")
        print(f"   Dataset: {dataset_name}")
        print(f"   Device: {self.device}")
        print(f"   Profile samples: {profile_samples}")
        print()
        
        # Load dataset and metadata
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset and extract sample data for testing."""
        print(f"📊 Loading dataset...")
        
        # Load metadata
        self.metadata = LeRobotDatasetMetadata(self.dataset_name)
        print(f"   Episodes: {self.metadata.total_episodes}")
        print(f"   Frames: {self.metadata.total_frames}")
        
        # Load dataset
        self.dataset = LeRobotDataset(self.dataset_name, video_backend="pyav")
        
        # Get sample observation for testing
        sample = self.dataset[0]
        self.sample_observation = {}
        for key, value in sample.items():
            if key.startswith("observation.") and isinstance(value, torch.Tensor):
                self.sample_observation[key] = value.unsqueeze(0)  # Add batch dimension
        
        print(f"   Sample observation keys: {list(self.sample_observation.keys())}")
        
        # Extract features
        self.features = dataset_to_policy_features(self.metadata.features)
        self.output_features = {key: ft for key, ft in self.features.items() if ft.type is FeatureType.ACTION}
        self.input_features = {key: ft for key, ft in self.features.items() if key not in self.output_features}
        
        print(f"   Input features: {list(self.input_features.keys())}")
        print(f"   Output features: {list(self.output_features.keys())}")
        print()
    
    def measure_fps_and_memory(self, policy, obs_dict: Dict[str, torch.Tensor], model_name: str) -> Dict[str, Any]:
        """Measure FPS and memory usage for a policy."""
        print(f"   🔍 Profiling {model_name} performance...")
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            gpu_memory_before = 0
        
        # Calculate model parameter memory (approximate)
        param_count = sum(p.numel() for p in policy.parameters())
        # Assume float32 parameters (4 bytes each)
        param_memory_mb = param_count * 4 / 1024 / 1024
        
        # Cold start - first inference (often slower)
        policy.eval()
        with torch.no_grad():
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            cold_start = time.time()
            action = policy.select_action(obs_dict)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            cold_time = time.time() - cold_start
        
        # Measure memory after first inference
        memory_after_inference = process.memory_info().rss / 1024 / 1024  # MB
        if self.device.type == "cuda":
            gpu_memory_after_inference = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            gpu_memory_after_inference = 0
        
        # Warm up with a few runs
        with torch.no_grad():
            for _ in range(min(10, self.profile_samples // 10)):
                policy.select_action(obs_dict)
        
        # Measure warm inference performance
        times = []
        with torch.no_grad():
            for i in range(self.profile_samples):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                action = policy.select_action(obs_dict)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
                
                # Progress indicator for long profiling
                if i % (self.profile_samples // 4) == 0 and self.profile_samples > 50:
                    print(f"     Progress: {i}/{self.profile_samples}")
        
        # Final memory measurement
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        if self.device.type == "cuda":
            gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            gpu_memory_after = 0
        
        # Calculate statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # FPS calculations
        mean_fps = 1.0 / mean_time
        max_fps = 1.0 / min_time  # Best case FPS
        min_fps = 1.0 / max_time  # Worst case FPS
        
        # Memory usage calculations
        process_memory_used = memory_after_inference - memory_before
        gpu_memory_used = gpu_memory_after_inference - gpu_memory_before
        
        performance_stats = {
            'cold_start_time': cold_time,
            'cold_start_fps': 1.0 / cold_time,
            'mean_inference_time': mean_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'mean_fps': mean_fps,
            'max_fps': max_fps,
            'min_fps': min_fps,
            'process_memory_used_mb': process_memory_used,
            'gpu_memory_used_mb': gpu_memory_used,
            'param_memory_mb': param_memory_mb,
            'total_memory_estimate_mb': max(process_memory_used, param_memory_mb),
            'action_shape': action.shape,
            'samples': self.profile_samples
        }
        
        # Print immediate results with better memory info
        total_mem = max(process_memory_used, param_memory_mb)
        print(f"     ⚡ Mean FPS: {mean_fps:.1f} | Max FPS: {max_fps:.1f}")
        print(f"     🕐 Cold start: {cold_time*1000:.1f}ms | Warm: {mean_time*1000:.1f}±{std_time*1000:.1f}ms")
        print(f"     💾 Memory: ~{total_mem:.1f}MB | Params: {param_memory_mb:.1f}MB | GPU: {gpu_memory_used:.1f}MB")
        
        return performance_stats
    
    def test_act_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test ACT model (your working baseline)."""
        print("🤖 Testing ACT Model...")
        
        try:
            # Try new ACT import structure first
            try:
                from lerobot.policies.act.configuration_act import ACTConfig
                from lerobot.policies.act.modeling_act import ACTPolicy
            except ImportError:
                # Fallback to old structure
                from lerobot.common.policies.act.configuration_act import ACTConfig
                from lerobot.common.policies.act.modeling_act import ACTPolicy
            
            # Create config with your proven settings
            config = ACTConfig(
                input_features=self.input_features,
                output_features=self.output_features,
                chunk_size=100,
                n_action_steps=100,
                dim_model=512,
                n_heads=8,
                dim_feedforward=3200,
                n_encoder_layers=4,
                n_decoder_layers=1,
                vision_backbone="resnet18",
                use_vae=True,
                latent_dim=32,
            )
            
            # Create policy
            policy = ACTPolicy(config, dataset_stats=self.metadata.stats)
            policy.to(self.device)
            policy.eval()
            
            # Test basic inference first
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                action = policy.select_action(obs_dict)
            
            # Measure performance
            performance = self.measure_fps_and_memory(policy, obs_dict, "ACT")
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config,
                'performance': performance
            }
            
            print(f"   ✅ ACT inference successful!")
            print(f"   Parameters: {result['parameters']:,}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ ACT failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_diffusion_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test Diffusion Policy model."""
        print("🌊 Testing Diffusion Policy...")
        
        try:
            # Try new Diffusion import structure first
            try:
                from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
                from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
            except ImportError:
                # Fallback to old structure
                from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
                from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
            
            # Create config with minimal required parameters
            config = DiffusionConfig(
                input_features=self.input_features,
                output_features=self.output_features,
                horizon=16,  # Prediction horizon
                n_action_steps=8,  # Number of action steps to execute
                num_inference_steps=10,  # Diffusion denoising steps
            )
            
            # Create policy
            policy = DiffusionPolicy(config, dataset_stats=self.metadata.stats)
            policy.to(self.device)
            policy.eval()
            
            # Test basic inference first
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                action = policy.select_action(obs_dict)
            
            # Measure performance
            performance = self.measure_fps_and_memory(policy, obs_dict, "Diffusion")
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config,
                'performance': performance
            }
            
            print(f"   ✅ Diffusion inference successful!")
            print(f"   Parameters: {result['parameters']:,}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Diffusion failed: {e}")
            traceback.print_exc()
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_smolvla_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test SmolVLA model (your focus!)."""
        print("🧠 Testing SmolVLA Model...")
        
        try:
            # Try new SmolVLA import structure first
            try:
                from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            except ImportError:
                # Fallback to old structure
                from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
                from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            
            if use_pretrained:
                # Use pretrained SmolVLA
                print("   Loading pretrained SmolVLA...")
                policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
                config = policy.config
                
                # Update config with your dataset features
                config.input_features = self.input_features
                config.output_features = self.output_features
                
            else:
                # Create fresh config
                config = SmolVLAConfig(
                    input_features=self.input_features,
                    output_features=self.output_features,
                    # SmolVLA specific settings (minimal config)
                    # Note: use_quantization removed - not supported in this version
                )
                
                # Create policy
                policy = SmolVLAPolicy(config, dataset_stats=self.metadata.stats)
            
            policy.to(self.device)
            policy.eval()
            
            # Test basic inference first
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                # SmolVLA requires a task description (it's a VLA model!)
                obs_dict["task"] = "grab red cube and put to left"
                action = policy.select_action(obs_dict)
            
            # Measure performance
            performance = self.measure_fps_and_memory(policy, obs_dict, "SmolVLA")
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config,
                'pretrained': use_pretrained,
                'model_type': 'VLA',
                'performance': performance
            }
            
            print(f"   ✅ SmolVLA VLA inference successful!")
            print(f"   Parameters: {result['parameters']:,}")
            print(f"   Pretrained: {use_pretrained}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ SmolVLA failed: {e}")
            traceback.print_exc()
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_pi0_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test π0 (Pi Zero) VLA model - NOW OFFICIALLY AVAILABLE!"""
        print("🥧 Testing π0 (Pi Zero) VLA Model...")
        
        try:
            # Try new π0 import structure first
            try:
                from lerobot.policies.pi0.modeling_pi0 import PI0Policy
                from lerobot.policies.pi0.configuration_pi0 import PI0Config
            except ImportError:
                # Fallback to old structure
                from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
                from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
            
            if use_pretrained:
                # Use official π0 pretrained model from LeRobot
                print("   Loading official π0 pretrained model...")
                policy = PI0Policy.from_pretrained("lerobot/pi0")
                config = policy.config
            else:
                # Create fresh config
                
                config = PI0Config(
                    input_features=self.input_features,
                    output_features=self.output_features,
                    # π0 VLA specific settings
                    chunk_size=10,  # VLA models use smaller chunks
                    n_action_steps=10,
                )
                
                # Create policy
                policy = PI0Policy(config, dataset_stats=self.metadata.stats)
            
            policy.to(self.device)
            policy.eval()
            
            # Test basic inference first
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                # π0 requires a task description (it's a VLA model!)
                obs_dict["task"] = "grab red cube and put to left"
                action = policy.select_action(obs_dict)
            
            # Measure performance
            performance = self.measure_fps_and_memory(policy, obs_dict, "π0")
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config,
                'pretrained': use_pretrained,
                'model_type': 'VLA',
                'performance': performance
            }
            
            print(f"   ✅ π0 VLA inference successful!")
            print(f"   Parameters: {result['parameters']:,}")
            print(f"   Pretrained: {use_pretrained}")
            print(f"   Type: Vision-Language-Action Model")
            
            return result
            
        except Exception as e:
            print(f"   ❌ π0 failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_pi0fast_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test π0-FAST (Pi Zero FAST) VLA model - AUTOREGRESSIVE VERSION!"""
        print("⚡ Testing π0-FAST (Pi Zero FAST) VLA Model...")
        
        try:
            # Try new π0-FAST import structure first
            try:
                from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
                from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
            except ImportError:
                # Fallback to old structure
                from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
                from lerobot.common.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
            
            if use_pretrained:
                # Use official π0-FAST pretrained model from LeRobot
                print("   Loading official π0-FAST pretrained model...")
                policy = PI0FASTPolicy.from_pretrained("lerobot/pi0fast")  # Assuming this path
                config = policy.config
            else:
                # Create fresh config
                
                config = PI0FASTConfig(
                    input_features=self.input_features,
                    output_features=self.output_features,
                    # π0-FAST specific settings (autoregressive)
                    chunk_size=10,
                    n_action_steps=10,
                )
                
                # Create policy
                policy = PI0FASTPolicy(config, dataset_stats=self.metadata.stats)
            
            policy.to(self.device)
            policy.eval()
            
            # Test basic inference first
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                # π0-FAST requires a task description (it's a VLA model!)
                obs_dict["task"] = "grab red cube and put to left"
                action = policy.select_action(obs_dict)
            
            # Measure performance
            performance = self.measure_fps_and_memory(policy, obs_dict, "π0-FAST")
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config,
                'pretrained': use_pretrained,
                'model_type': 'VLA-FAST',
                'performance': performance
            }
            
            print(f"   ✅ π0-FAST VLA inference successful!")
            print(f"   Parameters: {result['parameters']:,}")
            print(f"   Pretrained: {use_pretrained}")
            print(f"   Type: Autoregressive Vision-Language-Action Model (5x faster training)")
            
            return result
            
        except Exception as e:
            print(f"   ❌ π0-FAST failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_vqbet_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test VQBet model."""
        print("🎰 Testing VQBet Model...")
        
        try:
            # Try new VQBet import structure first
            try:
                from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
                from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
            except ImportError:
                # Fallback to old structure
                from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
                from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
            
            # Create config with only required parameters
            config = VQBeTConfig(
                input_features=self.input_features,
                output_features=self.output_features,
            )
            
            # Create policy
            policy = VQBeTPolicy(config, dataset_stats=self.metadata.stats)
            policy.to(self.device)
            policy.eval()
            
            # Test basic inference first
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                action = policy.select_action(obs_dict)
            
            # Measure performance
            performance = self.measure_fps_and_memory(policy, obs_dict, "VQBet")
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config,
                'performance': performance
            }
            
            print(f"   ✅ VQBet inference successful!")
            print(f"   Parameters: {result['parameters']:,}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ VQBet failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def run_inference_tests(self, models_to_test: list, use_pretrained: bool = False) -> Dict[str, Any]:
        """Run inference tests for specified models."""
        print(f"🚀 Running Model Inference Tests with FPS Profiling")
        print("=" * 60)
        
        # Available test methods
        test_methods = {
            'act': self.test_act_model,
            'diffusion': self.test_diffusion_model,
            'smolvla': self.test_smolvla_model,
            'pi0': self.test_pi0_model,
            'pi0fast': self.test_pi0fast_model,
            'vqbet': self.test_vqbet_model,
        }
        
        results = {}
        
        for model_name in models_to_test:
            if model_name not in test_methods:
                print(f"⚠️  Unknown model: {model_name}")
                results[model_name] = {'status': 'UNKNOWN'}
                continue
                
            print()
            try:
                results[model_name] = test_methods[model_name](use_pretrained)
                # Clean up memory between tests
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"   💥 Unexpected error testing {model_name}: {e}")
                results[model_name] = {'status': 'ERROR', 'error': str(e)}
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 INFERENCE TEST & PERFORMANCE SUMMARY")
        print("=" * 80)
        
        successful = []
        failed = []
        
        # Performance comparison table
        print(f"{'Model':<12} {'Status':<8} {'Params':<12} {'Mean FPS':<10} {'Max FPS':<10} {'Cold Start':<12} {'Memory':<12}")
        print("-" * 80)
        
        for model_name, result in results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                successful.append(model_name)
                params = result.get('parameters', 0)
                perf = result.get('performance', {})
                mean_fps = perf.get('mean_fps', 0)
                max_fps = perf.get('max_fps', 0)
                cold_start = perf.get('cold_start_time', 0) * 1000  # Convert to ms
                memory = perf.get('total_memory_estimate_mb', 0) # Use total_memory_estimate_mb
                
                print(f"{model_name:<12} {'✅ OK':<8} {params/1e6:>8.1f}M {mean_fps:>8.1f} {max_fps:>8.1f} {cold_start:>9.1f}ms {memory:>9.1f}MB")
                
            else:
                failed.append(model_name)
                error = result.get('error', 'Unknown error')
                print(f"{model_name:<12} {'❌ FAIL':<8} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
        
        print("-" * 80)
        print(f"\n🎯 Results: {len(successful)}/{len(results)} models working")
        
        if successful:
            print(f"✅ Working: {', '.join(successful)}")
            
            # Find fastest model
            fastest_model = None
            fastest_fps = 0
            for model_name in successful:
                result = results[model_name]
                if 'performance' in result:
                    fps = result['performance'].get('mean_fps', 0)
                    if fps > fastest_fps:
                        fastest_fps = fps
                        fastest_model = model_name
            
            if fastest_model:
                print(f"🏆 Fastest model: {fastest_model} ({fastest_fps:.1f} FPS)")
                
            # Real-time capability analysis
            print(f"\n⚡ Real-time Performance Analysis (Target: >30 FPS for real-time)")
            realtime_capable = []
            for model_name in successful:
                result = results[model_name]
                if 'performance' in result:
                    fps = result['performance'].get('mean_fps', 0)
                    if fps >= 30:
                        realtime_capable.append(f"{model_name} ({fps:.1f} FPS)")
            
            if realtime_capable:
                print(f"🚀 Real-time capable: {', '.join(realtime_capable)}")
            else:
                print(f"⚠️  No models achieve 30+ FPS. Consider optimizations or GPU inference.")
                
        if failed:
            print(f"❌ Failed: {', '.join(failed)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test inference for multiple model architectures with FPS profiling")
    parser.add_argument("--dataset", default="bearlover365/red_cube_always_in_same_place", 
                       help="Dataset to test with")
    parser.add_argument("--models", default="act,diffusion,pi0,pi0fast,smolvla,vqbet",
                       help="Comma-separated list of models to test")
    parser.add_argument("--use-pretrained", action="store_true",
                       help="Use pretrained models where available (SmolVLA)")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--quick", action="store_true", help="Quick test (50 samples)")
    parser.add_argument("--detailed", action="store_true", help="Detailed profiling (500 samples)")
    
    args = parser.parse_args()
    
    # Set profiling samples based on mode
    if args.quick:
        profile_samples = 50
    elif args.detailed:
        profile_samples = 500
    else:
        profile_samples = 100  # Default
    
    # Parse models list
    models_to_test = [m.strip() for m in args.models.split(",")]
    
    print("🧪 MULTI-MODEL INFERENCE TESTING WITH FPS PROFILING")
    print("=" * 60)
    print("🎯 Goal: Verify all model types work and measure performance")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🤖 Models: {', '.join(models_to_test)}")
    print(f"📦 Pretrained: {args.use_pretrained}")
    print(f"🔍 Profile samples: {profile_samples}")
    print()
    
    try:
        # Create tester
        tester = ModelInferenceTester(args.dataset, args.device, profile_samples)
        
        # Run tests
        results = tester.run_inference_tests(models_to_test, args.use_pretrained)
        
        # Count successful models
        successful_count = sum(1 for r in results.values() if r.get('status') == 'SUCCESS')
        
        if successful_count == len(models_to_test):
            print(f"\n🎉 ALL MODELS WORKING! Ready for training pipeline!")
            print(f"💡 Tip: Use fastest models for real-time deployment on CPU")
            return 0
        else:
            print(f"\n⚠️  {successful_count}/{len(models_to_test)} models working. Fix failed models before training.")
            return 1
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 