#!/usr/bin/env python3
"""
Multi-Model Inference Test Script

Tests inference for all available lerobot model architectures before training.
This is like a "unit test" to ensure all model types work with your pipeline.

Available models to test:
- ACT (already working)
- Diffusion
- SmolVLA (your focus)  
- PI0 (pi zero)
- PI0FAST
- TDMPC
- VQBet

Usage:
    # Test all models
    python test_model_inference.py
    
    # Test specific models only
    python test_model_inference.py --models act,diffusion,smolvla
    
    # Test with your dataset
    python test_model_inference.py --dataset bearlover365/red_cube_always_in_same_place
    
    # Test with pretrained models (for SmolVLA)
    python test_model_inference.py --models smolvla --use-pretrained
"""

import argparse
import torch
import warnings
from pathlib import Path
import traceback
from typing import Dict, Any
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

# Import factory functions
from lerobot_some_original_code.lerobot_original_factory import get_policy_class, make_policy_config


class ModelInferenceTester:
    """Test inference for different model architectures."""
    
    def __init__(self, dataset_name: str, device: str = "auto"):
        self.dataset_name = dataset_name
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🧪 Model Inference Tester")
        print(f"   Dataset: {dataset_name}")
        print(f"   Device: {self.device}")
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
    
    def test_act_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test ACT model (your working baseline)."""
        print("🤖 Testing ACT Model...")
        
        try:
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
            
            # Test inference
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                action = policy.select_action(obs_dict)
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config
            }
            
            print(f"   ✅ ACT inference successful!")
            print(f"   Action shape: {action.shape}")
            print(f"   Parameters: {result['parameters']:,}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ ACT failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_diffusion_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test Diffusion Policy model."""
        print("🌊 Testing Diffusion Policy...")
        
        try:
            from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
            from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
            
            # Create config with reasonable defaults
            config = DiffusionConfig(
                input_features=self.input_features,
                output_features=self.output_features,
                horizon=16,  # Prediction horizon
                n_action_steps=8,  # Number of action steps to execute
                num_inference_steps=10,  # Diffusion denoising steps
                down_dims=[256, 512, 1024],  # U-Net architecture
                kernel_size=5,
                n_groups=8,
                cond_predict_scale=True,
            )
            
            # Create policy
            policy = DiffusionPolicy(config, dataset_stats=self.metadata.stats)
            policy.to(self.device)
            policy.eval()
            
            # Test inference
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                action = policy.select_action(obs_dict)
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config
            }
            
            print(f"   ✅ Diffusion inference successful!")
            print(f"   Action shape: {action.shape}")
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
                    # SmolVLA specific settings
                    use_quantization=False,  # Disable for testing
                    # Add more config as needed
                )
                
                # Create policy
                policy = SmolVLAPolicy(config, dataset_stats=self.metadata.stats)
            
            policy.to(self.device)
            policy.eval()
            
            # Test inference
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                action = policy.select_action(obs_dict)
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config,
                'pretrained': use_pretrained
            }
            
            print(f"   ✅ SmolVLA inference successful!")
            print(f"   Action shape: {action.shape}")
            print(f"   Parameters: {result['parameters']:,}")
            print(f"   Pretrained: {use_pretrained}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ SmolVLA failed: {e}")
            traceback.print_exc()
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_pi0_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test PI0 (Pi Zero) model."""
        print("🥧 Testing PI0 (Pi Zero) Model...")
        
        try:
            from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
            from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
            
            # Create config
            config = PI0Config(
                input_features=self.input_features,
                output_features=self.output_features,
                # PI0 specific settings
                chunk_size=10,  # PI0 typically uses smaller chunks
                n_action_steps=10,
            )
            
            # Create policy
            policy = PI0Policy(config, dataset_stats=self.metadata.stats)
            policy.to(self.device)
            policy.eval()
            
            # Test inference
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                action = policy.select_action(obs_dict)
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config
            }
            
            print(f"   ✅ PI0 inference successful!")
            print(f"   Action shape: {action.shape}")
            print(f"   Parameters: {result['parameters']:,}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ PI0 failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_vqbet_model(self, use_pretrained: bool = False) -> Dict[str, Any]:
        """Test VQBet model."""
        print("🎰 Testing VQBet Model...")
        
        try:
            from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
            from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
            
            # Create config
            config = VQBeTConfig(
                input_features=self.input_features,
                output_features=self.output_features,
                # VQBet specific settings
                dim_model=256,
                n_heads=8,
                window_size=10,
                n_action_steps=10,
            )
            
            # Create policy
            policy = VQBeTPolicy(config, dataset_stats=self.metadata.stats)
            policy.to(self.device)
            policy.eval()
            
            # Test inference
            with torch.no_grad():
                obs_dict = {k: v.to(self.device) for k, v in self.sample_observation.items()}
                action = policy.select_action(obs_dict)
            
            result = {
                'status': 'SUCCESS',
                'action_shape': action.shape,
                'action_dtype': action.dtype,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'config': config
            }
            
            print(f"   ✅ VQBet inference successful!")
            print(f"   Action shape: {action.shape}")
            print(f"   Parameters: {result['parameters']:,}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ VQBet failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def run_inference_tests(self, models_to_test: list, use_pretrained: bool = False) -> Dict[str, Any]:
        """Run inference tests for specified models."""
        print(f"🚀 Running Model Inference Tests")
        print("=" * 50)
        
        # Available test methods
        test_methods = {
            'act': self.test_act_model,
            'diffusion': self.test_diffusion_model,
            'smolvla': self.test_smolvla_model,
            'pi0': self.test_pi0_model,
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
            except Exception as e:
                print(f"   💥 Unexpected error testing {model_name}: {e}")
                results[model_name] = {'status': 'ERROR', 'error': str(e)}
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 INFERENCE TEST SUMMARY")
        print("=" * 50)
        
        successful = []
        failed = []
        
        for model_name, result in results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                successful.append(model_name)
                params = result.get('parameters', 0)
                action_shape = result.get('action_shape', 'Unknown')
                print(f"✅ {model_name:12} | {params:>12,} params | Action: {action_shape}")
            else:
                failed.append(model_name)
                error = result.get('error', 'Unknown error')
                print(f"❌ {model_name:12} | Failed: {error[:50]}...")
        
        print(f"\n🎯 Results: {len(successful)}/{len(results)} models working")
        if successful:
            print(f"✅ Working: {', '.join(successful)}")
        if failed:
            print(f"❌ Failed: {', '.join(failed)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test inference for multiple model architectures")
    parser.add_argument("--dataset", default="bearlover365/red_cube_always_in_same_place", 
                       help="Dataset to test with")
    parser.add_argument("--models", default="act,diffusion,smolvla,pi0,vqbet",
                       help="Comma-separated list of models to test")
    parser.add_argument("--use-pretrained", action="store_true",
                       help="Use pretrained models where available (SmolVLA)")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Parse models list
    models_to_test = [m.strip() for m in args.models.split(",")]
    
    print("🧪 MULTI-MODEL INFERENCE TESTING")
    print("=" * 40)
    print("🎯 Goal: Verify all model types work before training")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🤖 Models: {', '.join(models_to_test)}")
    print(f"📦 Pretrained: {args.use_pretrained}")
    print()
    
    try:
        # Create tester
        tester = ModelInferenceTester(args.dataset, args.device)
        
        # Run tests
        results = tester.run_inference_tests(models_to_test, args.use_pretrained)
        
        # Count successful models
        successful_count = sum(1 for r in results.values() if r.get('status') == 'SUCCESS')
        
        if successful_count == len(models_to_test):
            print(f"\n🎉 ALL MODELS WORKING! Ready for training pipeline!")
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