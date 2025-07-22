#!/usr/bin/env python3
"""
Clean Training Script for LeRobot Policies

This script focuses purely on training without dataset analysis/visualization clutter.
For dataset analysis, use: python analyse_dataset.py
For policy evaluation, use: python visualize_policy.py

Usage:
    python train.py --dataset "bearlover365/red_cube_always_in_same_place"
    python train.py --dataset "my_dataset" --root "./data" --steps 5000
"""

import argparse
import torch
import warnings
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot_notebook_pipeline.dataset_utils.training import train_model
from lerobot_notebook_pipeline.dataset_utils.visualization import AddGaussianNoise
from torchvision import transforms

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def setup_device():
    """Setup and return the appropriate device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    return device


def load_dataset(dataset_name, root=None, video_backend="pyav", use_augmentation=True, augmentation_std=0.02):
    """Load and configure the dataset with optional augmentation."""
    print(f"📦 Loading dataset: {dataset_name}")
    
    # Load dataset metadata for configuration
    if root:
        dataset_metadata = LeRobotDatasetMetadata(dataset_name, root=root)
    else:
        dataset_metadata = LeRobotDatasetMetadata(dataset_name)
    
    # Configure policy features
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # Remove wrist image if present (optional)
    if "observation.wrist_image" in input_features:
        input_features.pop("observation.wrist_image")
    
    # Configure ACT policy
    cfg = ACTConfig(
        input_features=input_features, 
        output_features=output_features, 
        chunk_size=10, 
        n_action_steps=10
    )
    
    # Get delta timestamps for action chunking
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    
    # Setup data augmentation
    transform = None
    if use_augmentation:
        transform = transforms.Compose([
            AddGaussianNoise(mean=0., std=augmentation_std),
            transforms.Lambda(lambda x: x.clamp(0, 1))
        ])
        print(f"🎨 Using data augmentation (Gaussian noise σ={augmentation_std})")
    
    # Load dataset with configuration
    try:
        if root:
            dataset = LeRobotDataset(
                dataset_name, 
                root=root,
                delta_timestamps=delta_timestamps, 
                image_transforms=transform,
                video_backend=video_backend
            )
        else:
            dataset = LeRobotDataset(
                dataset_name,
                delta_timestamps=delta_timestamps, 
                image_transforms=transform,
                video_backend=video_backend
            )
        print(f"✅ Dataset loaded: {len(dataset)} steps across {len(dataset.meta.episodes)} episodes")
        
    except Exception as e:
        print(f"⚠️  Failed to load dataset with augmentation: {e}")
        print("🔄 Falling back to dataset without image transforms...")
        
        if root:
            dataset = LeRobotDataset(dataset_name, root=root, delta_timestamps=delta_timestamps, video_backend=video_backend)
        else:
            dataset = LeRobotDataset(dataset_name, delta_timestamps=delta_timestamps, video_backend=video_backend)
        print(f"✅ Dataset loaded without augmentation: {len(dataset)} steps")
    
    return dataset, dataset_metadata, cfg


def create_policy(cfg, dataset_metadata, device):
    """Create and configure the policy."""
    print("🧠 Initializing policy...")
    
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params_million = total_params / 1_000_000
    
    print(f"✅ Policy initialized with {total_params:,} parameters ({total_params_million:.2f}M)")
    return policy


def create_dataloader(dataset, device, batch_size=64, num_workers=4):
    """Create the training dataloader."""
    print(f"📦 Creating dataloader (batch_size={batch_size})...")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    
    print(f"✅ Dataloader ready with {len(dataloader)} batches")
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Train LeRobot policy')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name or path')
    parser.add_argument('--root', type=str, default=None,
                       help='Root directory for local datasets')
    parser.add_argument('--output-dir', type=str, default='./ckpt/trained_policy',
                       help='Directory to save the trained policy')
    parser.add_argument('--steps', type=int, default=3000,
                       help='Number of training steps')
    parser.add_argument('--log-freq', type=int, default=100,
                       help='Logging frequency')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--augmentation-std', type=float, default=0.02,
                       help='Standard deviation for Gaussian noise augmentation')
    parser.add_argument('--video-backend', type=str, default='pyav',
                       help='Video backend (pyav or cv2)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    print("🚀 LeRobot Policy Training")
    print("=" * 40)
    print(f"Dataset: {args.dataset}")
    print(f"Training steps: {args.steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Setup
        device = setup_device()
        
        # Load dataset and configuration
        dataset, dataset_metadata, cfg = load_dataset(
            args.dataset, 
            root=args.root,
            video_backend=args.video_backend,
            use_augmentation=not args.no_augmentation,
            augmentation_std=args.augmentation_std
        )
        
        # Create policy
        policy = create_policy(cfg, dataset_metadata, device)
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset, device, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
        print(f"⚙️  Optimizer: Adam (lr={args.lr})")
        
        # Train
        print(f"\n🔥 Starting training...")
        train_model(policy, dataloader, optimizer, args.steps, args.log_freq, device)
        
        # Save policy
        print(f"\n💾 Saving policy to {args.output_dir}")
        policy.save_pretrained(args.output_dir)
        print("✅ Policy saved successfully!")
        
        print(f"\n🎉 Training complete!")
        print(f"   Steps: {args.steps}")
        print(f"   Model saved to: {args.output_dir}")
        print(f"\n💡 Next steps:")
        print(f"   - Evaluate: python visualize_policy.py --policy-path {args.output_dir} --dataset {args.dataset}")
        if args.root:
            print(f"     (add --root {args.root})")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 