#!/usr/bin/env python3
"""
Cloud Setup Script for LeRobot Notebook Pipeline (Python version)
Works on Lightning AI, Google Colab, AWS, etc.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors gracefully."""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up LeRobot Notebook Pipeline on Cloud...")
    
    # Upgrade pip
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Install dependencies
    if Path("requirements-cloud.txt").exists():
        run_command("pip install -r requirements-cloud.txt", "Installing Python dependencies")
    else:
        print("⚠️  requirements-cloud.txt not found, using fallback dependencies...")
        fallback_packages = [
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "lerobot==0.2.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "pytest>=7.4.0",
            "opencv-python>=4.8.0"
        ]
        for package in fallback_packages:
            run_command(f"pip install '{package}'", f"Installing {package}")
    
    # Install this package
    if Path("pyproject.toml").exists():
        run_command("pip install -e .", "Installing lerobot_notebook_pipeline package")
    
    # Create directories
    print("📁 Creating output directories...")
    directories = ["models", "data", "videos", "single_episode_experiments"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✓ {dir_name}/")
    
    # Test installation
    print("🧪 Testing installation...")
    try:
        import torch
        import lerobot
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ LeRobot: {lerobot.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        # Test package imports
        from lerobot_notebook_pipeline.dataset_utils.analysis import get_dataset_stats
        from lerobot_notebook_pipeline.dataset_utils.visualization import plot_action_histogram
        print("✅ Package imports working")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    # Run quick tests
    if Path("tests").exists():
        print("🏃 Running quick test suite...")
        success = run_command("python -m pytest tests/ -v --tb=short --durations=0", "Running tests")
        if not success:
            print("⚠️  Some tests failed, but setup is complete")
    
    print("\n✅ Setup complete! You can now run your experiments.")
    print("\nQuick start commands:")
    print("  pytest tests/ -v --durations=10    # Run tests with timing")
    print("  python analyse_dataset.py . --fast # Quick dataset analysis") 
    print("  python train.py --help            # See training options")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 