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
        # Use sys.executable to ensure we're using the correct pip
        if cmd.startswith("pip "):
            cmd = f"{sys.executable} -m {cmd}"
        print(f"Executing command: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed.")
        if e.stdout:
            print(f"Output:\n{e.stdout}")
        if e.stderr:
            print(f"Error details:\n{e.stderr}")
        # Exit the script if a command fails
        sys.exit(1)


def main():
    print("🚀 Setting up LeRobot Notebook Pipeline on Cloud...")
    
    # --- 1. Upgrade pip ---
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # --- 2. Install Dependencies ---
    if Path("requirements-cloud.txt").exists():
        run_command("pip install -r requirements-cloud.txt", "Installing Python dependencies from requirements-cloud.txt")
    else:
        print("⚠️  requirements-cloud.txt not found. Please ensure the file exists.")
        sys.exit(1)
        
    # --- 3. Install this package in editable mode ---
    if Path("pyproject.toml").exists():
        run_command("pip install -e .", "Installing lerobot_notebook_pipeline package")
    
    # --- 4. Create output directories ---
    print("📁 Creating output directories...")
    directories = ["models", "data", "videos", "single_episode_experiments"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✓ {dir_name}/")
    
    # --- 5. Test Installation ---
    print("🧪 Verifying installation...")
    try:
        import torch
        import lerobot
        import cv2
        
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ LeRobot: {lerobot.__version__}")
        print(f"  ✓ OpenCV: {cv2.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        
        # Test package imports
        from lerobot_notebook_pipeline.dataset_utils.analysis import get_dataset_stats
        from lerobot_notebook_pipeline.dataset_utils.visualization import plot_action_histogram
        print("  ✓ Package imports are working correctly.")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        print("   This might be due to an installation issue. Please check the logs above.")
        return False
    
    # --- 6. Run quick tests ---
    if Path("tests").exists():
        print("🏃 Running quick test suite...")
        # We use a simple command here; `run_command` will handle the verbose output
        success = run_command("pytest tests/ -v -s --tb=short --durations=0", "Running tests with pytest")
        if not success:
            print("⚠️  Some tests failed. Please review the output above.")
            # We don't exit here, as the user might want to debug
    
    print("\n✅ Setup complete! You can now run your experiments.")
    print("\nQuick start commands:")
    print("  pytest tests/ -v --durations=10    # Run tests with timing")
    print("  python analyse_dataset.py . --fast # Quick dataset analysis") 
    print("  python train.py --help            # See training options")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1) 