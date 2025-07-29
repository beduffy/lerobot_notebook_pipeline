#!/bin/bash
# Cloud Setup Script for LeRobot Notebook Pipeline
# Works on Lightning AI, Google Colab, AWS, etc.

set -e  # Exit on any error

echo "🚀 Setting up LeRobot Notebook Pipeline on Cloud..."

# Update system packages (if running as root/sudo)
if command -v apt-get &> /dev/null; then
    echo "📦 Updating system packages..."
    apt-get update -qq || echo "⚠️  Could not update system packages (might not have sudo access)"
fi

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-cloud.txt

# Install this package in development mode
echo "🔧 Installing lerobot_notebook_pipeline package..."
pip install -e .

# Create necessary directories
echo "📁 Creating output directories..."
mkdir -p models
mkdir -p data
mkdir -p videos
mkdir -p single_episode_experiments

# Test the installation
echo "🧪 Testing installation..."
python -c "
import torch
import lerobot
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ LeRobot: {lerobot.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')

# Test our package imports
from lerobot_notebook_pipeline.dataset_utils.analysis import get_dataset_stats
from lerobot_notebook_pipeline.dataset_utils.visualization import plot_action_histogram
print('✅ Package imports working')
"

# Run a quick test
echo "🏃 Running quick test suite..."
python -m pytest tests/ -v --tb=short --durations=0

echo "✅ Setup complete! You can now run your experiments."
echo ""
echo "Quick start commands:"
echo "  pytest tests/ -v --durations=10    # Run tests with timing"
echo "  python analyse_dataset.py . --fast # Quick dataset analysis"
echo "  python train.py --help            # See training options" 