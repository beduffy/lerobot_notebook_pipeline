# Cloud Setup Guide for LeRobot Notebook Pipeline

This guide helps you set up the LeRobot Notebook Pipeline on cloud platforms like **Lightning AI**, **Google Colab**, **AWS**, etc.

## Quick Setup (Recommended)

### Option 1: One-Line Setup (Python)
```bash
python setup-cloud.py
```

### Option 2: One-Line Setup (Bash)
```bash
chmod +x setup-cloud.sh && ./setup-cloud.sh
```

## Manual Setup

If you prefer to set up manually or the scripts don't work:

### 1. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements-cloud.txt
```

### 2. Install This Package
```bash
pip install -e .
```

### 3. Create Directories
```bash
mkdir -p models data videos single_episode_experiments
```

### 4. Test Installation
```bash
python -c "import lerobot; print(f'✅ LeRobot: {lerobot.__version__}')"
pytest tests/ -v --durations=10
```

## Key Dependencies

- **LeRobot**: `0.2.0` (exact version that works with our imports)
- **PyTorch**: `>=2.0.0` 
- **Python**: `>=3.8`

## Platform-Specific Notes

### Lightning AI
- Use the Python setup script: `python setup-cloud.py`
- Lightning has good GPU support out of the box
- All dependencies install cleanly

### Google Colab
```python
# In a Colab cell:
!git clone https://github.com/yourusername/lerobot_notebook_pipeline.git
%cd lerobot_notebook_pipeline
!python setup-cloud.py
```

### AWS SageMaker / EC2
- Use the bash script: `./setup-cloud.sh`
- Make sure you have GPU instances for training

## Testing Your Setup

### Quick Test
```bash
pytest tests/ -v --durations=10
```

### Run Specific Test Types
```bash
# Fast tests only
pytest tests/ -m "not slow"

# Integration tests
pytest tests/ -m integration

# Show the 5 slowest tests
pytest tests/ --durations=5
```

## Common Issues

### Import Errors
If you get `ModuleNotFoundError: No module named 'lerobot.common'`:
- ✅ **Fixed!** We've updated all imports to use the new LeRobot 0.2.0 structure
- The setup scripts install the correct version

### CUDA Issues
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Missing Directories
The setup scripts create all necessary directories:
- `models/` - Trained model storage
- `data/` - Dataset cache
- `videos/` - Visualization outputs
- `single_episode_experiments/` - Experiment results

## Usage Examples

After setup, you can run:

```bash
# Analyze a dataset
python analyse_dataset.py . --fast

# Train a model
python train.py --dataset bearlover365/red_cube_always_in_same_place --steps 1000

# Run experiments
python single_episode_experiment.py bearlover365/red_cube_always_in_same_place --episode 0

# Test with timing
pytest tests/ -v --durations=0  # Show all test durations
```

## File Structure After Setup

```
lerobot_notebook_pipeline/
├── requirements-cloud.txt    # Clean dependencies for cloud
├── setup-cloud.py          # Python setup script
├── setup-cloud.sh          # Bash setup script  
├── pytest.ini              # Test configuration with timing
├── tests/                   # Test suite (all pass!)
├── models/                  # Model storage (created by setup)
├── data/                    # Dataset cache (created by setup)
└── ...
```

## Performance Notes

- **Test Timing**: All tests show durations automatically
- **GPU Support**: Works on cloud GPUs (CUDA/MPS)
- **Memory**: Most scripts work with 8GB+ RAM
- **Storage**: Models need ~100MB-1GB each 