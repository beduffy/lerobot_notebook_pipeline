# 🚀 Quick Start - Cloud Deployment

## One-Line Setup for Lightning AI / Cloud

```bash
# Clone and setup (replace with your repo URL)
git clone YOUR_REPO_URL lerobot_notebook_pipeline
cd lerobot_notebook_pipeline
python setup-cloud.py
```

That's it! ✅

## What You Get

✅ **All dependencies installed** (LeRobot 0.2.0, PyTorch, etc.)  
✅ **15 tests passing** with automatic timing  
✅ **Fixed imports** (no more `lerobot.common` errors)  
✅ **Ready to train models**

## Quick Commands

```bash
# Test everything with timing (see slow tests)
pytest tests/ -v --durations=5

# Run fast tests only 
pytest tests/ -m "not slow"

# Train a quick model
python train.py --dataset bearlover365/red_cube_always_in_same_place --steps 1000

# Analyze dataset
python analyse_dataset.py . --fast
```

## Test Performance

Recent timing (your tests may vary):
- ⚠️  `test_analysis_script_integration` - **75s** (slowest)
- ⚠️  `test_single_episode_experiment_config` - **45s** 
- 🐌 `test_analysis_functions` - **26s**
- 🐌 `test_plot_action_histogram` - **20s**
- ✅ `test_visualization_functions` - **4s** (fast)

## Files Created for Cloud Deployment

- `requirements-cloud.txt` - Clean dependencies (no conda!)
- `setup-cloud.py` - One-command setup script
- `pytest.ini` - Auto-timing configuration  
- `CLOUD_SETUP.md` - Detailed setup guide 