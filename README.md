# 🤖 LeRobot Notebook Pipeline

[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](./tests/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive, production-ready toolkit for robot learning with **non-interactive execution** and **optimized performance**. All visualization functions are fully automated - no user interaction required!

## 🚀 Key Features & Performance Improvements

### ⚡ **Fully Non-Interactive**
- **Zero user interaction**: All plots automatically saved as PNG files
- **Automated execution**: Perfect for remote servers, CI/CD, and batch processing
- **No blocking**: Scripts run to completion without waiting for user input

### 🎯 **Speed Optimizations**
- **Fast mode**: 10x faster analysis with `--fast` flag (10% data sampling)
- **Smart sampling**: Custom ratios with `--sample-ratio` for balanced speed/accuracy
- **Optimized animations**: 1/4 image size, 5x frame skipping, 20 frames (vs 100)
- **Parallel processing**: Efficient data loading and processing pipelines

### 📊 **Comprehensive Analysis**
- **Detailed timing**: Every operation shows timing breakdowns
- **Organized output**: `--output-dir` for clean file management
- **Multiple formats**: PNG plots, detailed console output, structured analysis

### 🧪 **Fast Tests**
- **~32 seconds**: Complete test suite runs quickly
- **No hanging**: Tests complete automatically without user interaction
- **Consistent performance**: Reliable timing across runs

## 🏗️ Project Structure

```
lerobot_notebook_pipeline/
├── dataset_utils/                   # Core utility modules
│   ├── analysis.py                 # Dataset analysis functions
│   ├── visualization.py            # Visualization functions
│   └── training.py                 # Training utilities
├── train.py                        # 🎯 Clean training script
├── train_ultimate.py               # 📚 Jupyter-style training (educational)
├── analyse_dataset.py              # 🔍 Comprehensive dataset analysis
├── demo_visualizations.py          # 🎨 Visualization showcase
├── visualize_policy.py             # 🤖 Policy evaluation
└── tests/                          # Test suite
```

## 🚀 Quick Start

### Training a Model
```bash
# Simple training
python train.py --dataset "bearlover365/red_cube_always_in_same_place"

# Advanced training with custom parameters
python train.py \
  --dataset "bearlover365/red_cube_always_in_same_place" \
  --steps 5000 \
  --lr 1e-4 \
  --batch-size 32 \
  --output-dir "./my_trained_model"
```

### Analyzing Your Dataset
```bash
# Comprehensive analysis
python analyse_dataset.py --dataset "bearlover365/red_cube_always_in_same_place"

# Quick visualization demo
python demo_visualizations.py --dataset "bearlover365/red_cube_always_in_same_place"

# Specific visualization types
python demo_visualizations.py \
  --dataset "bearlover365/red_cube_always_in_same_place" \
  --demo-type augmentation
```

### Evaluating a Trained Policy
```bash
python visualize_policy.py \
  --policy-path "./my_trained_model" \
  --dataset "bearlover365/red_cube_always_in_same_place"
```

## 📚 Scripts Overview

### 🎯 `train.py` - Clean Training
**Purpose**: Focused, production-ready training script  
**Best for**: Actual model training, experimentation, production use

**Features**:
- ✅ Streamlined training workflow
- ✅ Comprehensive command-line options
- ✅ Automatic data augmentation
- ✅ Clean progress logging
- ✅ Configurable everything (lr, batch size, steps, etc.)

```bash
python train.py --dataset "my_dataset" --steps 5000 --lr 2e-4
```

### 📚 `train_ultimate.py` - Educational Training
**Purpose**: Jupyter-style training with explanations  
**Best for**: Learning, understanding, step-by-step exploration

**Features**:
- ✅ Detailed explanations and markdown cells
- ✅ Quick dataset overview
- ✅ Compatibility with Jupyter notebooks
- ✅ Educational comments and analysis

### 🔍 `analyse_dataset.py` - Comprehensive Analysis
**Purpose**: Deep dataset understanding and quality assessment  
**Best for**: Understanding your data before training

**Features**:
- ✅ Complete dataset statistics with detailed timing
- ✅ Episode-by-episode analysis
- ✅ Overfitting risk assessment
- ✅ Action pattern analysis with optimized plotting
- ✅ Non-interactive execution with PNG output
- ✅ Fast mode and data sampling options
- ✅ Organized output directories

```bash
# Full analysis with organized output
python analyse_dataset.py dataset_path --output-dir ./analysis_results

# Fast analysis (10% data sampling)
python analyse_dataset.py dataset_path --fast

# Custom analysis with specific episodes
python analyse_dataset.py dataset_path \
  --episodes 0 1 2 \
  --sample-idx 5 \
  --sample-ratio 0.2 \
  --output-dir ./results
```

### 🎨 `demo_visualizations.py` - Visualization Showcase
**Purpose**: Explore all visualization capabilities  
**Best for**: Understanding visualization options, creating demos

**Features**:
- ✅ Modular demo types (basic, episodes, augmentation, actions, comprehensive)
- ✅ Beautiful visualizations saved as PNG files
- ✅ Episode trajectory analysis with optimized animations
- ✅ Data augmentation comparison
- ✅ Non-interactive execution with organized output

```bash
# Show everything, save to organized directory
python demo_visualizations.py dataset_path --output-dir ./demo_plots

# Just augmentation effects
python demo_visualizations.py dataset_path --demo augmentation

# Fast episode visualization with optimizations
python demo_visualizations.py dataset_path --demo episodes --output-dir ./results
```

### 🤖 `visualize_policy.py` - Policy Evaluation
**Purpose**: Evaluate and visualize trained policy performance  
**Best for**: Model evaluation, debugging, performance analysis

**Features**:
- ✅ Comprehensive error metrics
- ✅ Per-joint performance analysis
- ✅ Prediction vs ground truth plots
- ✅ Error distribution analysis

```bash
python visualize_policy.py \
  --policy-path "./ckpt/my_model" \
  --dataset "my_dataset" \
  --episode 0
```

## 🛠️ Development Workflow

### 1. **Dataset Analysis First**
```bash
# Understand your data
python analyse_dataset.py --dataset "my_dataset"

# Explore visualizations
python demo_visualizations.py --dataset "my_dataset"
```

### 2. **Train Your Model**
```bash
# Clean training for production
python train.py --dataset "my_dataset" --steps 5000

# Or educational training for learning
python train_ultimate.py  # (run in Jupyter or as script)
```

### 3. **Evaluate Performance**
```bash
# Comprehensive policy evaluation
python visualize_policy.py \
  --policy-path "./ckpt/trained_policy" \
  --dataset "my_dataset"
```

## 🧪 Available Functions

### Dataset Analysis (`dataset_utils/analysis.py`)
- `get_dataset_stats()` - Basic dataset statistics
- `analyze_episodes()` - Episode-by-episode breakdown
- `compare_episodes()` - Cross-episode comparison
- `analyze_action_patterns()` - Action dynamics analysis
- `analyze_overfitting_risk()` - Smart overfitting assessment
- `visualize_sample()` - Single sample visualization

### Visualization (`dataset_utils/visualization.py`)
- `plot_all_action_histograms()` - Action distribution analysis
- `visualize_episode_trajectory()` - Episode trajectory plots
- `create_training_animation()` - Episode animations
- `visualize_augmentations()` - Augmentation effects
- `compare_augmentation_effects()` - Multi-strategy comparison
- `AddGaussianNoise` - Data augmentation class

### Training (`dataset_utils/training.py`)
- `train_model()` - Enhanced training loop with analytics

## 🎯 Use Cases

### For Researchers
- Use `analyse_dataset.py` to understand data quality and diversity
- Use `train.py` for clean, reproducible experiments
- Use `visualize_policy.py` for comprehensive evaluation

### For Students/Learning
- Use `train_ultimate.py` for step-by-step understanding
- Use `demo_visualizations.py` to explore capabilities
- Use `analyse_dataset.py` to learn about data analysis

### For Production
- Use `train.py` for automated training pipelines
- Use `visualize_policy.py` for model validation
- Use the utility functions for custom workflows

## 🔧 Configuration Options

### Training Parameters
- `--steps`: Number of training steps (default: 3000)
- `--lr`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size (default: 64)
- `--no-augmentation`: Disable data augmentation
- `--augmentation-std`: Noise level for augmentation

### Analysis Parameters
- `--episodes`: Which episodes to analyze
- `--sample-idx`: Specific sample to visualize
- `--skip-animation`: Skip episode animations

### Visualization Parameters
- `--demo-type`: Type of demo (all, basic, episodes, augmentation, actions)
- `--show-episode-animation`: Include episode animations

## 🚨 Error Handling & Tips

### Common Issues
1. **PIL/Pillow Version Issues**: The scripts handle PIL compatibility automatically
2. **Memory Issues**: Use smaller batch sizes or fewer episodes for analysis
3. **CUDA Issues**: Scripts auto-detect and fall back to CPU
4. **Slow Plotting**: Use fast mode or data sampling for large datasets

### Performance Optimization
1. **Non-interactive mode**: All plots are automatically saved as PNG files - no user interaction required
2. **Use fast mode** for quick analysis: `--fast` (10% data sampling)
3. **Custom sampling**: `--sample-ratio 0.2` (20% of data)
4. **Skip animations** for faster analysis: `--skip-animation`
5. **Optimized animations**: Reduced frames (20 vs 100), frame skipping (every 5th), and image resizing (1/4 size)
6. **Timing information**: All functions show detailed timing breakdowns
7. **Organized output**: Use `--output-dir` to save plots to specific directories

```bash
# Fast analysis (10% data sampling, saves plots to ./results/)
python analyse_dataset.py --dataset "my_dataset" --fast --output-dir ./results

# Custom sampling (20% of data)
python analyse_dataset.py --dataset "my_dataset" --sample-ratio 0.2

# Skip slow operations
python analyse_dataset.py --dataset "my_dataset" --skip-animation --episodes 0

# Demo with organized output
python demo_visualizations.py --dataset "my_dataset" --output-dir ./demo_plots
```

### Animation Optimizations
The `create_training_animation` function now includes several performance optimizations:
- **Reduced frame count**: Default 20 frames (was 100)
- **Frame skipping**: Takes every 5th frame by default
- **Image resizing**: Images resized to 1/4 size (0.25 factor)
- **Lower DPI**: Saved at 80 DPI for smaller file sizes
- **No user interaction**: Fully automated, no need to press 'q' or interact with plots

### Best Practices
1. **Always analyze your dataset first** with `analyse_dataset.py`
2. **Start with small step counts** for testing
3. **Use the visualization tools** to understand your data before training
4. **Check overfitting warnings** seriously - they're usually accurate
5. **Use fast mode** for initial exploration, full analysis for final insights

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test individual components
python -m pytest tests/test_analysis.py -v
```

## 📈 Performance Monitoring

All scripts include:
- ✅ Progress tracking with ETAs
- ✅ Memory usage monitoring
- ✅ Automatic device detection
- ✅ Graceful error handling
- ✅ Informative logging

## 🎉 Examples

### Complete Workflow Example
```bash
# 1. Analyze your dataset
python analyse_dataset.py --dataset "bearlover365/red_cube_always_in_same_place"

# 2. Train a model
python train.py \
  --dataset "bearlover365/red_cube_always_in_same_place" \
  --steps 3000 \
  --output-dir "./my_cube_model"

# 3. Evaluate the model
python visualize_policy.py \
  --policy-path "./my_cube_model" \
  --dataset "bearlover365/red_cube_always_in_same_place"
```

### Visualization Exploration
```bash
# See everything
python demo_visualizations.py --dataset "bearlover365/red_cube_always_in_same_place"

# Focus on specific aspects
python demo_visualizations.py --dataset "my_dataset" --demo-type episodes --episodes 0 1
python demo_visualizations.py --dataset "my_dataset" --demo-type augmentation
python demo_visualizations.py --dataset "my_dataset" --demo-type actions
```

---

This modular approach separates concerns cleanly while maintaining all the powerful analysis and visualization capabilities. Each script has a specific purpose and can be used independently or as part of a complete workflow. 