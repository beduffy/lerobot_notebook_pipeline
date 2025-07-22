# LeRobot Notebook Pipeline

A comprehensive toolkit for robot learning with LeRobot, featuring separated training and analysis capabilities for better modularity and ease of use.

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
- ✅ Complete dataset statistics
- ✅ Episode-by-episode analysis
- ✅ Overfitting risk assessment
- ✅ Action pattern analysis
- ✅ Actionable recommendations

```bash
python analyse_dataset.py \
  --dataset "my_dataset" \
  --episodes 0 1 2 \
  --sample-idx 5
```

### 🎨 `demo_visualizations.py` - Visualization Showcase
**Purpose**: Explore all visualization capabilities  
**Best for**: Understanding visualization options, creating demos

**Features**:
- ✅ Modular demo types (basic, episodes, augmentation, actions)
- ✅ Beautiful visualizations
- ✅ Episode trajectory analysis
- ✅ Data augmentation comparison

```bash
# Show everything
python demo_visualizations.py --dataset "my_dataset"

# Just augmentation effects
python demo_visualizations.py --dataset "my_dataset" --demo-type augmentation
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

### Best Practices
1. **Always analyze your dataset first** with `analyse_dataset.py`
2. **Start with small step counts** for testing
3. **Use the visualization tools** to understand your data before training
4. **Check overfitting warnings** seriously - they're usually accurate

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