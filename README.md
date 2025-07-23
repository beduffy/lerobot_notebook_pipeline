# LeRobot Notebook Pipeline

A comprehensive analysis and training pipeline for LeRobot datasets with modular utilities.

## 🚀 **Quick Start: Single Episode Training**

**NEW**: Unified training script that auto-detects environment and adjusts settings:

```bash
# Local quick test (auto-detects CPU settings)
python train_single_episode.py --episode 0 --steps 50

# Cloud training (auto-detects GPU settings) 
python train_single_episode.py --episode 0 --steps 2000 --cloud --upload-model --wandb

# Custom configuration
python train_single_episode.py --episode 0 --steps 1000 --batch-size 16 --lr 1e-3
```

**Features:**
- ✅ **Auto-detection**: Automatically adjusts batch size, workers, logging based on environment
- ✅ **Full transparency**: See every step - dataset loading, model architecture, training progress
- ✅ **Cloud ready**: W&B logging, HuggingFace upload, optimized for GPU training
- ✅ **Local friendly**: Efficient settings for CPU development and testing

## 📊 Analysis Tools

Fast, non-interactive analysis and visualization tools:

```bash
# Comprehensive dataset analysis
python analyse_dataset.py . --fast

# Visualization demos (saves PNG files)
python demo_visualizations.py . --demo all

# Model evaluation
python evaluate_model.py ./trained_model --episode 0
```

## 🔧 Installation

1. **Install LeRobot** (follow their instructions)
2. **Install this package**:
   ```bash
   pip install -e .
   ```
3. **Ready to go!**

## 📁 Dataset Utils

The package provides modular utilities in `lerobot_notebook_pipeline.dataset_utils`:

- **`analysis.py`**: Dataset statistics, episode analysis, overfitting detection
- **`visualization.py`**: Action histograms, trajectories, animations (optimized, non-interactive)
- **`training.py`**: Training utilities and progress tracking

## 🧪 Testing

Fast, comprehensive tests (< 1 minute):

```bash
pytest tests/ -v
```

All tests save plots as PNG files instead of interactive displays.

## 🎯 Research Focus

This pipeline is designed for systematic research into **robot generalization**:

- **Single Episode Training**: Train on exactly 1 demonstration to test memorization vs generalization
- **Episode Comparison**: Compare model performance across different episodes
- **Transparent Analysis**: Full visibility into dataset characteristics and model behavior
- **Scalable**: Works locally for development, scales to cloud for serious training

## 📈 Performance

- **Analysis**: < 3 minutes for full dataset analysis with optimizations
- **Training**: Scales from CPU (development) to GPU (production) automatically
- **Tests**: < 1 minute for complete test suite
- **Visualization**: Optimized animations with frame skipping and resizing

## 🌟 Key Features

- **Non-interactive execution**: No `plt.show()` calls, all plots saved as files
- **Performance optimized**: Data sampling, animation optimization, progress tracking
- **Environment adaptive**: Auto-detects cloud vs local and adjusts accordingly
- **Research focused**: Built for systematic experimentation on generalization

---

*Built for understanding robot learning through systematic experimentation.* 