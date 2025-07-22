# Plan for lerobot_notebook_pipeline

This document outlines the plan for developing the `lerobot_notebook_pipeline` repository.

## 1. Dataset Analysis

- [ ] Create a new script or notebook (`analyse_dataset.py` or `analyse_dataset.ipynb`) to inspect and visualize the training data.
- [ ] Load the dataset and print summary statistics (e.g., number of episodes, steps per episode, action/observation space dimensions).
- [ ] Visualize sample observations and actions.

## 2. Refactor Training Script

- [ ] Split the existing `train_ultimate.py` into two separate scripts:
    - `analyse_dataset.py`: For the analysis work described above.
    - `train.py`: For the core training loop.
- [ ] Ensure that the training script can still be run independently.

## 3. Policy Visualization

- [ ] Create a script (`visualize_policy.py`) to load a trained policy.
- [ ] Run the policy in a simulated environment.
- [ ] Render the environment to visualize the policy's actions.
- [ ] Save a video or gif of the visualization.




# Other todos
TODO clear space on lightning and wandb, 
TODO maybe don't save models to wandb?
TODO understand when voice says episode over, it stops recording then, so can I reset sock position with arm or? 
TODO ACT and smolVLA FPS. That matters right?
TODO evaluate smolVLA, diffusion. 
TODO train diffusion and compare to others
TODO do train eval splits
