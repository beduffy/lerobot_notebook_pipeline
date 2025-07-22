# Plan for lerobot_notebook_pipeline

This document outlines the plan for developing the `lerobot_notebook_pipeline` repository.

KEEP train_ultimate.py and periodically sync with train_ultimate.ipynb with jupytext

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
TODO show first and last image or resize a lot the animation and make it use less than 10Mb
TODO maybe don't save models to wandb?
TODO understand when voice says episode over, it stops recording then, so can I reset sock position with arm or? 
TODO ACT and smolVLA FPS. That matters right?
TODO evaluate smolVLA, diffusion. 
TODO train diffusion and compare to others
TODO do train eval splits
TODO iterate faster. Create loads of little functions that can take in a LeRobotDataset and do something. 
- Question: 1 demonstration of picking up cube in one place is enough? Why do they recommend 5 per position, to be more general?
    - chop dataset, Train on 1 and see
- Later: data augmentation see how it impacts results, but 100% will be my requirement maybe.
- Later: pick up 4 different positions specified with pencil and put cube in center.
- What is the reason why ACT needs longer training etc? Can we just add noise to all data and it will be more robust to getting exact overfitting? action noise?
Later can I move cube to different positions
how many positions do you need between? how do NNs interpolate?

- I wanted to deeply understand everything (with notebook training)
    - even though cube is in same place for 10 episodes, could training on 1 episode make it good?
    - will eval still work now after weeks?
        - How to make eval running setup from 1 bedroom to next easier? Buy powered USB hub?
        - Could I move table? how robust are we? How to make robustness/generalisation? That’s my biggest issue, everyone’s biggest issue with these methods… Short answer is put it into the data
            - Why does my intuition sayobust to table moving? just because it is 1 cube?