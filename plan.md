# what I used to do
task_name="red_cube_always_in_same_place"
# run in both laptop and in cloud
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER

python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/${task_name} \
    --dataset.num_episodes=10 \
    --dataset.single_task="Grab red cube and put to left"

# train in cloud
task_name="red_cube_always_in_same_place"
run_count=1
python lerobot/scripts/train.py \
  --dataset.repo_id=bearlover365/${task_name} \
  --policy.type=act \
  --output_dir=outputs/train/${task_name}_${run_count} \
  --job_name=${task_name} \
  --policy.device=cuda \
  --wandb.enable=true \
  --dataset.video_backend=pyav

python lerobot_original_train.py \
  --dataset.repo_id=bearlover365/${task_name} \
  --policy.type=act \
  --output_dir=outputs/train/${task_name}_${run_count} \
  --job_name=${task_name} \
  --policy.device=cuda \
  --wandb.enable=true \
  --dataset.video_backend=pyav

huggingface-cli upload ${HF_USER}/${task_name} outputs/train/${task_name}_${run_count}/checkpoints/last/pretrained_model --repo-type=model

# evaluate on laptop
python -m lerobot.record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
  --policy.path=${HF_USER}/${task_name} \
  --dataset.repo_id=${HF_USER}/eval_${task_name} \
  --dataset.single_task="Grab red cube and put to left" \
  --dataset.num_episodes=1 \
  --display_data=true



# Plan for lerobot_notebook_pipeline
# document
This document outlines the plan for developing the `lerobot_notebook_pipeline` repository.

KEEP train_ultimate.py and periodically sync with train_ultimate.ipynb with jupytext

conda activate robosuite

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
TODO obviously the coolest goal of all would be to get some zero shot VLA to work! pizero over network or smolVLA. with fine tuning if needed but that defeats the entire purpose of VLA to me. 
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
- TODO i cannot specify root folder because of huggingface caching issues... that sucks for folder structure but works
- decide between train_ultimate or split or whatever. 



- I wanted to deeply understand everything (with notebook training)
    - even though cube is in same place for 10 episodes, could training on 1 episode make it good?
    - will eval still work now after weeks?
        - How to make eval running setup from 1 bedroom to next easier? Buy powered USB hub?
        - Could I move table? how robust are we? How to make robustness/generalisation? That’s my biggest issue, everyone’s biggest issue with these methods… Short answer is put it into the data
            - Why does my intuition say it will be robust to table moving? just because it is 1 cube?
search how Ville or others did interpretability https://github.com/villekuosmanen/physical-AI-interpretability and https://villekuosmanen.medium.com/opening-the-black-box-of-robot-ai-983ab2754eec
try bbox affordances
- keep one episode for validation and just to see how much that differs (get plots for that), and train on rest


# new doc

# Plan for lerobot_notebook_pipeline

This document outlines the plan for developing the `lerobot_notebook_pipeline` repository to **understand generalization in robots** through systematic experimentation.

## 🎯 Current Status (July 2024)
✅ **FOUNDATION COMPLETE**: Non-interactive pipeline with optimized performance  
✅ **Infrastructure**: Fast tests (~32s), automated plotting, organized output  
✅ **Tools Ready**: `analyse_dataset.py`, `demo_visualizations.py`, `train.py`  

## 🧠 Core Research Questions (Your North Star)

### **Primary Question**: What makes robot policies generalize?
- How much data is actually needed? (1 episode vs 5 vs 50?)
- What types of variation matter most? (lighting, position, orientation, background)
- How do different architectures (ACT, Diffusion, SmolVLA) handle generalization?
- Can data augmentation replace real variation?

### **Secondary Questions**: 
- Why does ACT need longer training? Is it just overfitting?
- How robust are policies to environmental changes? (table position, lighting)
- What's the minimum viable dataset for a simple task?

## 🚀 Phase 1: Systematic Generalization Study (NEXT 2-4 WEEKS)

### **1.1 Data Collection Campaign** 
**Goal**: Build the ultimate generalization test dataset

**Red Cube Experiment Series**:
- [ ] **Baseline**: 10 episodes, cube in exact same position (DONE - you have this!)
- [ ] **Position Variation**: 4 positions marked with pencil, 5 episodes each (20 total)
- [ ] **Lighting Variation**: Same position, different lighting conditions (morning/noon/evening/lamp)
- [ ] **Background Variation**: Same position, different objects on table
- [ ] **Combined Variation**: Mix everything (position + lighting + background)

**Tools to build**:
- [ ] `collect_systematic_data.py` - Script to guide data collection with prompts
- [ ] `dataset_compare.py` - Compare multiple datasets side by side
- [ ] `generalization_test.py` - Automated testing across variations

### **1.2 Single Episode Deep Dive**
**Goal**: Answer "Is 1 demonstration enough?"

- [ ] Train ACT on just 1 episode from your existing dataset
- [ ] Train with different augmentation levels (none → heavy)
- [ ] Test on slight position variations
- [ ] Document exactly where/when it fails

### **1.3 Architecture Comparison**
**Goal**: Compare how different models handle the same data

- [ ] **ACT** (your current setup)
- [ ] **Diffusion Policy** (lerobot supports this)
- [ ] **SmolVLA** (if possible with your data format)
- [ ] Document training time, data efficiency, robustness

**Tools to build**:
- [ ] `train_comparison.py` - Train multiple architectures on same data
- [ ] `architecture_analysis.py` - Compare model behaviors and predictions

## 🔬 Phase 2: Understanding Mechanisms (WEEKS 3-6)

### **2.1 Data Augmentation Deep Dive**
- [ ] Systematic study: What augmentations actually help?
- [ ] Visual noise vs action noise vs temporal noise
- [ ] Can augmentation replace real variation?

### **2.2 Overfitting Analysis**
- [ ] Train models to convergence, monitor validation performance
- [ ] Visualize what the model is actually looking at (attention maps, grad-cam)
- [ ] Test "distribution shift" - how far can you move the cube?

### **2.3 Minimal Viable Datasets**
- [ ] What's the absolute minimum data for cube pickup?
- [ ] How does performance scale with dataset size?
- [ ] Quality vs quantity trade-offs

## 🏗️ Phase 3: Scale Up Experiments (WEEKS 6-10)

### **3.1 Multi-Task Learning**
- [ ] Cube pickup + sock folding on same arm
- [ ] Transfer learning between tasks
- [ ] Multi-task vs single-task comparison

### **3.2 Environmental Robustness**
- [ ] Test in different rooms/lighting
- [ ] Move table position
- [ ] Add distractors and clutter

### **3.3 Real-World Deployment**
- [ ] Mobile manipulation setup (arm on wheels)
- [ ] Multiple camera angles
- [ ] Continuous learning / adaptation

## 🛠️ Infrastructure Improvements Needed

### **High Priority** (Support Phase 1)
- [ ] **`experiment_manager.py`**: Track experiments, hyperparameters, results
- [ ] **`dataset_builder.py`**: Systematic dataset creation and management  
- [ ] **`evaluation_suite.py`**: Standardized testing across conditions
- [ ] **`visualization_dashboard.py`**: Compare results across experiments

### **Medium Priority** (Support Phase 2-3)
- [ ] Integration with Weights & Biases for experiment tracking
- [ ] Automated video generation for policy comparisons
- [ ] Statistical analysis tools for significance testing
- [ ] Model interpretability tools (attention visualization)

### **Future Infrastructure**
- [ ] Integration with simulation environments (MuJoCo, Isaac)
- [ ] Support for multiple robot platforms
- [ ] Distributed training setup
- [ ] Real-time policy deployment tools

## 📊 Success Metrics & Deliverables

### **Phase 1 Deliverables**:
- [ ] **"Generalization Report"**: Comprehensive analysis of what makes policies robust
- [ ] **Open dataset**: Well-documented cube pickup dataset with variations
- [ ] **Benchmark suite**: Standardized tests for policy evaluation
- [ ] **Architecture comparison**: Head-to-head performance analysis

### **Research Papers / Blog Posts**:
- [ ] "How Much Data Do You Really Need? A Systematic Study"
- [ ] "ACT vs Diffusion vs Transformers: Generalization Showdown"  
- [ ] "The Minimal Viable Robot Dataset"
- [ ] "From 1 Demo to Production: A Scaling Study"

## 🎯 Immediate Next Steps (This Week)

1. **Start Single Episode Experiment** (2-3 hours)
   ```bash
   # Use your existing dataset, train on just episode 0
   python train.py dataset_path --episodes 0 --steps 10000 --output-dir ./single_episode_exp
   ```

2. **Build Data Collection Script** (1-2 hours)
   ```bash
   # Create guided data collection for systematic study
   python collect_systematic_data.py --task cube_pickup --variations position,lighting
   ```

3. **Design Position Marking System** (30 minutes)
   - Use pencil/tape to mark 4 cube positions
   - Document each position with photos
   - Create consistent pickup/placement protocol

## 🧠 Research Philosophy

**"Build greatness through systematic understanding"**

- Every experiment should answer a specific question
- Document everything - failures are as valuable as successes  
- Compare apples to apples - same evaluation metrics across experiments
- Start simple, add complexity systematically
- Measure what matters - task success, robustness, data efficiency

## 💡 Key Insights from Your Notes

1. **Focus on fundamentals**: You're right to want to understand everything deeply
2. **Infrastructure pays off**: Your pipeline work enables rapid experimentation now
3. **Generalization is THE question**: Position variation, lighting, robustness - these are the core challenges
4. **Start with 1 cube**: Simple tasks can teach profound lessons about learning
5. **Compare architectures**: Different models may have very different data requirements

---

**Remember**: You're not just training robots - you're building intuition about how learning works in the physical world. Each experiment brings you closer to understanding what makes robots truly useful in the real world! 🤖✨