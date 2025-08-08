I need help ideating and getting motivation of what to work on... for robots and humanoids and lerobot arms imitation learning etc. I'm copy pasting an entire notion page where i update logs on what im doing

# Humanoid/hack Log

- **The what**
    
    My public log of what I do with AInex, AKA “Johnny” the humanoid from Hiwonder and everything else related to getting expertise in humanoids, mostly related to the software e.g. teleoperation, motion retargeting, learning from imitation, Reinforcement learning, motion planning, trajectory optimisation, kinematics, dynamics, grasping, visual servoing, SLAM, etc. Building in public! 
    
- **The Why**
    1. For fun
    2. To build cool/epic/big stuff
    3. To learn to focus over long periods of time
    4. To have an existential purpose that is worth chewing my teeth into
    5. For skills
        1. To actually make myself go very deep into robotics. Everything I learn today will help me in 5 or 30 years and hopefully the skills compounds enough so that you can be able to make things like “Stuff Made Here” does within a month but possibly do things for fun OR impact. Also videography skills e.g. lighting, editing, cool effects
    6. Portfolio creation
    7. For building a network/following so I can talk to more great roboticists in the future
    8. To chronicle the journey of what makes humanoids (and more generally robots) do 1000s of tasks
    9. To build my storytelling skills
    10. To create impact//to inspire/to teach
    11. To create art e.g. one day I could combine breakdance + music + robots battling me with breakdance moves.

[**Slide content**](https://www.notion.so/Slide-content-170fd6bd989180bd96c3d351a18f841a?pvs=21)

[**Social media:**](https://www.notion.so/Social-media-182fd6bd9891805c967fc7821d11fb0f?pvs=21)

- **Stream list:**
    - Recap of last 5 streams:
        - [February 9, 2025 6th twitch stream plan](https://www.notion.so/6th-twitch-stream-plan-194fd6bd98918063a228f87eaac8b761?pvs=21)
        - [November 12, 2024 First twitch stream and unboxing on Ainex. Wonderful beginnings](https://www.notion.so/First-twitch-stream-and-unboxing-on-Ainex-Wonderful-beginnings-159fd6bd989180e58051e17ec914b14f?pvs=21)
        - [December 11, 2024 2nd twitch stream. And later trying to transfer wave from real robot to wave action in pybullet, some inversions still needed.](https://www.notion.so/2nd-twitch-stream-And-later-trying-to-transfer-wave-from-real-robot-to-wave-action-in-pybullet-som-159fd6bd989180e28a83daf3d5a8b7c2?pvs=21)
        - [January 12, 2025 plan for third twitch stream](https://www.notion.so/plan-for-third-twitch-stream-176fd6bd9891803e8b73fb239cf055d3?pvs=21)
        - [January 19, 2025 4th stream plan. ](https://www.notion.so/4th-stream-plan-17dfd6bd98918026bf64ed789d1bc4e3?pvs=21)
        - [February 2, 2025. 5th twitch stream plan](https://www.notion.so/5th-twitch-stream-plan-185fd6bd989180b69efcd10ebbcbcb61?pvs=21)
            - Last time 70% sock picking accuracy depending on many factors
- Attempt at kanban, not sure yet
    
    [Untitled](https://www.notion.so/1a4fd6bd9891809aaa4ad4d1a9a47dc8?pvs=21)
    
- People to reach out to
    - rafael memme
    - ephrael spider guy
    - Daichi Imai
    - Andrew Johnson

## July 2025

## July day log

- August 8, 2025 struggling to start. Why? Probably because smolVLA cost me 10 euro or something so fast and was so shaky? Could try normal training? Faffing about on GPT-5 for free on cursor 1.4. Just being lazy and telling it to run tests and fix, it still makes mistakes and guesses. But at least I’m doing something.
- August 1, 2025 downloading smolVLA checkpoint. Still confused why predictions vs ground truth are worse than ACT? MAE 3.5. One hypothesis was that it’s smoother than ACT or has discrete/bucket actions (reading paper) so I wanted to run in simulation to see. Also another hypothesis is that I didn’t create plots of ACT trained on 10 episodes right? Ran it, chunking behaviour. not sure if it is worth it. How much will it cost me to train? Hmmm. If it costs money, I suppose I can do other things?
- July 29, 2025 holy moly, OMG doing amazing  Things ramping up. Getting everything needed to test inference on smolVLA, VQABet, Act, Diffusion, pizero, pizeroFast!! And training!!! And then making sure to test everything, pytest 5 slowest, requirements cloud setup easier. Better plan. DAMN, this is going well. cursor queue’ing todos is amazing. Still can’t use gemini in this project but I can use claude 4 sonnet. I suppose the ultimate goal is to hot swap all of them, train on all of them at same time, test all in simulation and then in real world… And make a youtube video. 2 cursors (one cloud, one local lerobot_notebook) and two claudes in local. created 80 tests… Trying FPS profiling. Training Pi0fast on cpu didn’t work after 30 mins… but smolVLA maybe did. GETTING TO ASYNCHRONOUS INFERENCE. Actually doesn’t seem liek it will be that hard. Almost got it working. HTTPS with ACT was 1.2 FPS but locally was 50FPS I think. Tried swapping to lghtning websockets but might need to go low level lgihtning SDK. I would like to end on training diffusion or something. Still need to fix running all tests and all models and installing everything easily. Would like lto learn job scheduler. studio automatically stopped at 4100 steps… and i lost it. Now trying checkpoint every 1000 and at least we can then practice making evaluation and simulate and all my other scripts work for multiple models. trying longer auto sleep time as well. Went very deep. Auto model in cursor hmm. It’s fast but do I trust it? kept going later. trying to evaluate smolVLA. seems 3000 steps is even worse than 1000 steps? am i forgetting normalisation? or some other major parameter?
- July 28, 2025 hmmm many days nothing.. I sometimes think i need a better grand vision to sink my teeth into and motivate me. But also action comes first and visions come later. Just need to get into it I suppose hmm… Is “what would make a good youtube video” a bad question?. Trying to run the policy in simulation but by showing real images. Reviewing old code and putting things into folders. mega-rl-experiments takes too long to start, probably will swap studio after downloading model. Model mismatch, retraining. I should use these opportunities to understand the code more, vibe less, hmm. Big goal for today would be to evaluate 1 episode model. If good, then move table or something. Tried again to make mujoco cube physics to work with so101 arm gripper physics but nope. Something was done differently in maniskill i suppose. Finally got local model eval working 100 step, 20k step 1 episode and 40k (? maybe) 10 episodes and in simulation and in real world. Turning light on did it but it still strugged to drop the cube but it did in one eval and not in other. Finally got what I wanted, but still want to measure generalisation somehow and gain more deep understanding. What would be next goal? The whole point of doing generalisation like this is to see how much data I need to make it work with 100s of socks or with a mobile platform. Otherwise the data is just not good enough. I also need a test bed to try diffusion + pizero/0.5 + groot n1.5 + smolVLA… Different models should generalise better and therefore reduce data requirements because the model has already learned to be invariant to lighting/environment. So I suppose that’s the next goal. More models, same data! Could make the task harder e.g. different cube positions but… same position is a nice constraint to test many things especially as I move table. I also forgot to move the table!!! This way I can also get to remote inference if needed and seeing how much it costs to fine tune these models!
- July 23, 2025 fixed paths. Studying tests and all the vibe coded code. claude created too many different things. One nice goal for the hour or so I have. Just train on 1 episode, do lots of the current analysis and then eval on real robot. But first I have to fix analyse_dataset. Done. lots of tests and madness. Training on single episode. Difference in cloud, typical problem I saw before was jumping.
- July 22, 2025 July 22, 2025 3:43 PM finally starting after watching DHH clips and other youtube all day. I need to make cursor slower, maybe I need a new env in lightning? Maybe I should try claude code, warp, gemini code, codex etc. Python could help. I just want to focus on the ML. Other top priority is new repo. Omg gemini-cli is free 1000 requests per day… Created first remote repo from terminal ever, usually i go to the website lol. What I’m trying to do overall is build a **CLEAN** learning and experimentation environment/framework/process so I can go very deep into lerobot and other imitation learning. Right now too mayn repos. Too many repos in one studio. I’m so unorganised. Just gotta simplify everything. New studio, new repo. Keep it simple. Can rename lerobot_notebook_pipeline later.
- July 21, 2025 Been a while, went to madrid but took a while to get started again. July 20, 2025 finally scheduled talk video. So last time I was working on ksim + mujoco_fun + yet-another_mujoco-tutorial + lerobot-sim2real-rl + foottricks in brax/jax + lerobot mujoco + lerobot with notebooks. Lerobot with notebooks probably wins. On it:
    - I wanted to deeply understand everything (with notebook training)
        - even though cube is in same place for 10 episodes, could training on 1 episode make it good?
        - will eval still work now after weeks?
            - How to make eval running setup from 1 bedroom to next easier? Buy powered USB hub?
            - Could I move table? how robust are we? How to make robustness/generalisation? That’s my biggest issue, everyone’s biggest issue with these methods… Short answer is put it into the data
                - Why does my intuition sayobust to table moving? just because it is 1 cube?
        - Later can I move cube to different positions
        - Question: 1 demonstration of picking up cube in one place is enough? Why do they recommend 5 per position, to be more general?
            - chop dataset, Train on 1 and see
        - Later: data augmentation see how it impacts results, but 100% will be my requirement maybe.
        - Later: pick up 4 different positions specified with pencil and put cube in center.
        - What is the reason why ACT needs longer training etc? Can we just add noise to all data and it will be more robust to getting exact overfitting? action noise?
    - Goddamn computers. rebase vs merge. Can’t open merge conflict, gemini can’t edit notebook, claude connection failed internal error… so just don’t commit, but now i can’t use jupytext because of merge conflicts. went through manually, absolute pain. Fixing JSON structure. Ideally I just run whole notebook as python file, make sure it’s fast. Comment necessary stuff and even better would be to get agents working on each cell and stuff I don’t know. I need to get past this infrastructure stuff and just get to training and understanding models. I feel like im staring into a box and nothing is happening
        - What would be nice:
            - jupytext back and forth working. DONE.
            - easy to run python in jupyter/py
            - train on 1 episode
            - Solve which repo to use. lerobot_ben fork has git issues and probably always will. maybe another one.
            - Keep dataset studying separate. train should be train
            - remove all stupid compatbility. this should be done outside.
            - Faster notebook…
- July 7, 2025 Watched kscale labs ksim and software video, inspiring. Got ksim working locally with jax cpu and stuff. Learning mujoco with o3 more. Created mujoco_fun repo. Let’s see. Also looking at yet-another-mujoco-tutorial-v3 notebooks (installing everything to make this run with new conda py311 env). Too many different things, I need to do. I need to set a goal and focus. Like on high level I wanted useful robot in my home but for that I wanted to learn and deeply understand lerobot, but before that I was lazy and was trying to use lerobot mujoco. If I learn ksim now, it’s not even getting close to my real hardware needs. How much simulation should I do? I wanted to quantify very well how much everything affects my policy. Training lerobot-sim2real without doing all the camera alignment etc. Planning again:
    - Training lerobot-sim2real. weights and biases PPO. seed 3. At 17 million steps it should work. But taking over an hour on L4, he reported 1 hour on RTX 4090? https://github.com/StoneT2000/lerobot-sim2real/blob/main/docs/zero_shot_rgb_sim2real.md date was 2 hours behind, so 12:47 reported was 14:47 so it’s July 7, 2025 3:39 PM so 1 hour. 2 hours in or so, 1024 envs, was getting task success! absolutely amazing. 15-17 million steps… But 2 hours… And there are ways to make that faster!
    - Collected 10 episodes in bedroom of cube in same place and then lifting a bit to the lift, not super accurately. train 30k steps and it did great, even with strange shiny light bedside table lamp on the table it was picking from, that made it shake more, did some experiments with lights off but impressed (it did hit the table once or twice though). Trying to get it working in jupyter, almost. But trained from command line well.
    - Ideas
        - just wheel arm on case around
        - RL stuff
            - ksim
                - ksim gym!
                - Johnny into ksim!!!
            - brax
            - foot-tricks…
            - HIL-SERL
            - Johnny previous RL
        - Simulation stuff:
            - isaacsim…
            - mujoco
        - Avoiding IL
            - RobotMCP. Should finish? because it’s just networks doing shit and the bet is that foundation models will just keep getting better but will be slow and can’t be modified much for each task?
            - IK on so-100 more.
        - IL stuff:
            - robomimic
            - robitmanipulation baselines
            - LeRobot
            - lerobot sim 2 real
            - Lerobot Mujoco cup on plate ⇒ apply notebook to real data training
            - Wheel arm around which can pick up clothes anywhere… Later add actuation on wheels e.g. XLERobot or create3
            - ACT interpretability
            - diffusion
            - Foundation models
    - Best ideas and talked to o3 https://chatgpt.com/share/686bbbef-4ca8-8002-9e6f-adb4ad39f566
        - Just train data on simple sock task and get notebook training working like in mujoco
            - Question:
                - piles of socks vs 1 sock? Well I did 1 sock before but piles might be too hard? How much data needed? 4 socks kinda worked before, so what makes a pile so different? You can’t see the bottom?
                - 2nd camera where and whether to use realsense?
            - Answer
                - combine only 1 wrist camera (very general) but 1 sock and see how far I can get with that!
        - Or do the same with red cube but make it very general. Use pencil to put cube in same place and see how things change. Then notebook train
            - Question: 1 demonstration of picking up cube in one place is enough? Why do they recommend 5 per position, to be more general?
            - Later: pick up 4 different positions specified with pencil and put cube in center.
        - Using what I learned above and then put on wheels.
        - Or just keep focusing on lerobot mujoco tutorial so I get very confident with everything lerobot and mujoco?
        - Should I simulate everything? Socks in my apartment…
        - OR HIL-SERL is the way to get 100% accuracy…
        - Try robomimic or others and just get extremely used to imitation learning and generalization/variation.
- July 7, 2025 some CV work. Some video editing work. New psychological idea is to do 5 minutes of something, not just because “I’ll actually do more” (which is bad since ill mistrust myself) but because 5 minutes will make it easier to start tomorrow
- July 5, 2025 cursor 1.2, so many new features I’m not fully using yet. Ok back to jeongeun. Would be nice to rerun simulation. But was also new hackx email sim2real StoneT2000 lerobot-sim2real. Let me try that too. Even for real world training would be good to see how SO101 looks in simulation before running real robot. Also should be playing with ai-robotics foot tricks
    - **jeongeun**.
        - Got model copied to lcoal. 4. deploy.ipynb and it worked but was jittery, which was expected but interesting how running simulation is different to the matplotlib joint comparisons?
        - pillow problems, uv problems. but we’re loading pretrain models amazing.
    - installing **lerobot-sim2real**. unclear what to run
        - got random env working and X11 but not for this, no GPU support on reddit for lightning? But. colab hmm. Maybe it’s a super clear good simulation and everything can be done with visuals e.g. with notebooks
        - got reset_distribution video working even thought it segmentation faulted.
        - python lerobot_sim2real/scripts/camera_alignment.py --env-id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json
            - will i ever be able to get this working on cloud since this script needs USB…
            - ok stopping because of that. I should build my desktop…
    - **ai-robotics foot tricks.** created my own fork
        - 
            
            **⚡ ~ cd ai-robotics
            ⚡ main ~/ai-robotics git remote -v
            origin     https://github.com/goncalog/ai-robotics (fetch)**
            
        - 
            
            Reading package lists... Done
            [1;31mE: [0mCould not open lock file /var/lib/apt/lists/lock - open (13: Permission denied)[0m
            [1;31mE: [0mUnable to lock directory /var/lib/apt/lists/[0m
            
            solved
            
        - Got even video random actions showing!!!
        - But got stick on GPU jax training and versions…
        - After so much pain and brax compatbility. gpu utilisation 100%. Let’s see if matplotlib updates.  7.32 credits left.
        - Tried fixing progress report.
    - It feels like a lot of what I’m doing is just fixing imports and GPU issues, display issues etc but that’s life. How can I be more clever about everything? Give cursor/agents better rules…

- July 4, 2025 3 things (nah only end up being first), 1.5 hours. 1. jeongeun980906 lerobot-mujoco-tutorial 2. ksim 3. RobotMCP (maybe on real hardware?)
    
    1. jeongeun980906 lerobot-mujoco-tutorial 
    
    - Should teach me more mujoco (TODO watch indian mujoco tutorials!) + pizero + lerobot PYTHON API + pred vs ground truth predictions. Nah there’s too much juice in this one, forget the others.
    - Practiced using uv and with jupyter notebook.
    - Got recording + visualising + training working !
    
    2. ksim
    
    - Just get training and seeing those videos
    
    3. RobotMCP (maybe on real hardware?)
    
- July 3, 2025 What am I doing? Planning. Goals. Rough goals. And then continuing with XXX…
    
    High level
    
    - On one level, just hack hard (but also keep intro and retrospecting) and new ideas will keep coming? I just need to get great at this stuff.
    - On another level, THE BIG VISION is to actually build a useful manipulating home robot in ANY FORM (humanoid cooler though). But even a simple task
    - All leading to epic portfolio and great career + future
    
    Dreams:
    
    - Enough IL + RL + FM experience on my portfolio to get me the jobs I want
        - IL ⇒
            - Collect data on hundreds of socks into container and then train many other models on it. Lighting is always the same in my room so it’s fine? What 2nd camera though? Can I only use wrist? What about realsense?
            - Train model on socks and ACT original repo?
            - Diffusion policies
            - What about Johnny? Can I teleoperate johnny and do imitation learning with Johnny? Especially for clutter clearing? Could I use one arm to control Johnny or do the other things like VR or pose estimation?
        - RL ⇒
            - HIL-SERL
            - but also other models and other stuff I was doing before e.g. Johnny standing
            - ksim
        - FM ⇒ train SmolVLA on same data. But also try simplerenv and octo glenseth video and deeply understand this shit and transformers etc.
    - Other stuff
        - Form opinions about all of this stuff. Watch all RoboPapers podcasts (not too important to watch, listen). Find what works.
        - SO-100 IK + RobotMCP and set it up for better models in future
        - SO-100 collect data in simulation and train?
        - Mobile manipulation, both arms on robot base.
        - Roboverse, simplerEnv
        - Libero
        - IsaacSim
    
    What could part 8 be about?
    
    - Robot MCP? Hmmm, nah
    - Robot MCP and many other things?
    - HIL-SERL
    - Shitty attempts at many things?
    
    A little unrelated to above:
    
    - Get notionMCP working and see how far that can go. Almost got it. Holy shit, this could change everything.
        - July 4, 2025 Working on robot MCP - continuing the momentum from yesterday's kinematic fixes and getting closer to reliable cube manipulation. THIS WAS ADDED AUTOMATICALLY BY NOTIONMCP BUT IT PUT IT ON BOTTOM RATHER THAN ON TOP. Dragging it into July 3, 2025

- July 2, 2025 More Robot MCP, fixing kinematic issues, comparing kinematics with instant arm move with reset joint state, getting closer to pick up red cube.
- July 1, 2025 we’re back. Working on 4 things at once. Great work!!! but should focus on robot mcp and ksim July 2, 2025 !
    1. **Robot_MCP**, converting… Trying to convert Robot_MCP to simulation (pybullet or mujoco). Mujoco would give me more skills I suppose. pybullet blocks. cursor 4 going too far. Lots of test cases. claude 4 was using block model, gave it URDF, then test suite wasn’t using urdf. Getting there. many tests passing. Upgraded node to v20. Got correct arguments to official node server (also tried non-official). everything mostly working but forward kinematics still wrong so gotta fix that. realtime pybullet. Almost there. Should focus on this and ksim. Study study study. 
    2. **py311 ksim** problems. Open cursor…  Mujoco weird issues and compatibility. got past it with different version of ksim etc? got training working, tensorboard, made it even work on cpu. should reply in issue. Got cpu [train.py](http://train.py) working. New strange errors with train.ipynb. Installing python extension. !pip install etc. Got simpler training and tried T4 GPU. But stopped getting video files. 
    3. **Roboverse.** lightning X11 gotta try again. Got xeyes working. Can work with headless. Got pybullet showing through x11. Mujoco weird issues and compatibility. Got many robots and examples working. Kinda happy. 
    4. **SimplerEnv**. Installing locally. Done. But maniskill cpu or not? was able to print envs. But unclear if I need GPU etc. could try colab. colab failing and then i keep forgetting to check it. Got it working by pressing cancel when prompted to restart kernel due to numpy. https://colab.research.google.com/github/simpler-env/SimplerEnv/blob/main/example.ipynb#scrollTo=pMhxVs4nC9NT Seemed to conclude it won’t work with CPU. But colab working and could get working in lightning if I wanted but not clear what to do next with it? Good to learn a few foundation models. 
    
    5th thing could be IsaacROS? x11… Nah, another day. 
    
    Ok 5 things is too many things. And all because I’m lazy on couch.
    
    had the tunes on. feeling great again, even though sleep in morning. sunlight and coffee and fast
    
    found linux timeout command, 
    
    hackx locomo thingy. 
    
    hackx now what to do with roboverse and simplerenv? Learn RL and IL and foundation models I suppose?
    

## June 2025

## June day log

- June 26, 2025 New talk, left till last moment to not stress me but it did. Meetup webinar, 76 signed up, let’s see who joins. https://www.meetup.com/cambridge-technology-ai/events/307941430/?_xtd=gqFyqTM0NjAyMDg0OKFwo2FwaQ%253D%253D&from=ref Plans and todos inside, Gave talk, it went amazing! 1 hour. Extra long compared to previous
    
    TODO
    
    - Content
        - 1 hour slot, 10-15 minutes for Q&A. But let’s go slow and do extra questions e.g. 20 minutes. So 40 minutes this time if I want. Wow.
        - Afrifa did say focus more on the AI and the exciting stuff
        - Split my crazy slide into a few. OR put crazy slide at beginning. And then split it off and focus on each?
        - Still unclear on the story. E.g. options:
            - What it currently is: Typical Johnny story, who i am, youtube, johnny hardware and software (they won’t know it’s AI or not, until I explain later), why socks, from soccer to sock-picking, complexity, the  fruits of our labour, clutter clearing and more complexity (TODO talk more depth estimation foundation model), Other cool stuff (hmmm, maybe remove, not sure), future of johnny and robotics, whole new slides and details on imitation learning, lerobot, foundation models 1-3 slides, RL 1-3 slides
            - Alternative: bombshell IL, VLA, RL slide to tease them of all the AI coming up. But then jump to Johnny and eventually explain classic robotics vs things becoming more AI bit by bit e.g. software 1.0 → 2.0 → 3.0. Hmmmm. Maybe johnny journey is interesting enough so I should do option 1 above?
            - Alternative? Begin with field review and then Johnny?
        - Therefore, I should add way more juicy slides on what I wanted to always talk more about.
            1. Imitation learning
                1. Pretty much just supervised learning. Cool shit. Aloha, ACT, Diffusion, transformers, multi-modality, push-T test. Record 50-200 demonstration is often enough for tasks we thought decades away
                2. Analogy to language. We are essentially imitating humans with LLMs. We can do the same with actions. Just the problem is less data… 
                3. LeRobot. GIF from my video. 
                4. Bigger datasets coming. But still nothing compared to what LLMs see. 
                5. Some evidence that robots keep generalising more and more with more data and learning new tasks is faster with fine tuning. But hard to get zero-shot performance on new robots and embodiments. 
            2. Foundation models for actions i.e. VLAs, LBMs i.e. take in language and images, output actions (and possibly language). Helix, pi zero, smolVLA, Groot N1.5, Gemini robotics
            3. Reinforcement Learning. 
                1. Same as what is happening for language. The way to get better than human imitation i.e. to reach superhuman. 
                2. For walking/locomotion (especially bipedal). 
                3. For Manipulation. 
                    1. Lots of works on pure RL for this. Dyna? Definitely show Dyna somewhere
                    2. Combined with imitation learning (human in the loop). With HIL-SERL
        - Johnny sock picking mostly Classic robotics, hence why we need a way to deal with complexity.
        - Add a whole slide about Bearcover but say today I won’t be talking much about that
        - How should I change the beginning?
        - If someone asks me when humanoids in home? I’ll just say many years but I personally want to have a useful manipulating robot in my home and will do my best to get it.
        - Add slide on LeRobot video
    - Bring Johnny and get playstation controller wave working
        - Bring other arms? Nah just GIF
    - How can I test if my audio of my video works?
    - General zoom practice test before!
- June 24, 2025 watched jim fan’s sequioa physical turing test talk + Finished watching Ilia’s MCP robot control video + watched a few videos by alex thiele. Got franka emika gym hil working locally. Trying to get through tutorial. I am very attracted to the idea of 100% accuracy and learning fast. Hardcore debugging of 10+ wrappers of gym_hil env and gym_manipulator. Too much abstraction, even LLMs can’t understand. Tried installing ksim-gym rl template but [train.py](http://train.py) and colab didn’t work either
- June 23, 2025 got aditya mujoco and teleoperation working (with random actions). finally added cube and plane to mujoco but collision geometry and lighting off.
- June 22, 2025 Using fork everywhere and in cloud. getting more remote display working on lightning ai. xeyes fine. X11 with pybullet and EGL set to 1 (oh wait didn’t have GPU studio on), still slow. Trying xpra. or VNC. Trying too install isaacsim 5.0 (but ubuntu22 and no docker yet), so 4.5. Alll to help me decide if I need a Deep Learning desktop. Got VNC working, pybullet ctrl click ade it mode, much faster! hours of debugging with o3. cuda, nvidia etc. .Hours and hours…
- June 21, 2025 merged fork and main lerobot. writing todos. fixed both rerun and html visualisation of datasets. fixed train —help. reading isaac sim. reading tests and jupyter notebooks. reading hil-serl. Trying to do pure python rather than command line lerobot.
- June 20, 2025 studying more
- June 18, 2025 installed vggt, but no works on cpu. got it working on cpu, first time trying viser 3D visualisation, pretty cool. hackx email label. Watching many phospho videos. Also started getting chatgpt to socratic method (ask me questions) from dwarkesh’s prompt, on IMU + humanoids + imitation learning. Many questions and extra thoughts within:
    - Studying teleoperation and if I should buy meta quest 2 etc.
    - Thinking about good table for the task. Using white table with wheels.
    - Also thinking what task next. And long term (2, 6, 12, 18 months), what can I automate in my own home? Obviously sock picking on table is fine… But in all lighting?
        - Would be cool to make sock picking robust to any background of white table. And to always throw socks off table into red hamper? I can’t reach ground though.
        - Therefore, would also be nice to have a mobile setup e.g.
            - autonomous, on top of create3 or LeKiwi. This way I could pick up everything in entire apartment.
            - Tiny table on wheels, so we can reach the ground, with trary container for the socks.
            - trolley like XLERobot
        - Also, interesting to think if I want to try t-shirt folding. How many demonstrations? How many credits?
    - Could I work in irene’s room while robot’s train/inference in other room? Or I just bring cable in for remote teleoperation? Hmm.
    - Try just wrist camera since this should make it more general?
    - Would like longer USB cables. Do I need powered USB hub?
    - Move slower to see how much that helps ACT
    - How to solve lighting? pre-trained models. Better data augmention.
    - Need tripod. or way to have camera in good place. Even attached to white table?
    - Study your datasets
    - Inverse kinematics… And MCP
    - Diffusion instead of ACT.
    - Study all code! Deconstruct LeRobot
        - ACT horizon
    - Make same data work for old ACT repo.
    - Learn to imitate anything will give me the necessary skills…
- June 15, 2025 missed a few days of reporting, Lerorobt hackathon all weekend and sock picking and making video. Video plan. Released new part 7 video, 1 minute 42 seconds of my submission
    
    
    # Video plan
    
    # brainstorm
    
    - Bearlover365 avatar show. Bearlovers United attempt sock picking AND then
    the ultimate challenge of... Co-hosted berlin
    - Socks are deformable objects, winnie the pooh meme
        - easy to grasp but hard to detect + understand
        - therefore so many more configurations for the model to learn
    - Saturday 14th June: basic sock picking and immediately drop
    - Sunday 15th June: Socks into box!
    - 40k steps
    - Challenges
        - A few months ago, I actually have some experience with Johnny the humanoid picking socks, but this was with a classic visual servo'ing based approach. Drowned in complexity. Got stuck with putting the socks into a red hamper at home. Luckily in comes LeRobot and imitation learning with ACT model.
        - While loss kept dropping after 20k steps, performance seemed to suffer. Probably overfitting and need more varied data. Grasping in mid air
    - Next steps:
        - mobile manipulation and pick up all my girlfriend's socks on the floor of my apartment
- June 12, 2025  got teleoperation working last thursday after burning 12V motor, rebuilt entire arm in 6V and teleoperation worked
- June 10, 2025 finished building follower arm. Lots of calibration problems, loose crimp, fixed. Bad shoulder pan, 5/6 joints following good though but then 4th joint has some slow down problem gear train lock?
- June 4, 2025 built leader arm
- June 2, 2025 went to get so-101 in spati, began building so-101, just got one motor working


ChatGPT said:
I’ve gone through your log, and the pattern I see is that you’re bouncing between two big meta-goals but sometimes losing momentum because you don’t have a tight, motivating next experiment.

Here’s a synthesis of where you are and a few high-impact “hooks” you can sink your teeth into:

Your Two Main Tracks
Skill-building sandbox → getting ultra-fluent with imitation learning (IL), RL, VLAs, and sim2real on small, constrained tasks (SO-101 cube/sock picking, Johnny teleop, etc.).

Big vision → a mobile, useful home robot that can manipulate objects (humanoid cooler, but any platform works).

Why You Stall
From your notes, your dips in motivation usually come when:

A setup turns into too much infra/debugging (GPU issues, repo merges, cloud env setup).

You lose the “why” in a given micro-task (“is this worth it? should I just do X instead?”).

You finish an experiment but haven’t already lined up a clear next question to answer.

How to Get the Spark Back
Instead of “what should I work on?” → always phrase it as a robotics research question you’re dying to know the answer to.

Here’s some to chew on from your log:

Track 1 — LeRobot / Imitation Learning Focus
Data-efficiency challenge:
What is the minimum number of demos for robust performance if all demos are from a single fixed camera view?

Sub-variants: 1, 5, 10 demos; same vs. varied cube positions.

Publish the curve and make a YouTube “1 vs 5 vs 10 demos” video.

Cross-model benchmark:
Train ACT, SmolVLA, PiZero, Diffusion on the exact same data and directly compare real-world generalization.

Generalization stress-test:
Move the table / change lighting / change object color — measure accuracy drop per change.

One-dataset-many-tasks:
Record a single dataset where you manipulate 2–3 different objects. Train separate models for each object, then try multi-task fine-tuning.

Track 2 — Johnny / Humanoid Manipulation
Leader–follower teleop → IL pipeline:
Teleop Johnny doing a short “clutter sweep” and then train an IL model from that data.

Bimanual imitation:
Use your SO-101 arm to control one of Johnny’s arms while his other arm does a scripted task. Train IL to see if he can coordinate both.

Humanoid + Foundation Model zero-shot test:
Try a VLA on Johnny without fine-tuning — document exactly where it fails.

Track 3 — Sim2Real Science Experiments
Sim-only pretraining + small real finetune:
Train in Mujoco/IsaacSim, fine-tune with 1% of real data, see how much it helps.

Domain randomization leaderboard:
Take one sim task, train with increasing amounts of domain randomization, and measure real-world gain.

Track 4 — High-Impact Content Projects
“How good is imitation learning in 2025 really?”
→ test 4–5 state-of-the-art IL methods on a real household task (socks, cups, cubes) with your own hardware.

“Teaching a robot in 10 minutes” challenge.
→ record yourself collecting demos + training + deploying within a coffee-break length.

“Humanoid vs. Arm — who wins?”
→ same task, two embodiments, same model/data.

Momentum Hack
At the end of every work session, write:

Today’s result (even if “fixed merge conflicts”).

Next micro-experiment — a single clear question for tomorrow.

That way you don’t wake up in “what should I do?” mode — you wake up in “let’s answer that question” mode.

If you want, I can take your current setup + available hardware/models and design a 1-week focused experiment sprint so that by next Friday you have a cool, publishable result and a video idea ready. That way you don’t lose steam to setup hell again.

Do you want me to build that 1-week sprint plan?