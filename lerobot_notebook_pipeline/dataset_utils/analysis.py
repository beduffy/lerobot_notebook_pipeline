import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np
import time


def get_dataset_stats(dataset: LeRobotDataset):
    """
    Analyzes a LeRobotDataset and returns a dictionary of summary statistics.

    Args:
        dataset: A LeRobotDataset instance.

    Returns:
        A dictionary containing summary statistics.
    """
    stats = {}
    stats["num_steps"] = len(dataset)
    stats["num_episodes"] = len(dataset.meta.episodes)

    # Get observation and action space dimensions
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            stats[f"{key}_shape"] = tuple(value.shape)
            stats[f"{key}_dtype"] = value.dtype

    # Get dataset statistics
    stats["dataset_stats"] = dataset.meta.stats

    return stats


def analyze_episodes(dataset: LeRobotDataset):
    """
    Analyzes individual episodes in the dataset.
    
    Args:
        dataset: A LeRobotDataset instance.
        
    Returns:
        Dictionary with per-episode analysis
    """
    start_time = time.time()
    print("🔍 Analyzing individual episodes...")
    
    episode_analysis = {}
    num_episodes = len(dataset.meta.episodes)
    print(f"   Processing {num_episodes} episodes...")
    
    data_loading_time = 0
    computation_time = 0
    
    for ep_idx in range(num_episodes):
        ep_start = time.time()
        
        # Get episode data indices
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        episode_length = to_idx - from_idx
        
        if ep_idx % max(1, num_episodes // 4) == 0:  # Progress every 25%
            print(f"   📈 Processing episode {ep_idx}/{num_episodes} (length: {episode_length})...")
        
        # Analyze episode actions
        load_start = time.time()
        episode_actions = []
        episode_states = []
        
        for i in range(from_idx, to_idx):
            sample = dataset[i]
            episode_actions.append(sample['action'].numpy())
            if 'observation.state' in sample:
                episode_states.append(sample['observation.state'].numpy())
        
        episode_actions = np.array(episode_actions)
        episode_states = np.array(episode_states) if episode_states else None
        data_loading_time += time.time() - load_start
        
        # Calculate episode statistics
        comp_start = time.time()
        episode_analysis[ep_idx] = {
            'length': episode_length,
            'from_idx': from_idx,
            'to_idx': to_idx,
            'action_mean': np.mean(episode_actions, axis=0),
            'action_std': np.std(episode_actions, axis=0),
            'action_range': np.max(episode_actions, axis=0) - np.min(episode_actions, axis=0),
            'max_action_change': np.max(np.abs(np.diff(episode_actions, axis=0)), axis=0) if len(episode_actions) > 1 else np.zeros(episode_actions.shape[1])
        }
        
        if episode_states is not None:
            episode_analysis[ep_idx].update({
                'state_mean': np.mean(episode_states, axis=0),
                'state_std': np.std(episode_states, axis=0),
                'state_range': np.max(episode_states, axis=0) - np.min(episode_states, axis=0)
            })
        computation_time += time.time() - comp_start
        
        if ep_idx % max(1, num_episodes // 4) == 0:
            ep_time = time.time() - ep_start
            print(f"      Episode {ep_idx} completed in {ep_time:.2f}s")
    
    # Print summary
    lengths = [ep['length'] for ep in episode_analysis.values()]
    total_time = time.time() - start_time
    
    print(f"📊 Episode Analysis Summary:")
    print(f"   Number of episodes: {num_episodes}")
    print(f"   Episode lengths - Mean: {np.mean(lengths):.1f}, Std: {np.std(lengths):.1f}")
    print(f"   Shortest episode: {min(lengths)} steps")
    print(f"   Longest episode: {max(lengths)} steps")
    print(f"⏱️  Timing breakdown:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Data loading: {data_loading_time:.2f}s ({data_loading_time/total_time*100:.1f}%)")
    print(f"   Computation: {computation_time:.2f}s ({computation_time/total_time*100:.1f}%)")
    
    return episode_analysis


def compare_episodes(dataset: LeRobotDataset, episode_indices=None):
    """
    Compare action trajectories across episodes.
    
    Args:
        dataset: A LeRobotDataset instance
        episode_indices: List of episode indices to compare (default: first 3)
    """
    start_time = time.time()
    
    if episode_indices is None:
        episode_indices = list(range(min(3, len(dataset.meta.episodes))))
    
    print(f"📈 Comparing episodes {episode_indices}...")
    
    # Get action dimensions
    print("   🔍 Analyzing action dimensions...")
    sample_start = time.time()
    sample = dataset[0]
    action_dim = sample['action'].shape[0]
    joint_names = [f'Joint {i+1}' for i in range(action_dim-1)] + ['Gripper'] if action_dim == 7 else [f'Dim {i}' for i in range(action_dim)]
    print(f"      Action space: {action_dim} dimensions ({time.time() - sample_start:.3f}s)")
    
    # Setup plotting
    print("   🎨 Setting up plot structure...")
    plot_setup_start = time.time()
    fig, axes = plt.subplots(action_dim, 1, figsize=(15, 3*action_dim))
    if action_dim == 1:
        axes = [axes]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    print(f"      Plot setup completed ({time.time() - plot_setup_start:.3f}s)")
    
    # Load and plot episode data
    data_loading_time = 0
    plotting_time = 0
    
    for ep_num, ep_idx in enumerate(episode_indices):
        print(f"   📊 Processing episode {ep_idx} ({ep_num+1}/{len(episode_indices)})...")
        
        # Get episode data
        load_start = time.time()
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        episode_length = to_idx - from_idx
        
        episode_actions = []
        for i in range(from_idx, to_idx):
            episode_actions.append(dataset[i]['action'].numpy())
        
        episode_actions = np.array(episode_actions)
        data_load_time = time.time() - load_start
        data_loading_time += data_load_time
        print(f"      Data loaded: {episode_length} steps ({data_load_time:.3f}s)")
        
        # Plot each joint
        plot_start = time.time()
        for joint_idx in range(action_dim):
            axes[joint_idx].plot(episode_actions[:, joint_idx], 
                               label=f'Episode {ep_idx}', 
                               color=colors[ep_idx % len(colors)],
                               alpha=0.8)
            axes[joint_idx].set_title(f'{joint_names[joint_idx]} - Episode Comparison')
            axes[joint_idx].set_xlabel('Time Step')
            axes[joint_idx].set_ylabel('Action Value')
            axes[joint_idx].legend()
            axes[joint_idx].grid(True, alpha=0.3)
        plot_time = time.time() - plot_start
        plotting_time += plot_time
        print(f"      Plotting completed ({plot_time:.3f}s)")
    
    # Finalize plot
    print("   🎨 Finalizing plot...")
    finalize_start = time.time()
    plt.tight_layout()
    plt.suptitle('🔄 Episode Action Trajectory Comparison', fontsize=16, y=1.02)
    finalize_time = time.time() - finalize_start
    
    print("   📺 Displaying plot...")
    show_start = time.time()
    plt.show()
    show_time = time.time() - show_start
    
    total_time = time.time() - start_time
    print(f"⏱️  Episode comparison timing:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Data loading: {data_loading_time:.2f}s ({data_loading_time/total_time*100:.1f}%)")
    print(f"   Plotting: {plotting_time:.2f}s ({plotting_time/total_time*100:.1f}%)")
    print(f"   Plot finalization: {finalize_time:.3f}s ({finalize_time/total_time*100:.1f}%)")
    print(f"   Display: {show_time:.3f}s ({show_time/total_time*100:.1f}%)")


def analyze_action_patterns(dataset: LeRobotDataset):
    """
    Analyze action patterns and dynamics in the dataset.
    
    Args:
        dataset: A LeRobotDataset instance
    """
    start_time = time.time()
    print("🎯 Analyzing action patterns...")
    
    # Collect all actions
    print("   📊 Loading action data...")
    load_start = time.time()
    all_actions = []
    
    dataset_len = len(dataset)
    for i in range(dataset_len):
        if i % max(1, dataset_len // 10) == 0:  # Progress every 10%
            print(f"      Loading actions: {i}/{dataset_len} ({i/dataset_len*100:.1f}%)")
        action = dataset[i]['action'].numpy()
        all_actions.append(action)
    
    all_actions = np.array(all_actions)
    load_time = time.time() - load_start
    print(f"      Action data loaded: {all_actions.shape} ({load_time:.2f}s)")
    
    # Calculate derivatives
    print("   🧮 Computing action derivatives...")
    deriv_start = time.time()
    action_velocities = []
    action_accelerations = []
    
    # Calculate velocities (action changes)
    if len(all_actions) > 1:
        action_velocities = np.diff(all_actions, axis=0)
        print(f"      Velocities computed: {action_velocities.shape}")
    
    # Calculate accelerations (velocity changes)
    if len(action_velocities) > 1:
        action_accelerations = np.diff(action_velocities, axis=0)
        print(f"      Accelerations computed: {action_accelerations.shape}")
    
    deriv_time = time.time() - deriv_start
    print(f"      Derivatives computed ({deriv_time:.3f}s)")
    
    action_dim = all_actions.shape[1]
    joint_names = [f'Joint {i+1}' for i in range(action_dim-1)] + ['Gripper'] if action_dim == 7 else [f'Dim {i}' for i in range(action_dim)]
    
    # Create analysis plots
    print("   📈 Creating action distribution plots...")
    plot1_start = time.time()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Action distributions
    dims_to_plot = min(4, action_dim)
    for i in range(dims_to_plot):
        row, col = i // 2, i % 2
        axes[row, col].hist(all_actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'{joint_names[i]} - Action Distribution')
        axes[row, col].set_xlabel('Action Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('📊 Action Value Distributions', fontsize=16, y=1.02)
    
    show1_start = time.time()
    plt.show()
    plot1_time = time.time() - plot1_start
    show1_time = time.time() - show1_start
    print(f"      Action distributions plotted ({plot1_time - show1_time:.3f}s + {show1_time:.3f}s display)")
    
    if len(action_velocities) > 0:
        # 2. Action velocity distributions
        print("   ⚡ Creating velocity distribution plots...")
        plot2_start = time.time()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i in range(dims_to_plot):
            row, col = i // 2, i % 2
            axes[row, col].hist(action_velocities[:, i], bins=50, alpha=0.7, edgecolor='black', color='orange')
            axes[row, col].set_title(f'{joint_names[i]} - Velocity Distribution')
            axes[row, col].set_xlabel('Action Velocity')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('⚡ Action Velocity Distributions', fontsize=16, y=1.02)
        
        show2_start = time.time()
        plt.show()
        plot2_time = time.time() - plot2_start
        show2_time = time.time() - show2_start
        print(f"      Velocity distributions plotted ({plot2_time - show2_time:.3f}s + {show2_time:.3f}s display)")
    
    # Print statistics
    print("   📊 Computing summary statistics...")
    stats_start = time.time()
    print(f"\n📊 Action Pattern Analysis:")
    print(f"   Total action steps: {len(all_actions)}")
    print(f"   Action dimensions: {action_dim}")
    
    for i, joint_name in enumerate(joint_names):
        print(f"\n   {joint_name}:")
        print(f"      Range: [{all_actions[:, i].min():.3f}, {all_actions[:, i].max():.3f}]")
        print(f"      Mean: {all_actions[:, i].mean():.3f}")
        print(f"      Std: {all_actions[:, i].std():.3f}")
        
        if len(action_velocities) > 0:
            print(f"      Max velocity: {np.abs(action_velocities[:, i]).max():.3f}")
            print(f"      Avg |velocity|: {np.abs(action_velocities[:, i]).mean():.3f}")
    
    stats_time = time.time() - stats_start
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Action pattern analysis timing:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Data loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
    print(f"   Derivative computation: {deriv_time:.3f}s ({deriv_time/total_time*100:.1f}%)")
    if len(action_velocities) > 0:
        plotting_time = plot1_time + plot2_time
        print(f"   Plotting: {plotting_time:.2f}s ({plotting_time/total_time*100:.1f}%)")
    else:
        print(f"   Plotting: {plot1_time:.2f}s ({plot1_time/total_time*100:.1f}%)")
    print(f"   Statistics: {stats_time:.3f}s ({stats_time/total_time*100:.1f}%)")


def analyze_overfitting_risk(dataset: LeRobotDataset):
    """
    Analyze the risk of overfitting based on dataset characteristics.
    
    Args:
        dataset: A LeRobotDataset instance
    """
    print("⚠️  Analyzing overfitting risk...")
    
    episode_analysis = analyze_episodes(dataset)
    num_episodes = len(episode_analysis)
    
    # Check episode similarity
    if num_episodes > 1:
        action_correlations = []
        
        for i in range(num_episodes - 1):
            for j in range(i + 1, num_episodes):
                ep1_data = episode_analysis[i]
                ep2_data = episode_analysis[j]
                
                # Calculate correlation between episode means
                corr = np.corrcoef(ep1_data['action_mean'], ep2_data['action_mean'])[0, 1]
                action_correlations.append(corr)
        
        avg_correlation = np.mean(action_correlations)
        
        print(f"\n🎯 Overfitting Risk Assessment:")
        print(f"   Number of episodes: {num_episodes}")
        print(f"   Average action correlation between episodes: {avg_correlation:.3f}")
        
        if num_episodes < 5:
            print("   ⚠️  HIGH RISK: Very few episodes - model will likely overfit")
        elif avg_correlation > 0.9:
            print("   ⚠️  HIGH RISK: Episodes are very similar - limited diversity")
        elif avg_correlation > 0.7:
            print("   ⚠️  MEDIUM RISK: Episodes are somewhat similar")
        else:
            print("   ✅ LOW RISK: Good episode diversity")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if num_episodes < 10:
            print("   - Collect more demonstration episodes")
        if avg_correlation > 0.8:
            print("   - Add more diversity to demonstrations (different starting positions, speeds, etc.)")
        
        print("   - Use data augmentation to increase effective dataset size")
        print("   - Monitor validation loss during training")
        print("   - Consider early stopping to prevent overfitting")
    
    else:
        print("   ⚠️  CRITICAL RISK: Only one episode - model will definitely overfit!")
        print("   💡 Single episode training is only useful for debugging/testing")


def visualize_sample(dataset: LeRobotDataset, index: int):
    """
    Visualizes a single sample from a LeRobotDataset.

    Args:
        dataset: A LeRobotDataset instance.
        index: The index of the sample to visualize.
    """
    sample = dataset[index]

    # Visualize the observation image
    image_key = None
    for key in sample.keys():
        if "image" in key and isinstance(sample[key], torch.Tensor):
            image_key = key
            break

    if image_key:
        image = sample[image_key]
        if isinstance(image, torch.Tensor):
            # Convert to numpy and transpose if necessary
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = image.permute(1, 2, 0).numpy()
            else:
                image = image.numpy()

            plt.imshow(image)
            plt.title(f"Observation Image ({image_key}) at Index {index}")
            plt.show()

    # Print the action
    if "action" in sample:
        action = sample["action"]
        print(f"Action at Index {index}: {action}") 