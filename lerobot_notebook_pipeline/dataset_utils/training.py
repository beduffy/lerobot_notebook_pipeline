import torch
import time
from collections import deque

def train_model(policy, dataloader, optimizer, training_steps, log_freq, device):
    """
    Trains a policy for a specified number of steps.

    Args:
        policy: The policy to train.
        dataloader: The dataloader for the training data.
        optimizer: The optimizer for the policy.
        training_steps: The number of training steps.
        log_freq: The frequency at which to log training progress.
        device: The device to train on.
    """
    step = 0
    done = False
    start_time = time.time()
    loss_history = deque(maxlen=100)  # Keep track of recent losses
    best_loss = float('inf')

    print(f"🚀 Starting training for {training_steps} steps...")
    print(f"📊 Logging every {log_freq} steps")
    print("=" * 80)

    while not done:
        for batch in dataloader:
            # Move batch to device
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            # Forward pass
            loss, _ = policy.forward(inp_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Track loss
            current_loss = loss.item()
            loss_history.append(current_loss)
            if current_loss < best_loss:
                best_loss = current_loss

            # Logging and progress tracking
            if step % log_freq == 0:
                elapsed_time = time.time() - start_time
                steps_per_second = (step + 1) / elapsed_time if elapsed_time > 0 else 0
                remaining_steps = training_steps - step
                eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                eta_minutes = eta_seconds / 60
                
                # Calculate average loss over recent steps
                avg_recent_loss = sum(loss_history) / len(loss_history) if loss_history else current_loss
                
                print(f"🔥 Step {step:4d}/{training_steps} | "
                      f"Loss: {current_loss:.3f} | "
                      f"Avg: {avg_recent_loss:.3f} | "
                      f"Best: {best_loss:.3f} | "
                      f"Speed: {steps_per_second:.1f} steps/s | "
                      f"ETA: {eta_minutes:.1f}m")
                
                # Check for potential overfitting warning
                if step > 500 and len(loss_history) == 100:
                    recent_improvement = loss_history[0] - loss_history[-1]
                    if recent_improvement < 0.001:  # Very little improvement
                        print("⚠️  Warning: Loss plateaued - possible overfitting with single demo!")

            step += 1
            if step >= training_steps:
                done = True
                break

    total_time = time.time() - start_time
    print("=" * 80)
    print(f"✅ Training completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🎯 Final loss: {current_loss:.3f}")
    print(f"🏆 Best loss: {best_loss:.3f}")
    print(f"⚡ Average speed: {training_steps/total_time:.1f} steps/second") 