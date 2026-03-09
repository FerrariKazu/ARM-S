# ═══════════════════════════════════════════════════════
# FILE: training/generate_dataset.py
# PURPOSE: The "Data Factory"; automatically runs the simulation for hours to record AI training data.
# LAYER: Data Pipeline Layer
# INPUTS: High-level motion policies (how to walk, reach, lift).
# OUTPUTS: Compressed training files (.npz) split into Train/Validation/Test groups.
# CALLED BY: Manual developer execution (e.g. ./generate_data.sh)
# ═══════════════════════════════════════════════════════

#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
from typing import Dict, List, Any

# Ensure we can import from the 'src' directory relative to this file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sim.srl_env import SRLEnv
from src.sim.human_motion_policies import CarryPolicy, ReachPolicy, AssemblyPolicy, OverheadPolicy
from src.perception.emg_simulator import EMGSimulator

def main():
    """
    WHAT THIS DOES: Coordinates a mass-scale recording session of the robot.
    
    WHY: AI models are 'Hungry' for data. To teach the arm how to react, 
         we must show it 10,000+ examples of different human movements.
         
    HOW IT WORKS (step by step):
      1. Initializes the Physics Engine (MuJoCo) and the Bio-Sensor Sim.
      2. Loops through 4 different human behaviors (e.g. 'Carrying', 'Assembly').
      3. For each behavior, it takes 2,500 snapshots of muscle and body data.
      4. Compresses all 10,000 snapshots into a single file for the AI to study.
    """
    print("--- ARM-S Dataset Generator v1 ---")
    
    # 1. SETUP: Prepare the laboratory environment
    # render_mode=None makes the simulation run 10x faster because we don't draw any pictures
    env = SRLEnv(render_mode=None)
    obs, info = env.reset()
    # Connect the Bio-Sim to the physical world
    emg_sim = EMGSimulator(env.data)
    
    # Define the 4 'Storylines' we want to record
    policies = {
        "carry": CarryPolicy(),
        "reach": ReachPolicy(),
        "assembly": AssemblyPolicy(),
        "overhead": OverheadPolicy()
    }
    
    # Map text labels to computer numbers (IDs 0-6)
    label_map = {
        "reach_left": 0, "reach_right": 1, "lift": 2, "hold": 3,
        "release": 4, "stabilize": 5, "handoff": 6
    }
    
    # SETTINGS: 2500 samples per policy = 10,000 total training examples
    samples_per_policy = 2500
    total_samples = samples_per_policy * len(policies)
    
    # ALLOCATE MEMORY: Pre-create empty boxes to store our sensor results
    emg_windows = np.zeros((total_samples, 25, 16), dtype=np.float32) # Muscle waves
    body_states = np.zeros((total_samples, 24), dtype=np.float32)    # Spine/Wrist angles
    joint_states = np.zeros((total_samples, 14), dtype=np.float32)   # Robot motor states
    intent_labels = np.zeros(total_samples, dtype=np.int32)          # The 'Correct Answer'
    times_to_action = np.zeros(total_samples, dtype=np.float32)      # Future-look timing
    
    current_idx = 0
    start_time = time.time()
    
    # 2. GENERATION LOOP: The "Recording Session"
    for p_name, policy in policies.items():
        print(f"Collecting data for policy: {p_name}...")
        env.reset()
        t_sim = 0.0 # Virtual clock for this specific recording
        
        for step in range(samples_per_policy):
            # A. CONSULT THE SCRIPT: Ask the movement policy where the human moves next
            policy_out = policy.step(t_sim)
            targets = policy_out["joint_targets"]
            
            # B. EXECUTE PHYSICS: Map the human targets to the simulator
            action = np.zeros(14)
            # Normalize angles to a [-1, 1] range for the neural net
            action[:12] = np.clip(targets / 3.14, -1, 1) 
            
            # Run one millisecond of physics
            obs, reward, _, _, info = env.step(action)
            # Generate the biological 'Sparks' (EMG) based on that movement
            emg_sim.update()
            
            # 3. EXTRACTION: The "Snapshot" phase
            # Save the last 500ms of muscle activity
            emg_windows[current_idx] = emg_sim.get_window()
            
            # Save the human's physical posture (Spine, Wrists, IMU balance)
            state_vec = np.zeros(24, dtype=np.float32)
            state_vec[0:7] = env.data.qpos[0:7] # Trunk position
            # Note: Wrist poses are currently mocked with small noise (future sensor integration)
            state_vec[7:14] = np.random.normal(0, 0.01, 7) 
            state_vec[14:21] = np.random.normal(0, 0.01, 7) 
            state_vec[21] = t_sim
            state_vec[22:24] = env.data.sensordata[0:2] # Simulated Vestibular (Balance) data
            body_states[current_idx] = state_vec
            
            # Save the state of the robot arm itself
            joint_states[current_idx] = np.concatenate([env.data.qpos[7:19], [0, 0]])
            
            # Save the 'Ground Truth' labels for future AI training
            intent_labels[current_idx] = label_map[policy_out["intent_label"]]
            times_to_action[current_idx] = policy_out["time_to_next_action"]
            
            # Update the clock (50 captures per virtual second)
            t_sim += (1.0 / 50.0) 
            current_idx += 1
            
    total_time = time.time() - start_time
    print(f"Data collection complete in {total_time:.1f}s.")
    
    # 4. STORAGE: Organizing the massive library of results
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, "srl_dataset_v1.npz")
    # Squash the data to save 50%+ disk space
    np.savez_compressed(full_path, 
                        emg=emg_windows, state=body_states, joints=joint_states,
                        labels=intent_labels, time_to_action=times_to_action)
    
    # DATA SHUFFLING
    # Randomly shuffle all 10,000 examples so the AI doesn't learn 'The Order'
    indices = np.random.permutation(total_samples)
    train_end = int(0.8 * total_samples) # 80% for study
    val_end = int(0.9 * total_samples)   # 10% for the 'Pop Quiz'
    
    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:] # 10% for the 'Final Exam'
    }
    
    # Create the 3 separate files for training, validation, and testing
    for split_name, split_idx in splits.items():
        path = os.path.join(output_dir, f"srl_dataset_{split_name}.npz")
        np.savez_compressed(path,
                            emg=emg_windows[split_idx],
                            state=body_states[split_idx],
                            joints=joint_states[split_idx],
                            labels=intent_labels[split_idx],
                            time_to_action=times_to_action[split_idx])

    # 5. REPORT CARD: Print a summary for the human developer
    print("\n--- Dataset Summary ---")
    print(f"Total Samples: {total_samples}")
    
    print("\nClass Distribution (Balance Check):")
    rev_map = {v: k for k, v in label_map.items()}
    for label_id in range(7):
        count = np.sum(intent_labels == label_id)
        percent = (count / total_samples) * 100
        print(f"  {rev_map[label_id]:<15}: {count:>5} ({percent:>5.1f}%)")
        
    print("\nFeature Statistics (Mean / Std):")
    print(f"  EMG Signals: {np.mean(emg_windows):.3f} / {np.std(emg_windows):.3f}")
    print(f"  Body States: {np.mean(body_states):.3f} / {np.std(body_states):.3f}")
    
    # Provide a rough estimate for how long the AI model will take to 'Learn' this data
    est_time_min = (15000 * 0.02) / 60.0
    print(f"\nEst. Intent Transformer training time on RTX 4060: ~{est_time_min:.1f} minutes")
    print(f"Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
