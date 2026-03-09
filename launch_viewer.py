# ═══════════════════════════════════════════════════════
# FILE: launch_viewer.py
# PURPOSE: The "Simulation Cinema"; opens a 3D window to watch the robot move in real-time.
# LAYER: Simulation Layer (Visualizer)
# INPUTS: Choosing a 'Human Behavior' script (e.g. Carry or Reach) via terminal command.
# OUTPUTS: A high-fidelity 3D MuJoCo window and optional frame-by-frame photo recordings.
# CALLED BY: Manual developer execution (e.g. python launch_viewer.py --policy reach)
# ═══════════════════════════════════════════════════════

#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import cv2

# Ensure we can import from the 'src' directory relative to this file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.sim.human_motion_policies import CarryPolicy, ReachPolicy, AssemblyPolicy, OverheadPolicy
from src.perception.emg_simulator import EMGSimulator

def get_args():
    """
    WHAT THIS DOES: Sets up the 'Command Panel' for the simulation.
    
    WHY: So engineers can toggle recording or switch policies without editing code.
    """
    parser = argparse.ArgumentParser(description="SRL Robot Policy Viewer")
    # Tweakable settings for the human's goal
    parser.add_argument("--policy", type=str, default="reach", 
                        choices=["carry", "reach", "assembly", "overhead"],
                        help="Which human behavior script to run")
    # Tweakable settings for data capture
    parser.add_argument("--record", action="store_true", help="Record every frame as a photo")
    parser.add_argument("--fast", action="store_true", help="Lower resolution to save disk space")
    parser.add_argument("--no-contacts", action="store_true", help="Disable collision arrows")
    return parser.parse_args()

def main():
    """
    WHAT THIS DOES: The "Director's Chair"; orchestrates the 3D world creation and playback.
    
    WHY: This is the main entry point for a human to 'SEE' the robot's logic in action.
    
    HOW IT WORKS (step by step):
      1. Loads the 3D world model (MJCF XML).
      2. Snapshots the human behaviors and the bio-sensor sim.
      3. Launches the MuJoCo viewer window.
      4. Executes a repeating 50Hz loop: Read intent -> Move motors -> Step physics -> Update Screen.
    """
    args = get_args()
    
    # 1. WORLD SETUP: Load the digital physics lab
    model_path = os.path.join(os.path.dirname(__file__), "models/mjcf/srl_robot.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Reset everything to the 'Home' pose (Keyframe 0)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 2. STORY SETUP: Pick the human personality
    policies = {
        "carry": CarryPolicy(),
        "reach": ReachPolicy(),
        "assembly": AssemblyPolicy(),
        "overhead": OverheadPolicy()
    }
    policy = policies[args.policy]
    # Attach a digital 'Muscle Sensor' to the human
    emg_sim = EMGSimulator(data)
    
    # 3. CAMERA SETUP: Configure the recording equipment
    renderer = None
    if args.record:
        renderer = mujoco.Renderer(model)
        record_dir = os.path.join(os.path.dirname(__file__), "training/data/recordings", f"{args.policy}_{int(time.time())}")
        os.makedirs(record_dir, exist_ok=True)
        print(f"Recording enabled. Saving to: {record_dir}")

    # Identify the 'Claws' (Hands) for position tracking
    r_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_gripper")
    l_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_gripper")

    print(f"Launching MuJoCo viewer for policy: {args.policy}")
    
    # GRAPHICS OPTIMIZATION: Disable heavy visual effects to keep simulation at 60FPS
    model.vis.quality.shadowsize = 0  
    model.vis.map.shadowclip = 0
    model.vis.quality.offsamples = 0
    model.vis.quality.numslices = 16  
    model.vis.quality.numstacks = 16
    model.vis.map.fogstart = 100      
    model.vis.map.fogend = 200

    if args.fast and args.record:
        model.vis.global_.offwidth = 640
        model.vis.global_.offheight = 480

    # Subtly tint the floor so the robot segments pop visually
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id != -1:
        model.geom_rgba[floor_id] = [0.1, 0.1, 0.1, 0.4] 
    
    # THE MAIN STAGE: Launch the interactive window
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        
        # Enable collision visualization (Red arrows for contact forces)
        if not args.no_contacts:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        
        # Position the 'Director's Camera' for a cinematic side view
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.2
        viewer.cam.lookat = [0, 0, 1.1] # Focus on human shoulder height
        
        start_time = time.time()
        frame_count = 0
        
        # 4. ACTION LOOP: The core simulation sequence
        while viewer.is_running():
            loop_start = time.perf_counter()
            t_sim = time.time() - start_time
            
            # --- THINK ---
            # Ask the motion script where the human should be right now
            policy_out = policy.step(t_sim)
            targets = policy_out["joint_targets"]
            
            # --- ACT ---
            # Send the goal angles to the physical motor actuators
            data.ctrl[:16] = targets
            
            # --- PHYSICS ---
            # Let the MuJoCo engine calculate the next 1 millisecond of time
            mujoco.mj_step(model, data)
            # Update the bio-sensor 'Sparks'
            emg_out = emg_sim.update()
            
            # --- DASHBOARD: Terminal HUD ---
            # Clear terminal for a smooth 'HUD' update
            os.system('clear')
            print(f"--- ARM-S VIEWER | Policy: {args.policy.upper()} ---")
            print(f"Intent: {policy_out['intent_label'].upper()}")
            print(f"Next Action: {policy_out['time_to_next_action']:.2f}s")
            
            # COORDINATE TRACKING: Print exactly where the hands are in space
            r_pos = data.site_xpos[r_ee_id]
            l_pos = data.site_xpos[l_ee_id]
            
            torso_pos = data.xpos[1] # Reference coordinate for the human's base
            # Calculate how far the arms are reaching (as % of max length)
            r_ext = np.linalg.norm(r_pos - torso_pos) / 0.75 * 100
            l_ext = np.linalg.norm(l_pos - torso_pos) / 0.75 * 100
            
            gripper_state = "OPEN" if "release" in policy_out['intent_label'] else "CLOSED"
            
            print(f"R-Gripper: [{r_pos[0]:.2f}, {r_pos[1]:.2f}, {r_pos[2]:.2f}] | Ext: {min(100, r_ext):.1f}%")
            print(f"L-Gripper: [{l_pos[0]:.2f}, {l_pos[1]:.2f}, {l_pos[2]:.2f}] | Ext: {min(100, l_ext):.1f}%")
            print(f"Segments: 8 per arm | Grippers: {gripper_state}")
            
            # CAMERA MOTION: Slowly orbit the scene for a 360 effect
            viewer.cam.azimuth += 0.3 / 200 * 50 
            
            # ASCII VISUALIZATION: Draw a bar-chart in text for muscle activity
            print("\nEMG ACTIVITY (Live Pulse):")
            emg_norm = np.clip(emg_out / 5.0, 0, 1)
            bar_len = 20
            for i in range(8):
                # Calculate bar filling for Right and Left channels
                r_fill = int(emg_norm[i] * bar_len)
                l_fill = int(emg_norm[i+8] * bar_len)
                r_bar = "█" * r_fill + "-" * (bar_len - r_fill)
                l_bar = "█" * l_fill + "-" * (bar_len - l_fill)
                print(f"CH{i:02d}(R): {r_bar} | CH{i+8:02d}(L): {l_bar}")
            
            # Update the 3D window
            viewer.sync()
            
            # --- PHOTOGRAPHY: Optional frame recording ---
            if args.record:
                renderer.update_scene(data)
                frame = renderer.render()
                frame_path = os.path.join(record_dir, f"frame_{frame_count:05d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1
            
            # STABILITY: Hold the simulation at exactly 50 logic steps per second
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, 0.02 - elapsed))

if __name__ == "__main__":
    main()
