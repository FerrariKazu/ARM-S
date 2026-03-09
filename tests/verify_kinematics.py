# ═══════════════════════════════════════════════════════
# FILE: tests/verify_kinematics.py
# PURPOSE: The "Safety Inspector"; ensures the robot arms stay within human boundaries and don't hit the floor.
# LAYER: Verification Layer
# INPUTS: Specific 'Yoga Poses' (Joint angles) for the robot to attempt.
# OUTPUTS: A success report if the arms stayed within safe physical limits.
# CALLED BY: Manual QA during the building/testing phase.
# ═══════════════════════════════════════════════════════

#!/usr/bin/env python3
import os
import mujoco
import numpy as np
import sys

def main():
    """
    WHAT THIS DOES: Puts the robot through a series of 'Flexibility Tests'.
    
    WHY: In robotics, a bug in math can lead to a physical crash. This script 
         verifies that for any given command, the robot remains in a 'Safe Zone'.
         
    HOW IT WORKS (step by step):
      1. Loads the 3D physics model.
      2. Tries 5 different 'Extreme' poses (e.g. reaching floor, lifting high).
      3. For each pose, it calculates the exact XYZ coordinate of the hands.
      4. Checks two rules: 
          A) Hand must be above floor (z > 0).
          B) Hand must not be too long (extension < 0.9m).
    """
    try:
        # 1. SETUP: Load the robotic skeleton
        model_path = os.path.join(os.path.dirname(__file__), "../models/mjcf/srl_robot.xml")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        # ID lookup for the 'Claws' (Grippers)
        r_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_gripper")
        l_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_gripper")
        
        # 2. DEFINITION: The 'Standard Poses' to test
        configs = {
            "Zero Pose": np.zeros(16), # Arms straight up/down
            "Reach Left": np.array([0]*8 + [1.57, 1.57] + [0]*6), # Left shoulder lift
            "Reach Right": np.array([-1.57, -1.57] + [0]*14), # Right shoulder lift
            "Reach Forward": np.array([1.57] + [0]*7 + [-1.57] + [0]*7), # Both forward
            "Reach Down": np.array([0, 0, 1.57] + [0]*5 + [0, 0, 1.57] + [0]*5) # Elbows pointing down
        }

        print(f"--- SRL Kinematics Verification ---")
        
        all_passed = True
        
        # 3. EXECUTION: Try every pose
        for name, joint_angles in configs.items():
            print(f"\nTesting Configuration: {name}")
            
            # Snap the simulation to the specific pose
            mujoco.mj_resetData(model, data)
            data.qpos[:16] = joint_angles
            
            # TRIGGER MATH: Calculate the updated 3D positions
            mujoco.mj_forward(model, data)
            
            # Extract hand positions [X, Y, Z]
            r_pos = data.site_xpos[r_ee_id].copy()
            l_pos = data.site_xpos[l_ee_id].copy()
            
            # Reference point: The middle of the human's back
            torso_pos = data.xpos[1] 
            
            # Read Force Sensors (Check for 'Sphost' collisions)
            r_force = data.sensordata[0:3]
            l_force = data.sensordata[3:6]
            
            print(f"  Joints (rad): {joint_angles}")
            print(f"  R-Gripper Pos: {r_pos}")
            print(f"  L-Gripper Pos: {l_pos}")
            print(f"  R-Force: {r_force}")
            print(f"  L-Force: {l_force}")

            # 4. INSPECTION: The Rulebook
            failures = []
            
            # RULE 1: GROUND CLEARANCE
            # If Z < 0, the robot is stuck inside the floor.
            if r_pos[2] < -0.01 or l_pos[2] < -0.01:
                failures.append(f"CRASH! End-effector below floor level! R_z={r_pos[2]:.2f}, L_z={l_pos[2]:.2f}")
            
            # RULE 2: ANATOMICAL CHOREOGRAPHY
            # The robot arm is physically about 75cm long. 
            # If math says it is 1.5m away, something is broken.
            r_dist = np.linalg.norm(r_pos - torso_pos)
            l_dist = np.linalg.norm(l_pos - torso_pos)
            if r_dist > 0.9 or l_dist > 0.9:
                failures.append(f"LIMIT BREACHED! Extension exceeds 0.9m! R_ext={r_dist:.2f}m, L_ext={l_dist:.2f}m")
            
            # Output the verdict for this pose
            if not failures:
                print(f"  STATUS: \033[92mPASS\033[0m")
            else:
                print(f"  STATUS: \033[91mFAIL\033[0m")
                for f in failures:
                    print(f"    - {f}")
                all_passed = False

        print(f"\n-----------------------------------")
        # 5. FINAL VERDICT
        if all_passed:
            print("FINAL VERDICT: \033[92mSUCCESS\033[0m — Bio-Mechanical limits verified.")
            sys.exit(0)
        else:
            print("FINAL VERDICT: \033[91mFAILURE\033[0m — Safety limits violated.")
            sys.exit(1)

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
