# ═══════════════════════════════════════════════════════
# FILE: verify_install.py
# PURPOSE: The "Doctor's Checkup"; ensures the simulator computer is healthy and ready to run.
# LAYER: Infrastructure / Setup Layer
# INPUTS: Installed software libraries (MuJoCo, Torch, Python).
# OUTPUTS: A system health report and a success confirmation if physics is working.
# CALLED BY: Manual execution after initial installation or updates.
# ═══════════════════════════════════════════════════════

#!/usr/bin/env python3
import sys
import time
import os
import mujoco
import dm_control
import gymnasium
import torch
import transforms3d

def main():
    """
    WHAT THIS DOES: Audits the robot's 'Operating System' and hardware.
    
    WHY: Robotics requires complex dependencies. If one library (like CUDA) 
         is missing, the entire simulation will fail or run extremely slow.
         
    HOW IT WORKS (step by step):
      1. Prints out the exact versions of every major library installed.
      2. Checks if a powerful Graphics Card (GPU) is available for AI math.
      3. Loads a basic 'Humanoid' skeleton to verify MuJoCo is working.
      4. Runs 100 fast-forward steps of physics to measure CPU speed.
    """
    try:
        # 1. VERSION AUDIT: Print the library 'Birth Certificates'
        print(f"--- Software Versions ---")
        print(f"Python: {sys.version.split()[0]}")
        print(f"MuJoCo: {mujoco.__version__}")
        # dm_control helps with smooth kinematics math
        print(f"dm_control: Detected (Path: {os.path.dirname(dm_control.__file__)})")
        print(f"Gymnasium: {gymnasium.__version__}")
        print(f"Torch: {torch.__version__}")
        print(f"transforms3d: {transforms3d.__version__}")
        print(f"--------------------------")

        # 2. AI ENGINE CHECK: Verify the 'Jet Engine' (GPU)
        if not torch.cuda.is_available():
            print("ERROR: torch.cuda.is_available() is False. GPU acceleration won't work.", file=sys.stderr)
            sys.exit(1)
        
        gpu_name = torch.cuda.get_device_name(0)
        print(f"SUCCESS: CUDA is available. GPU: {gpu_name}")

        # 3. PHYSICS SMOKE TEST: Load a virtual laboratory
        mujoco_dir = os.path.dirname(mujoco.__file__)
        potential_model_paths = [
            os.path.join(mujoco_dir, "models", "humanoid.xml"),
            os.path.join(mujoco_dir, "sample", "humanoid.xml"),
            "humanoid.xml" 
        ]
        
        model = None
        for path in potential_model_paths:
            if os.path.exists(path):
                # Try to load a real human skeleton model
                model = mujoco.MjModel.from_xml_path(path)
                print(f"Loaded MuJoCo model from: {path}")
                break
        
        if model is None:
            # SAFETY FALLBACK: Use a simple Red Box if no human skeleton is found
            print("WARNING: humanoid.xml not found. Using basic test XML.")
            base_xml = """
            <mujoco model="test_sim">
                <worldbody>
                    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
                    <body name="box" pos="0 0 1">
                        <freejoint name="root"/>
                        <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
                    </body>
                </worldbody>
            </mujoco>
            """
            model = mujoco.MjModel.from_xml_string(base_xml)

        # Create the simulation workspace
        data = mujoco.MjData(model)
        
        # 4. BENCHMARK: Stress test the math
        n_steps = 100
        print(f"Running {n_steps} simulation steps...")
        
        start_time = time.perf_counter()
        for _ in range(n_steps):
            # Advance the clock by 1 tick
            mujoco.mj_step(model, data)
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / n_steps
        
        # 5. VERDICT: Report findings
        print(f"Simulation SUCCESS.")
        # Robot needs < 1ms per step for real-time operation
        print(f"Average step time: {avg_time_ms:.4f} ms")
        
        sys.exit(0)

    except Exception as e:
        print(f"CRITICAL FAILURE: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
