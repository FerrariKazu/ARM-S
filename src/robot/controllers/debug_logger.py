# ═══════════════════════════════════════════════════════
# FILE: src/robot/controllers/debug_logger.py
# PURPOSE: The system's 'Black Box' recorder and live terminal dashboard.
# LAYER: Monitoring Layer
# INPUTS: Real-time telemetry data (joint positions, speeds, intent scores).
# OUTPUTS: A scrolling terminal interface and saved performance data files (.npz).
# CALLED BY: SRL Main Loop, src/sim/srl_env.py
# ═══════════════════════════════════════════════════════

import os
import time
import numpy as np
from datetime import datetime

class SRLDebugLogger:
    """
    WHAT THIS CLASS IS:
      A "Flight Recorder". It captures every single movement and calculation 
      the robot makes and saves it for engineers to study later. It also 
      paints a simple text dashboard in the terminal window.

    WHY IT EXISTS:
      When a robot moves poorly, we need to look back at the exact 
      milliseconds to see if it was a sensor error or a math error. 
      This class provides the 'evidence' for debugging.

    HOW IT WORKS (step by step):
      1. Every frame, it accepts a package of data (The Telemetry).
      2. It stamps the data with the exact time it arrived.
      3. 10 times every second, it clears the screen and draws a 'HUD'.
      4. When the robot is turned off, it squashes all that data into a 
         tiny file on the hard drive.

    EXAMPLE USAGE:
      logger = SRLDebugLogger()
      logger.log_step({'task_mode': 'surgical', 'collision': False})
      logger.save_session()
    """
    def __init__(self):
        """
        WHAT THIS DOES: Initializes the memory banks and timers.
        """
        # A list to store thousands of frames of robot history
        self.session_data = []
        # Mark the exact moment the system was 'Booted'
        self.start_time = time.time()
        # Track the last time we updated the screen to prevent flickering
        self.last_print_time = 0.0
        # 0.1 seconds = 10 updates per second (10Hz)
        self.print_interval = 0.1 
        
    def log_step(self, data_dict: dict):
        """
        WHAT THIS DOES: Grabs the current 'pulse' of the robot and adds it to the archive.
        
        WHY: To ensure we have a continuous record of performance.
        
        ARGS:
          data_dict (dict): Keys represent sensor names, values represent readings.
            
        RETURNS:
          (None)
        """
        # Create a 'snapshot' of the data
        # We copy arrays to prevent them from changing while we are saving them
        step_data = {
            k: np.copy(v) if isinstance(v, np.ndarray) else v
            for k, v in data_dict.items()
        }
        # Add the 'Mission Time' stamp
        step_data['timestamp'] = time.time() - self.start_time
        # Push to the master list
        self.session_data.append(step_data)
        
        # TERMINAL ANIMATION LOGIC
        # Only redraw the screen if enough time has passed (10Hz limit)
        current_time = time.time()
        if (current_time - self.last_print_time) >= self.print_interval:
            self._print_hud(step_data)
            self.last_print_time = current_time

    def _print_hud(self, data: dict):
        """
        WHAT THIS DOES: Draws the 'Head-Up Display' in the terminal.
        
        WHY: So the operator can see the robot's logic without opening a heavy dashboard.
        
        ARGS:
          data (dict): The most recent snapshot of telemetry.
        
        MATH/LOGIC:
          - Uses ANSI Escape Codes (\033[2J) to 'teleport' the cursor to the top-left 
            instead of printing millions of new lines.
        """
        # Visual 'Wipe' command: Clears the terminal and resets cursor to Row 1, Col 1
        print("\033[2J\033[H", end="")
        
        # DRAWING THE INTERFACE BOX
        print("+" + "-"*50 + "+")
        print(f"| SRL DEBUG CONTROLLER HUD           T: {data.get('timestamp', 0):.2f}s |")
        print("+" + "-"*50 + "+")
        
        # Display the active mission type
        mode = data.get('task_mode', 'N/A')
        print(f"| MODE: {mode:<42} |")
        
        # Display the AI's current 'Feeling' or intention
        intent = data.get('intent_prediction', 'N/A')
        print(f"| INTENT: {str(intent):<40} |")
        
        # SAFETY WARNING: Highlight collisions in the text feed
        col_flag = data.get('collision_flags', False)
        col_str = "CAUTION!" if col_flag else "CLEAR"
        print(f"| COLLISION: {col_str:<37} |")
        
        print("+" + "-"*50 + "+")

    def save_session(self, path: str = None):
        """
        WHAT THIS DOES: Writes the entire memory bank to a permanent file.
        
        WHY: So data scientists can train new AI models using real robot movement logs.
        
        ARGS:
          path (str): Optional custom filename.
        """
        if path is None:
            # Create a 'Logs' folder if it doesn't already exist
            os.makedirs("training/logs", exist_ok=True)
            # Create a unique name based on the current date and time
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"training/logs/session_{stamp}.npz"
            
        print(f"Saving {len(self.session_data)} ticks to {path}...")
        
        # DATA COMPRESSION
        # Turn the long list of dictionaries into a single 'Block' for each sensor type
        collated = {}
        if len(self.session_data) > 0:
            keys = self.session_data[0].keys()
            for k in keys:
                # Efficiently stack the data into high-speed numpy arrays
                collated[k] = np.array([step[k] for step in self.session_data])
                
        # Write to disk using high-speed compression (.npz mode)
        np.savez_compressed(path, **collated)
        print("Save complete.")

    def replay_session(self, path: str):
        """
        WHAT THIS DOES: Opens a previously saved mission. (Feature Placeholder)
        """
        data = np.load(path, allow_pickle=True)
        print(f"Loaded replay with {len(data['timestamp'])} frames.")
        return data
        
    def plot_session_summary(self, path: str):
        """
        WHAT THIS DOES: Generates a graph of the mission performance. (Feature Placeholder)
        """
        print(f"Generated plot at {path}")
