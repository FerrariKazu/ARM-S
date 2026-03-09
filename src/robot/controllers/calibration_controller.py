# ═══════════════════════════════════════════════════════
# FILE: src/robot/controllers/calibration_controller.py
# PURPOSE: Personalizes the robot's 'Bio-Interface' to the unique muscle signature of each human.
# LAYER: Sensing Layer
# INPUTS: Raw electricity signals from human muscles (EMG).
# OUTPUTS: A mathematical 'User Profile' that ensures the robot understands subtle vs strong movements.
# CALLED BY: SRL Setup, src/sim/srl_env.py
# ═══════════════════════════════════════════════════════

import json
import os
import time
import numpy as np

class EMGCalibrationController:
    """
    WHAT THIS CLASS IS:
      The "Signal Translator". Every human has different muscle strengths; 
      one person's "weak nudge" might look like another's "strong pull". 
      This class runs a 60-second test to learn your unique body language.

    WHY IT EXISTS:
      Without calibration, the robot arms would be too sensitive for athletes 
      and too stiff for people with average muscle strength. This class 
      ensures the robot is 'Tuned' perfectly to the current operator.

    HOW IT WORKS (step by step):
      1. REST PHASE: Measures your muscle 'Noise' while you are totally relaxed.
      2. ACTIVE PHASE: Asks you to flex as hard as possible to find your 'Maximum Power'.
      3. MATH: Creates a scale where (Noise = 0%) and (Max = 100%).
      4. MEMORY: Saves this 'User Profile' to a file so you only have to calibrate once.

    EXAMPLE USAGE:
      calibrator = EMGCalibrationController()
      calibrator.run_calibration(sensor_simulator)
      calibrator.save_profile("user_01")
    """
    def __init__(self):
        """
        WHAT THIS DOES: Sets the default 'Neutral' state for new users.
        """
        # The electricity level when the user is doing absolutely nothing
        self.noise_floor = np.zeros(16)
        # The peak electricity level reached during a maximum effort flex
        self.signal_peak = np.ones(16)
        # Truth flag to prevent normalization before we know the user's limits
        self.is_calibrated = False

    def run_calibration(self, emg_simulator, duration: int = 60):
        """
        WHAT THIS DOES: Guided 2-part test to build the user's Profile.
        
        WHY: To establish the 'Mathematical Floor' and 'Ceiling' of user intent.
        
        ARGS:
          emg_simulator (obj): The hardware or software sensor feed.
          duration (int): Total testing time in seconds.
        """
        print(f"Starting EMG calibration procedure ({duration}s)...")
        # Split time equally between Relaxing and Flexing
        rest_duration = duration // 2
        active_duration = duration - rest_duration
        
        # --- PHASE 1: THE FLOOR ---
        print(f"Phase 1: REST ({rest_duration}s). Please relax all muscles.")
        time.sleep(0.5) # Allow user a moment to get comfortable
        # Collect samples at 50Hz (50 frames per second)
        raw_rest = emg_simulator.get_raw_sensor_data(size=rest_duration * 50)
        # Calculate average noise and add a 50% buffer to prevent accidental arm movement
        self.noise_floor = np.mean(np.abs(raw_rest), axis=0) * 1.5 
        
        # --- PHASE 2: THE CEILING ---
        print(f"Phase 2: ACTIVE ({active_duration}s). Perform maximum contractions.")
        time.sleep(0.5)
        raw_active = emg_simulator.get_raw_sensor_data(size=active_duration * 50)
        # Boost active signal in simulation to simulate adrenaline/high-effort spikes
        raw_active *= 3.0 
        # Record the single highest pulse reached during the flex
        self.signal_peak = np.max(np.abs(raw_active), axis=0)
        
        # SAFETY CHECK: Ensure peak is always higher than noise floor to avoid divide-by-zero
        self.signal_peak = np.maximum(self.signal_peak, self.noise_floor + 1e-4)
        
        self.is_calibrated = True
        print("Calibration completed successfully.")

    def normalize(self, raw_emg: np.ndarray) -> np.ndarray:
        """
        WHAT THIS DOES: Converts raw sensor numbers into a percentage (0% to 100%).
        
        WHY: The Intelligence layer (AI) only understands percentages, not raw voltages.
        
        ARGS:
          raw_emg (np.ndarray): raw data coming from the skin sensors.
            
        RETURNS:
          (np.ndarray): Cleaned signal where 0=Relaxed and 1=Maximum Flex.
        
        MATH:
          - (Value - Floor) / (Ceiling - Floor)
        """
        # If we haven't calibrated, return the raw data as a safety fallback
        if not self.is_calibrated:
            return raw_emg
            
        # 1. Take absolute value (EMG signals oscillate positive/negative)
        abs_emg = np.abs(raw_emg)
        # 2. Rescale between the learned Floor and learned Peak
        norm_emg = (abs_emg - self.noise_floor) / (self.signal_peak - self.noise_floor)
        # 3. Clamp results between 0 and 1 so errors don't confuse the AI
        return np.clip(norm_emg, 0.0, 1.0)

    def save_profile(self, user_id: str, path: str = None):
        """
        WHAT THIS DOES: Writes the user's settings to a permanent JSON file.
        
        WHY: So users don't have to spend 60 seconds recalibrating every single day.
        """
        if path is None:
            # Create a standard directory for biometric profiles
            os.makedirs("config/user_profiles", exist_ok=True)
            path = f"config/user_profiles/{user_id}.json"
            
        data = {
            "user_id": user_id,
            "noise_floor": self.noise_floor.tolist(), # convert to list for JSON compatibility
            "signal_peak": self.signal_peak.tolist()
        }
        
        # Save as a human-readable text file
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved calibration profile to {path}")

    def load_profile(self, user_id: str, path: str = None):
        """
        WHAT THIS DOES: Reads a user's settings from the hard drive.
        """
        if path is None:
            path = f"config/user_profiles/{user_id}.json"
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Convert JSON lists back into high-speed math (numpy) arrays
        self.noise_floor = np.array(data["noise_floor"])
        self.signal_peak = np.array(data["signal_peak"])
        # Activate the normalization pipeline
        self.is_calibrated = True
        print(f"Loaded calibration profile from {path}")
