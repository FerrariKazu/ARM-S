# ═══════════════════════════════════════════════════════
# FILE: src/robot/controllers/speed_controller.py
# PURPOSE: Governor that enforces maximum speed limits for arm joints depending on the task.
# LAYER: Execution Layer (Motion Context)
# INPUTS: Requested task mode (e.g. 'surgical') and planned movement trajectories.
# OUTPUTS: Speed-capped trajectory paths and smoothed velocity transitions.
# CALLED BY: SRL Motion Planner, src/sim/srl_env.py
# ═══════════════════════════════════════════════════════

import numpy as np

class SpeedController:
    """
    WHAT THIS CLASS IS:
      The robot's "Speed Governor". It decides how fast the arms are allowed 
      to move based on the environment and the specific job being done.

    WHY IT EXISTS:
      Moving at 100% speed while performing surgery is dangerous, but 
      moving at 10% speed during an emergency is ineffective. This class 
      ensures the robot's physical performance matches the task's safety needs.

    HOW IT WORKS (step by step):
      1. Maintains a lookup table of "Speed Envelopes" for different activities.
      2. Takes a planned path and "stretches" it out so it never exceeds the limit.
      3. Smooths out transitions so that switching from "Slow" to "Fast" doesn't 
         cause the robot arms to jerk or whip.

    EXAMPLE USAGE:
      governor = SpeedController()
      safe_path = governor.scale_trajectory(raw_path, mode="surgical")
    """
    def __init__(self):
        """
        WHAT THIS DOES: Defines the maximum velocity allowed for each task type.
        
        WHY: To establish 'Legal Limits' that the software layers cannot override.
        """
        # Dictionary mapping the descriptive task name to physical velocity (radians/sec)
        self.mode_limits = {
            "surgical": 0.3,   # Precision Mode: slow, steady, and high-impedance
            "carry": 0.8,      # Standard Mode: everyday assistance
            "emergency": 2.0   # Reactive Mode: high-torque, high-speed interception
        }
        # Fallback speed if the system is in an undefined state
        self.default_limit = 1.0

    def get_velocity_limit(self, task_mode: str) -> float:
        """
        WHAT THIS DOES: Looking up the speed cap for a specific mode.
        
        WHY: Used as a reference for other controllers that need to know 
             the current machine constraints.
             
        ARGS:
          task_mode (str): The label of the current operational state.
          
        RETURNS:
          (float): Max allowed radians per second.
        """
        # Convert to lowercase to prevent typos from causing speed-limit bypasses
        return self.mode_limits.get(task_mode.lower(), self.default_limit)

    def scale_trajectory(self, traj: np.ndarray, mode: str) -> np.ndarray:
        """
        WHAT THIS DOES: Downscales a fast path to fit inside the legal speed envelope.
        
        WHY: Prevents the motion planner from outputting 'impossible' or 'unsafe' commands.
        
        ARGS:
          traj (np.ndarray): Array of planned motor speeds.
          mode (str): The current task name.
            
        RETURNS:
          (np.ndarray): The resized path that matches the speed cap.
        
        MATH:
          - We calculate the "Overspeed Ratio" (Limit / Actual Speed).
          - If Ratio < 1.0, we multiply the whole trajectory by that ratio.
        """
        limit = self.get_velocity_limit(mode)
        # Find the peak speed in the requested movement
        magnitudes = np.linalg.norm(traj, axis=-1, keepdims=True)
        
        # Zero-check to avoid 'Infinity' errors in stationary movements
        magnitudes[magnitudes == 0] = 1e-6
        
        # Calculate the percentage we need to 'throttle' the speeds
        # .clip(0, 1) ensures we slow down fast paths, but NEVER speed up slow paths.
        scale_factors = np.clip(limit / magnitudes, 0.0, 1.0)
        
        # Apply the governor to every joint in the sequence
        return traj * scale_factors

    def smooth_acceleration(self, current_vel: float, target_vel: float, dt: float) -> float:
        """
        WHAT THIS DOES: Gently ramps speed up or down over time.
        
        WHY: Abruptly jumping from 0 to 2.0 rad/s would snap the robot's transmission.
        
        ARGS:
          current_vel (float): Real-world speed right now.
          target_vel (float): Speed requested by the AI.
          dt (float): Time elapsed.
            
        RETURNS:
          (float): The next speed step allowed by the motor's acceleration limits.
        
        MATH/LOGIC:
          - Uses a physics-based exponential approach to approach the target.
          - Clamps the result by a Hard Max Acceleration constant (5.0 rad/s^2).
        """
        if dt <= 0:
            return current_vel
            
        difference = target_vel - current_vel
        # Max acceleration constraint to protect the physical motors
        max_accel = 5.0
        
        # Use an exponential ramp formula to close the gap smoothly
        # 10.0 constant determines the 'snappiness' of the response
        step = difference * (1.0 - np.exp(-10.0 * dt)) 
        
        # HARD PHYSICAL CONSTRAINT
        # Ensure the jump in speed is never greater than what physics allows in 'dt' time
        max_step = max_accel * dt
        step = np.clip(step, -max_step, max_step)
        
        return current_vel + step
