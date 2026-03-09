# ═══════════════════════════════════════════════════════
# FILE: src/robot/controllers/pid_controller.py
# PURPOSE: Precision feedback loops to ensure smooth, organic movement in robot joints.
# LAYER: Execution Layer
# INPUTS: Position error (target minus current angle) and time delta.
# OUTPUTS: Corrective torque/velocity signal clamped to safe motor limits.
# CALLED BY: SRL Control Loop, src/sim/srl_env.py
# ═══════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import os

class PIDController:
    """
    WHAT THIS CLASS IS:
      A "Smart Cruise Control" for robot joints. It adjusts motor power 
      automatically to reach a target position without overshooting or vibrating.

    WHY IT EXISTS:
      Raw motor signals are twitchy and dangerous. This PID controller 
      mathematically filters the commands so the physical arms move with 
      human-like smoothness.

    HOW IT WORKS (step by step):
      1. Proportional (P): Pushes harder the further away it is from the goal.
      2. Integral (I): Adds up small errors over time to overcome physical friction.
      3. Derivative (D): Acts like a shock absorber, slowing down as it reaches the goal.
      4. Clamping: Ensures the math never accidentally asks the motor to exceed its power.

    EXAMPLE USAGE:
      # Create a controller for Joint 1
      j1_ctrl = PIDController(kp=5.0, ki=0.1, kd=0.5)
      
      # In the control loop
      torque = j1_ctrl.compute(target_angle - current_angle, time_delta)
    """
    def __init__(self, kp: float, ki: float, kd: float, output_limits: tuple = (-50.0, 50.0), integral_limits: tuple = (-5.0, 5.0)):
        """
        WHAT THIS DOES: Configures the 'personality' of the controller.
        
        WHY: Different joints (shoulders vs wrists) need different levels of stiffness (P) and damping (D).
        
        ARGS:
          kp (float): Stiffness - how aggressively to correct error.
          ki (float): Sustained effort - how much to boost power against gravity/friction.
          kd (float): Braking - how much to resist fast changes to prevent vibration.
          output_limits (tuple): Min/Max power allowed to be sent to the hardware.
          integral_limits (tuple): Bounds for anti-windup to prevent 'runaway' integral buildup.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.reset()
        
    def reset(self):
        """
        WHAT THIS DOES: Clears the memory of past errors.
        
        WHY: If the robot was stuck and now it's free, we don't want it to 'snap' violently 
             using old accumulated error data.
        """
        # Wipe the running total of past errors
        self._integral = 0.0
        # Forget the last error recorded
        self._prev_error = 0.0
        
    def tune(self, kp: float, ki: float, kd: float):
        """
        WHAT THIS DOES: Updates the math gains while the robot is running.
        
        WHY: Allows 'Auto-Tuning' features to make the robot stiffer or softer depending on the task.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
    def compute(self, error: float, dt: float) -> float:
        """
        WHAT THIS DOES: Calculates the exact power level to send to the motor right now.
        
        WHY: This is the heartbeat of physical motion control.
        
        ARGS:
          error (float): Distance to the goal in radians.
          dt (float): Time since the last check in seconds.
            
        RETURNS:
          output (float): Normalized motor power signal.
        
        MATH/LOGIC:
          - P-term: error * kp 
          - I-term: running_total * ki (capped by integral_limits to prevent 'windup')
          - D-term: (change_in_error / time) * kd
        """
        # Safety check: if time stopped, physics should stop too
        if dt <= 0:
            return 0.0
            
        # PROPORTIONAL: The immediate reaction
        p_term = self.kp * error
        
        # INTEGRAL: Fixed persistence
        # Accumulate error over time to overcome static friction or gravity 'sag'
        self._integral += error * dt
        i_term = self.ki * self._integral
        # ANTI-WINDUP: Cap the force so it doesn't build up infinite power if the arm is blocked
        if self.integral_limits:
            i_term = np.clip(i_term, self.integral_limits[0], self.integral_limits[1])
            
        # DERIVATIVE: Predictive braking
        # Calculate how fast we are approaching the goal to determine how much to brake
        d_error = (error - self._prev_error) / dt
        d_term = self.kd * d_error
        
        # Save current state for the next calculation cycle
        self._prev_error = error
        
        # SUMMATION: Combine the three independent forces
        output = p_term + i_term + d_term
        
        # HARDWARE BOUNDS: Clamp to ensures we don't fry the motor or snap the cables
        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
            
        return output

if __name__ == "__main__":
    # DEMO: VISUALIZING SMOOTH VS JERKY MOTION
    dt = 0.01  # 100Hz high-speed simulation
    time_steps = np.arange(0, 5.0, dt)
    # Define a target that steps from 1.0 down to -0.5
    target = np.ones_like(time_steps)
    target[250:] = -0.5 
    
    # 1. THE TWITCHY ROBOT (P-only)
    # High stiffness but no shock absorbers (D=0); expect overshoot and vibration
    jerky_pid = PIDController(kp=10.0, ki=0.0, kd=0.0)
    jerky_pos = 0.0
    jerky_response = []
    
    # 2. THE ORGANIC ROBOT (Tuned)
    # Balanced stiffness with active braking; expect smooth convergence
    smooth_pid = PIDController(kp=5.0, ki=1.0, kd=0.5)
    smooth_pos = 0.0
    smooth_response = []
    
    # Execute the comparison loop
    for t_idx, t in enumerate(time_steps):
        # Simulation step for jerky
        err_j = target[t_idx] - jerky_pos
        ctrl_j = jerky_pid.compute(err_j, dt)
        jerky_pos += ctrl_j * dt # basic euler physics integration
        jerky_response.append(jerky_pos)
        
        # Simulation step for smooth
        err_s = target[t_idx] - smooth_pos
        ctrl_s = smooth_pid.compute(err_s, dt)
        smooth_pos += ctrl_s * dt 
        smooth_response.append(smooth_pos)
        
    # Generate the validation plot
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, target, 'k--', label='Target')
    plt.plot(time_steps, jerky_response, 'r-', alpha=0.7, label='Jerky Response (P-only)')
    plt.plot(time_steps, smooth_response, 'b-', linewidth=2, label='Smooth Response (Tuned PID)')
    plt.title('PID Controller: Smooth vs Jerky Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the proof of performance to the notebooks directory
    os.makedirs('notebooks', exist_ok=True)
    plt.savefig('notebooks/pid_demo.png')
    print("Demo saved to notebooks/pid_demo.png")
