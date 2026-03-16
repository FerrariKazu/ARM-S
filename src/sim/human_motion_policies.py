# ═══════════════════════════════════════════════════════
# FILE: src/sim/human_motion_policies.py
# PURPOSE: The "Actor's Script" for the simulated human; defines how they move for different jobs.
# LAYER: Simulation Layer
# INPUTS: The current time in the simulation (seconds).
# OUTPUTS: A plan for where the human's joints should move and what they are trying to do.
# CALLED BY: training/generate_dataset.py, src/sim/srl_env.py
# ═══════════════════════════════════════════════════════

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class HumanMotionPolicy(ABC):
    """
    WHAT THIS CLASS IS:
      A template for "Human Personalities". It defines the basic math needed 
      to move a body naturally, like smooth acceleration and label definitions.

    WHY IT EXISTS:
      In a simulation, the human isn't real. This class (and its children) 
      acts as the "Ghost in the Machine" that moves the human model so the 
      robot has a realistic partner to react to.

    HOW IT WORKS (step by step):
      1. Receives the current time CPU clock.
      2. Calculates high-level joint angles using smooth math (curves).
      3. Returns the goal positions along with a 'Label' (like "LIFTING").

    EXAMPLE USAGE:
      # This is an abstract class, so you use one of its children:
      actor = ReachPolicy()
      motion_data = actor.step(current_time)
    """
    def __init__(self, num_arm_joints: int = 16):
        """
        WHAT THIS DOES: Records how many joints the actor has to control.
        """
        self.num_arm_joints = num_arm_joints
        # Standard dictionary of labels the AI is trained to recognize
        self.intent_labels = ["reach_left", "reach_right", "lift", "hold", "release", "stabilize", "handoff"]

    @abstractmethod
    def step(self, t: float) -> Dict[str, Any]:
        """
        WHAT THIS DOES: A placeholder for the specific movement math.
        
        WHY: Every child (like Carry vs reach) must define its own movement pattern.
        """
        pass

    def _smooth_step(self, t: float, start_val: float, end_val: float, duration: float) -> float:
        """
        WHAT THIS DOES: An 'Ease-In, Ease-Out' math formula.
        
        WHY: Humans don't snap their bodies instantly; they accelerate and decelerate. 
             This makes the simulation look organic.
             
        ARGS:
          t (float): Current time in the movement.
          start_val (float): Initial angle.
          end_val (float): Target angle.
          duration (float): Seconds to complete the move.
        
        MATH:
          - Uses Cubic Hermite Interpolation (3x^2 - 2x^3) to create an S-curve.
        """
        # Calculate percentage of completion (0.0 to 1.0)
        x = np.clip(t / duration, 0, 1)
        # Apply the S-Curve formula to the gap between start and end
        return start_val + (end_val - start_val) * (3 * x**2 - 2 * x**3)

class CarryPolicy(HumanMotionPolicy):
    """
    WHAT THIS CLASS IS:
      The "Walking Robot" behavior. It simulates a person carrying a heavy 
      box while walking slowly.
      
    HOW IT WORKS: 
      Uses sine waves to create a 'swaying' motion that looks like a gait.
    """
    def step(self, t: float) -> Dict[str, Any]:
        """
        WHAT THIS DOES: Calculates the swaying shoulder/elbow angles.
        """
        # Set a walking pace of 0.5 steps per second
        gait_freq = 0.5 
        # Convert time to a repeating circle (0 to 2*PI)
        phase = t * 2 * np.pi * gait_freq
        
        # Initialize a silent pose (all zeros)
        targets = np.zeros(self.num_arm_joints)
        
        # Apply rhythmic swaying to simulate the human's hips and shoulders moving
        targets[1] = -0.5 + 0.05 * np.sin(phase) # Right shoulder swaying
        targets[2] = 1.0 + 0.05 * np.cos(phase)  # Right elbow stabilized
        targets[9] = 0.5 + 0.05 * np.sin(phase)  # Left shoulder mirroring
        targets[10] = 1.0 + 0.05 * np.cos(phase) # Left elbow stabilized
        
        return {
            "joint_targets": targets,
            "intent_label": "hold",
            "time_to_next_action": 2.0
        }

class ReachPolicy(HumanMotionPolicy):
    """
    WHAT THIS CLASS IS:
      The "Item Picker" behavior. Alternately reaches forward with the 
      Left and Right arms.
    """
    def __init__(self, num_arm_joints: int = 16):
        super().__init__(num_arm_joints)
        self.cycle_time = 3.0

    def step(self, t: float) -> Dict[str, Any]:
        """
        WHAT THIS DOES: Ramps the shoulders forward and upward.
        """
        cycle = (t % self.cycle_time) / self.cycle_time  # 0.0 to 1.0
        
        # Phase 1 (0.0-0.4): right arm lifts and reaches forward
        # Phase 2 (0.4-0.6): hold at extension  
        # Phase 3 (0.6-1.0): return smoothly
        
        targets = np.zeros(16)
        
        if cycle < 0.4:
            progress = cycle / 0.4  # 0 to 1
            # Apply cosine easing for smooth start/stop
            smooth = 0.5*(1 - np.cos(np.pi*progress))
            # Right arm: lift (joint2 negative = up), reach forward (joint1)
            targets[0] =  0.4 * smooth   # shoulder out slightly
            targets[1] = -0.6 * smooth   # lift upward
            targets[2] = -0.4 * smooth   # curve forward
            targets[3] = -0.2 * smooth   # wrist follow
        elif cycle < 0.6:
            # Phase 2: HOLD extended position
            targets[0] =  0.4
            targets[1] = -0.6
            targets[2] = -0.4
            targets[3] = -0.2
        else:
            # Phase 3: RETURN smoothly to home
            progress = (cycle - 0.6) / 0.4
            smooth = 0.5*(1 - np.cos(np.pi*progress))
            targets[0] =  0.4 * (1 - smooth)
            targets[1] = -0.6 * (1 - smooth)
            targets[2] = -0.4 * (1 - smooth)
            targets[3] = -0.2 * (1 - smooth)
        
        # Left arm stays in neutral rest position (offset slightly for variety)
        targets[8:12] = [0.1, -0.2, -0.1, 0.0]
        
        return {
            'joint_targets': targets,
            'intent_label': 'reach_right' if cycle < 0.6 else 'release',
            'time_to_next_action': (1.0 - cycle) * self.cycle_time
        }

class AssemblyPolicy(HumanMotionPolicy):
    """
    WHAT THIS CLASS IS:
      The "Fine Motor" behavior. Simulates a worker doing precise 
      electronics work near their chest.
    """
    def step(self, t: float) -> Dict[str, Any]:
        """
        WHAT THIS DOES: Keeps arms tight to the body with tiny wrist wiggles.
        """
        # Faster wiggles for high-concentration work
        freq = 1.2
        phase = t * 2 * np.pi * freq
        
        targets = np.zeros(self.num_arm_joints)
        
        # Position arms in 'Precision Triangle' in front of chest
        targets[0] = 0.4 
        targets[8] = -0.4 
        
        # MICRO-MOVEMENTS: Tiny high-speed oscillations to keep the AI alert
        targets[4] = 0.1 * np.sin(phase) # Right wrist 'poking'
        targets[12] = 0.1 * np.cos(phase) # Left wrist 'poking'
        
        return {
            "joint_targets": targets,
            "intent_label": "stabilize",
            "time_to_next_action": 0.5
        }

class OverheadPolicy(HumanMotionPolicy):
    """
    WHAT THIS CLASS IS:
      The "Lifting" behavior. Simulates putting something on a high shelf.
    """
    def step(self, t: float) -> Dict[str, Any]:
        """
        WHAT THIS DOES: Smoothly lifts the arms above 90 degrees and holds them.
        """
        cycle = 4.0
        t_cycle = t % cycle
        
        targets = np.zeros(self.num_arm_joints)
        
        if t_cycle < 2.0:
            # THE LIFT: Ramp from neutral to full extension
            arm_lift = self._smooth_step(t_cycle, 0.5, 2.0, 2.0)
            label = "lift"
        else:
            # THE HOLD: Freeze at the peak height
            arm_lift = 2.0
            label = "hold"
            
        # Move both shoulder motors to the lift height
        targets[1] = -arm_lift 
        targets[9] = arm_lift  
        
        return {
            "joint_targets": targets,
            "intent_label": label,
            "time_to_next_action": max(0, 2.0 - (t_cycle % 2.0))
        }

if __name__ == "__main__":
    # BOOTSTRAP VALIDATION
    import time
    
    # Initialize all personalities
    policies = [CarryPolicy(), ReachPolicy(), AssemblyPolicy(), OverheadPolicy()]
    policy_names = ["Carry", "Reach", "Assembly", "Overhead"]
    
    print("Testing Human Motion Policies (Simulated first 3 seconds)...")
    for name, policy in zip(policy_names, policies):
        print(f"\n--- Policy: {name} ---")
        # Sample the movement every 1 second
        for t in np.arange(0, 3.1, 1.0):
            result = policy.step(t)
            print(f"t={t:.1f}s: Label={result['intent_label']}, SumTargets={np.sum(result['joint_targets']):.3f}")
