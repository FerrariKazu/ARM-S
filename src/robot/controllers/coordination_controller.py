# ═══════════════════════════════════════════════════════
# FILE: src/robot/controllers/coordination_controller.py
# PURPOSE: The 'Conductor' that ensures the Left and Right arms work together as a team.
# LAYER: Planning Layer
# INPUTS: The separate intentions of the Left and Right arms.
# OUTPUTS: A single coordinated 'Game Plan' for both arms to follow.
# CALLED BY: SRL Task Orchestrator, src/sim/srl_env.py
# ═══════════════════════════════════════════════════════

import numpy as np

class DualArmCoordinationController:
    """
    WHAT THIS CLASS IS:
      The "Teamwork Brain". Most robot scripts control one arm; this class 
      manages two, ensuring they don't hit each other and that they help 
      each other carry heavy things.

    WHY IT EXISTS:
      Working with two arms is hard. If the human wants to lift a large box, 
      the arms must move in perfect sync. This class provides the 4 'Playbooks' 
      (Modes) for dual-arm cooperation.

    HOW IT WORKS (step by step):
      1. Receives the destination for the Left arm and the Right arm.
      2. Checks the "Cooperation Mode" (e.g., Mirroring or Handoff).
      3. Tweaks the destinations so they make sense for a duo (e.g., making 
         heights identical in Cooperative mode).
      4. Monitors the distance between hands for smooth object transfers.

    EXAMPLE USAGE:
      conductor = DualArmCoordinationController()
      conductor.set_mode("SYMMETRIC")
      left, right = conductor.compute_coordinated_action(L_aim, R_aim, sensors)
    """
    def __init__(self):
        """
        WHAT THIS DOES: Starts the robot in 'Independent' mode by default.
        """
        # Starting mode: Arms act like two separate robots
        self.mode = "INDEPENDENT"
        # Minimum distance (10cm) before we consider a 'Physical Handoff' possible
        self.handoff_threshold_m = 0.1
        
    def set_mode(self, mode: str):
        """
        WHAT THIS DOES: Switches the cooperation strategy.
        
        WHY: Different jobs (carrying vs reaching) require different levels of teamwork.
        
        ARGS:
          mode (str): One of INDEPENDENT, SYMMETRIC, COOPERATIVE, or HANDOFF.
        """
        valid_modes = ["INDEPENDENT", "SYMMETRIC", "COOPERATIVE", "HANDOFF"]
        mode = mode.upper()
        if mode in valid_modes:
            self.mode = mode
            print(f"[Coordination] Mode set to {self.mode}")
        else:
            # Fallback to prevent system crashes on invalid command strings
            print(f"[Coordination] Unknown mode: {mode}")

    def compute_coordinated_action(
        self, 
        left_intent: dict, 
        right_intent: dict, 
        obs: dict
    ) -> tuple[dict, dict]:
        """
        WHAT THIS DOES: The main 'Negotiator'. It modifies arm plans to honor the team mode.
        
        WHY: To prevent 'Single-Arm' thinking from causing errors in 'Dual-Arm' tasks.
        
        ARGS:
          left_intent (dict): where the left arm *wants* to go.
          right_intent (dict): where the right arm *wants* to go.
          obs (dict): current sensor data (e.g., hand positions).
            
        RETURNS:
          (tuple): Revised dictionaries for (left_action, right_action).
        
        MATH/LOGIC:
          - SYMMETRIC: Inverts the X-axis so the left arm mirrors the right arm's work.
          - COOPERATIVE: Averages the Z-height so both arms lift a level load.
          - HANDOFF: Proximity trigger that swaps gripper states (one opens, one closes).
        """
        # Start with the user's original intentions
        left_action = left_intent.copy()
        right_action = right_intent.copy()
        
        if self.mode == "INDEPENDENT":
            # No changes needed - arms act completely autonomously
            return left_action, right_action
            
        elif self.mode == "SYMMETRIC":
            # MIRROR MODE (Useful for repetitive assembly or showing off)
            # The Right arm handles the 'Strategy'; the Left arm follows the mirror image
            if 'target_pos' in right_action:
                mirror_pos = np.copy(right_action['target_pos'])
                # Swap the sign on the X-axis (Left vs Right side of the human)
                mirror_pos[0] = -mirror_pos[0] 
                left_action['target_pos'] = mirror_pos
                # Ensure the left claw follows the right claw's opening/closing
                left_action['gripper'] = right_action.get('gripper', 0.0)
                
        elif self.mode == "COOPERATIVE":
            # BOX LIFT MODE (Ensures the load doesn't tilt and drop)
            # If both arms are moving, synchronize their height (Z-axis)
            if 'target_pos' in right_action and 'target_pos' in left_action:
                # Find the 'Middle ground' height
                avg_z = (right_action['target_pos'][2] + left_action['target_pos'][2]) / 2.0
                # Snap both arms to the horizontal average
                right_action['target_pos'][2] = avg_z
                left_action['target_pos'][2] = avg_z
                
        elif self.mode == "HANDOFF":
            # PASSING MODE (Passing an object from Right tool to Left tool)
            # Monitor the spatial gap between the physical hands
            if 'ee_pos_left' in obs and 'ee_pos_right' in obs:
                # Calculate 3D distance between grippers
                dist = np.linalg.norm(np.array(obs['ee_pos_left']) - np.array(obs['ee_pos_right']))
                # If they are within 10cm, perform the automated 'Relay Race' swap
                if dist < self.handoff_threshold_m:
                    print("[Coordination] Handoff proximity reached! Synchronizing grippers.")
                    # LEFT arm takes the object (Closes)
                    left_action['gripper'] = 1.0  
                    # RIGHT arm lets go (Opens)
                    right_action['gripper'] = 0.0 
                    
        return left_action, right_action
