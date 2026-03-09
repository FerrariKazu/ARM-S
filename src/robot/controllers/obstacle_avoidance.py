# ═══════════════════════════════════════════════════════
# FILE: src/robot/controllers/obstacle_avoidance.py
# PURPOSE: Virtual 'Force Field' that prevents the robot from ever touching the human's head or face.
# LAYER: Execution Layer (Safety)
# INPUTS: 3D positions (XYZ) of the robot's hands (end-effectors).
# OUTPUTS: Repulsion force vectors that push the arm away from danger zones.
# CALLED BY: SRL Control Loop, src/sim/srl_env.py
# ═══════════════════════════════════════════════════════

import numpy as np
from typing import List, Tuple

class ObstacleAvoidanceController:
    """
    WHAT THIS CLASS IS:
      A "Safety Shield" for the human operator. It monitors where the robot 
      arms are and creates invisible repelling forces if they get too close 
      to the human's head.

    WHY IT EXISTS:
      In Supernumerary Robotics, the arms are mounted directly to the human. 
      This controller ensures that even if the AI makes a mistake, the hardware 
      physically cannot swing into the human's face.

    HOW IT WORKS (step by step):
      1. Defines "Forbidden Zones" as spheres around the human's head and neck.
      2. Measures the distance between the robot's hand and these spheres.
      3. If the distance is less than 15cm, it calculates a 'push' vector.
      4. The closer the arm gets, the exponentially harder the vector pushes back.

    EXAMPLE USAGE:
      detector = ObstacleAvoidanceController()
      push_force = detector.compute_repulsion_vector(hand_xyz)
      # apply push_force to the motor torque to steer away
    """
    def __init__(self, forbidden_zones: List[Tuple[float, float, float, float]] = None):
        """
        WHAT THIS DOES: Defines the locations of the human's critical body parts.
        
        WHY: To create a mathematical map of where the robot is never allowed to go.
        
        ARGS:
          forbidden_zones (list): optional list of (x, y, z, radius) spheres.
        """
        if forbidden_zones is None:
            # DEFAULT SAFETY CONFIGURATION
            # Defined in meters relative to the backpack center
            self.forbidden_zones = [
                (0.0, 0.0, 0.35, 0.15),  # The Main Head Sphere
                (0.1, 0.0, 0.30, 0.1)    # The Lower Face/Chin Shield
            ]
        else:
            self.forbidden_zones = forbidden_zones
            
        # The 'Outer Buffer' where the robot starts to feel a gentle push
        self.repulsion_threshold = 0.15 

    def check_proximity(self, ee_pos: np.ndarray) -> float:
        """
        WHAT THIS DOES: Calculates the exact gap between the hand and the nearest hazard.
        
        WHY: Used as a simple 'Yes/No' check for safety audits.
        
        ARGS:
          ee_pos (np.ndarray): current [x, y, z] of the gripper.
            
        RETURNS:
          min_dist (float): gap in meters (negative means a collision has occurred).
        """
        # Ensure input is a usable numpy math object
        ee_pos = np.asarray(ee_pos)
        min_dist = float('inf')
        
        # Iterate through all configured danger zones
        for zone in self.forbidden_zones:
            center = np.array(zone[0:3])
            radius = zone[3]
            # Simple Euclidean distance from hand to center of sphere
            dist_to_center = np.linalg.norm(ee_pos - center)
            # Subtract radius to find distance to the actual 'skin' of the zone
            dist_to_boundary = dist_to_center - radius
            
            # Keep track of only the single most dangerous object nearby
            if dist_to_boundary < min_dist:
                min_dist = dist_to_boundary
                
        return min_dist

    def compute_repulsion_vector(self, ee_pos: np.ndarray) -> np.ndarray:
        """
        WHAT THIS DOES: Generates a 3D directional arrow pointing away from danger.
        
        WHY: This vector is added to the motor power to 'steer' the robot around the human.
        
        ARGS:
          ee_pos (np.ndarray): current [x, y, z] of the gripper.
            
        RETURNS:
          total_repulsion (np.ndarray): [x, y, z] force vector.
        
        MATH/LOGIC:
          - Uses Exponential Scaling: Force = e ^ (-dist).
          - This ensures a soft nudge at 15cm, but a massive physical stop at 2cm.
        """
        ee_pos = np.asarray(ee_pos)
        total_repulsion = np.zeros(3)
        
        for zone in self.forbidden_zones:
            center = np.array(zone[0:3])
            radius = zone[3]
            # Vector pointing from the hazard center TO the robot hand
            offset = ee_pos - center
            dist_to_center = np.linalg.norm(offset)
            
            # Divide-by-zero protection if the hand is exactly at the center (unlikely but safe)
            if dist_to_center < 1e-6:
                offset = np.array([0, 0, 1.0]) 
                dist_to_center = 1.0
                
            # Normalize the vector to get the pure direction of the push
            dir_vector = offset / dist_to_center
            dist_to_boundary = dist_to_center - radius
            
            # Activate only if the arm has entered the 'Warning Zone' buffer
            if dist_to_boundary < self.repulsion_threshold:
                # EXPONENTIAL RAMP-UP
                if dist_to_boundary <= 0:
                    # EMERGENCY CLAMP: If hand is inside the head, apply maximum possible push
                    magnitude = 1000.0 
                else:
                    # Calculate a smooth force curve so the robot doesn't jerk
                    # -15.0 constant controls how 'stiff' the air feels near the zone
                    magnitude = np.exp(-15.0 * dist_to_boundary / self.repulsion_threshold)
                
                # Add this specific repulsion to the global tally for all zones
                total_repulsion += dir_vector * magnitude
                
        return total_repulsion

    def is_safe(self, trajectory: np.ndarray) -> bool:
        """
        WHAT THIS DOES: Bench-tests a planned path before the robot actually moves.
        
        WHY: It's better to realize a path is dangerous in math than in reality.
        
        ARGS:
          trajectory (np.ndarray): An array of many [x, y, z] points representing a future path.
            
        RETURNS:
          (bool): True if every single point is outside the danger zones.
        """
        # Loop through every waypoint in the proposed movement
        for i in range(trajectory.shape[0]):
            # If even one point is a collision, reject the whole path
            if self.check_proximity(trajectory[i]) <= 0:
                return False
        return True
