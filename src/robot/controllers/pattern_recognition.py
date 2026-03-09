# ═══════════════════════════════════════════════════════
# FILE: src/robot/controllers/pattern_recognition.py
# PURPOSE: Translates raw arm movement "wiggles" into meaningful human gestures like 'Reaching'.
# LAYER: Extraction Layer (Heuristics)
# INPUTS: A history window of the robot's hand positions (last 20 frames).
# OUTPUTS: A text label (REACH, LIFT, etc.) and a confidence percentage.
# CALLED BY: SRL Logic, dashboard_controllers.py
# ═══════════════════════════════════════════════════════

import numpy as np

class ArmPatternRecognizer:
    """
    WHAT THIS CLASS IS:
      A "Gesture Translator". It looks at how the robot's hands have moved 
      over the last 400 milliseconds and matches those movements to a list 
      of known human-like activities.

    WHY IT EXISTS:
      Sometimes the robot needs to know the *intent* of a move without 
      asking a heavy AI model. This class provides instant, rule-based 
      recognition for basic motions like "Lifting something up" or 
      "Sweeping across a table".

    HOW IT WORKS (step by step):
      1. Takes the last 20 positions of the robot's hand.
      2. Calculates how fast it's moving (Velocity) and how fast it's speeding up (Acceleration).
      3. Determines the overall direction (e.g., mostly Up, mostly Left).
      4. Compares these numbers to human-defined "Rules" to find the best match.

    EXAMPLE USAGE:
      recognizer = ArmPatternRecognizer()
      label, confidence = recognizer.classify(recent_hand_positions)
      print(f"I think the user is doing a: {label}")
    """
    def __init__(self):
        """
        WHAT THIS DOES: Defines the vocabulary of gestures the system understands.
        """
        self.known_patterns = [
            "REACH", "LIFT", "HOLD", "LOWER", "SWEEP", "STABILIZE", "HANDOFF", "UNKNOWN"
        ]
        
    def extract_features(self, ee_history: np.ndarray) -> np.ndarray:
        """
        WHAT THIS DOES: Turns a long list of coordinates into a tidy summary of movement math.
        
        WHY: Raw data is too noisy; we need 'averages' to see the true pattern.
        
        ARGS:
          ee_history (np.ndarray): 20 frames of [x, y, z] coordinates.
            
        RETURNS:
          features (np.ndarray): A 12-number summary vector.
        
        MATH/LOGIC:
          - Velocity: The difference between one frame and the next.
          - Acceleration: The difference between one 'speed' and the next.
          - Direction Cosines: Arrows that point exactly where the movement is headed.
        """
        # Safety check: we need exactly 20 frames to make an accurate guess
        assert ee_history.shape == (20, 3), "ee_history must be shape (20, 3)"
        
        # CALCULATE DYNAMICS
        # np.diff finds the 'jump' between points (Velocity)
        vel = np.diff(ee_history, axis=0)
        # np.diff on velocity finds the change in speed (Acceleration)
        acc = np.diff(vel, axis=0)
        
        # Smooth out noise by taking the average across the 20-frame window
        mean_vel = np.mean(vel, axis=0)
        mean_acc = np.mean(acc, axis=0)
        
        # DISPLACEMENT
        # Subtract the start point from the end point to see the net progress
        displacement = ee_history[-1] - ee_history[0]
        
        # DIRECTION SCANNING
        # Normalize the displacement to get a pure 'pointing' vector
        norm = np.linalg.norm(displacement)
        if norm < 1e-6:
            # If the hand didn't move, the direction is zero
            dir_cosines = np.zeros(3)
        else:
            # Scale the vector to length 1.0
            dir_cosines = displacement / norm
            
        # Compile everything into a single flat list of 12 numbers
        features = np.concatenate([
            mean_vel,      # [0-2]
            mean_acc,      # [3-5]
            displacement,  # [6-8]
            dir_cosines    # [9-11]
        ])
        
        return features

    def classify(self, ee_history: np.ndarray) -> tuple[str, float]:
        """
        WHAT THIS DOES: Matches the distilled movement math to a specific gesture.
        
        WHY: This is the decision-making step that tells the robot how to assist.
        
        ARGS:
          ee_history (np.ndarray): recent 3D movement data.
            
        RETURNS:
          label (str): The name of the gesture.
          confidence (float): How sure we are (0.0 to 1.0).
        
        MATH/LOGIC (The Rulebook):
          - Proximity Rule: If distance < 2cm, call it a 'HOLD'.
          - Verticality Rule: If movement is >70% Upwards, call it a 'LIFT'.
          - Planar Rule: If movement is mostly side-to-side, call it a 'SWEEP'.
        """
        # Step 1: Simplify the data
        features = self.extract_features(ee_history)
        
        # Extract specific useful dimensions for the rules below
        mean_vel_z = features[2]   # Up/Down speed
        mean_acc_z = features[5]   # Up/Down thrust
        disp_mag = np.linalg.norm(features[6:9]) # Total distance traveled
        dir_z = features[11]       # Verticality score (-1.0 to 1.0)
        
        # RULE 1: THE STATIONARY CHECK
        # If the hand has moved less than 2cm total
        if disp_mag < 0.02:
            # Check if the motors are vibrating (high acceleration)
            if np.mean(np.abs(features[3:6])) > 0.05:
                return ("STABILIZE", 0.8) # Machine is working hard to stay still
            else:
                return ("HOLD", 0.95) # Machine is idle
                
        # RULE 2: THE VERTICALITY CHECK
        # If more than 70% of the movement intensity is on the Z-axis
        if abs(dir_z) > 0.7:
            if mean_vel_z > 0:
                # Upward movement
                return ("LIFT", min(1.0, 0.5 + abs(mean_vel_z)*10))
            else:
                # Downward movement
                return ("LOWER", min(1.0, 0.5 + abs(mean_vel_z)*10))
                
        # RULE 3: THE TABLE SWEEP
        # If movement is horizontal (>5cm) and verticality is low (<30%)
        xy_disp = np.linalg.norm(features[6:8])
        if xy_disp > 0.05 and abs(dir_z) < 0.3:
            return ("SWEEP", 0.85)
            
        # RULE 4: THE REACH
        # General outward progress of >10cm that doesn't fit other specialized rules
        if disp_mag > 0.1:
            return ("REACH", 0.7)
            
        # If no logic matches, admit defeat to prevent wrong automated reactions
        return ("UNKNOWN", 0.0)
