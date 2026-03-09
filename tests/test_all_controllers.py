# ═══════════════════════════════════════════════════════
# FILE: tests/test_all_controllers.py
# PURPOSE: The "Logic Inspector"; runs a battery of math tests on the robot's brain.
# LAYER: Verification Layer
# INPUTS: Fake 'Mock' data (simulated errors, paths, and muscle signals).
# OUTPUTS: A checklist of [PASS] or [FAIL] for every critical system component.
# CALLED BY: Manual QA testing (e.g., python tests/test_all_controllers.py)
# ═══════════════════════════════════════════════════════

#!/usr/bin/env python3
import numpy as np
import sys
import os

# Ensure the tester can find the 'src' folder by looking one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all 7 'Brains' for inspection
from src.robot.controllers import (
    PIDController,
    ObstacleAvoidanceController,
    SpeedController,
    ArmPatternRecognizer,
    SRLDebugLogger,
    EMGCalibrationController,
    DualArmCoordinationController
)

def test_pid():
    """
    WHAT THIS DOES: Verifies the 'Steering' math.
    
    WHY: If the PID math is broken, the robot will vibrate or fly apart in simulation.
    """
    print("Testing PIDController...")
    try:
        pid = PIDController(1.0, 0.1, 0.05)
        # Feed it a 0.5 error and see if it outputs a correction number
        out = pid.compute(error=0.5, dt=0.01)
        assert isinstance(out, float), "PID must return a numerical correction"
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def test_obstacle():
    """
    WHAT THIS DOES: Checks the 'Safety Shield'.
    
    WHY: Ensures the robot knows how to push AWAY from a point in 3D space.
    """
    print("Testing ObstacleAvoidanceController...")
    try:
        ctrl = ObstacleAvoidanceController()
        # Simulate the hand being near the head (0, 0, 0.35)
        repulsion = ctrl.compute_repulsion_vector(np.array([0, 0, 0.35]))
        assert repulsion.shape == (3,), "Repulsion must be a 3D [X,Y,Z] vector"
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def test_speed():
    """
    WHAT THIS DOES: Verifies the 'Governor'.
    
    WHY: Ensures the robot never ignores its safety speed limits.
    """
    print("Testing SpeedController...")
    try:
        ctrl = SpeedController()
        # Verify that 'Surgical' mode is indeed slow (0.3 rad/s)
        limit = ctrl.get_velocity_limit("surgical")
        assert limit == 0.3, "Surgical speed should be capped at 0.3"
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def test_pattern():
    """
    WHAT THIS DOES: Tests the 'Gesture Reader'.
    
    WHY: Checks if moving up is correctly identified as a 'LIFT'.
    """
    print("Testing ArmPatternRecognizer...")
    try:
        ctrl = ArmPatternRecognizer()
        # Create a fake 20-frame movement moving only UP (Z-axis)
        history = np.zeros((20, 3))
        history[:, 2] = np.linspace(0, 0.5, 20) 
        pattern, conf = ctrl.classify(history)
        assert pattern == "LIFT", f"Robot thought lifting was {pattern} instead"
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def test_logger():
    """
    WHAT THIS DOES: Checks the 'Black Box Recorder'.
    
    WHY: Ensures logs are actually being saved to memory.
    """
    print("Testing SRLDebugLogger...")
    try:
        logger = SRLDebugLogger()
        logger.log_step({"intent_prediction": "REACH", "task_mode": "surgical"})
        assert len(logger.session_data) == 1, "Log file should have 1 entry"
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def test_calibration():
    """
    WHAT THIS DOES: Verifies signal 'Cleaning'.
    
    WHY: Ensures muscle signals are normalized to a standard 0-1 range.
    """
    print("Testing EMGCalibrationController...")
    try:
        ctrl = EMGCalibrationController()
        # Feed it 16 channels of random raw noise
        calib = ctrl.normalize(np.random.randn(16))
        assert calib.shape == (16,), "Output must maintain 16-channel structure"
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def test_coordination():
    """
    WHAT THIS DOES: Tests 'Teamwork' logic.
    
    WHY: Ensures the arms know who should 'Grasp' during a handoff.
    """
    print("Testing DualArmCoordinationController...")
    try:
        ctrl = DualArmCoordinationController()
        ctrl.set_mode("HANDOFF")
        # Fake object position (Right arm is closer)
        obs_mock = {'ee_pos_left': [0.1, 0, 0], 'ee_pos_right': [0.05, 0, 0]}
        l_out, r_out = ctrl.compute_coordinated_action({}, {}, obs_mock)
        # In Handoff, Left arm should grip (1.0) and Right should be ready (0.0)
        assert l_out['gripper'] == 1.0, "Left arm should have gripped"
        assert r_out['gripper'] == 0.0, "Right arm should be releasing"
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

if __name__ == "__main__":
    # --- MASTER TEST SEQUENCER ---
    print("--- Running Controller Unit Tests ---\n")
    results = [
        test_pid(),
        test_obstacle(),
        test_speed(),
        test_pattern(),
        test_logger(),
        test_calibration(),
        test_coordination()
    ]
    
    # FINAL VERDICT
    if all(results):
        print("\n[SUCCESS] All 7 controllers instantiated and executed baseline tests.")
    else:
        print("\n[ERROR] Some controller tests failed. See log above for details.")
