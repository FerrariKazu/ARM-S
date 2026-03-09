# ═══════════════════════════════════════════════════════
# FILE: src/architecture.py
# PURPOSE: The "Blueprint" of the entire ARM-S project; defines how data flows between layers.
# LAYER: Core Architecture
# INPUTS: N/A (Defines constants and data structures)
# OUTPUTS: Standardized Python objects for Intent, Sensing, and Control.
# CALLED BY: Almost every file in the src/ directory.
# ═══════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import sys

# --- BASE GEOMETRY BLOCKS ---
# These mock the industry-standard 'geometry_msgs' used in ROS 2.

@dataclass
class Point:
    """WHAT THIS IS: A single coordinate in 3D space [X, Y, Z]."""
    x: float
    y: float
    z: float

@dataclass
class Quaternion:
    """WHAT THIS IS: A 4-number math object used to describe rotation without 'Gimbal Lock'."""
    x: float
    y: float
    z: float
    w: float

@dataclass
class Pose:
    """WHAT THIS IS: A complete description of an object's location AND where it is pointing."""
    position: Point
    orientation: Quaternion

# --- INTELLECTUAL DATA CONTRACTS ---
# These define what "Knowledge" looks like in the ARM-S system.

@dataclass
class IntentPrediction:
    """
    WHAT THIS CLASS IS:
      The result of the AI's "Deep Thought". It contains what the AI thinks 
      the human is doing and how sure it is.
    """
    confidence: float # 0.0 to 1.0
    action_type: str  # e.g. "REACH", "GRASP"
    target_position: Point
    emg_features: List[float]

@dataclass
class BodyState:
    """
    WHAT THIS CLASS IS:
      A digital snapshot of the human operator's current physical pose.
    """
    spine_pose: Pose
    wrist_poses: List[Pose]
    imu_data: List[float] # RAW balance data from body sensors

@dataclass
class TaskContext:
    """
    WHAT THIS CLASS IS:
      The 'Rules of Engagement' for the current job. 
    """
    mode: str              # e.g. "surgical", "heavy_lift"
    urgency_level: float   # 0 (relax) to 1 (emergency)
    human_proximity_alert: bool # True if robot is dangerously close to human skin

@dataclass
class ArmCommand:
    """
    WHAT THIS CLASS IS:
      The final 'Instruction Set' sent to the physical robot motors.
    """
    arm_id: int
    joint_positions: List[float]
    joint_velocities: List[float]
    gripper_force: float

# --- SYSTEM LAYER DEFINITIONS ---
# These represent the 4 logical 'Brains' that make up ARM-S.

@dataclass
class SensingLayer:
    """PURPOSE: Turns raw electrical noise into a clean human pose."""
    input_imu_raw: List[float]
    input_emg_raw: List[float]
    output_body_state: BodyState

@dataclass
class IntentLayer:
    """PURPOSE: Guesses the human's goal by looking at their body language."""
    input_body_state: BodyState
    output_prediction: IntentPrediction

@dataclass
class PlanningLayer:
    """PURPOSE: Maps out exactly where the robot arms should go to help the human."""
    input_prediction: IntentPrediction
    input_context: TaskContext
    output_target_poses: List[Pose]

@dataclass
class ExecutionLayer:
    """PURPOSE: Turns the 'Plan' into high-speed motor vibrations and movements."""
    input_target_poses: List[Pose]
    output_command: ArmCommand

# --- CONFIGURATION & SAFETY GATES ---

@dataclass
class LatencyBudget:
    """
    WHAT THIS CLASS IS:
      The "Time Bank". It defines exactly how many milliseconds 
      each layer is allowed to take.
      
    WHY IT EXISTS:
      If the 'Sensing' layer takes too long, the robot will be 'laggy', 
      which can cause physical injury to the human operator.
    """
    sensing_ms: float = 10.0  # Time to read sensors
    intent_ms: float = 35.0   # Time to run AI model
    planning_ms: float = 30.0 # Time to find a path
    execution_ms: float = 15.0 # Time to send motor packets

    @property
    def total_ms(self) -> float:
        """Calculates the end-to-end reaction time of the system."""
        return self.sensing_ms + self.intent_ms + self.planning_ms + self.execution_ms

@dataclass
class SystemConfig:
    """
    WHAT THIS CLASS IS:
      The central 'Settings Page' for the entire robot simulation.
    """
    # Number of motor checks per second (100 is smooth)
    control_frequency_hz: int = field(default=100) 
    # Bio-signal resolution (1kHz is pro-grade)
    emg_sample_rate_hz: int = field(default=1000) 
    num_emg_channels: int = field(default=8)
    num_arms: int = field(default=2)
    joints_per_arm: int = field(default=7)
    # Distance in meters before the software panics and stops
    safety_distance_threshold: float = field(default=0.5) 
    default_mode: str = field(default="collaborative")
    # AI Tuning parameters
    tuneable_parameters: Dict[str, float] = field(default_factory=lambda: {
        "p_gain": 0.8,
        "d_gain": 0.1,
        "intent_threshold": 0.75
    })

def validate_architecture(budget: LatencyBudget):
    """
    WHAT THIS DOES: Runs a 'Stress Test' on the system timing.
    
    WHY: To ensure we never ship code that is too slow to be safe.
    
    ARGS:
      budget (LatencyBudget): the current timing measurements.
    
    MATH:
      - Checks if Total Latency < 100ms (The 'Human Sync' limit).
    """
    # 100ms is the maximum delay before a human feels 'disconnected' from the arms
    max_total = 100.0
    total = budget.total_ms
    
    # --- PRINT SUMMARY TABLE ---
    print(f"\n{'Layer':<20} | {'Budget (ms)':<15}")
    print("-" * 38)
    print(f"{'Sensing':<20} | {budget.sensing_ms:<15.2f}")
    print(f"{'Intent':<20} | {budget.intent_ms:<15.2f}")
    print(f"{'Planning':<20} | {budget.planning_ms:<15.2f}")
    print(f"{'Execution':<20} | {budget.execution_ms:<15.2f}")
    print("-" * 38)
    print(f"{'TOTAL':<20} | {total:<15.2f}")
    
    # --- FINAL VERDICT ---
    if total < max_total:
        print(f"\nSUCCESS: Total latency {total:.2f}ms is under the {max_total}ms limit.")
    else:
        # If the code is slow, we crash the program so it can't be used dangerously
        print(f"\nFAILURE: Total latency {total:.2f}ms exceeds the {max_total}ms limit!", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # BOOTSTRAP ARCHITECTURE VALIDATION
    current_budget = LatencyBudget()
    config = SystemConfig()
    
    print(f"ARM-S System Architecture Initialized")
    print(f"Control Frequency: {config.control_frequency_hz} Hz")
    
    # Ensure the blueprint is physically possible
    validate_architecture(current_budget)
