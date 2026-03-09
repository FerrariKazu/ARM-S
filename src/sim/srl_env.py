# ═══════════════════════════════════════════════════════
# FILE: src/sim/srl_env.py
# PURPOSE: Gymnasium wrapper around MuJoCo for training and testing the robot arms.
# LAYER: Execution Layer (Simulation)
# INPUTS: 18-member action vector (16 joints, 2 grippers).
# OUTPUTS: 128-member observation vector, scalar reward, and termination flags.
# CALLED BY: training/train_rl.py, manual test scripts.
# ═══════════════════════════════════════════════════════

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from typing import Dict, Any, Tuple, Optional, List

class SRLEnv(gym.Env):
    """
    WHAT THIS CLASS IS:
      A "Gymnasium Environment" that acts as the physical playground for the robot. 
      It takes math (actions) and returns what the robot sees and feels (observations).

    WHY IT EXISTS:
      Reinforcement Learning (RL) needs a standard interface to talk to the robot. 
      This class turns the complex MuJoCo simulator into a simple "turn-based" game 
      the AI can play to learn how to help the human.

    HOW IT WORKS (step by step):
      1. Loads the doc_ock robot 3D model from the MJCF file.
      2. Initializes the "Action Space" (wheels/motors) and "Observation Space" (eyes/sensors).
      3. In every Step: Moves the robot motors, checks for collisions, and calculates a score.
      4. Tracks synthetic muscle activity (EMG) to simulate physical interaction.

    EXAMPLE USAGE:
      env = SRLEnv(render_mode="human")
      obs, info = env.reset()
      for _ in range(1000):
          action = env.action_space.sample()
          obs, reward, done, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, task_mode: str = "collaborative", render_mode: Optional[str] = None):
        """
        WHAT THIS DOES: Sets up the simulation world, model, and interface boundaries.
        
        WHY: To ensure the AI knows exactly how many 'buttons' it can press and what it can 'see'.
        
        ARGS:
          task_mode (str): Defines the goal (e.g. 'carry', 'reach').
          render_mode (str): How to draw the world ('human' for window, 'rgb_array' for pixels).
        
        RETURNS:
          (None)
        """
        super().__init__()
        
        # Store configuration for behavior variation in reward functions
        self.task_mode = task_mode
        self.render_mode = render_mode
        
        # Load the physical definition of the robot (segments, weights, joints)
        model_path = os.path.join(os.path.dirname(__file__), "../../models/mjcf/srl_robot.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        # Create a workspace to hold current joint positions and speeds
        self.data = mujoco.MjData(self.model)
        
        # DEFINE ACTION BOUNDARIES
        # 16 ARM JOINTS + 2 GRIPPERS - We use -1 to 1 to simplify AI learning math
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        
        # DEFINE VISION BOUNDARIES
        # 128 dimensions covering positions, speeds, forces, and EMG signals
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)
        
        # Initialize rendering engine if requested by the user
        self.renderer = None
        if self.render_mode == "human":
            # launch_passive allows the simulation loop to continue while the window is open
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        WHAT THIS DOES: Teleports the robot back to its "Ready" position and resets the clock.
        
        WHY: RL training relies on thousands of 'episodes'; we must start with a clean slate every time.
        
        ARGS:
          seed (int): Optional number to make 'random' movements repeatable.
          options (dict): Custom start logic.
        
        RETURNS:
          observation (np.ndarray): The starting state of the robot.
          info (dict): Extra debug data.
        """
        # Ensure repeatable randomness if a seed is provided
        super().reset(seed=seed)
        
        # Wipe all joint history and positions back to XML defaults
        mujoco.mj_resetData(self.model, self.data)
        
        # Jitter the human base slightly to ensure the AI doesn't just memorize one exact spot
        if seed is not None:
            np.random.seed(seed)
        self.data.qpos[0:3] += np.random.uniform(-0.05, 0.05, size=3)
        
        # Establish the document standard base pose: First 2 joints of each arm at 0.3 rad
        # to ensure the arms aren't clipping into the backpack on frame 0.
        self.data.qpos[7:9] = 0.3
        self.data.qpos[13:15] = 0.3
        
        # Propagate pose changes throughout the kinematic chain
        mujoco.mj_forward(self.model, self.data)
        
        # Get the first 'photograph' of the world
        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        WHAT THIS DOES: Advances the clock, applies forces, and returns the result.
        
        WHY: This is the core "pulse" of the simulation.
        
        ARGS:
          action (np.ndarray): Target positions for the 18 motors.
        
        RETURNS:
          observation (np.ndarray): New state.
          reward (float): How good this action was (- points for hitting heads).
          done (bool): If the task is finished.
          truncated (bool): If time ran out.
          info (dict): Collision and energy telemetry.
        
        MATH/LOGIC:
          - Scaling: AI outputs -1 to 1; we scale by PI (3.14) to reach full joint range.
          - Substepping: We run 5 physics steps per 1 logic step to prevent the arms from 
            vibrating or 'exploding' due to high forces.
        """
        # Convert AI's abstract [-1, 1] intention to real-world joint angles
        self.data.ctrl[:16] = action[:16] * 3.14 
        
        # CORE PHYSICS LOOP
        # We step 5 times internally to ensure the solver resolves contacts smoothly
        for _ in range(5): 
            mujoco.mj_step(self.model, self.data)
        
        # Harvest new sensor data
        observation = self._get_obs()
        # Calculate the score for this transition
        reward = self._compute_reward(action)
        
        terminated = False 
        truncated = False
        
        # Package telemetry for debugging and dashboarding
        info = {
            "is_collision": self.data.ncon > 0, # check if any contact points exist
            "energy_use": np.sum(np.abs(self.data.ctrl)) # proxy for battery drain
        }
        
        # Sync the 3D window pixels if visual mode is active
        if self.render_mode == "human":
            self.viewer.sync()
            
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        WHAT THIS DOES: Generates a picture of the simulation.
        
        WHY: Used for recording proof-of-work videos and debugging.
        """
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                # Lazy initialization to save memory if not rendering
                self.renderer = mujoco.Renderer(self.model)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self):
        """
        WHAT THIS DOES: Safely kills the 3D window and cleans up memory.
        """
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.close()

    def _get_obs(self) -> np.ndarray:
        """
        WHAT THIS DOES: Compiles all physical data into a flat number array.
        
        WHY: AI models require a single fixed-length vector as input.
        
        RETURNS:
          obs (np.ndarray): 128-dimensional state vector.
        """
        obs = np.zeros(128, dtype=np.float32)
        
        # 1. TORSO STATE
        # Since the base is welded, we provide static coordinates to anchor the AI's spatial awareness
        obs[0:3] = [0.0, 0.0, 0.9]
        obs[3:7] = [1.0, 0.0, 0.0, 0.0] # Identity quaternion
        
        # 2. JOINT TELEMETRY
        # Extract Right and Left arm states (pos/vel) from the MuJoCo data structure
        obs[7:15] = self.data.qpos[0:8]   # 8 joint angles (Right)
        obs[15:23] = self.data.qvel[0:8]  # 8 joint speeds (Right)
        
        obs[23:31] = self.data.qpos[8:16]  # 8 joint angles (Left)
        obs[31:39] = self.data.qvel[8:16]  # 8 joint speeds (Left)
        
        # 3. FORCE PERCEPTION
        # Read touch/pressure sensors to detect if the arms are straining against an object
        obs[45:51] = self.data.sensordata[0:6]
        
        # 4. BIOLOGICAL OVERLAY
        # Inject the 'muscle' signals so the robot knows what its human counterpart is doing
        obs[43:51] = self._simulate_emg()
        
        return obs

    def _simulate_emg(self) -> np.ndarray:
        """
        WHAT THIS DOES: Creates fake muscle signals based on how fast the arms move.
        
        WHY: Used for training the Intent layer before real EMG hardware is connected.
        
        RETURNS:
          emg (np.ndarray): 8-channel signal vector.
        
        MATH:
          - (ABS(Speed) + GaussianNoise) creates a signature that looks like real firing neurons.
        """
        # Map the first 8 right-arm velocities to 8 EMG channels
        vells = self.data.qvel[0:8] 
        # Add 5% white noise to prevent the AI from becoming too rigid
        noise = np.random.normal(0, 0.05, size=8)
        emg = np.abs(vells) + noise
        return emg.astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        """
        WHAT THIS DOES: calculates a score between -Infinity and +Infinity.
        
        WHY: High rewards tell the robot "do more of this"; low rewards mean "stop doing this!"
        
        ARGS:
          action (np.ndarray): current motor commands.
          
        RETURNS:
          (float): Total weighted score.
        
        MATH/LOGIC:
          - Assistance: Negative distance from target (higher=better).
          - Collision: Massive penalty (-10) to force the AI to respect human safety.
          - Energy: Scolds the AI for vibrating the motors (-0.01 * square of power).
        """
        # 1. Proximity Reward: Penalize the absolute distance from zero position (placeholder)
        assistance_reward = -np.sum(np.abs(self.data.qpos[7:19])) * 0.1
        
        # 2. Safety First: Any contact results in a heavy loss to ensure 'Safety-by-Design'
        collision_penalty = -10.0 if self.data.ncon > 0 else 0.0
        
        # 3. Efficiency: Penalize high-effort actions to encourage smooth, battery-saving paths
        energy_penalty = -np.sum(np.square(action)) * 0.01
        
        return assistance_reward + collision_penalty + energy_penalty

if __name__ == "__main__":
    # BOOTSTRAP TEST LOOP
    env = SRLEnv(render_mode=None)
    obs, _ = env.reset()
    print(f"Env initialized. Observation shape: {obs.shape}")
    
    # Run 10 random steps to verify the physics engine isn't crashing
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, *_ = env.step(action)
        print(f"Step Reward: {reward:.4f}")
    
    env.close()
    print("Test complete.")
