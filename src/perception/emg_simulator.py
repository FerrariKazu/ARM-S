# ═══════════════════════════════════════════════════════
# FILE: src/perception/emg_simulator.py
# PURPOSE: Bridges the gap between Physics and Biology by creating realistic 'Muscle signals'.
# LAYER: Sensing Layer
# INPUTS: Robot joint speeds (qvel) and motor forces (qfrc) from MuJoCo.
# OUTPUTS: A 16-channel stream of synthetic electrical pulses (EMG).
# CALLED BY: SRL Sensor Pipeline, Intent Recognition Model.
# ═══════════════════════════════════════════════════════

import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple
from collections import deque
import matplotlib.pyplot as plt

class EMGSimulator:
    """
    WHAT THIS CLASS IS:
      A "Bio-Signal Generator". It looks at how hard the simulated robot is 
      working and generates fake electricity signals that look exactly like 
      what real human muscle sensors would output.

    WHY IT EXISTS:
      Actual EMG sensors are noisy, drift over time, and are hard to wear. 
      This class allows us to train our AI models in a perfect simulation 
      before we ever have to strap a real user into the robotic hardware.

    HOW IT WORKS (step by step):
      1. Reads the current speed and force of every joint in the robot.
      2. Maps those physical numbers to specific muscles (e.g., Biceps, Triceps).
      3. Adds "Real World Problems" like electrical noise and sensor drift.
      4. Stores a rolling 500ms history window for the AI to 'look back' on.

    EXAMPLE USAGE:
      sim = EMGSimulator(physics_data)
      signal_frame = sim.update()
      history = sim.get_window()
    """
    
    def __init__(self, data: mujoco.MjData, sampling_rate_hz: int = 50, window_ms: int = 500):
        """
        WHAT THIS DOES: Configures the sensor array and digital memory.
        
        WHY: To establish the 'Resolution' and 'Memory Depth' of our bio-intelligence.
        
        ARGS:
          data (obj): The live MuJoCo physics workspace.
          sampling_rate_hz (int): How many sensor checks per second (50Hz = standard).
          window_ms (int): How much history to remember (500ms = human reaction time).
        """
        self.data = data
        self.fs = sampling_rate_hz
        # Convert milliseconds (500) into frame counts (25)
        self.window_size = int((window_ms / 1000) * self.fs)
        self.num_channels = 16
        
        # DOUBLE-ENDED QUEUE (DEQUE): Efficiently discards old data as new data arrives
        self.buffer = deque(maxlen=self.window_size)
        
        # Initialize the 'Short term memory' with flatlines (zeros)
        for _ in range(self.window_size):
            self.buffer.append(np.zeros(self.num_channels))
            
        # Simulation playback state
        self.time = 0.0
        # Simulated electrode movement - 0.01Hz wave causes slow signal 'wandering'
        self.drift_freq = 0.01 
        # Physical saturation point: real sensors max out at 5.0 Volts
        self.hardware_limit = 5.0 
        
        # ANATOMY MAP: 8 sensors on the Human Right arm, 8 on the Left
        self.labels = [
            "R_Bicep", "R_Tricep", "R_Flex1", "R_Flex2", "R_Flex3", "R_Ext1", "R_Ext2", "R_Ext3",
            "L_Bicep", "L_Tricep", "L_Flex1", "L_Flex2", "L_Flex3", "L_Ext1", "L_Ext2", "L_Ext3"
        ]

    def update(self) -> np.ndarray:
        """
        WHAT THIS DOES: Calculates the 'Sparks' of electricity for the current millisecond.
        
        WHY: This is the primary sensor loop that feeds the AI Intent layer.
            
        RETURNS:
          clipped (np.ndarray): 16 channels of noisy, drifted sensor data.
        
        MATH/LOGIC:
          - Signal = (Velocity * Gain) + (Force * Sensitivity) + Noise.
          - We use Absolute Value because EMG voltage oscillates around zero.
        """
        self.time += 1.0 / self.fs
        
        # Grab high-resolution physical state from MuJoCo
        qvel = self.data.qvel[0:16] # Joint speeds
        forces = self.data.qfrc_actuator[0:16] # Amount of motor torque currently used
        
        raw_signal = np.zeros(self.num_channels)
        
        # --- BIO-MAPPING LOGIC ---
        # Right Arm (Channels 0-7)
        # Scaled by 0.8 to leave 20% headroom below hardware saturation
        raw_signal[0] = np.abs(qvel[0]) * 0.8 + np.abs(forces[0]) * 0.05 
        raw_signal[1] = np.abs(qvel[1]) * 0.7 + np.abs(forces[1]) * 0.05 
        # Mix in randomness to simulate user variations in grip strength
        raw_signal[2:5] = np.abs(qvel[2]) * np.random.uniform(0.5, 0.9, size=3) 
        raw_signal[5:8] = np.abs(qvel[3]) * np.random.uniform(0.4, 0.8, size=3) 
        
        # Left Arm (Channels 8-15)
        raw_signal[8] = np.abs(qvel[8]) * 0.8 + np.abs(forces[8]) * 0.05 
        raw_signal[9] = np.abs(qvel[9]) * 0.7 + np.abs(forces[9]) * 0.05
        raw_signal[10:13] = np.abs(qvel[10]) * np.random.uniform(0.5, 0.9, size=3) 
        raw_signal[13:16] = np.abs(qvel[11]) * np.random.uniform(0.4, 0.8, size=3)

        # --- REAL-WORLD ARTIFACTS ---
        # 1. GAUSSIAN NOISE: Simulates skin impedance and radio interference
        noise = np.random.normal(0, 0.05, size=self.num_channels)
        
        # 2. ELECTRODE DRIFT: Simulates the sensor slowly unsticking from the skin
        drift = 0.2 * np.sin(2 * np.pi * self.drift_freq * self.time)
        
        # 3. MOTION ARTIFACTS: Large physical jolts cause sudden 'Spikes' in voltage
        qacc_norm = np.linalg.norm(self.data.qacc[0:16])
        motion_spike = 0.0
        if qacc_norm > 5.0:
            # If the robot arm vibrates too hard, the sensor goes haywire temporarily
            motion_spike = np.random.uniform(0, 0.5)
            
        # Composite the final noisy signal
        combined = raw_signal + noise + drift + motion_spike
        
        # 4. VOLTAGE LIMIT: Physically, the amplifier cannot output more than 5.0V
        clipped = np.clip(combined, -self.hardware_limit, self.hardware_limit)
        
        # Add the new frame to our sliding history window
        self.buffer.append(clipped)
        return clipped

    def get_window(self) -> np.ndarray:
        """
        WHAT THIS DOES: Returns the last 500ms of signal as a single block.
        
        WHY: Sequence-based AI models (Transformers) need to see 'Trends', not just 'Now'.
            
        RETURNS:
          (np.ndarray): matrix of shape (25, 16).
        """
        return np.array(self.buffer)

    def visualize_signals(self, ax: plt.Axes):
        """
        WHAT THIS DOES: Paints the 16 channels onto a diagnostic chart.
        
        WHY: Humans need a visual way to check if the sensors are 'Live' and 'Healthy'.
        
        ARGS:
          ax (matplotlib axis): The canvas to draw the waveforms on.
        """
        window = self.get_window()
        # Create a time axis from -500ms to 0ms (Present Day)
        t = np.linspace(-500, 0, self.window_size)
        
        ax.clear()
        # STACKED PLOTTING: We add vertical offsets (i*2.5) so signals don't overlap
        for i in range(self.num_channels):
            ax.plot(t, window[:, i] + (i * 2.5), label=self.labels[i])
            
        ax.set_title("16-Channel EMG Simulation (500ms Window)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude / Channel Offset")
        ax.grid(True, alpha=0.3)
        # Position the text labels exactly next to each corresponding wave
        ax.set_yticks(np.arange(0, self.num_channels * 2.5, 2.5))
        ax.set_yticklabels(self.labels)

if __name__ == "__main__":
    # VALIDATION DEMO
    import os
    model_path = os.path.join(os.path.dirname(__file__), "../../models/mjcf/srl_robot.xml")
    try:
        # Boot the physics engine
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        # Start the Bio-Sim
        sim = EMGSimulator(data)
        
        print("Generating 5 seconds of EMG data...")
        # Simulate vigorous random movement for testing
        data.qvel[0:16] = np.random.uniform(-2, 2, size=16)
        
        # Execute 250 logic steps (5s @ 50Hz)
        for i in range(250): 
            # Inject jitter into velocities to create an organic waveform
            data.qvel[0:16] += np.random.normal(0, 0.1, size=16)
            mujoco.mj_forward(model, data)
            sim.update()
            
        # Draw the final result for human verification
        fig, ax = plt.subplots(figsize=(12, 8))
        sim.visualize_signals(ax)
        plt.tight_layout()
        
        # Save a screenshot for the project documentation
        os.makedirs("../../artifacts", exist_ok=True)
        plt.savefig("../../artifacts/emg_demo.png")
        print(f"Demo plot saved to artifacts/emg_demo.png")
        
    except Exception as e:
        print(f"Error during demo: {e}")
