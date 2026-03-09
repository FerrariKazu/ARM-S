# ═══════════════════════════════════════════════════════
# FILE: dashboard.py
# PURPOSE: The "Mission Control" visualizer; provides a real-time HUD for robot health.
# LAYER: Monitoring Layer (Main GUI)
# INPUTS: High-speed telemetry from all 4 system layers (Sensing -> Execution).
# OUTPUTS: An animated, dark-themed 7-panel performance dashboard.
# CALLED BY: Manual developer execution (e.g., ./present.sh)
# ═══════════════════════════════════════════════════════

#!/usr/bin/env python3
import matplotlib
# CRITICAL: Force 'TkAgg' backend to ensure live animation window opens correctly on WSL2/Linux
matplotlib.use('TkAgg')

import os
import datetime
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# --- SENSOR CONFIGURATION ---
# These parameters define the 'Resolution' of the visual graphs.
T = 300       # Number of visible horizontal points in the wave charts
N_CH = 16    # Number of muscle channels to display
# Create varying wave speeds so the graph doesn't look like a single repeating line
FREQS = np.linspace(0.5, 4.0, N_CH)
PHASES = np.random.uniform(0, 2*np.pi, N_CH)
# Vertical gap between stacked muscle waves for readability
OFFSETS = np.arange(N_CH) * 0.8
emg_t = [0.0]  # Global time tracker for the wave animation

# --- DESIGN SYSTEM ---
# These HSL-adjacent hex codes create the "Cyberpunk / Tech" aesthetic requested.
BG_COLOR = "#050A0F"    # Deep space black/blue
CARD_COLOR = "#0D1721"  # Slightly lighter panel background
TEXT_COLOR = "#E0E6ED"  # Crisp white-blue text
CYAN = "#00F5FF"        # System Highlight / Status Glow
ACCENT = "#FF3131"      # Warning / Error Red
GREEN = "#00FF66"       # Success / Safety Green

class SRLDashboard:
    """
    WHAT THIS CLASS IS:
      The central Graphical User Interface (GUI). It acts as the 'Display Driver' 
      for the entire robot simulation.

    WHY IT EXISTS:
      Robotics is complex. A human cannot understand 16 muscle channels and 
      dual-arm coordinates by reading raw text. This dashboard translates 
      those millions of numbers into 'Pictures' that a human can verify instantly.

    HOW IT WORKS (step by step):
      1. Initializes a 3x3 grid using Matplotlib GridSpec.
      2. Creates 6 specialized panels (Workspace, EMG, AI Intent, Latency, etc.).
      3. Starts a high-speed 'Animate' loop that redraws the screen 10 times a second.
      4. Manages 'Logit' math to ensure AI predictions look smooth and don't flicker.

    EXAMPLE USAGE:
      dash = SRLDashboard()
      plt.show() # Opens the mission control window
    """
    def __init__(self):
        """
        WHAT THIS DOES: Boots the visual engine and applies the project 'Theme'.
        """
        # Apply the dark-mode text palette to all graph elements
        plt.rcParams['text.color'] = TEXT_COLOR
        plt.rcParams['axes.labelcolor'] = TEXT_COLOR
        plt.rcParams['xtick.color'] = TEXT_COLOR
        plt.rcParams['ytick.color'] = TEXT_COLOR
        
        # Create the master window (The Stage)
        self.fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
        # Partition the window into a 3x3 layout
        self.gs = GridSpec(3, 3, figure=self.fig)
        self.fig.suptitle("ARM-S: Autonomous Robotic Motion System — SRL Prototype", 
                          fontsize=22, fontweight='bold', y=0.98, color=CYAN)
        
        # DATA INITIALIZATION
        self.intent_classes = ["reach_left", "reach_right", "lift", "hold", "release", "stabilize", "handoff"]
        # Logits represent 'Uncertainty'—we track them to create smooth sliding animations between AI guesses
        self.current_logits = np.zeros(7)
        # Pre-calculate the robot's physical reach boundary (The 'Bubble')
        self.workspace_points = self._generate_workspace()
        
        # ACTIVATE PANELS
        self._init_workspace_panel()    # The 3D view
        self._init_emg_panel()          # The Live waves
        self._init_intent_panel()       # The AI brain view
        self._init_latency_panel()      # The Performance view
        self._init_architecture_panel() # The System map
        self._init_dataset_panel()      # The History view
        
        # DRAW FOOTER
        today = datetime.date.today().strftime("%Y-%m-%d")
        self.fig.text(0.5, 0.02, f"Phase 3 — Verified | Deployment: WSL2-SRL-PROTOTYPE | Date: {today}", 
                     ha='center', fontsize=10, color="#506070")
        
        # Force a layout recalculation so panels don't overlap headers
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _generate_workspace(self):
        """
        WHAT THIS DOES: Calculates the 'Safety Bubble' in 3D.
        
        WHY: To visualize exactly where the robot arm is physically able to reach.
        
        MATH:
          - Generates 2000 random points in a sphere.
          - Filters out anything 'Behind' the human (x < -0.1).
          - Colors dots by distance from the shoulder mount.
        """
        num_pts = 2000
        # Sample points in a torso-centered envelope
        phi = np.random.uniform(0, 2 * np.pi, num_pts)
        theta = np.random.uniform(0, np.pi, num_pts)
        rad = np.random.uniform(0.2, 0.85, num_pts) # Reach up to 85cm
        
        # Convert Sphere coordinates to 3D Cartesian [X, Y, Z]
        x = rad * np.sin(theta) * np.cos(phi)
        y = rad * np.sin(theta) * np.sin(phi)
        z = rad * np.cos(theta) + 1.0 # Lift the bubble to shoulder height (1.0m)
        
        # RELEVANCE FILTER: Remove points that would be 'Inside' the user's torso
        mask = (x > -0.1) | (np.abs(y) > 0.2)
        pts = np.column_stack([x[mask], y[mask], z[mask]])
        # Find distance for heatmap color scaling
        dist = np.linalg.norm(pts - np.array([-0.1, 0, 1.0]), axis=1)
        return pts, dist

    def _init_workspace_panel(self):
        """PURPOSE: 3D Visualization of reachable space."""
        self.ax_work = self.fig.add_subplot(self.gs[0:2, 0], projection='3d', facecolor=BG_COLOR)
        self.ax_work.set_title("Robot Workspace Envelope", fontsize=14, color=CYAN)
        pts, dist = self.workspace_points
        # viridis cmap: Yellow = Close/Safe, Blue = Far/Extended
        self.scatter = self.ax_work.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=dist, cmap='viridis', s=2, alpha=0.6)
        self.ax_work.set_xlim(-1, 1); self.ax_work.set_ylim(-1, 1); self.ax_work.set_zlim(0, 1.5)
        self.ax_work.axis('off') # Cleaner look for the 3D 'Ghost'

    def _init_emg_panel(self):
        """PURPOSE: Stacked waveform view of muscle electricity."""
        self.ax_emg = self.fig.add_subplot(self.gs[0, 1:3], facecolor=CARD_COLOR)
        self.ax_emg.set_title("Live EMG Signals (16 Channels)", fontsize=14, color=CYAN)
        
        self.emg_lines = []
        colors = plt.cm.tab20(np.linspace(0, 1, 16))
        x = np.linspace(0, 6, T) # Draw 6 seconds of history
        for i in range(N_CH):
            # Pre-draw the wavy lines at their vertical offsets
            y = 0.3 * np.sin(2*np.pi*FREQS[i]*x + PHASES[i]) + OFFSETS[i]
            line, = self.ax_emg.plot(x, y, color=colors[i], lw=0.8, alpha=0.9)
            self.emg_lines.append(line)

        self.ax_emg.set_ylim(-1, 14)
        self.ax_emg.set_xlim(0, 6)
        self.ax_emg.set_axis_off()

    def _init_intent_panel(self):
        """PURPOSE: Bar chart of AI's current decision process."""
        self.ax_intent = self.fig.add_subplot(self.gs[1, 1], facecolor=CARD_COLOR)
        self.ax_intent.set_title("Intent Prediction", fontsize=14, color=CYAN)
        # horizontal bars initialized to 0%
        self.intent_bars = self.ax_intent.barh(self.intent_classes, np.zeros(7), color='#2A3F5F')
        self.ax_intent.set_xlim(0, 1)
        self.ax_intent.spines['top'].set_visible(False)
        self.ax_intent.spines['right'].set_visible(False)

    def _init_latency_panel(self):
        """PURPOSE: Real-time 'Reaction Time' auditor."""
        self.ax_lat = self.fig.add_subplot(self.gs[1, 2], facecolor=CARD_COLOR)
        self.ax_lat.set_title("Latency Budget (ms)", fontsize=14, color=CYAN)
        stages = ["Sense", "Intent", "Plan", "Exec"]
        budget = [5, 10, 15, 5]
        actual = [3.2, 8.5, 12.1, 4.4] # Mock data placeholder
        
        # Draw background 'Shadow' bars for the limit
        self.ax_lat.barh(stages, budget, color='#303030', label='Budget', alpha=0.5)
        # Draw active performance bars (GREEN = OK)
        self.ax_lat.barh(stages, actual, color=GREEN, label='Actual', height=0.6)
        self.ax_lat.set_xlim(0, 20)
        self.ax_lat.legend(frameon=False, loc='lower right', fontsize=8)

    def _init_architecture_panel(self):
        """PURPOSE: Schematic flow-chart of the software brain."""
        self.ax_arch = self.fig.add_subplot(self.gs[2, 0:2], facecolor=CARD_COLOR)
        self.ax_arch.set_title("Pipeline Architecture", fontsize=14, color=CYAN)
        self.ax_arch.set_xlim(0, 100); self.ax_arch.set_ylim(0, 40)
        self.ax_arch.axis('off')
        
        layers = ["Sensing", "Intent", "Planning", "Execution"]
        msgs = ["BodyState", "IntentPred", "Trajectory", "PWM/Pos"]
        
        # Procedurally draw boxes and arrows
        for i, (layer, msg) in enumerate(zip(layers, msgs)):
            # Create high-quality rounded rects
            rect = patches.FancyBboxPatch((5 + i*22, 15), 18, 10, boxstyle="round,pad=1", facecolor='#203040', edgecolor=CYAN)
            self.ax_arch.add_patch(rect)
            self.ax_arch.text(14 + i*22, 20, layer, ha='center', va='center', fontweight='bold')
            if i < 3:
                # Add flow arrows and data-packet labels between boxes
                self.ax_arch.arrow(23 + i*22, 20, 4, 0, head_width=2, head_length=1, fc=GREEN, ec=GREEN)
                self.ax_arch.text(25 + i*22, 24, msg, fontsize=8, color="#A0A0A0", ha='center')

    def _init_dataset_panel(self):
        """PURPOSE: Breakdown of historical training data."""
        self.ax_data = self.fig.add_subplot(self.gs[2, 2], facecolor=CARD_COLOR)
        self.ax_data.set_title("Dataset Statistics (v1)", fontsize=14, color=CYAN)
        
        labels = self.intent_classes
        counts = [1276, 1224, 1301, 3699, 0, 2500, 0] 
        
        self.ax_data.bar(labels, counts, color='#3498DB')
        self.ax_data.set_xticks(range(7))
        self.ax_data.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        self.ax_data.spines['top'].set_visible(False)
        self.ax_data.spines['right'].set_visible(False)

    def animate(self, i):
        """
        WHAT THIS DOES: The "Heartbeat" of the dashboard.
        
        WHY: To make the charts feel 'Alive' and show movement in real-time.
        """
        # 1. ROTATE 3D VIEW: Spins the workspace bubble for a perspective sweep
        self.ax_work.view_init(elev=20, azim=i*0.5)
        
        # 2. SCROLLING EMG WAVES: Slides the waveform data to the left
        emg_t[0] += 0.05  # advance virtual clock by 50ms
        t = emg_t[0]
        x = np.linspace(t, t + 6, T)
        for ch, line in enumerate(self.emg_lines):
            # Calculate new wave position + random jitter to simulate muscle heat
            y = 0.3 * np.sin(2*np.pi*FREQS[ch]*x + PHASES[ch])
            y += 0.05 * np.random.randn(T) 
            line.set_xdata(x)
            line.set_ydata(y + OFFSETS[ch])
        self.ax_emg.set_xlim(t, t + 6)
            
        # 3. SMOOTH BRAIN TRANSITIONS: 
        # Simulated AI noise that periodically 'Spikes' into a confident decision.
        self.current_logits += np.random.randn(7) * 0.05
        if i > 0 and i % 120 == 0:
            spike_idx = np.random.randint(7)
            self.current_logits[spike_idx] += 2.5
            
        # MATH: Convert raw scores into % (Softmax)
        exp_logits = np.exp(self.current_logits - np.max(self.current_logits)) 
        probs = exp_logits / np.sum(exp_logits)
        
        # VISUAL EFFECTS: Pulse the winning bar for 'Cinematic' feedback
        pulse = 0.7 + 0.3 * np.abs(np.sin(i * 0.5))
        winner_idx = np.argmax(probs)
        
        for idx, bar in enumerate(self.intent_bars):
            bar.set_width(probs[idx])
            if idx == winner_idx:
                bar.set_color(CYAN) # Winning prediction glows Cyan
                bar.set_alpha(pulse) # Winning prediction pulses
            else:
                bar.set_color('#2A3F5F')
                bar.set_alpha(0.6)
            
        return self.emg_lines + list(self.intent_bars)

if __name__ == "__main__":
    # BOOTSTRAPP MISSION CONTROL
    dash = SRLDashboard()
    # 360 frames @ 10Hz = 36 second total animation loop
    dash.fig.anim = FuncAnimation(dash.fig, dash.animate, frames=360, interval=100, blit=True)
    # Block and show the window
    plt.show()
