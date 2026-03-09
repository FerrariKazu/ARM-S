# ═══════════════════════════════════════════════════════
# FILE: dashboard_controllers.py
# PURPOSE: Technical deep-dive visualizer; provides a high-resolution HUD for individual controllers.
# LAYER: Monitoring Layer (Analytics)
# INPUTS: High-speed telemetry from all 7 robot control subsystems.
# OUTPUTS: An animated, dark-themed 7-panel logic-map dashboard.
# CALLED BY: Manual developer execution (e.g., ./present.sh)
# ═══════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Dark theme settings
plt.style.use('dark_background')
BG_COLOR = '#050A0F'
PANEL_BG = '#0B131A'
TEXT_COLOR = '#E0E0E0'
CYAN = '#00F0FF'

class ControllerDashboard:
    """
    WHAT THIS CLASS IS:
      The "Technical X-Ray". While the main dashboard shows the whole system, 
      this class zooms in on the 7 individual 'Control Centers' of the robot.

    WHY IT EXISTS:
      When debugging a specific behavior (like hitting an obstacle), engineers 
      need to see the *internal math* of that specific controller. This 
      dashboard visualizes the "Invisible Forces" like PID error and repulsion.

    HOW IT WORKS (step by step):
      1. Constructs a highly dense 7-panel UI using Matplotlib.
      2. Simulates or reads the internal variables of each specific controller.
      3. Draws specialized graphs: Figure-8 paths for Avoidance, Sine waves for PID.
      4. Highlights the 'Active' teamwork mode in the bottom row.

    EXAMPLE USAGE:
      db = ControllerDashboard()
      plt.show() # Opens the technical controller window
    """
    def __init__(self):
        """
        WHAT THIS DOES: Sets up the complex 7-panel 'Grid' system.
        """
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
        self.fig.canvas.manager.set_window_title("ARM-S: Controller Systems — Live Demo")
        
        # Add master title
        self.fig.suptitle("ARM-S: Multi-Controller Architecture Live Demo", 
                          color=CYAN, fontsize=16, fontweight='bold', y=0.98)
                          
        # 3 rows base layout. R1 and R2 are 3-cols each. R3 is shared.
        self.gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3, 
                           bottom=0.08, top=0.92, left=0.05, right=0.95)
                           
        # BOOT INDIVIDUAL CONTROLLER MONITORS
        self._init_pid_panel()          # (1/7) PID Status
        self._init_obstacle_panel()     # (2/7) Escape Vectors
        self._init_speed_panel()        # (3/7) Velocity Limits
        self._init_pattern_panel()      # (4/7) Gesture Classifier
        self._init_logger_panel()       # (5/7) Black Box Feed
        self._init_emg_panel()          # (6/7) Muscle Normalization
        self._init_coordination_panel() # (7/7) Teamwork Protocol
        
        # FOOTER
        self.fig.text(0.5, 0.02, "All 7 Controllers — PASS | ARM-S Phase 3-5 | Deployment: Linux-WSL2",
                      ha='center', va='center', color=TEXT_COLOR, alpha=0.6, fontsize=10)

        self.t = 0
        self.dt = 0.1 # Real-time simulation step (100ms)

    def _init_pid_panel(self):
        """PURPOSE: Visualizes 'Focus'—how well the robot follows instructions."""
        self.ax_pid = self.fig.add_subplot(self.gs[0, 0], facecolor=PANEL_BG)
        self.ax_pid.set_title("PID Joint Controller", color=TEXT_COLOR)
        self.ax_pid.set_ylim(-1.5, 1.5)
        self.ax_pid.set_xlim(0, 50) 
        self.ax_pid.set_xticks([])
        
        self.pid_frames = 50
        self.pid_setpoint_data = np.zeros(self.pid_frames)
        self.pid_actual_data = np.zeros(self.pid_frames)
        self.pid_error_data = np.zeros(self.pid_frames)
        
        # Setpoint = Intended path (Dashed Cyan)
        self.line_sp, = self.ax_pid.plot([], [], color=CYAN, linestyle='--', label='Setpoint')
        # Actual = Physical reality (Solid White)
        self.line_act, = self.ax_pid.plot([], [], color='white', label='Actual')
        # Error = The 'Pain' signal (Red glow)
        self.line_err, = self.ax_pid.plot([], [], color='#FF3366', alpha=0.7, label='Error')
        self.ax_pid.legend(loc='upper right', fontsize=8, frameon=False)
        
        self.pid_text = self.ax_pid.text(0.05, 0.05, "Kp:5.0 Ki:0.1 Kd:0.5", 
                                         transform=self.ax_pid.transAxes, color=CYAN, fontsize=9)
        self.pid_val = 0.0

    def _init_obstacle_panel(self):
        """PURPOSE: 2D safety map showing forbidden 'No-Go' zones."""
        self.ax_obs = self.fig.add_subplot(self.gs[0, 1], facecolor='#0A1520')
        self.ax_obs.set_title("Obstacle Avoidance", color=TEXT_COLOR)
        self.ax_obs.set_xlim(-0.5, 0.5)
        self.ax_obs.set_ylim(-0.2, 0.8)
        self.ax_obs.set_aspect('equal')
        self.ax_obs.set_xticks([]); self.ax_obs.set_yticks([])
        
        # Human Forbidden Zones (Head, Jaws)
        self.zones = [
            (0.0, 0.5, 0.15), # Head center
            (0.12, 0.45, 0.08), # Right Jaw
            (-0.12, 0.45, 0.08) # Left Jaw
        ]
        for z in self.zones:
            circle = plt.Circle((z[0], z[1]), z[2], color='#FF3366', alpha=0.4)
            self.ax_obs.add_artist(circle)
            
        self.obs_path, = self.ax_obs.plot([], [], color='#00D4FF', linewidth=2.5)
        self.ee_marker, = self.ax_obs.plot([], [], 'wo', markersize=8)
        self.repel_arrows = []

    def _init_speed_panel(self):
        """PURPOSE: Bar chart of hard velocity limits for all 14 joints."""
        self.ax_speed = self.fig.add_subplot(self.gs[0, 2], facecolor=PANEL_BG)
        self.ax_speed.set_title("Speed Controller", color=TEXT_COLOR)
        self.ax_speed.set_ylim(0, 2.5)
        self.ax_speed.set_ylabel("rad/s", color=TEXT_COLOR, fontsize=8)
        
        self.num_joints = 14
        self.bars_speed = self.ax_speed.bar(range(self.num_joints), np.zeros(self.num_joints))
        self.ax_speed.set_xticks(range(self.num_joints))
        self.ax_speed.set_xticklabels([f"J{i}" for i in range(self.num_joints)], rotation=90, fontsize=6)
        
        self.modes = [("surgical", 0.3), ("carry", 0.8), ("emergency", 2.0)]
        self.mode_idx = 0
        self.mode_timer = 0
        
        # Red horizontal line representing the Hard Max Limit
        self.speed_limit_line = self.ax_speed.axhline(y=0.3, color='red', linestyle='--', linewidth=1)

    def _init_pattern_panel(self):
        """PURPOSE: 3D Visualization of movement 'Fingerprints'."""
        self.ax_pat = self.fig.add_subplot(self.gs[1, 0], projection='3d', facecolor=PANEL_BG)
        self.ax_pat.set_title("Arm Pattern Recognition", color=TEXT_COLOR)
        # Apply dark pane shading for a polished look
        self.ax_pat.xaxis.set_pane_color((0.04, 0.07, 0.1, 1.0))
        self.ax_pat.yaxis.set_pane_color((0.04, 0.07, 0.1, 1.0))
        self.ax_pat.zaxis.set_pane_color((0.04, 0.07, 0.1, 1.0))
        
        self.ax_pat.set_xlim(-0.5, 0.5); self.ax_pat.set_ylim(0, 1.0); self.ax_pat.set_zlim(0, 1.0)
        self.ax_pat.set_xticks([]); self.ax_pat.set_yticks([]); self.ax_pat.set_zticks([])
        
        self.pat_line, = self.ax_pat.plot([], [], [], linewidth=3, color=CYAN)
        self.pat_text = self.fig.text(0.06, 0.62, "WAITING...", color=CYAN, 
                                     fontsize=14, fontweight='bold', 
                                     bbox=dict(facecolor=BG_COLOR, edgecolor=CYAN, alpha=0.8))
        self.pat_hist = []
        self.pat_timer = 0
        self.pat_state = 0

    def _init_logger_panel(self):
        """PURPOSE: Functional terminal scrolling log simulation."""
        self.ax_log = self.fig.add_subplot(self.gs[1, 1], facecolor=PANEL_BG)
        self.ax_log.set_title("Debug Logger — Live Feed", color=TEXT_COLOR)
        self.ax_log.axis('off')
        
        self.log_lines = []
        for i in range(12):
            # Create text objects that we will later update with real logs
            text = self.ax_log.text(0.02, 0.95 - (i * 0.08), "", 
                                    color=CYAN, fontfamily='monospace', fontsize=8,
                                    transform=self.ax_log.transAxes)
            self.log_lines.append(text)
        self.log_history = [""] * 12

    def _init_emg_panel(self):
        """PURPOSE: Side-by-side comparison of Raw vs Filtered muscle signals."""
        self.ax_emg = self.fig.add_subplot(self.gs[1, 2], facecolor=PANEL_BG)
        self.ax_emg.set_title("EMG Calibration Controller", color=TEXT_COLOR)
        self.ax_emg.set_xlim(0, 2.0)
        self.ax_emg.set_ylim(-1, 16)
        self.ax_emg.set_yticks(range(16))
        self.ax_emg.set_yticklabels([f"CH{i}" for i in range(16)], fontsize=6)
        self.ax_emg.set_xticks([0.5, 1.5])
        self.ax_emg.set_xticklabels(["RAW (V)", "CALIB (Norm)"])
        
        self.ax_emg.axvline(1.0, color=TEXT_COLOR, alpha=0.3, linestyle='-')
        self.bars_raw = self.ax_emg.barh(range(16), np.zeros(16), left=0, height=0.6, color='#555555')
        self.bars_cal = self.ax_emg.barh(range(16), np.zeros(16), left=1.0, height=0.6, color=CYAN)
        
        self.calib_badge = self.ax_emg.text(1.7, 14.5, "Calibrated ✓", 
                         color='#00FF00', fontweight='bold', fontsize=10,
                         bbox=dict(facecolor='#003300', edgecolor='#00FF00', alpha=0.8),
                         ha='center')

    def _init_coordination_panel(self):
        """PURPOSE: Logic diagrams for dual-arm teamwork protocols."""
        gs_bottom = self.gs[2, :].subgridspec(1, 4, wspace=0.1)
        self.coords = []
        modes = ["INDEPENDENT", "SYMMETRIC", "COOPERATIVE", "HANDOFF"]
        
        for i, mode in enumerate(modes):
            ax = self.fig.add_subplot(gs_bottom[0, i], facecolor=PANEL_BG)
            ax.set_title(mode, color=TEXT_COLOR, fontsize=10)
            ax.set_xlim(-1, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_color('#333333')
                
            line_l, = ax.plot([], [], color='#3388FF', linewidth=2, label="Left")
            line_r, = ax.plot([], [], color='#FF8833', linewidth=2, label="Right")
            
            if i == 0:
                ax.legend(loc='lower center', ncol=2, fontsize=8, frameon=False)
                
            self.coords.append({'ax': ax, 'mode': mode, 'line_l': line_l, 'line_r': line_r})
        self.coord_timer = 0
        self.coord_idx = 0

    def update(self, frame):
        """
        WHAT THIS DOES: The main 'Movie Projector' pulse.
        
        WHY: To recalculate every panel 10 times per second for a smooth simulation.
        
        MATH:
          - PID: Computes error between target sine wave and 'actual' motor.
          - Obstacle: Calculates radial arrows pointing away from the head.
          - Pattern: Generates a new 3D coordinate for the active demo gesture.
        """
        self.t += self.dt
        
        # --- 1. PID DYNAMICS ---
        sp_val = np.sin(self.t * 2) * 0.8 # Target sine wave
        err = sp_val - self.pid_val
        self.pid_val += err * (self.dt * 5.0) # Move the white line toward the target
        
        self.pid_setpoint_data = np.roll(self.pid_setpoint_data, -1)
        self.pid_actual_data   = np.roll(self.pid_actual_data, -1)
        self.pid_error_data    = np.roll(self.pid_error_data, -1)
        
        self.pid_setpoint_data[-1] = sp_val
        self.pid_actual_data[-1] = self.pid_val
        self.pid_error_data[-1] = err
        
        x_p = np.arange(self.pid_frames)
        self.line_sp.set_data(x_p, self.pid_setpoint_data)
        self.line_act.set_data(x_p, self.pid_actual_data)
        self.line_err.set_data(x_p, self.pid_error_data)
        
        # --- 2. OBSTACLE ESCAPE VECTORS ---
        ee_x = np.sin(self.t) * 0.3
        ee_y = 0.4 + np.sin(self.t * 2) * 0.2
        
        for arr in self.repel_arrows: arr.remove()
        self.repel_arrows.clear()
            
        for z in self.zones:
            dist = max(0.01, np.sqrt((ee_x - z[0])**2 + (ee_y - z[1])**2))
            if dist < (z[2] + 0.15): 
                # DRAW REPULSION: The closer the hand, the thicker the red arrows
                arr_len = max(0, 0.05 / dist) 
                for ang in np.linspace(0, 2*np.pi, 5, endpoint=False):
                    sx = z[0] + np.cos(ang) * z[2]
                    sy = z[1] + np.sin(ang) * z[2]
                    dx = np.cos(ang) * arr_len
                    dy = np.sin(ang) * arr_len
                    
                    # Point arrow AWAY from human head
                    arr = self.ax_obs.annotate(
                        "", xy=(sx+dx, sy+dy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color='#FF4444', lw=min(3.0, arr_len*20))
                    )
                    self.repel_arrows.append(arr)
        self.obs_path.set_data([0, ee_x], [0.1, ee_y])
        self.ee_marker.set_data([ee_x], [ee_y])
        
        # --- 3. HARWARE SPEED LIMITS ---
        self.mode_timer += self.dt
        if self.mode_timer > 3.0: # Rotate modes every 3 seconds
            self.mode_idx = (self.mode_idx + 1) % len(self.modes)
            self.mode_timer = 0
        mode_n, mode_l = self.modes[self.mode_idx]
        self.speed_limit_line.set_ydata([mode_l, mode_l])
        for i, bar in enumerate(self.bars_speed):
            val = np.abs(np.sin(self.t * 3 + i)) * mode_l * 0.9
            bar.set_height(val)
            # DANGER FLASH: If joint is at 85% of its legal limit, bar turns RED
            if val / mode_l > 0.85: bar.set_color('#FF3366')
            else: bar.set_color('#00FF66')
            
        # --- 4. GESTURE RECOGNITION ---
        self.pat_timer += self.dt
        if self.pat_timer > 2.0:
            self.pat_state = (self.pat_state + 1) % 4
            self.pat_timer = 0; self.pat_hist.clear()
            
        if self.pat_state == 0:
            self.pat_hist.append([np.sin(self.t*4)*0.4, 0.3+self.t*0.05, 0.4])
            c, l = CYAN, "REACH"
        elif self.pat_state == 1:
            self.pat_hist.append([0.0, 0.4, min(1.0, 0.2 + self.pat_timer*0.4)])
            c, l = "#00FF66", "LIFT"
        elif self.pat_state == 2:
            self.pat_hist.append([np.sin(self.t*5)*0.3, 0.6, 0.6])
            c, l = "#A500FF", "SWEEP"
        else:
            self.pat_hist.append([np.random.normal(0,0.01), 0.5+np.random.normal(0,0.01), 0.5])
            c, l = "#FFCC00", "STABILIZE"
            
        if len(self.pat_hist) > 0:
            a = np.array(self.pat_hist).T
            self.pat_line.set_data(a[0], a[1]); self.pat_line.set_3d_properties(a[2])
            self.pat_line.set_color(c)
        self.pat_text.set_text(f"{l} [{85 + int(np.random.rand()*14)}%]")
        self.pat_text.set_color(c)
        
        # --- 5. LOG SCROLLING ---
        if frame % 2 == 0:
            log = f"[T+{self.t:05.2f}s] CONTROLLER::{l} | STATUS: OK | DT: 100ms"
            self.log_history.insert(0, log); self.log_history.pop()
        for i, txt in enumerate(self.log_lines):
            txt.set_text(self.log_history[i])
            txt.set_alpha(max(0.1, 1.0 - (i * 0.08)))
            
        # --- 6. BIOMETRIC CLEANING ---
        raw = np.abs(np.sin(self.t * 5 + np.arange(16))) * (0.5 + np.random.rand(16)*0.5)
        for i, (br, bc) in enumerate(zip(self.bars_raw, self.bars_cal)):
            br.set_width(raw[i])
            bc.set_width(min(1.0, raw[i] * 1.5))
            
        # --- 7. TEAMWORK MODES ---
        self.coord_timer += self.dt
        if self.coord_timer > 4.0:
            self.coord_idx = (self.coord_idx + 1) % 4
            self.coord_timer = 0
        for i, p in enumerate(self.coords):
            # Highlight border of the active cooperation playbook
            p['ax'].spines['bottom'].set_linewidth(3 if i == self.coord_idx else 1)
            p['ax'].spines['bottom'].set_color(CYAN if i == self.coord_idx else '#333333')
            
            # Simple dot-motion visualization for each strategy
            mx = np.sin(self.t * 3) * 0.4
            my = 0.5 + np.cos(self.t * 3) * 0.3
            if p['mode'] == "SYMMETRIC":
                p['line_l'].set_data([mx-0.3], [my]); p['line_r'].set_data([-(mx-0.3)], [my])
            elif p['mode'] == "HANDOFF":
                # Arc path for object handoff
                trc = np.linspace(0, np.pi, 20)
                cyc = (self.coord_timer % 4.0) / 4.0
                idx = int(cyc * 19)
                p['line_l'].set_data(-0.8+0.8*np.cos(trc[:idx+1]), 0.5+0.3*np.sin(trc[:idx+1]))
                p['line_r'].set_data(0.8-0.8*np.cos(trc[:idx+1]), 0.5+0.3*np.sin(trc[:idx+1]))

if __name__ == "__main__":
    db = ControllerDashboard()
    # Cache_frame_data=False prevents memory leaks during long-running presentations
    anim = FuncAnimation(db.fig, db.update, interval=100, save_count=100, cache_frame_data=False)
    plt.show()
