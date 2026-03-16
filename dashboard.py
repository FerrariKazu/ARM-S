#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')

import os
import datetime
import time
import numpy as np
import scipy.signal
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# --- VISUAL THEME ---
BG_COLOR = "#050A0F"
CARD_COLOR = "#0D1721"
TEXT_COLOR = "#E0E6ED"
CYAN = "#00F5FF"
ACCENT = "#FF3131"
GREEN = "#00FF66"

# --- SIMULATION SPECS ---
N_CH = 16
T = 300
INTENT_CLASSES = ["reach_left", "reach_right", "lift", "hold", "release", "stabilize", "handoff"]
COLORS_INTENT = [CYAN, '#3498DB', GREEN, 'white', '#E67E22', '#F1C40F', '#9B59B6']

# Muscle group colors for 16 channels
CH_COLORS = []
for i in range(16):
    if i < 3: CH_COLORS.append('#FF3333')       # Bicep (Red)
    elif i < 6: CH_COLORS.append(CYAN)          # Tricep (Cyan)
    elif i < 11: CH_COLORS.append(GREEN)        # Flexors (Green)
    else: CH_COLORS.append('#9B59B6')           # Extensors (Purple)

# Projection Matrix for 3D->2D Skeleton
PROJ_MAT = np.array([[1, 0, -0.5],
                     [0, 1, -0.3]])

class V3Dashboard:
    def __init__(self):
        plt.rcParams['text.color'] = TEXT_COLOR
        plt.rcParams['axes.labelcolor'] = TEXT_COLOR
        plt.rcParams['xtick.color'] = TEXT_COLOR
        plt.rcParams['ytick.color'] = TEXT_COLOR
        
        self.fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
        self.fig.canvas.manager.set_window_title("ARM-S V3 Clinical Monitor")
        
        # 5-Row Layout
        self.gs = GridSpec(5, 3, height_ratios=[0.06, 1, 1, 1, 0.08], figure=self.fig)
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.07, hspace=0.35, wspace=0.3)
        self.fig.suptitle("ARM-S: AUTONOMOUS ROBOTIC MOTION SYSTEM (CLINICAL)", 
                          fontsize=18, fontweight='bold', y=0.98, color=CYAN)
                          
        self.blitted_artists = []
        self.start_time = time.time()
        self.last_frame = time.time()
        self.frame_count = 0
        
        # Simulated Data State
        self.current_logits = np.zeros(7)
        self.emg_buffer = deque(np.zeros((32, 16)), maxlen=32)
        
        # Panels
        self._init_top_hud()
        self._init_skeleton_panel()
        self._init_neural_spikes()
        self._init_attention()
        self._init_polar_confidence()
        self._init_waterfall()
        self._init_threat_panel()
        self._init_coherence()
        self._init_spectrogram()
        self._init_bottom_hud()

    # --- 9. TOP HUD ---
    def _init_top_hud(self):
        self.ax_top = self.fig.add_subplot(self.gs[0, :], facecolor=CARD_COLOR)
        self.ax_top.axis('off')
        
        self.vital_texts = []
        metrics = ["LATENCY: ---ms", "INTENT: ---", "EMG SNR: >22dB", "JOINTS: 16/16", 
                   "COLLISIONS: CLEAR", "GRASP: OPEN", "UPTIME: 0s", "MODEL: INT8 ONNX"]
        
        for i, m in enumerate(metrics):
            x = 0.05 + i * (0.95 / 8)
            t = self.ax_top.text(x, 0.5, m, ha='center', va='center', fontsize=10, 
                                 fontweight='bold', fontfamily='monospace')
            self.vital_texts.append(t)
            self.blitted_artists.append(t)

    # --- 1. SKELETON (2D-Projected) ---
    def _init_skeleton_panel(self):
        self.ax_skel = self.fig.add_subplot(self.gs[1:3, 0], facecolor=BG_COLOR)
        self.ax_skel.set_title("Kinematic Track (2D Proj)", color=CYAN)
        self.ax_skel.set_xlim(-1.5, 1.5)
        self.ax_skel.set_ylim(-1.5, 1.5)
        self.ax_skel.axis('off')
        
        # Bones & Joints
        self.skel_lines = []
        for _ in range(5): # Torso, L-arm, R-arm, Head
            line, = self.ax_skel.plot([], [], lw=3, color='#FAD6B1') # Skin tone
            self.skel_lines.append(line)
            self.blitted_artists.append(line)
            
        self.skel_ee_l, = self.ax_skel.plot([], [], 'o', color=CYAN, markersize=8)
        self.skel_ee_r, = self.ax_skel.plot([], [], 'o', color=CYAN, markersize=8)
        self.blitted_artists.extend([self.skel_ee_l, self.skel_ee_r])

    # --- 2. NEURAL SPIKE RASTER ---
    def _init_neural_spikes(self):
        self.ax_spikes = self.fig.add_subplot(self.gs[1, 1], facecolor=CARD_COLOR)
        self.ax_spikes.set_title("Neural Spike Motor Recruitment", color=CYAN)
        
        self.spike_lines = []
        self.t_x = np.linspace(0, 1, 100)
        self.spike_data = np.zeros((16, 100))
        
        for i in range(16):
            line, = self.ax_spikes.plot(self.t_x, self.spike_data[i] + i*2, color=CH_COLORS[i], lw=0.8)
            self.spike_lines.append(line)
            self.blitted_artists.append(line)
            
        self.ax_spikes.set_ylim(-1, 32)
        self.ax_spikes.axis('off')

    # --- 8. NEURAL DECODE ATTENTION ---
    def _init_attention(self):
        self.ax_att = self.fig.add_subplot(self.gs[1, 2], facecolor=CARD_COLOR)
        self.ax_att.set_title("Transformer Attention (L3:H2)", color=CYAN, fontsize=10)
        self.att_data = np.random.rand(16, 25) * 0.2
        self.att_im = self.ax_att.imshow(self.att_data, aspect='auto', cmap='magma', vmin=0, vmax=1)
        self.ax_att.axis('off')
        self.blitted_artists.append(self.att_im)

    # --- 5. CONFIDENCE DECAY RINGS ---
    def _init_polar_confidence(self):
        self.ax_polar = self.fig.add_subplot(self.gs[2, 1], projection='polar')
        self.ax_polar.set_facecolor(CARD_COLOR)
        
        self.angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        self.ax_polar.set_xticks(self.angles)
        self.ax_polar.set_xticklabels([l[:4].upper() for l in INTENT_CLASSES], fontsize=7, color=TEXT_COLOR)
        self.ax_polar.set_yticks([]) # No radial rings
        self.ax_polar.set_ylim(0, 1)
        
        # Setup ghost rings
        self.ghost_polys = []
        for _ in range(5): # 5 historical fading rings
            poly, = self.ax_polar.fill([], [], alpha=0.0)
            self.ghost_polys.append(poly)
            self.blitted_artists.append(poly)
            
        self.polar_history = deque(maxlen=5)

    # --- 3. LATENCY WATERFALL ---
    def _init_waterfall(self):
        self.ax_water = self.fig.add_subplot(self.gs[2, 2], facecolor=CARD_COLOR)
        self.ax_water.set_title("Latency Waterfall (ms)", color=CYAN)
        
        self.water_data = np.ones((6, 100)) * 5.0 # nominal latency
        self.water_im = self.ax_water.imshow(self.water_data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=25)
        self.ax_water.set_yticks(range(6))
        self.ax_water.set_yticklabels(["SNS", "NTN", "PLN", "IK", "CMD", "HW"], fontsize=7)
        self.ax_water.set_xticks([])
        self.blitted_artists.append(self.water_im)

    # --- 4. THREAT ASSESSMENT ---
    def _init_threat_panel(self):
        self.ax_threat = self.fig.add_subplot(self.gs[3, 0], facecolor=CARD_COLOR)
        self.ax_threat.set_title("Safety Monitoring", color=CYAN)
        self.ax_threat.set_xlim(-1, 1); self.ax_threat.set_ylim(-1, 1)
        self.ax_threat.axis('off')
        
        colors = ['#FF0000', '#FF8C00', '#FFD700', GREEN, '#112233']
        radii = [0.1, 0.3, 0.5, 0.7, 0.9]
        for c, r in zip(colors, radii):
            circ = patches.Circle((0, 0), r, fill=False, edgecolor=c, lw=1.5, alpha=0.5)
            self.ax_threat.add_patch(circ)
            
        self.threat_ee_l, = self.ax_threat.plot([], [], 'o', color='#3498DB', markersize=6)
        self.threat_ee_r, = self.ax_threat.plot([], [], 'o', color=CYAN, markersize=6)
        self.threat_txt = self.ax_threat.text(0, -0.9, "CLEAR", ha='center', color=GREEN, fontweight='bold')
        self.blitted_artists.extend([self.threat_ee_l, self.threat_ee_r, self.threat_txt])

    # --- 6. SIGNAL COHERENCE ---
    def _init_coherence(self):
        self.ax_coh = self.fig.add_subplot(self.gs[3, 1], facecolor=CARD_COLOR)
        self.ax_coh.set_title("Channel Cross-Correlation", color=CYAN)
        
        self.coh_data = np.zeros((16, 16))
        self.coh_im = self.ax_coh.imshow(self.coh_data, aspect='auto', cmap='plasma', interpolation='nearest', vmin=0, vmax=1)
        self.ax_coh.axis('off')
        self.blitted_artists.append(self.coh_im)

    # --- 10. SPECTROGRAM ---
    def _init_spectrogram(self):
        self.ax_spec = self.fig.add_subplot(self.gs[3, 2], facecolor=CARD_COLOR)
        self.ax_spec.set_title("EMG Spectrogram (Freq Analysis)", color=CYAN)
        
        self.spec_cache = np.zeros((17, 32))
        self.spec_im = self.ax_spec.imshow(self.spec_cache, aspect='auto', cmap='inferno', origin='lower', vmin=0, vmax=100)
        self.ax_spec.set_yticks([])
        self.ax_spec.set_xticks([])
        self.blitted_artists.append(self.spec_im)

    # --- 7. INTENT TIMELINE ---
    def _init_bottom_hud(self):
        self.ax_bot = self.fig.add_subplot(self.gs[4, :], facecolor=BG_COLOR)
        self.ax_bot.axis('off')
        
        self.time_blocks = []
        for i in range(120):
            p = patches.Rectangle((i*0.008, 0.2), 0.007, 0.6, facecolor=BG_COLOR)
            self.ax_bot.add_patch(p)
            self.time_blocks.append(p)
            self.blitted_artists.append(p)
            
        self.ax_bot.text(0.96, 0.5, "NOW", va='center', fontsize=8, color=TEXT_COLOR)
        line = self.ax_bot.axvline(0.958, ymin=0, ymax=1, color='white', lw=1)
        self.blitted_artists.append(line)

    def update(self, frame):
        self.frame_count += 1
        t_now = time.time()
        dt = (t_now - self.last_frame) * 1000.0
        self.last_frame = t_now
        sys_uptime = t_now - self.start_time
        
        # --- 1. SKELETON (Proj 2D) ---
        skel_3d = np.array([
            [0,0,1], [0.3,0.2,0.8], [0.5,0.4,0.6 + 0.3*np.sin(sys_uptime)], # R arm
            [0,0,1], [-0.3,0.2,0.8], [-0.5,0.4,0.6 + 0.3*np.cos(sys_uptime)] # L arm
        ])
        for i in range(2):
            pts = PROJ_MAT @ skel_3d[i*3:i*3+3].T
            self.skel_lines[i].set_data(pts[0], pts[1])
            if i==0: self.skel_ee_r.set_data([pts[0,-1]], [pts[1,-1]])
            else: self.skel_ee_l.set_data([pts[0,-1]], [pts[1,-1]])

        # --- 2. NEURAL SPIKES ---
        self.spike_data = np.roll(self.spike_data, -1, axis=1)
        activations = np.zeros(16)
        
        # Chain reaction probability firing
        for i in range(16):
            base = 0.1 * np.sin(self.t_x[-1]*10 + i)
            spike = 0
            if np.random.rand() < 0.05: # Primary trigger
                spike = 2.0
                activations[i] = 1.0
                # Trigger neighbor correlation
                if i < 15 and np.random.rand() < 0.6: 
                    self.spike_data[i+1, -1] = 2.0
                    activations[i+1] = 1.0
            
            self.spike_data[i, -1] = base + spike
            self.spike_lines[i].set_ydata(self.spike_data[i] + i*2)
            
        self.emg_buffer.append(activations)

        # --- 8. ATTENTION STREAM ---
        shift = int(sys_uptime*10) % 25
        base_att = np.roll(np.random.rand(16, 25), shift, axis=1) * 0.2
        self.att_data = np.roll(self.att_data, -1, axis=1)
        self.att_data[:, -1] = activations * 0.8 + 0.2
        self.att_im.set_data(self.att_data)

        # --- INTENT MATH ---
        self.current_logits += np.random.randn(7) * 0.1
        if self.frame_count % 30 == 0: self.current_logits[np.random.randint(7)] += 4.0
        exp_l = np.exp(self.current_logits - np.max(self.current_logits))
        probs = exp_l / np.sum(exp_l)
        win_idx = np.argmax(probs)
        
        # --- 5. CONFIDENCE GHOSTS ---
        self.polar_history.append((probs, win_idx))
        for g_idx, g_poly in enumerate(self.ghost_polys):
            if g_idx < len(self.polar_history):
                ph_probs, ph_win = self.polar_history[g_idx]
                r = np.append(ph_probs, ph_probs[0])
                t = np.append(self.angles, self.angles[0])
                g_poly.set_xy(np.column_stack([t, r]))
                
                # Fading effect. Oldest = lowest alpha
                alpha = 0.5 * ((g_idx + 1) / 5.0) 
                g_poly.set_facecolor(COLORS_INTENT[ph_win])
                g_poly.set_alpha(alpha)

        # --- 7. TIMELINE ---
        for i in range(119):
            self.time_blocks[i].set_facecolor(self.time_blocks[i+1].get_facecolor())
        self.time_blocks[-1].set_facecolor(COLORS_INTENT[win_idx])

        # --- 3. WATERFALL ---
        self.water_data = np.roll(self.water_data, -1, axis=1)
        new_lat = [2, 8, 12, 1, 1, dt]
        if dt > 50: new_lat[np.random.randint(6)] = 30 # random lag visual
        self.water_data[:, -1] = new_lat
        self.water_im.set_data(self.water_data)

        # --- 4. THREAT ---
        ee_dist = 0.8 * np.abs(np.cos(sys_uptime*0.8))
        self.threat_ee_r.set_data([ee_dist], [0])
        self.threat_ee_l.set_data([-ee_dist], [0])
        
        if ee_dist < 0.3:
            self.ax_threat.set_facecolor('#300000') # Dark red flash
            self.threat_txt.set_text("CRITICAL")
            self.threat_txt.set_color(ACCENT)
        else:
            self.ax_threat.set_facecolor(CARD_COLOR)
            self.threat_txt.set_text("CLEAR")
            self.threat_txt.set_color(GREEN)

        # --- 6. COHERENCE ---
        self.coh_data *= 0.92
        if self.frame_count % 8 == 0:
            spk = self.emg_buffer[-1]
            self.coh_data += np.outer(spk, spk)
            self.coh_data = np.clip(self.coh_data, 0, 1)
        self.coh_im.set_data(self.coh_data)

        # --- 10. SPECTROGRAM ---
        if self.frame_count % 15 == 0:
            audio = np.sum(list(self.emg_buffer), axis=1)
            f, t_s, Sxx = scipy.signal.spectrogram(audio, fs=50, nperseg=16, noverlap=8)
            # Resize if needed (naive stretching to fit 17x32 cache if matrices differ)
            if Sxx.shape[0] > 0 and Sxx.shape[1] > 0:
                h_scale = np.resize(Sxx, (17, 32)) * 10.0
                self.spec_cache = h_scale
            self.spec_im.set_data(self.spec_cache)

        # --- VITAL SIGNS ---
        v_texts = [
            f"LATENCY: {dt:02.0f}ms", f"INTENT: {INTENT_CLASSES[win_idx][:4].upper()}",
            "EMG SNR: 25dB", "JOINTS: 16/16",
            "COLLISIONS: ALERT" if ee_dist < 0.3 else "COLLISIONS: CLEAR",
            "GRASP: LOCK", f"UPTIME: {int(sys_uptime)}s", "MODEL: ONNX"
        ]
        
        has_error = (dt > 100) or (ee_dist < 0.3)
        self.ax_top.set_facecolor('#500000' if has_error else CARD_COLOR)
        
        for i, t in enumerate(self.vital_texts):
            t.set_text(v_texts[i])
            if has_error: t.set_color('white')
            else: t.set_color(CYAN)

        return self.blitted_artists

if __name__ == "__main__":
    dash = V3Dashboard()
    # blit=True is critical for FPS explicitly returning blitted_artists
    ani = FuncAnimation(dash.fig, dash.update, interval=80, blit=True, cache_frame_data=False)
    plt.show()
