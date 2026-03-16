#!/usr/bin/env python3
import os
import time
import collections
import numpy as np

# Try EGL first for GPU rendering, fall back to GLFW
try:
    os.environ['MUJOCO_GL'] = 'egl'
    import mujoco as _test_mj
    _test_mj.MjModel.from_xml_string('<mujoco/>')
    import mujoco
    import mujoco.viewer
except Exception:
    os.environ['MUJOCO_GL'] = 'glfw'
    import mujoco
    import mujoco.viewer

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POSE LIBRARY — define ALL poses as numpy arrays of shape (16,)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POSES = {
  # Both arms relaxed, slight droop, natural resting stance
  'HOME': np.array([
     0.1, -0.2, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0,   # right arm
    -0.1, -0.2, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0    # left arm
  ]),

  # Right arm reaches forward-down toward workbench (picking up cube)
  'REACH_R_TABLE': np.array([
     0.15, -0.35, -0.55, -0.45, -0.2,  0.1,  0.0,  0.0,
    -0.1,  -0.2,  -0.1,  0.0,   0.0,  0.0,  0.0,  0.0
  ]),

  # Right arm lifts object up and slightly back overhead
  'LIFT_R_HIGH': np.array([
     0.2, -1.1, -0.6, -0.3, -0.15, 0.1,  0.0,  0.0,
    -0.1, -0.2, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0
  ]),

  # Right arm extended overhead dramatically — Spider-Man pose
  'SPIDERMAN_R': np.array([
     0.4,   -1.3, -0.5, -0.2, -0.1,  0.05, 0.0,  0.0,
    -0.3, -0.8, -0.4, -0.2,  0.0,  0.0,  0.0,  0.0
  ]),

  # Both arms reach toward each other for handoff — center of body
  'HANDOFF_R': np.array([
     0.7, -0.7, -0.4, -0.2,  0.0,  0.0,  0.0,  0.0,
    -0.7, -0.7, -0.4, -0.2,  0.0,  0.0,  0.0,  0.0
  ]),

  # Left arm receives and lifts to opposite side
  'LIFT_L_HIGH': np.array([
     0.1, -0.2, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0,
    -0.2, -1.1, -0.6, -0.3, -0.15, 0.1,  0.0,  0.0
  ]),

  # Left arm reaches down to table for sphere
  'REACH_L_TABLE': np.array([
     0.1, -0.2, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0,
    -0.15,-0.35,-0.55,-0.45, -0.2,  0.1,  0.0,  0.0
  ]),

  # Both arms fully extended outward — dramatic display
  'WINGSPAN': np.array([
     0.8, -0.5, -0.2,  0.0,  0.0,  0.0,  0.0,  0.0,
    -0.8, -0.5, -0.2,  0.0,  0.0,  0.0,  0.0,  0.0
  ]),

  # Both arms crossed in front — guarding/carrying
  'CROSS_CARRY': np.array([
     0.9, -0.6, -0.5, -0.3,  0.0,  0.0,  0.0,  0.0,
    -0.9, -0.6, -0.5, -0.3,  0.0,  0.0,  0.0,  0.0
  ]),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATE MACHINE — full sequence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEQUENCE = [
  # (state_name, duration_sec, target_pose, grasp_action, release_action, camera_preset)
  ('IDLE',         2.5, 'HOME',          None,             None,            'WIDE'),
  ('REACH_CUBE',   2.0, 'REACH_R_TABLE', None,             None,            'CLOSE_R'),
  ('GRASP_CUBE',   0.8, 'REACH_R_TABLE', 'grasp_right',    None,            'CLOSE_R'),
  ('LIFT_CUBE',    2.0, 'LIFT_R_HIGH',   'grasp_right',    None,            'TRACKING'),
  ('SPIDERMAN',    2.5, 'SPIDERMAN_R',   'grasp_right',    None,            'DRAMATIC'),
  ('PRE_HANDOFF',  1.5, 'HANDOFF_R',     'grasp_right',    None,            'CLOSE_C'),
  ('HANDOFF',      1.0, 'HANDOFF_R',     'grasp_left',     'grasp_right',   'CLOSE_C'),
  ('LIFT_L',       2.0, 'LIFT_L_HIGH',   'grasp_left',     None,            'TRACKING'),
  ('WINGSPAN',     1.5, 'WINGSPAN',      'grasp_left',     None,            'WIDE'),
  ('REACH_SPHERE', 2.0, 'REACH_L_TABLE', None,             None,            'CLOSE_L'),
  ('GRASP_SPHERE', 0.8, 'REACH_L_TABLE', 'grasp_sphere_l', None,            'CLOSE_L'),
  ('LIFT_SPHERE',  2.0, 'LIFT_L_HIGH',   'grasp_sphere_l', None,            'TRACKING'),
  ('RELEASE',      1.5, 'SPIDERMAN_R',   None,             'grasp_sphere_l','DRAMATIC'),
  ('RESET',        2.0, 'HOME',          None,             None,            'WIDE'),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERPOLATION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ease_in_out(t):
    # Cubic ease-in-out — much smoother than cosine for fast moves
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        p = 2.0 * t - 2.0
        return 0.5 * p * p * p + 1.0

def ease_out_elastic(t):
    # Slight spring overshoot at end — organic feel
    c4 = (2.0 * np.pi) / 3.0
    if t == 1.0: return 1.0
    return pow(2.0, -8.0*t) * np.sin((t*8.0 - 0.75) * c4) + 1.0

smoothed_ctrl = POSES['HOME'].copy()
ALPHA = 0.10  # exponential smoothing — lower = silkier

def update_ctrl(target, use_elastic=False):
    global smoothed_ctrl
    smoothed_ctrl = ALPHA * target + (1.0 - ALPHA) * smoothed_ctrl
    return smoothed_ctrl

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRASP MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GraspManager:
    def __init__(self, model):
        # Build lookup: constraint name → index in model.eq_active
        self.eq_ids = {}
        for i in range(model.neq):
            self.eq_ids[model.equality(i).name] = i
        self.active = set()
    
    def grasp(self, data, name):
        if name in self.eq_ids:
            data.eq_active[self.eq_ids[name]] = True
            self.active.add(name)
    
    def release(self, data, name):
        if name in self.eq_ids:
            data.eq_active[self.eq_ids[name]] = False
            self.active.discard(name)
    
    def release_all(self, data):
        for name in list(self.active):
            self.release(data, name)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECT RESET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OBJECT_SPAWN = {
    'obj_cube':     np.array([0.18, -0.65, 0.82, 1, 0, 0, 0]),
    'obj_sphere':   np.array([-0.18,-0.65, 0.82, 1, 0, 0, 0]),
    'obj_cylinder': np.array([0.0,  -0.65, 0.86, 1, 0, 0, 0]),
}

def reset_objects(model, data, gm):
    gm.release_all(data)
    for name, spawn_pos in OBJECT_SPAWN.items():
        body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        jnt_id   = model.body(body_id).jntadr[0]
        if jnt_id >= 0:
            qpos_adr = model.jnt_qposadr[jnt_id]  # ← correct address
            data.qpos[qpos_adr:qpos_adr+7] = spawn_pos
            data.qvel[qpos_adr-1:qpos_adr+5] = 0
    mujoco.mj_forward(model, data)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CINEMATIC CAMERA PRESETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAM_PRESETS = {
    'WIDE':      {'azimuth': 200, 'elevation': -18, 'distance': 2.6, 'lookat': [0, -0.1, 1.0]},
    'CLOSE_R':   {'azimuth': 160, 'elevation': -25, 'distance': 1.5, 'lookat': [0.3, -0.3, 1.0]},
    'CLOSE_L':   {'azimuth': 240, 'elevation': -25, 'distance': 1.5, 'lookat': [-0.3,-0.3, 1.0]},
    'CLOSE_C':   {'azimuth': 180, 'elevation': -30, 'distance': 1.4, 'lookat': [0, -0.2, 1.1]},
    'DRAMATIC':  {'azimuth': 210, 'elevation': -10, 'distance': 2.0, 'lookat': [0,  0.0, 1.3]},
    'TRACKING':  {'azimuth': 195, 'elevation': -22, 'distance': 1.8, 'lookat': [0, -0.1, 1.2]},
}

cam_current = {k: v for k, v in CAM_PRESETS['WIDE'].items()}
cam_target  = {k: v for k, v in CAM_PRESETS['WIDE'].items()}
CAM_ALPHA   = 0.04  # very slow camera easing — cinematic feel
ORBIT_SPEED = 0.04  # degrees per frame continuous orbit

def update_camera(viewer, frame_idx):
    global cam_current
    # Ease toward target
    for key in ['elevation', 'distance', 'azimuth']:
        # Static analysis fix: ensure we treat as float
        target_val = float(cam_target[key])
        current_val = float(cam_current[key])
        cam_current[key] = current_val + CAM_ALPHA * (target_val - current_val)
    
    # Orbit continuously
    cam_current['azimuth'] = float(cam_current['azimuth']) + ORBIT_SPEED
    
    # Handheld shake (subtle)
    shake_x = 0.002 * np.sin(frame_idx * 0.15)
    shake_y = 0.002 * np.cos(frame_idx * 0.1)

    for i in range(3):
        target_l = float(cam_target['lookat'][i])
        current_l = float(cam_current['lookat'][i])
        cam_current['lookat'][i] = current_l + CAM_ALPHA * (target_l - current_l)
    
    viewer.cam.azimuth   = float(cam_current['azimuth']) + shake_x * 10
    viewer.cam.elevation = float(cam_current['elevation']) + shake_y * 10
    viewer.cam.distance  = float(cam_current['distance'])
    viewer.cam.lookat[:] = [v + s for v, s in zip(cam_current['lookat'], [shake_x, shake_y, 0])]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    global smoothed_ctrl, cam_target
    model_path = os.path.join(os.path.dirname(__file__), "models/mjcf/srl_robot.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    gm    = GraspManager(model)
    mujoco.mj_resetData(model, data)
    reset_objects(model, data, gm)

    seq_idx        = 0
    state_start    = time.time()
    frame_times    = collections.deque(maxlen=60)
    
    # GAP FIX: Initialize pose_start properly before loop
    smoothed_ctrl = POSES['HOME'].copy()
    pose_start     = POSES['HOME'].copy()
    held_object    = 'none'
    frame_idx      = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Initial camera
        viewer.cam.azimuth   = 200
        viewer.cam.elevation = -18
        viewer.cam.distance  = 2.6
        viewer.cam.lookat[:] = [0, -0.1, 1.0]
        
        while viewer.is_running():
            frame_start = time.time()
            
            # ── STATE MACHINE ──────────────────────────────
            state_name, duration, pose_key, grasp_on, grasp_off, cam_key = SEQUENCE[seq_idx]
            elapsed   = time.time() - state_start
            progress  = min(elapsed / duration, 1.0)
            
            # Interpolate pose
            t_eased  = ease_in_out(progress)
            target   = pose_start + t_eased * (POSES[pose_key] - pose_start)
            data.ctrl[:model.nu] = update_ctrl(target)
            
            # Pick up objects at 20% into state (arm is near object)
            if grasp_on and progress > 0.20:
                gm.grasp(data, grasp_on)
                if 'cube' in grasp_on:   held_object = 'CUBE'
                if 'sphere' in grasp_on: held_object = 'SPHERE'
                if 'cyl' in grasp_on:    held_object = 'CYLINDER'
            
            # Release object at 30% into state
            if grasp_off and progress > 0.30:
                gm.release(data, grasp_off)
                held_object = 'none'
            
            # RESET state: respawn objects
            if state_name == 'RESET' and progress > 0.5:
                reset_objects(model, data, gm)
                held_object = 'none'
            
            # Advance state when duration complete
            if elapsed >= duration:
                seq_idx     = (seq_idx + 1) % len(SEQUENCE)
                state_start = time.time()
                
                # GAP FIX: Use actual ctrl values, not global smoothed_ctrl to prevent jump
                pose_start  = data.ctrl[:model.nu].copy()
                
                # Update camera target
                new_cam = CAM_PRESETS[SEQUENCE[seq_idx][5]]
                cam_target.update(new_cam)
            
            # ── PHYSICS ────────────────────────────────────
            for _ in range(2):  # 2 physics steps per render = 100Hz
                mujoco.mj_step(model, data)
            
            # ── CAMERA ─────────────────────────────────────
            # Camera updates
            update_camera(viewer, frame_idx)
            
            # ── RENDER ─────────────────────────────────────
            viewer.sync()
            
            # ── FPS & HUD ──────────────────────────────────
            frame_times.append(time.time() - frame_start)
            fps = 1.0 / (sum(frame_times)/len(frame_times) + 1e-9)
            
            r_ext = abs(data.qpos[1]) / 1.3 * 100  # joint2 = vertical extent
            l_ext = abs(data.qpos[9]) / 1.3 * 100
            
            next_state = SEQUENCE[(seq_idx+1) % len(SEQUENCE)][0]
            time_left  = duration - elapsed
            
            # Print to terminal directly
            print(f"\033[H\033[J"  # clear terminal
                  f"╔══════════════════════════════════════════╗\n"
                  f"║  ARM-S  ·  AUTONOMOUS SIMULATION         ║\n"
                  f"╠══════════════════════════════════════════╣\n"
                  f"║  State  : {state_name:<30} ║\n"
                  f"║  Next   : {next_state:<24} {time_left:4.1f}s ║\n"
                  f"║  R-Arm  : {'█'*int(r_ext//10):<10} {r_ext:5.1f}%          ║\n"
                  f"║  L-Arm  : {'█'*int(l_ext//10):<10} {l_ext:5.1f}%          ║\n"
                  f"║  Holding: {held_object:<30} ║\n"
                  f"║  FPS    : {fps:<31.0f} ║\n"
                  f"╚══════════════════════════════════════════╝", flush=True)
            
            frame_idx += 1
            time.sleep(max(0.0, 0.01 - (time.time() - frame_start)))

if __name__ == "__main__":
    main()
