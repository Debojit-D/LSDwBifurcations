# ------------------------------------------------------------------------------
# MuJoCo Panda scene + DLSVelocityPlanner IK demo (for <general> position servos)
# ------------------------------------------------------------------------------

import mujoco
import numpy as np
from mujoco import viewer
import sys
import os
import time

# ---- make project root importable ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level up from scripts/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dls_velocity_control.dls_velocity_ctrl import DLSVelocityPlanner


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def rotmat_to_rpy(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix R (3x3) to roll-pitch-yaw (XYZ convention).
    Returns [roll, pitch, yaw] in radians.
    """
    # Guard for numerical drift
    R20 = np.clip(R[2, 0], -1.0, 1.0)

    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(R20)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo order [w x y z] → [x y z w]."""
    q = np.asarray(q_wxyz)
    return np.array([q[1], q[2], q[3], q[0]])


def _quat_log_error(q_t: np.ndarray, q_c: np.ndarray) -> np.ndarray:
    """
    Quaternion logarithmic error (axis-angle 3-vector) between target q_t
    and current q_c. Quaternions are [x y z w].
    """
    q_c_inv = np.array([-q_c[0], -q_c[1], -q_c[2], q_c[3]])
    x1, y1, z1, w1 = q_t
    x2, y2, z2, w2 = q_c_inv
    q_e = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])

    # Hemisphere continuity
    if q_e[3] < 0.0:
        q_e *= -1.0

    ang = 2.0 * np.arccos(np.clip(q_e[3], -1.0, 1.0))
    if ang < 1e-6:
        return np.zeros(3)
    axis = q_e[:3] / np.sin(ang / 2.0)
    return ang * axis


# ------------------------------------------------------------------------------
# Path to your MuJoCo XML model (MJCF or converted URDF file)
# ------------------------------------------------------------------------------

XML_PATH = "/home/debojit/Debojit_WS/LDSwBifurcations/robot_descriptions/franka_emika_panda/scene.xml"

# Load model & data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# ------------------------------------------------------------------------------
# Reset to "home" keyframe if available
# ------------------------------------------------------------------------------
key_name = "home"
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)

if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
else:
    # List available keyframes to help you pick the right name
    if model.nkey > 0:
        print(f'[WARN] Keyframe "{key_name}" not found. Available keyframes:')
        for i in range(model.nkey):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            print(f"  - {i}: {nm}")
    else:
        print("[WARN] No keyframes defined in this model.")
    # Fall back to a clean reset (default initial state)
    mujoco.mj_resetData(model, data)

# Initialize derived quantities
mujoco.mj_forward(model, data)

# ------------------------------------------------------------------------------
# Control gains & caps
# ------------------------------------------------------------------------------

# Let robot be visible for a short time before IK kicks in
CONTROL_START_TIME = 0.5  # seconds

# Task-space PD gains (these map error -> desired twist before capping)
POS_GAIN = 2.0       # [1/s]; tune for how aggressively you move
ORI_GAIN = 1.0       # [1/s]; for orientation alignment

# Cartesian caps
MAX_CARTESIAN_SPEED = 0.10   # m/s  (linear speed cap)
MAX_ANGULAR_SPEED = 0.5      # rad/s (~30 deg/s) (angular speed cap)

# Circle trajectory settings
CIRCLE_RADIUS = 0.2   # meters
CIRCLE_PERIOD = 8.0    # seconds per cycle
CIRCLE_CYCLES = 2

# The physics timestep is usually much faster than a useful display refresh.
# Rendering every simulation step can make the viewer feel unnecessarily slow.
VIEWER_FPS = 20.0


# ------------------------------------------------------------------------------
# Check sites and EE site availability
# ------------------------------------------------------------------------------

if model.nsite == 0:
    print("[WARN] Model has no <site> elements at all.")
    print("       IK needs an EE site (e.g., on the hand).")
    print("       Viewer will open, but no IK will be run.")
    ee_site_available = False
    ee_site_id = -1
else:
    # Change this to your actual EE site name if different
    EE_SITE_NAME = "right_center"
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)

    if ee_site_id == -1:
        print(f'[WARN] EE site "{EE_SITE_NAME}" not found in model.')
        print("       Available sites are:")
        for i in range(model.nsite):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            print(f"  - {i}: {nm}")
        print("       Viewer will open, but no IK will be run.")
        ee_site_available = False
    else:
        ee_site_available = True
        print(f'[INFO] Using EE site "{EE_SITE_NAME}" with id {ee_site_id}.')

# ------------------------------------------------------------------------------
# Create the DLSVelocityPlanner (only if EE site exists)
# ------------------------------------------------------------------------------

gripper_cfg = [
    {"actuator_id": 7},  # 0-based index of the tendon actuator (actuator8)
]

planner = None
if ee_site_available:
    gripper_cfg = [
        {"actuator_id": 7},  # 0-based index of the tendon actuator (actuator8)
    ]

    planner = DLSVelocityPlanner(
        model=model,
        data=data,
        kd=5.0,
        site_name=EE_SITE_NAME,   # must match the EE site in your XML
        damping=1e-2,
        gripper_cfg=gripper_cfg,
        for_multi=False,
        actuator_mode="position",   # IMPORTANT: Panda <general> are position servos
    )

    # Print initial EE pose (for debugging / sanity)
    ee_pos0 = data.site_xpos[planner.site_id].copy()
    ee_R0 = data.site_xmat[planner.site_id].reshape(3, 3)
    q_wxyz0 = np.zeros(4)
    mujoco.mju_mat2Quat(q_wxyz0, data.site_xmat[planner.site_id])
    rpy0 = rotmat_to_rpy(ee_R0)
    print("[INFO] Initial EE position (world):", ee_pos0)
    print("[INFO] Initial EE rotation matrix (row-major):\n", ee_R0)
    print("[INFO] Initial EE RPY (deg) [roll, pitch, yaw]:", np.degrees(rpy0))

    # ------------------------------------------------------------------------------
    # Choose an ABSOLUTE target EE position (world coordinates)
    # ------------------------------------------------------------------------------

    circle_center = ee_pos0 - np.array([CIRCLE_RADIUS, 0.0, 0.0])
    target_quat = _wxyz_to_xyzw(q_wxyz0)
    print("[INFO] Circle center (world):", circle_center)
    print("[INFO] Circle radius:", CIRCLE_RADIUS)
    print("[INFO] Circle cycles:", CIRCLE_CYCLES)
    print("[INFO] Holding initial EE orientation.")
else:
    print("[INFO] Skipping IK setup (no valid EE site).")

# ------------------------------------------------------------------------------
# Launch passive viewer (non-blocking) and run a control loop
# ------------------------------------------------------------------------------

v = viewer.launch_passive(model, data)
print("[INFO] Viewer started. Close the viewer window to exit.")

trace_complete = False
last_viewer_sync = time.perf_counter()
viewer_sync_period = 1.0 / VIEWER_FPS

while v.is_running():
    step_start = time.perf_counter()

    # Lock the viewer while we modify model/data
    with v.lock():
        # Simple time-based gating: run IK only after some sim time
        if ee_site_available and planner is not None and data.time > CONTROL_START_TIME:
            trace_time = data.time - CONTROL_START_TIME
            trace_progress = trace_time / CIRCLE_PERIOD

            if trace_progress >= CIRCLE_CYCLES:
                trace_complete = True
                print(f"[INFO] Completed {CIRCLE_CYCLES} circle cycles. Stopping viewer.")
            else:
                theta = 2.0 * np.pi * trace_progress
                target_pos = circle_center + CIRCLE_RADIUS * np.array([
                    np.cos(theta),
                    np.sin(theta),
                    0.0,
                ])

            # Current EE pose
            ee_pos = data.site_xpos[planner.site_id].copy()
            ee_R = data.site_xmat[planner.site_id].reshape(3, 3)

            # --- Position error & desired linear velocity -----------------
            if not trace_complete:
                pos_err = target_pos - ee_pos
                v_cmd = POS_GAIN * pos_err  # "desired" linear velocity (before cap)

                # --- Orientation error & desired angular velocity ---------
                q_t = np.asarray(target_quat, float)
                q_t /= np.linalg.norm(q_t)

                q_wxyz = np.zeros(4)
                mujoco.mju_mat2Quat(q_wxyz, data.site_xmat[planner.site_id])
                q_c = _wxyz_to_xyzw(q_wxyz)

                if np.dot(q_t, q_c) < 0.0:
                    q_t = -q_t

                ori_err = _quat_log_error(q_t, q_c)
                w_cmd = ORI_GAIN * ori_err  # "desired" angular velocity (before cap)

                # --- Cartesian speed caps ---------------------------------
                # Linear
                lin_speed = np.linalg.norm(v_cmd)
                if lin_speed > MAX_CARTESIAN_SPEED and lin_speed > 1e-9:
                    v_cmd *= MAX_CARTESIAN_SPEED / lin_speed

                # Angular
                ang_speed = np.linalg.norm(w_cmd)
                if ang_speed > MAX_ANGULAR_SPEED and ang_speed > 1e-9:
                    w_cmd *= MAX_ANGULAR_SPEED / ang_speed

                # --- Send twist to planner (DLS IK + position servo integration)
                planner.track_twist(v_cmd, w_cart=w_cmd)

        # Advance the simulation one step
        mujoco.mj_step(model, data)

    now = time.perf_counter()
    if now - last_viewer_sync >= viewer_sync_period:
        v.sync()
        last_viewer_sync = now

    if trace_complete:
        break

    sleep_time = model.opt.timestep - (time.perf_counter() - step_start)
    if sleep_time > 0:
        time.sleep(sleep_time)

v.close()
