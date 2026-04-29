import os
import sys
import time

RECORD = False
if RECORD:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from mujoco import viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.bifurcation import BifurcationParams, ds_3d, limit_norm, smoothstep
from utils.bifurcation.mujoco_recording import MP4Recorder
from utils.dls_velocity_control.dls_velocity_ctrl import DLSVelocityPlanner


XML_PATH = os.path.join(PROJECT_ROOT, "robot_descriptions/franka_emika_panda/scene.xml")
EE_SITE_NAME = "right_center"

CONTROL_START_TIME = 0.5
REACH_TIME = 5.0
RAMP_TIME = 4.0
CYCLE_PERIOD = 6.0
CYCLE_COUNT = 2.0

CYCLE_RADIUS = 0.04
M = 4.0
R_MAX = 2.0 * np.pi / CYCLE_PERIOD
MAX_CARTESIAN_SPEED = 0.12
ORI_GAIN = 1.0
MAX_ANGULAR_SPEED = 0.5
VIEWER_FPS = 60.0
RECORD_FPS = 20
RECORD_WIDTH = 320
RECORD_HEIGHT = 240
RECORD_PATH = os.path.join(PROJECT_ROOT, "bifurcation_unified.mp4")


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz)
    return np.array([q[1], q[2], q[3], q[0]])


def quat_log_error(q_t: np.ndarray, q_c: np.ndarray) -> np.ndarray:
    q_c_inv = np.array([-q_c[0], -q_c[1], -q_c[2], q_c[3]])
    x1, y1, z1, w1 = q_t
    x2, y2, z2, w2 = q_c_inv
    q_e = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])
    if q_e[3] < 0.0:
        q_e *= -1.0
    ang = 2.0 * np.arccos(np.clip(q_e[3], -1.0, 1.0))
    if ang < 1e-6:
        return np.zeros(3)
    return ang * q_e[:3] / np.sin(ang / 2.0)


def orientation_velocity(model, data, site_id: int, target_quat: np.ndarray) -> np.ndarray:
    q_t = np.asarray(target_quat, dtype=float)
    q_t /= np.linalg.norm(q_t)

    q_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(q_wxyz, data.site_xmat[site_id])
    q_c = wxyz_to_xyzw(q_wxyz)

    if np.dot(q_t, q_c) < 0.0:
        q_t = -q_t
    return limit_norm(ORI_GAIN * quat_log_error(q_t, q_c), MAX_ANGULAR_SPEED)


def make_model_and_planner():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    if site_id == -1:
        raise RuntimeError(f'EE site "{EE_SITE_NAME}" not found in {XML_PATH}')

    planner = DLSVelocityPlanner(
        model=model,
        data=data,
        kd=5.0,
        site_name=EE_SITE_NAME,
        damping=1e-2,
        gripper_cfg=[{"actuator_id": 7}],
        for_multi=False,
        actuator_mode="position",
    )
    return model, data, planner, site_id


def bifurcation_params(t: float, center: np.ndarray) -> BifurcationParams:
    if t < REACH_TIME:
        return BifurcationParams(center=center, rho0=0.0, M=M, R=0.0)

    ramp = smoothstep((t - REACH_TIME) / RAMP_TIME)
    return BifurcationParams(
        center=center,
        rho0=CYCLE_RADIUS * ramp,
        M=M,
        R=R_MAX * ramp,
    )


def main():
    model, data, planner, site_id = make_model_and_planner()

    ee_pos0 = data.site_xpos[site_id].copy()
    center = ee_pos0 + np.array([-0.08, 0.04, -0.06])

    q_wxyz0 = np.zeros(4)
    mujoco.mju_mat2Quat(q_wxyz0, data.site_xmat[site_id])
    target_quat = wxyz_to_xyzw(q_wxyz0)

    total_motion_time = REACH_TIME + RAMP_TIME + CYCLE_COUNT * CYCLE_PERIOD
    print("[INFO] Unified bifurcation DS demo")
    print("[INFO] Reach/limit-cycle center:", center)
    print("[INFO] Initial EE position:", ee_pos0)
    print("[INFO] Holding initial EE orientation.")

    if RECORD:
        recorder = MP4Recorder(
            model,
            RECORD_PATH,
            fps=RECORD_FPS,
            width=RECORD_WIDTH,
            height=RECORD_HEIGHT,
        )
        next_frame_time = 0.0
        done = False
        print("[INFO] Recording MP4:", RECORD_PATH)

        while not done:
            if data.time > CONTROL_START_TIME:
                motion_t = data.time - CONTROL_START_TIME
                if motion_t >= total_motion_time:
                    done = True
                else:
                    ee_pos = data.site_xpos[site_id].copy()
                    params = bifurcation_params(motion_t, center)
                    v_cmd = limit_norm(ds_3d(ee_pos, params), MAX_CARTESIAN_SPEED)
                    w_cmd = orientation_velocity(model, data, site_id, target_quat)
                    planner.track_twist(v_cmd, w_cart=w_cmd)

            mujoco.mj_step(model, data)

            if data.time >= next_frame_time:
                recorder.capture(data)
                next_frame_time += 1.0 / RECORD_FPS

        recorder.close()
        print("[INFO] Finished unified reach-to-limit-cycle recording.")
        return

    v = viewer.launch_passive(model, data)
    print("[INFO] Viewer started. It will close after the demo.")

    last_viewer_sync = time.perf_counter()
    viewer_sync_period = 1.0 / VIEWER_FPS
    done = False

    while v.is_running():
        step_start = time.perf_counter()

        with v.lock():
            if data.time > CONTROL_START_TIME:
                motion_t = data.time - CONTROL_START_TIME
                if motion_t >= total_motion_time:
                    done = True
                else:
                    ee_pos = data.site_xpos[site_id].copy()
                    params = bifurcation_params(motion_t, center)
                    v_cmd = limit_norm(ds_3d(ee_pos, params), MAX_CARTESIAN_SPEED)
                    w_cmd = orientation_velocity(model, data, site_id, target_quat)
                    planner.track_twist(v_cmd, w_cart=w_cmd)

            mujoco.mj_step(model, data)

        now = time.perf_counter()
        if now - last_viewer_sync >= viewer_sync_period:
            v.sync()
            last_viewer_sync = now

        if done:
            print("[INFO] Finished unified reach-to-limit-cycle demo.")
            break

        sleep_time = model.opt.timestep - (time.perf_counter() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    v.close()


if __name__ == "__main__":
    main()
