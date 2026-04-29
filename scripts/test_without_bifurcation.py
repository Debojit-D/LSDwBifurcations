import os
import sys
import time
import csv
from dataclasses import dataclass

RECORD = False
RECORD_DATA = True
CLOSE_SIMULATION = False

if RECORD:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from mujoco import viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.bifurcation import limit_norm
from utils.bifurcation.mujoco_recording import MP4Recorder
from utils.dls_velocity_control.dls_velocity_ctrl import DLSVelocityPlanner


# ---------------------------------------------------------------------
# Paths and MuJoCo names
# ---------------------------------------------------------------------

XML_PATH = os.path.join(
    PROJECT_ROOT,
    "robot_descriptions/franka_emika_panda/mjx_single_cube.xml",
)

GENERATED_XML_PATH = os.path.join(
    os.path.dirname(XML_PATH),
    "generated_hard_switch_task_scene.xml",
)

EE_SITE_CANDIDATES = ("right_center", "gripper")
CUBE_BODY_NAME = "box"
GRIPPER_ACTUATOR_ID = 7

CUBE_SPAWN_POS = np.array([0.5, 0.0, 0.03])
CUBE_SPAWN_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

TARGET_QUAT_XYZW = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------
# Timing, recording, and data logging
# ---------------------------------------------------------------------

CONTROL_START_TIME = 5.0

VIEWER_FPS = 60.0

RECORD_FPS = 20
RECORD_WIDTH = 480
RECORD_HEIGHT = 360

RECORD_PATH = os.path.join(
    PROJECT_ROOT,
    "hard_switch_pick_polish_place.mp4",
)

DATA_CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "mujoco_hard_switch_pick_polish_place_log.csv",
)

DATA_RECORD_EVERY_N_STEPS = 1


# ---------------------------------------------------------------------
# Task geometry
# ---------------------------------------------------------------------

APPROACH_CLEARANCE = 0.02
GRASP_Z_OFFSET = 0.0
LIFT_HEIGHT = 0.10
PLACE_APPROACH_HEIGHT = 0.10

POLISH_CENTER_OFFSETS = (
    np.array([-0.18, 0.40, LIFT_HEIGHT - 0.04]),
    np.array([0.04, -0.25, LIFT_HEIGHT - 0.04]),
)

POLISH_RADIUS = 0.045
POLISH_MARKER_RADIUS = POLISH_RADIUS + 0.05

PICK_CONTAINER_HALF_SIZE = 0.065
MARKER_Z = 0.002


# ---------------------------------------------------------------------
# Controller parameters
# ---------------------------------------------------------------------

POINT_GAIN = 2.0
CIRCLE_TRACKING_GAIN = 2.0

MAX_CARTESIAN_SPEED = 0.13
MAX_POLISH_SPEED = 0.10

ORI_GAIN = 2.5
MAX_ANGULAR_SPEED = 1.2

GOAL_TOLERANCE = 0.012
MIN_PHASE_TIME = 0.7

POLISH_CYCLE_PERIOD = 3.0
POLISH_CYCLES_PER_CENTER = 5.0
POLISH_OMEGA = 2.0 * np.pi / POLISH_CYCLE_PERIOD

OPEN_GRIPPER = 0.04
CLOSED_GRIPPER = 0.006

HOME_POS_TOL = 0.012


@dataclass(frozen=True)
class TaskPhase:
    name: str
    center: np.ndarray
    duration: float
    mode: str = "point"          # "point" or "circle"
    max_speed: float = MAX_CARTESIAN_SPEED
    gripper: float = OPEN_GRIPPER
    auto_advance: bool = True
    radius: float = 0.0
    omega: float = 0.0


@dataclass
class ControlInfo:
    phase_index: int
    phase_name: str
    phase_time: float
    phase_center: np.ndarray
    phase_duration: float
    phase_mode: str
    gripper_cmd: float
    max_speed: float
    radius: float
    omega: float
    raw_v_cmd: np.ndarray
    v_cmd: np.ndarray
    w_cmd: np.ndarray
    ref_pos: np.ndarray
    ref_vel: np.ndarray
    control_active: bool
    task_done: bool
    home_done: bool
    hard_switch_velocity_jump: float


# ---------------------------------------------------------------------
# Data logger
# ---------------------------------------------------------------------

class SimulationDataLogger:
    def __init__(self, enabled: bool, csv_path: str):
        self.enabled = enabled
        self.csv_path = csv_path
        self.rows = []
        self.exported = False

    def log(
        self,
        model,
        data,
        site_id: int,
        cube_body_id: int,
        control_info: ControlInfo,
    ) -> None:
        if not self.enabled:
            return

        ee_pos = data.site_xpos[site_id].copy()
        cube_pos = data.xpos[cube_body_id].copy()

        q_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(q_wxyz, data.site_xmat[site_id])
        q_xyzw = wxyz_to_xyzw(q_wxyz)

        center = np.asarray(control_info.phase_center, dtype=float)
        raw_v_cmd = np.asarray(control_info.raw_v_cmd, dtype=float)
        v_cmd = np.asarray(control_info.v_cmd, dtype=float)
        w_cmd = np.asarray(control_info.w_cmd, dtype=float)
        ref_pos = np.asarray(control_info.ref_pos, dtype=float)
        ref_vel = np.asarray(control_info.ref_vel, dtype=float)

        rel = ee_pos - center
        distance_to_center = float(np.linalg.norm(rel))
        xy_radius = float(np.linalg.norm(rel[:2]))
        radial_error_xy = xy_radius - float(control_info.radius)

        self.rows.append({
            "time": float(data.time),

            "phase_index": int(control_info.phase_index),
            "phase_name": str(control_info.phase_name),
            "phase_time": float(control_info.phase_time),
            "phase_duration": float(control_info.phase_duration),
            "phase_mode": str(control_info.phase_mode),

            "control_active": int(control_info.control_active),
            "task_done": int(control_info.task_done),
            "home_done": int(control_info.home_done),

            "phase_center_x": float(center[0]),
            "phase_center_y": float(center[1]),
            "phase_center_z": float(center[2]),

            # Kept for compatibility with the bifurcation plotting script.
            # For hard-switch case, rho0 simply means the target circle radius.
            "rho0": float(control_info.radius),
            "R": float(control_info.omega),
            "M": 0.0,

            "circle_radius": float(control_info.radius),
            "circle_omega": float(control_info.omega),

            "ee_x": float(ee_pos[0]),
            "ee_y": float(ee_pos[1]),
            "ee_z": float(ee_pos[2]),

            "ee_qx": float(q_xyzw[0]),
            "ee_qy": float(q_xyzw[1]),
            "ee_qz": float(q_xyzw[2]),
            "ee_qw": float(q_xyzw[3]),

            "cube_x": float(cube_pos[0]),
            "cube_y": float(cube_pos[1]),
            "cube_z": float(cube_pos[2]),

            "ref_x": float(ref_pos[0]),
            "ref_y": float(ref_pos[1]),
            "ref_z": float(ref_pos[2]),

            "ref_vx": float(ref_vel[0]),
            "ref_vy": float(ref_vel[1]),
            "ref_vz": float(ref_vel[2]),
            "ref_v_norm": float(np.linalg.norm(ref_vel)),

            "raw_v_cmd_x": float(raw_v_cmd[0]),
            "raw_v_cmd_y": float(raw_v_cmd[1]),
            "raw_v_cmd_z": float(raw_v_cmd[2]),
            "raw_v_cmd_norm": float(np.linalg.norm(raw_v_cmd)),

            "v_cmd_x": float(v_cmd[0]),
            "v_cmd_y": float(v_cmd[1]),
            "v_cmd_z": float(v_cmd[2]),
            "v_cmd_norm": float(np.linalg.norm(v_cmd)),

            "w_cmd_x": float(w_cmd[0]),
            "w_cmd_y": float(w_cmd[1]),
            "w_cmd_z": float(w_cmd[2]),
            "w_cmd_norm": float(np.linalg.norm(w_cmd)),

            "distance_to_phase_center": distance_to_center,
            "xy_radius_about_center": xy_radius,
            "radial_error_xy": radial_error_xy,

            "gripper_cmd": float(control_info.gripper_cmd),
            "max_speed": float(control_info.max_speed),

            "hard_switch_velocity_jump": float(control_info.hard_switch_velocity_jump),
        })

    def export_csv(self) -> None:
        if not self.enabled:
            return

        if self.exported:
            return

        if len(self.rows) == 0:
            print("[WARN] RECORD_DATA=True, but no data rows were logged.")
            self.exported = True
            return

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        fieldnames = list(self.rows[0].keys())

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        self.exported = True

        print(f"[INFO] Exported hard-switch simulation CSV: {self.csv_path}")
        print(f"[INFO] Number of logged samples: {len(self.rows)}")


# ---------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------

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


def orientation_velocity(
    model,
    data,
    site_id: int,
    target_quat: np.ndarray,
) -> np.ndarray:
    q_t = np.asarray(target_quat, dtype=float)
    q_t /= np.linalg.norm(q_t)

    q_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(q_wxyz, data.site_xmat[site_id])
    q_c = wxyz_to_xyzw(q_wxyz)

    if np.dot(q_t, q_c) < 0.0:
        q_t = -q_t

    w_cmd = ORI_GAIN * quat_log_error(q_t, q_c)
    return limit_norm(w_cmd, MAX_ANGULAR_SPEED)


# ---------------------------------------------------------------------
# Simple discontinuous baseline motion laws
# ---------------------------------------------------------------------

def point_attractor_velocity(
    ee_pos: np.ndarray,
    target_pos: np.ndarray,
) -> np.ndarray:
    return POINT_GAIN * (target_pos - ee_pos)


def circle_reference(
    center: np.ndarray,
    radius: float,
    omega: float,
    phase_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = omega * phase_time

    ref_pos = center + np.array([
        radius * np.cos(theta),
        radius * np.sin(theta),
        0.0,
    ])

    ref_vel = np.array([
        -radius * omega * np.sin(theta),
        radius * omega * np.cos(theta),
        0.0,
    ])

    return ref_pos, ref_vel


def circle_tracking_velocity(
    ee_pos: np.ndarray,
    center: np.ndarray,
    phase_time: float,
    radius: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_pos, ref_vel = circle_reference(
        center=center,
        radius=radius,
        omega=omega,
        phase_time=phase_time,
    )

    feedback = CIRCLE_TRACKING_GAIN * (ref_pos - ee_pos)
    raw_v_cmd = ref_vel + feedback

    return raw_v_cmd, ref_pos, ref_vel


def compute_phase_velocity(
    phase: TaskPhase,
    phase_time: float,
    ee_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if phase.mode == "point":
        raw_v_cmd = point_attractor_velocity(ee_pos, phase.center)
        ref_pos = phase.center.copy()
        ref_vel = np.zeros(3)
        return raw_v_cmd, ref_pos, ref_vel

    if phase.mode == "circle":
        return circle_tracking_velocity(
            ee_pos=ee_pos,
            center=phase.center,
            phase_time=phase_time,
            radius=phase.radius,
            omega=phase.omega,
        )

    raise ValueError(f"Unknown phase mode: {phase.mode}")


# ---------------------------------------------------------------------
# XML augmentation for markers
# ---------------------------------------------------------------------

def make_augmented_xml_path() -> str:
    source = open(XML_PATH, "r", encoding="utf-8").read()

    marker_xml = build_task_marker_xml()
    source = source.replace("<worldbody>", f"<worldbody>\n{marker_xml}", 1)

    with open(GENERATED_XML_PATH, "w", encoding="utf-8") as f:
        f.write(source)

    return GENERATED_XML_PATH


def build_task_marker_xml() -> str:
    polish_centers = [CUBE_SPAWN_POS + offset for offset in POLISH_CENTER_OFFSETS]

    c = CUBE_SPAWN_POS
    s = PICK_CONTAINER_HALF_SIZE
    z = MARKER_Z

    marker_lines = [
        f'    <geom name="pick_container_base" type="box" pos="{c[0]} {c[1]} {z}" '
        f'size="{s} {s} 0.002" rgba="0.1 0.45 1.0 0.28" contype="0" conaffinity="0"/>',

        f'    <geom name="pick_container_front" type="box" pos="{c[0]} {c[1] + s} {z + 0.006}" '
        f'size="{s} 0.004 0.008" rgba="0.1 0.45 1.0 0.55" contype="0" conaffinity="0"/>',

        f'    <geom name="pick_container_back" type="box" pos="{c[0]} {c[1] - s} {z + 0.006}" '
        f'size="{s} 0.004 0.008" rgba="0.1 0.45 1.0 0.55" contype="0" conaffinity="0"/>',

        f'    <geom name="pick_container_left" type="box" pos="{c[0] - s} {c[1]} {z + 0.006}" '
        f'size="0.004 {s} 0.008" rgba="0.1 0.45 1.0 0.55" contype="0" conaffinity="0"/>',

        f'    <geom name="pick_container_right" type="box" pos="{c[0] + s} {c[1]} {z + 0.006}" '
        f'size="0.004 {s} 0.008" rgba="0.1 0.45 1.0 0.55" contype="0" conaffinity="0"/>',
    ]

    circle_rgba = (
        "1.0 0.55 0.05 0.45",
        "0.85 0.1 0.95 0.45",
    )

    for i, center in enumerate(polish_centers):
        marker_lines.append(
            f'    <geom name="polish_disc_{i + 1}" type="cylinder" '
            f'pos="{center[0]} {center[1]} {z}" '
            f'size="{POLISH_MARKER_RADIUS} 0.002" '
            f'rgba="{circle_rgba[i]}" contype="0" conaffinity="0"/>'
        )

    return "\n".join(marker_lines)


# ---------------------------------------------------------------------
# MuJoCo setup
# ---------------------------------------------------------------------

def make_model_and_planner():
    model = mujoco.MjModel.from_xml_path(make_augmented_xml_path())
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    site_id = -1
    site_name = None

    for candidate in EE_SITE_CANDIDATES:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            candidate,
        )

        if site_id != -1:
            site_name = candidate
            break

    if site_id == -1:
        raise RuntimeError(
            f"None of the EE sites {EE_SITE_CANDIDATES} were found in {XML_PATH}"
        )

    cube_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        CUBE_BODY_NAME,
    )

    if cube_body_id == -1:
        raise RuntimeError(f'Cube body "{CUBE_BODY_NAME}" not found in {XML_PATH}')

    cube_joint_id = find_cube_free_joint(model, cube_body_id)
    set_cube_spawn_pose(model, data, cube_joint_id)

    mujoco.mj_forward(model, data)

    home_pos = data.site_xpos[site_id].copy()

    planner = DLSVelocityPlanner(
        model=model,
        data=data,
        kd=5.0,
        site_name=site_name,
        damping=1e-2,
        gripper_cfg=[{"actuator_id": GRIPPER_ACTUATOR_ID}],
        for_multi=False,
        actuator_mode="position",
    )

    return model, data, planner, site_id, cube_body_id, home_pos


def find_cube_free_joint(model, cube_body_id: int) -> int:
    for joint_id in range(model.njnt):
        if model.jnt_bodyid[joint_id] != cube_body_id:
            continue

        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id

    raise RuntimeError(f'Cube body "{CUBE_BODY_NAME}" has no free joint')


def set_cube_spawn_pose(model, data, cube_joint_id: int) -> None:
    qpos_adr = model.jnt_qposadr[cube_joint_id]
    qvel_adr = model.jnt_dofadr[cube_joint_id]

    data.qpos[qpos_adr : qpos_adr + 3] = CUBE_SPAWN_POS
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = CUBE_SPAWN_QUAT_WXYZ
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def set_gripper(data, value: float) -> None:
    data.ctrl[GRIPPER_ACTUATOR_ID] = value


# ---------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------

def should_advance(
    phase: TaskPhase,
    phase_time: float,
    ee_pos: np.ndarray,
) -> bool:
    if phase_time >= phase.duration:
        return True

    if not phase.auto_advance:
        return False

    if phase_time < MIN_PHASE_TIME:
        return False

    return np.linalg.norm(ee_pos - phase.center) < GOAL_TOLERANCE


def build_task(cube_pos0: np.ndarray) -> list[TaskPhase]:
    grasp_center = cube_pos0 + np.array([0.0, 0.0, GRASP_Z_OFFSET])
    above_pick = grasp_center + np.array([0.0, 0.0, APPROACH_CLEARANCE])
    lifted_pick = grasp_center + np.array([0.0, 0.0, LIFT_HEIGHT])

    polish_centers = [cube_pos0 + offset for offset in POLISH_CENTER_OFFSETS]

    above_place = grasp_center + np.array([0.0, 0.0, PLACE_APPROACH_HEIGHT])
    home_clearance = grasp_center + np.array([0.0, 0.0, LIFT_HEIGHT])

    phases = [
        TaskPhase(
            name="approach 2 cm above cube",
            center=above_pick,
            duration=5.0,
            mode="point",
            gripper=OPEN_GRIPPER,
        ),

        TaskPhase(
            name="descend to grasp",
            center=grasp_center,
            duration=3.0,
            mode="point",
            gripper=OPEN_GRIPPER,
        ),

        TaskPhase(
            name="close gripper",
            center=grasp_center,
            duration=1.2,
            mode="point",
            gripper=CLOSED_GRIPPER,
            auto_advance=False,
        ),

        TaskPhase(
            name="lift cube",
            center=lifted_pick,
            duration=4.0,
            mode="point",
            gripper=CLOSED_GRIPPER,
        ),
    ]

    for i, polish_center in enumerate(polish_centers):
        phases.append(
            TaskPhase(
                name=f"move to polish disc {i + 1} center",
                center=polish_center,
                duration=5.0,
                mode="point",
                gripper=CLOSED_GRIPPER,
                max_speed=MAX_CARTESIAN_SPEED,
            )
        )

        phases.append(
            TaskPhase(
                name=f"hard-switch circular polish disc {i + 1}",
                center=polish_center,
                duration=POLISH_CYCLES_PER_CENTER * POLISH_CYCLE_PERIOD,
                mode="circle",
                radius=POLISH_RADIUS,
                omega=POLISH_OMEGA,
                gripper=CLOSED_GRIPPER,
                max_speed=MAX_POLISH_SPEED,
                auto_advance=False,
            )
        )

    phases.extend([
        TaskPhase(
            name="return high above cube start",
            center=above_place,
            duration=5.0,
            mode="point",
            gripper=CLOSED_GRIPPER,
        ),

        TaskPhase(
            name="lower cube back in place",
            center=grasp_center,
            duration=3.0,
            mode="point",
            gripper=CLOSED_GRIPPER,
        ),

        TaskPhase(
            name="release cube",
            center=grasp_center,
            duration=1.0,
            mode="point",
            gripper=OPEN_GRIPPER,
            auto_advance=False,
        ),

        TaskPhase(
            name="retreat 2 cm above cube",
            center=above_pick,
            duration=3.0,
            mode="point",
            gripper=OPEN_GRIPPER,
        ),

        TaskPhase(
            name="clear cube before home",
            center=home_clearance,
            duration=3.0,
            mode="point",
            gripper=OPEN_GRIPPER,
        ),
    ])

    return phases


# ---------------------------------------------------------------------
# Runtime control helpers
# ---------------------------------------------------------------------

def make_idle_control_info(
    phase_index: int,
    phase_name: str,
    phase_center: np.ndarray,
    phase_duration: float = 0.0,
    phase_time: float = 0.0,
    gripper_cmd: float = OPEN_GRIPPER,
    task_done: bool = False,
    home_done: bool = False,
) -> ControlInfo:
    return ControlInfo(
        phase_index=phase_index,
        phase_name=phase_name,
        phase_time=phase_time,
        phase_center=np.asarray(phase_center, dtype=float),
        phase_duration=phase_duration,
        phase_mode="idle",
        gripper_cmd=gripper_cmd,
        max_speed=0.0,
        radius=0.0,
        omega=0.0,
        raw_v_cmd=np.zeros(3),
        v_cmd=np.zeros(3),
        w_cmd=np.zeros(3),
        ref_pos=np.asarray(phase_center, dtype=float),
        ref_vel=np.zeros(3),
        control_active=False,
        task_done=task_done,
        home_done=home_done,
        hard_switch_velocity_jump=0.0,
    )


def command_home_pose(
    model,
    data,
    planner,
    site_id: int,
    home_pos: np.ndarray,
    target_quat: np.ndarray,
) -> tuple[bool, ControlInfo]:
    set_gripper(data, OPEN_GRIPPER)

    ee_pos = data.site_xpos[site_id].copy()

    raw_v_cmd = point_attractor_velocity(ee_pos, home_pos)
    v_cmd = limit_norm(raw_v_cmd, MAX_CARTESIAN_SPEED)
    w_cmd = orientation_velocity(model, data, site_id, target_quat)

    planner.track_twist(v_cmd, w_cart=w_cmd)

    home_done = np.linalg.norm(ee_pos - home_pos) < HOME_POS_TOL

    control_info = ControlInfo(
        phase_index=-2,
        phase_name="return_home",
        phase_time=0.0,
        phase_center=home_pos.copy(),
        phase_duration=0.0,
        phase_mode="point",
        gripper_cmd=OPEN_GRIPPER,
        max_speed=MAX_CARTESIAN_SPEED,
        radius=0.0,
        omega=0.0,
        raw_v_cmd=raw_v_cmd.copy(),
        v_cmd=v_cmd.copy(),
        w_cmd=w_cmd.copy(),
        ref_pos=home_pos.copy(),
        ref_vel=np.zeros(3),
        control_active=True,
        task_done=True,
        home_done=home_done,
        hard_switch_velocity_jump=0.0,
    )

    return home_done, control_info


def run_control_step(
    model,
    data,
    planner,
    site_id: int,
    target_quat: np.ndarray,
    phases: list[TaskPhase],
    phase_index: int,
    phase_start_time: float,
    last_raw_v_cmd,
) -> tuple[int, float, bool, np.ndarray, ControlInfo]:
    ee_pos = data.site_xpos[site_id].copy()

    phase = phases[phase_index]
    phase_time = data.time - phase_start_time

    advanced = False

    if should_advance(phase, phase_time, ee_pos):
        phase_index += 1

        if phase_index >= len(phases):
            control_info = make_idle_control_info(
                phase_index=phase_index,
                phase_name="task_done",
                phase_center=ee_pos.copy(),
                task_done=True,
                home_done=False,
            )
            return phase_index, phase_start_time, True, last_raw_v_cmd, control_info

        phase = phases[phase_index]
        phase_start_time = data.time
        phase_time = 0.0
        advanced = True

        print(f"[INFO] Phase {phase_index + 1}/{len(phases)}: {phase.name}")

    set_gripper(data, phase.gripper)

    raw_v_cmd, ref_pos, ref_vel = compute_phase_velocity(
        phase=phase,
        phase_time=phase_time,
        ee_pos=ee_pos,
    )

    hard_switch_velocity_jump = 0.0
    if advanced and last_raw_v_cmd is not None:
        hard_switch_velocity_jump = float(np.linalg.norm(raw_v_cmd - last_raw_v_cmd))
        print(f"[INFO] Hard-switch velocity jump: {hard_switch_velocity_jump:.4f} m/s")

    v_cmd = limit_norm(raw_v_cmd, phase.max_speed)
    w_cmd = orientation_velocity(model, data, site_id, target_quat)

    planner.track_twist(v_cmd, w_cart=w_cmd)

    control_info = ControlInfo(
        phase_index=phase_index,
        phase_name=phase.name,
        phase_time=phase_time,
        phase_center=phase.center.copy(),
        phase_duration=phase.duration,
        phase_mode=phase.mode,
        gripper_cmd=phase.gripper,
        max_speed=phase.max_speed,
        radius=phase.radius,
        omega=phase.omega,
        raw_v_cmd=raw_v_cmd.copy(),
        v_cmd=v_cmd.copy(),
        w_cmd=w_cmd.copy(),
        ref_pos=ref_pos.copy(),
        ref_vel=ref_vel.copy(),
        control_active=True,
        task_done=False,
        home_done=False,
        hard_switch_velocity_jump=hard_switch_velocity_jump,
    )

    return phase_index, phase_start_time, False, raw_v_cmd.copy(), control_info


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    model, data, planner, site_id, cube_body_id, home_pos = make_model_and_planner()

    cube_pos0 = data.xpos[cube_body_id].copy()
    ee_pos0 = data.site_xpos[site_id].copy()

    phases = build_task(cube_pos0)
    target_quat = TARGET_QUAT_XYZW.copy()

    logger = SimulationDataLogger(
        enabled=RECORD_DATA,
        csv_path=DATA_CSV_PATH,
    )

    set_gripper(data, OPEN_GRIPPER)

    print("[INFO] Hard-switch pick-polish-place baseline demo")
    print("[INFO] No rho, no bifurcation DS, no smooth radius ramp.")
    print("[INFO] Point phases use simple first-order attraction.")
    print("[INFO] Polish phases hard-switch to explicit circular trajectory tracking.")
    print("[INFO] Cube start:", cube_pos0)
    print("[INFO] Initial EE position:", ee_pos0)
    print("[INFO] First target:", phases[0].center)
    print("[INFO] Target polishing radius:", POLISH_RADIUS)
    print("[INFO] Target polishing period:", POLISH_CYCLE_PERIOD)
    print(f"[INFO] Holding simulation for {CONTROL_START_TIME} seconds before motion.")

    if RECORD_DATA:
        print("[INFO] Data recording enabled.")
        print("[INFO] CSV will be exported to:", DATA_CSV_PATH)

    if RECORD:
        recorder = MP4Recorder(
            model,
            RECORD_PATH,
            fps=RECORD_FPS,
            width=RECORD_WIDTH,
            height=RECORD_HEIGHT,
        )

        next_frame_time = 0.0
        sim_step_count = 0

        phase_index = 0
        phase_start_time = CONTROL_START_TIME

        task_done = False
        home_done = False

        last_raw_v_cmd = None

        print("[INFO] Recording MP4:", RECORD_PATH)
        print(f"[INFO] Phase 1/{len(phases)}: {phases[0].name}")

        try:
            while not home_done:
                if not task_done and data.time > CONTROL_START_TIME:
                    (
                        phase_index,
                        phase_start_time,
                        task_done,
                        last_raw_v_cmd,
                        control_info,
                    ) = run_control_step(
                        model=model,
                        data=data,
                        planner=planner,
                        site_id=site_id,
                        target_quat=target_quat,
                        phases=phases,
                        phase_index=phase_index,
                        phase_start_time=phase_start_time,
                        last_raw_v_cmd=last_raw_v_cmd,
                    )

                    if task_done:
                        print("[INFO] Task complete. Returning robot to Cartesian home.")

                elif task_done:
                    home_done, control_info = command_home_pose(
                        model=model,
                        data=data,
                        planner=planner,
                        site_id=site_id,
                        home_pos=home_pos,
                        target_quat=target_quat,
                    )

                else:
                    control_info = make_idle_control_info(
                        phase_index=-1,
                        phase_name="hold_before_start",
                        phase_center=phases[0].center,
                        phase_duration=CONTROL_START_TIME,
                        phase_time=data.time,
                        gripper_cmd=OPEN_GRIPPER,
                    )

                if sim_step_count % DATA_RECORD_EVERY_N_STEPS == 0:
                    logger.log(
                        model=model,
                        data=data,
                        site_id=site_id,
                        cube_body_id=cube_body_id,
                        control_info=control_info,
                    )

                mujoco.mj_step(model, data)
                sim_step_count += 1

                if data.time >= next_frame_time:
                    recorder.capture(data)
                    next_frame_time += 1.0 / RECORD_FPS

        finally:
            recorder.close()
            logger.export_csv()

        print("[INFO] Finished hard-switch pick-polish-place recording.")
        return

    v = viewer.launch_passive(model, data)

    print("[INFO] Viewer started. CSV will export when viewer is closed.")
    print(f"[INFO] Phase 1/{len(phases)}: {phases[0].name}")

    last_viewer_sync = time.perf_counter()
    viewer_sync_period = 1.0 / VIEWER_FPS

    sim_step_count = 0

    phase_index = 0
    phase_start_time = CONTROL_START_TIME

    task_done = False
    home_done = False

    task_done_msg_printed = False
    home_done_msg_printed = False

    last_raw_v_cmd = None

    try:
        while v.is_running():
            step_start = time.perf_counter()

            with v.lock():
                if not task_done and data.time > CONTROL_START_TIME:
                    (
                        phase_index,
                        phase_start_time,
                        task_done,
                        last_raw_v_cmd,
                        control_info,
                    ) = run_control_step(
                        model=model,
                        data=data,
                        planner=planner,
                        site_id=site_id,
                        target_quat=target_quat,
                        phases=phases,
                        phase_index=phase_index,
                        phase_start_time=phase_start_time,
                        last_raw_v_cmd=last_raw_v_cmd,
                    )

                    if task_done and not task_done_msg_printed:
                        print("[INFO] Task complete. Returning robot to Cartesian home.")
                        task_done_msg_printed = True

                elif task_done:
                    home_done, control_info = command_home_pose(
                        model=model,
                        data=data,
                        planner=planner,
                        site_id=site_id,
                        home_pos=home_pos,
                        target_quat=target_quat,
                    )

                else:
                    control_info = make_idle_control_info(
                        phase_index=-1,
                        phase_name="hold_before_start",
                        phase_center=phases[0].center,
                        phase_duration=CONTROL_START_TIME,
                        phase_time=data.time,
                        gripper_cmd=OPEN_GRIPPER,
                    )

                if sim_step_count % DATA_RECORD_EVERY_N_STEPS == 0:
                    logger.log(
                        model=model,
                        data=data,
                        site_id=site_id,
                        cube_body_id=cube_body_id,
                        control_info=control_info,
                    )

                mujoco.mj_step(model, data)
                sim_step_count += 1

            now = time.perf_counter()
            if now - last_viewer_sync >= viewer_sync_period:
                v.sync()
                last_viewer_sync = now

            if home_done:
                if CLOSE_SIMULATION:
                    print("[INFO] Home pose reached. Closing simulation.")
                    break

                if not home_done_msg_printed:
                    print("[INFO] Home pose reached. Viewer remains open.")
                    print("[INFO] Close the viewer to export the CSV.")
                    home_done_msg_printed = True

            sleep_time = model.opt.timestep - (time.perf_counter() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        v.close()
        logger.export_csv()


if __name__ == "__main__":
    main()