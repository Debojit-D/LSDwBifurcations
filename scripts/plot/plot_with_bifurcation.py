import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Change this default path if you do not want to pass path from terminal
# ---------------------------------------------------------------------

DEFAULT_CSV_PATH = (
    "/home/iitgn-robotics/Debojit_WS/LSDwBifurcations/"
    "mujoco_bifurcation_pick_polish_place_log.csv"
)

POLISH_RAMP_TIME = 2.0

# Only highlight the portion that is actually near the circular orbit.
# This avoids coloring the entire transition trajectory.
CIRCULAR_RADIUS_TOL = 0.012

# Optional: skip a little bit after the ramp so the transient into the circle
# is not colored as part of the circular orbit.
CIRCULAR_HIGHLIGHT_EXTRA_DELAY = 0.3


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def get_csv_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return DEFAULT_CSV_PATH


def make_output_dir(csv_path: str) -> str:
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    out_dir = os.path.join(csv_dir, "mujoco_plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def clean_time_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="first")
    df = df.reset_index(drop=True)
    return df


def phase_change_times(df: pd.DataFrame):
    if "phase_name" not in df.columns:
        return []

    phase = df["phase_name"].astype(str).to_numpy()
    t = df["time"].to_numpy()

    idx = np.where(phase[1:] != phase[:-1])[0] + 1
    return [(t[i], phase[i]) for i in idx]


def add_phase_lines(ax, changes, add_labels=False):
    for tc, name in changes:
        ax.axvline(tc, linestyle="--", linewidth=1, alpha=0.45)

        if add_labels:
            y_top = ax.get_ylim()[1]
            ax.text(
                tc,
                y_top,
                name,
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=7,
                alpha=0.75,
            )


def place_center_label(ax, x: float, y: float, text: str, offset_frac: float = 0.02) -> None:
    """Place a small offset label near (x, y) with a subtle white background."""
    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    except Exception:
        xlim = (x - 0.05, x + 0.05)
        ylim = (y - 0.05, y + 0.05)

    dx = (xlim[1] - xlim[0]) * offset_frac
    dy = (ylim[1] - ylim[0]) * offset_frac
    ha = "left" if x >= (xlim[0] + xlim[1]) / 2.0 else "right"

    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        fontsize=7,
        ha=ha,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
    )


def save_and_close(fig, out_dir: str, filename: str):
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {path}")


def compute_acceleration_norm(df: pd.DataFrame):
    t = df["time"].to_numpy()

    vx = df["v_cmd_x"].to_numpy()
    vy = df["v_cmd_y"].to_numpy()
    vz = df["v_cmd_z"].to_numpy()

    v = np.column_stack([vx, vy, vz])

    if len(t) < 3:
        return np.zeros_like(t)

    dt = np.gradient(t)
    dt[dt <= 1e-9] = np.nan

    a = np.gradient(v, axis=0) / dt[:, None]
    a_norm = np.linalg.norm(a, axis=1)

    return np.nan_to_num(a_norm, nan=0.0, posinf=0.0, neginf=0.0)


def compute_jerk_norm(df: pd.DataFrame):
    t = df["time"].to_numpy()

    ax = np.gradient(df["v_cmd_x"].to_numpy(), t)
    ay = np.gradient(df["v_cmd_y"].to_numpy(), t)
    az = np.gradient(df["v_cmd_z"].to_numpy(), t)

    acc = np.column_stack([ax, ay, az])

    if len(t) < 4:
        return np.zeros_like(t)

    jerk = np.gradient(acc, axis=0) / np.gradient(t)[:, None]
    jerk_norm = np.linalg.norm(jerk, axis=1)

    return np.nan_to_num(jerk_norm, nan=0.0, posinf=0.0, neginf=0.0)


def get_active_df(df: pd.DataFrame):
    if "control_active" in df.columns:
        active = df[df["control_active"] == 1].copy()
    else:
        active = df.copy()

    return clean_time_series(active)


def get_polish_df(df: pd.DataFrame) -> pd.DataFrame:
    active = get_active_df(df)

    if "phase_name" not in active.columns:
        return pd.DataFrame()

    polish = active[
        active["phase_name"].astype(str).str.contains("polish", case=False, na=False)
    ].copy()

    return clean_time_series(polish)


def split_masked_segments(x: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """
    Convert a boolean mask into continuous x-y segments.
    This prevents matplotlib from drawing unwanted straight lines across gaps.
    """
    segments = []

    if len(mask) == 0 or not np.any(mask):
        return segments

    idx = np.where(mask)[0]

    split_points = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, split_points)

    for g in groups:
        if len(g) >= 2:
            segments.append((x[g], y[g]))

    return segments


def circular_orbit_mask(group: pd.DataFrame) -> np.ndarray:
    """
    Returns True only for samples that are actually on/near the circular orbit.
    This is the main fix: the full polishing phase is not highlighted anymore.
    """
    group = clean_time_series(group)

    t_local = group["time"].to_numpy(dtype=float) - float(group["time"].iloc[0])

    if "xy_radius_about_center" in group.columns:
        radius_actual = group["xy_radius_about_center"].to_numpy(dtype=float)
    else:
        dx = group["ee_x"].to_numpy(dtype=float) - group["phase_center_x"].to_numpy(dtype=float)
        dy = group["ee_y"].to_numpy(dtype=float) - group["phase_center_y"].to_numpy(dtype=float)
        radius_actual = np.sqrt(dx**2 + dy**2)

    if "rho0" in group.columns:
        target_radius = group["rho0"].to_numpy(dtype=float)
    else:
        target_radius = np.full(len(group), np.nanmax(radius_actual))

    # Avoid highlighting point-attractor/ramp part.
    time_mask = t_local >= (POLISH_RAMP_TIME + CIRCULAR_HIGHLIGHT_EXTRA_DELAY)

    # Highlight only near the final circular orbit.
    radius_mask = np.abs(radius_actual - target_radius) <= CIRCULAR_RADIUS_TOL

    # Target radius should be non-trivial.
    valid_radius_mask = target_radius > 1e-4

    return time_mask & radius_mask & valid_radius_mask


# ---------------------------------------------------------------------
# Plot 1: Most important plot — command smoothness
# ---------------------------------------------------------------------

def plot_command_smoothness(df: pd.DataFrame, out_dir: str):
    active = get_active_df(df)
    changes = phase_change_times(active)

    t = active["time"].to_numpy()

    if "v_cmd_norm" in active.columns:
        speed = active["v_cmd_norm"].to_numpy()
    else:
        speed = np.linalg.norm(
            active[["v_cmd_x", "v_cmd_y", "v_cmd_z"]].to_numpy(),
            axis=1,
        )

    acc_norm = compute_acceleration_norm(active)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(t, speed, linewidth=2)
    axes[0].set_ylabel(r"$\|v_{cmd}\|$ [m/s]")
    axes[0].set_title("Command Smoothness During MuJoCo Pick-Polish-Place Task")
    axes[0].grid(True)

    axes[1].plot(t, acc_norm, linewidth=2)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"$\|a_{cmd}\|$ [m/s$^2$]")
    axes[1].grid(True)

    add_phase_lines(axes[0], changes, add_labels=True)
    add_phase_lines(axes[1], changes, add_labels=False)

    save_and_close(fig, out_dir, "01_command_smoothness_velocity_acceleration.png")


# ---------------------------------------------------------------------
# Plot 2: End-effector 3D trajectory
# ---------------------------------------------------------------------

def plot_ee_trajectory_3d(df: pd.DataFrame, out_dir: str):
    active = get_active_df(df)

    x = active["ee_x"].to_numpy()
    y = active["ee_y"].to_numpy()
    z = active["ee_z"].to_numpy()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=2)

    if {"phase_center_x", "phase_center_y", "phase_center_z"}.issubset(active.columns):
        centers = active[["phase_center_x", "phase_center_y", "phase_center_z", "phase_name"]]
        centers = centers.drop_duplicates(
            subset=["phase_name", "phase_center_x", "phase_center_y", "phase_center_z"]
        )

        ax.scatter(
            centers["phase_center_x"],
            centers["phase_center_y"],
            centers["phase_center_z"],
            s=35,
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("End-Effector 3D Trajectory")

    save_and_close(fig, out_dir, "02_end_effector_3d_trajectory.png")


# ---------------------------------------------------------------------
# Plot 3: End-effector XY path over polish centers
# ---------------------------------------------------------------------

def plot_ee_xy_path(df: pd.DataFrame, out_dir: str):
    active = get_active_df(df)
    polish = get_polish_df(df)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Full path remains one neutral/base color.
    ax.plot(
        active["ee_x"].to_numpy(dtype=float),
        active["ee_y"].to_numpy(dtype=float),
        linewidth=2,
        color="tab:blue",
        label="End-effector path",
        zorder=2,
    )

    ax.scatter(
        active["ee_x"].iloc[0],
        active["ee_y"].iloc[0],
        s=50,
        label="Start",
        zorder=5,
    )

    ax.scatter(
        active["ee_x"].iloc[-1],
        active["ee_y"].iloc[-1],
        s=50,
        label="End",
        zorder=5,
    )

    # Only polish centers are labeled/scattered here.
    if {"phase_center_x", "phase_center_y", "phase_name"}.issubset(polish.columns) and not polish.empty:
        centers = polish[
            ["phase_center_x", "phase_center_y", "phase_name"]
        ].drop_duplicates(
            subset=["phase_center_x", "phase_center_y"]
        )

        ax.scatter(
            centers["phase_center_x"],
            centers["phase_center_y"],
            s=35,
            label="Polish centers",
            zorder=6,
        )

        for _, row in centers.iterrows():
            place_center_label(
                ax,
                float(row["phase_center_x"]),
                float(row["phase_center_y"]),
                str(row["phase_name"]),
            )

    # Highlight only the samples that are actually on the circular orbit.
    polish_colors = ["#f0ad4e", "#2ca02c"]
    polish_idx = 0

    if not polish.empty:
        for phase_name, group in polish.groupby("phase_name", sort=False):
            group = clean_time_series(group)

            mask = circular_orbit_mask(group)

            xg = group["ee_x"].to_numpy(dtype=float)
            yg = group["ee_y"].to_numpy(dtype=float)

            segments = split_masked_segments(xg, yg, mask)

            if not segments:
                continue

            color = polish_colors[polish_idx % len(polish_colors)]
            label = f"{phase_name} circular orbit"

            first_segment = True
            for xs, ys in segments:
                ax.plot(
                    xs,
                    ys,
                    linewidth=2.8,
                    color=color,
                    zorder=4,
                    label=label if first_segment else None,
                )
                first_segment = False

            polish_idx += 1

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("End-Effector XY Path")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(fontsize=8)

    save_and_close(fig, out_dir, "03_end_effector_xy_path.png")


# ---------------------------------------------------------------------
# Plot 4: Radial convergence during polishing
# ---------------------------------------------------------------------

def plot_polish_radial_error(df: pd.DataFrame, out_dir: str):
    if "phase_name" not in df.columns:
        print("[WARN] Cannot plot radial error: phase_name column missing.")
        return

    active = get_active_df(df)
    polish = active[active["phase_name"].astype(str).str.contains("polish", case=False, na=False)].copy()

    if polish.empty:
        print("[WARN] No polish phase found in CSV.")
        return

    fig, ax = plt.subplots(figsize=(11, 4))

    for phase_name, group in polish.groupby("phase_name"):
        group = clean_time_series(group)

        t = group["time"].to_numpy()
        t = t - t[0]

        if "radial_error_xy" in group.columns:
            radial_error = group["radial_error_xy"].to_numpy()
        else:
            dx = group["ee_x"].to_numpy() - group["phase_center_x"].to_numpy()
            dy = group["ee_y"].to_numpy() - group["phase_center_y"].to_numpy()
            radial_error = np.sqrt(dx**2 + dy**2) - group["rho0"].to_numpy()

        ax.plot(t, radial_error, linewidth=2, label=str(phase_name))

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Local polish phase time [s]")
    ax.set_ylabel("Radial error [m]")
    ax.set_title("Radial Error During Limit-Cycle Polishing")
    ax.grid(True)
    ax.legend()

    save_and_close(fig, out_dir, "04_polish_radial_error.png")


# ---------------------------------------------------------------------
# Plot 5: Radius achieved during polishing
# ---------------------------------------------------------------------

def plot_polish_radius(df: pd.DataFrame, out_dir: str):
    if "phase_name" not in df.columns:
        print("[WARN] Cannot plot polish radius: phase_name column missing.")
        return

    active = get_active_df(df)
    polish = active[active["phase_name"].astype(str).str.contains("polish", case=False, na=False)].copy()

    if polish.empty:
        print("[WARN] No polish phase found in CSV.")
        return

    fig, ax = plt.subplots(figsize=(11, 4))

    for phase_name, group in polish.groupby("phase_name"):
        group = clean_time_series(group)

        t = group["time"].to_numpy()
        t = t - t[0]

        if "xy_radius_about_center" in group.columns:
            radius = group["xy_radius_about_center"].to_numpy()
        else:
            dx = group["ee_x"].to_numpy() - group["phase_center_x"].to_numpy()
            dy = group["ee_y"].to_numpy() - group["phase_center_y"].to_numpy()
            radius = np.sqrt(dx**2 + dy**2)

        rho0 = group["rho0"].to_numpy()

        ax.plot(t, radius, linewidth=2, label=f"{phase_name} actual radius")
        ax.plot(t, rho0, linestyle="--", linewidth=1, label=f"{phase_name} target/ramped radius")

    ax.set_xlabel("Local polish phase time [s]")
    ax.set_ylabel("XY radius about polish center [m]")
    ax.set_title("Polishing Radius Tracking")
    ax.grid(True)
    ax.legend()

    save_and_close(fig, out_dir, "05_polish_radius_tracking.png")


# ---------------------------------------------------------------------
# Plot 6: Gripper command and phase timeline
# ---------------------------------------------------------------------

def plot_gripper_and_phase(df: pd.DataFrame, out_dir: str):
    active = clean_time_series(df.copy())
    changes = phase_change_times(active)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(active["time"], active["gripper_cmd"], linewidth=2)
    axes[0].set_ylabel("Gripper command [m]")
    axes[0].set_title("Gripper Command and Phase Timeline")
    axes[0].grid(True)

    if "phase_index" in active.columns:
        axes[1].step(active["time"], active["phase_index"], where="post", linewidth=2)
        axes[1].set_ylabel("Phase index")
    else:
        phase_codes = pd.factorize(active["phase_name"].astype(str))[0]
        axes[1].step(active["time"], phase_codes, where="post", linewidth=2)
        axes[1].set_ylabel("Phase code")

    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True)

    add_phase_lines(axes[0], changes, add_labels=True)
    add_phase_lines(axes[1], changes, add_labels=False)

    save_and_close(fig, out_dir, "06_gripper_and_phase_timeline.png")


# ---------------------------------------------------------------------
# Plot 7: Angular command smoothness
# ---------------------------------------------------------------------

def plot_angular_velocity(df: pd.DataFrame, out_dir: str):
    active = get_active_df(df)
    changes = phase_change_times(active)

    t = active["time"].to_numpy()

    if "w_cmd_norm" in active.columns:
        w_norm = active["w_cmd_norm"].to_numpy()
    else:
        w_norm = np.linalg.norm(
            active[["w_cmd_x", "w_cmd_y", "w_cmd_z"]].to_numpy(),
            axis=1,
        )

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(t, w_norm, linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|\omega_{cmd}\|$ [rad/s]")
    ax.set_title("Commanded Angular Velocity Norm")
    ax.grid(True)

    add_phase_lines(ax, changes, add_labels=True)

    save_and_close(fig, out_dir, "07_commanded_angular_velocity.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    csv_path = get_csv_path()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "Pass the path as:\n"
            "python plot_mujoco_log.py /path/to/mujoco_bifurcation_pick_polish_place_log.csv"
        )

    out_dir = make_output_dir(csv_path)

    df = pd.read_csv(csv_path)
    df = clean_time_series(df)

    required_cols = [
        "time",
        "ee_x",
        "ee_y",
        "ee_z",
        "v_cmd_x",
        "v_cmd_y",
        "v_cmd_z",
        "phase_name",
        "phase_index",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nCheck that RECORD_DATA=True was used in the MuJoCo script."
        )

    print("[INFO] Loaded CSV:", csv_path)
    print("[INFO] Number of samples:", len(df))
    print("[INFO] Time range:", df["time"].iloc[0], "to", df["time"].iloc[-1])
    print("[INFO] Output directory:", out_dir)

    plot_command_smoothness(df, out_dir)
    plot_ee_trajectory_3d(df, out_dir)
    plot_ee_xy_path(df, out_dir)
    plot_polish_radial_error(df, out_dir)
    plot_polish_radius(df, out_dir)
    plot_gripper_and_phase(df, out_dir)
    plot_angular_velocity(df, out_dir)

    print("[INFO] All plots generated.")


if __name__ == "__main__":
    main()