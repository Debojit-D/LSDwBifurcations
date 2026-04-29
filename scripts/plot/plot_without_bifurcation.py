import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Default CSV path for WITHOUT-BIFURCATION / HARD-SWITCH case
# ---------------------------------------------------------------------

DEFAULT_CSV_PATH = (
    "/home/iitgn-robotics/Debojit_WS/LSDwBifurcations/"
    "mujoco_hard_switch_pick_polish_place_log.csv"
)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def get_csv_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return DEFAULT_CSV_PATH


def make_output_dir(csv_path: str) -> str:
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    out_dir = os.path.join(csv_dir, "mujoco_hard_switch_plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def clean_time_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="first")
    df = df.reset_index(drop=True)
    return df


def clean_phase_display_name(name: str) -> str:
    """
    Cleans labels for plotting.

    Example:
    hard-switch circular polish disc 1 -> Circular Polish Disc 1
    move to polish disc 1 center       -> removed elsewhere by filtering circle phase only
    """
    name = str(name)

    replacements = {
        "hard-switch circular polish disc": "Circular Polish Disc",
        "Hard-Switch Circular Polish Disc": "Circular Polish Disc",
        "hard-switch Circular Polish Disc": "Circular Polish Disc",
        "Hard-switch Circular Polish Disc": "Circular Polish Disc",
        "circular polish disc": "Circular Polish Disc",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    name = name.replace("  ", " ").strip()

    # Fix capitalization if the CSV already contains lowercase names.
    if name.lower().startswith("circular polish disc"):
        parts = name.split()
        if len(parts) >= 4:
            name = "Circular Polish Disc " + parts[-1]

    return name


def get_active_df(df: pd.DataFrame) -> pd.DataFrame:
    if "control_active" in df.columns:
        active = df[df["control_active"] == 1].copy()
    else:
        active = df.copy()

    return clean_time_series(active)


def get_circle_df(df: pd.DataFrame) -> pd.DataFrame:
    active = get_active_df(df)

    if "phase_mode" in active.columns:
        circle = active[active["phase_mode"].astype(str).str.lower() == "circle"].copy()
    else:
        phase_name = active["phase_name"].astype(str)
        circle = active[
            phase_name.str.contains("circular", case=False, na=False)
            | phase_name.str.contains("hard-switch", case=False, na=False)
        ].copy()

    return clean_time_series(circle)


def phase_change_times(df: pd.DataFrame):
    if "phase_name" not in df.columns or len(df) < 2:
        return []

    phase = df["phase_name"].astype(str).to_numpy()
    t = df["time"].to_numpy()

    idx = np.where(phase[1:] != phase[:-1])[0] + 1
    return [(float(t[i]), clean_phase_display_name(phase[i])) for i in idx]


def add_phase_lines(ax, changes, add_labels=False):
    for tc, name in changes:
        ax.axvline(tc, linestyle="--", linewidth=1, alpha=0.45)

        if add_labels:
            y_min, y_max = ax.get_ylim()
            ax.text(
                tc,
                y_max,
                name,
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=7,
                alpha=0.75,
            )


def place_center_label(ax, x: float, y: float, text: str, offset_frac: float = 0.02) -> None:
    """
    Place a small offset label near (x, y) using axis-relative offset to avoid
    overwriting plotted markers. Adds a white translucent bbox for readability.
    """
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


def vector_norm_from_columns(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.linalg.norm(df[cols].to_numpy(dtype=float), axis=1)


def compute_acceleration(df: pd.DataFrame) -> np.ndarray:
    t = df["time"].to_numpy(dtype=float)
    v = df[["v_cmd_x", "v_cmd_y", "v_cmd_z"]].to_numpy(dtype=float)

    if len(t) < 3:
        return np.zeros_like(v)

    acc = np.zeros_like(v)

    for j in range(3):
        acc[:, j] = np.gradient(v[:, j], t)

    acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
    return acc


def compute_acceleration_norm(df: pd.DataFrame) -> np.ndarray:
    acc = compute_acceleration(df)
    return np.linalg.norm(acc, axis=1)


def compute_jerk_norm(df: pd.DataFrame) -> np.ndarray:
    t = df["time"].to_numpy(dtype=float)
    acc = compute_acceleration(df)

    if len(t) < 4:
        return np.zeros(len(t))

    jerk = np.zeros_like(acc)

    for j in range(3):
        jerk[:, j] = np.gradient(acc[:, j], t)

    jerk = np.nan_to_num(jerk, nan=0.0, posinf=0.0, neginf=0.0)
    return np.linalg.norm(jerk, axis=1)


def require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            "CSV is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nCheck that RECORD_DATA=True was used in the hard-switch MuJoCo script."
        )


def get_speed(df: pd.DataFrame) -> np.ndarray:
    if "v_cmd_norm" in df.columns:
        return df["v_cmd_norm"].to_numpy(dtype=float)

    return vector_norm_from_columns(df, ["v_cmd_x", "v_cmd_y", "v_cmd_z"])


def get_raw_speed(df: pd.DataFrame) -> np.ndarray:
    if "raw_v_cmd_norm" in df.columns:
        return df["raw_v_cmd_norm"].to_numpy(dtype=float)

    if {"raw_v_cmd_x", "raw_v_cmd_y", "raw_v_cmd_z"}.issubset(df.columns):
        return vector_norm_from_columns(
            df,
            ["raw_v_cmd_x", "raw_v_cmd_y", "raw_v_cmd_z"],
        )

    return get_speed(df)


def get_angular_speed(df: pd.DataFrame) -> np.ndarray:
    if "w_cmd_norm" in df.columns:
        return df["w_cmd_norm"].to_numpy(dtype=float)

    return vector_norm_from_columns(df, ["w_cmd_x", "w_cmd_y", "w_cmd_z"])


def get_radius_column(df: pd.DataFrame) -> np.ndarray:
    if "circle_radius" in df.columns:
        return df["circle_radius"].to_numpy(dtype=float)

    if "rho0" in df.columns:
        return df["rho0"].to_numpy(dtype=float)

    return np.zeros(len(df))


# ---------------------------------------------------------------------
# Plot 1: Main hard-switch smoothness / discontinuity plot
# ---------------------------------------------------------------------

def plot_command_smoothness(df: pd.DataFrame, out_dir: str):
    active = get_active_df(df)
    changes = phase_change_times(active)

    t = active["time"].to_numpy(dtype=float)
    speed = get_speed(active)
    raw_speed = get_raw_speed(active)
    acc_norm = compute_acceleration_norm(active)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t, raw_speed, linewidth=1.6, label=r"$\|v_{raw}\|$")
    axes[0].plot(t, speed, linewidth=1.8, label=r"$\|v_{cmd}\|$ after clipping")
    axes[0].set_ylabel("Velocity [m/s]")
    axes[0].set_title("Without-Bifurcation Command Discontinuity During Pick-Polish-Place")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, acc_norm, linewidth=1.8)
    axes[1].set_ylabel(r"$\|a_{cmd}\|$ [m/s$^2$]")
    axes[1].grid(True)

    jerk_norm = compute_jerk_norm(active)
    axes[2].plot(t, jerk_norm, linewidth=1.5)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel(r"$\|j_{cmd}\|$ [m/s$^3$]")
    axes[2].grid(True)

    add_phase_lines(axes[0], changes, add_labels=True)
    add_phase_lines(axes[1], changes, add_labels=False)
    add_phase_lines(axes[2], changes, add_labels=False)

    save_and_close(fig, out_dir, "01_hard_switch_command_velocity_acceleration_jerk.png")


# ---------------------------------------------------------------------
# Plot 2: Explicit hard-switch velocity jump plot
# ---------------------------------------------------------------------

def plot_hard_switch_velocity_jumps(df: pd.DataFrame, out_dir: str):
    if "hard_switch_velocity_jump" not in df.columns:
        print("[WARN] hard_switch_velocity_jump column not found. Skipping jump plot.")
        return

    active = get_active_df(df)
    jumps = active[active["hard_switch_velocity_jump"].to_numpy(dtype=float) > 1e-12].copy()

    fig, ax = plt.subplots(figsize=(11, 4))

    if jumps.empty:
        ax.text(
            0.5,
            0.5,
            "No non-zero hard-switch velocity jumps logged",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
    else:
        ax.stem(
            jumps["time"].to_numpy(dtype=float),
            jumps["hard_switch_velocity_jump"].to_numpy(dtype=float),
        )

        for _, row in jumps.iterrows():
            ax.text(
                row["time"],
                row["hard_switch_velocity_jump"],
                clean_phase_display_name(row["phase_name"]),
                rotation=90,
                verticalalignment="bottom",
                fontsize=7,
                alpha=0.75,
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity jump [m/s]")
    ax.set_title("Measured Velocity Jump at Without-Bifurcation Phase Transitions")
    ax.grid(True)

    save_and_close(fig, out_dir, "02_hard_switch_velocity_jumps.png")


# ---------------------------------------------------------------------
# Plot 3: End-effector 3D trajectory
# ---------------------------------------------------------------------

def plot_ee_trajectory_3d(df: pd.DataFrame, out_dir: str):
    active = get_active_df(df)

    x = active["ee_x"].to_numpy(dtype=float)
    y = active["ee_y"].to_numpy(dtype=float)
    z = active["ee_z"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=2, label="End-effector trajectory")

    if {"phase_center_x", "phase_center_y", "phase_center_z", "phase_name"}.issubset(active.columns):
        centers = active[
            ["phase_center_x", "phase_center_y", "phase_center_z", "phase_name"]
        ].drop_duplicates(
            subset=["phase_name", "phase_center_x", "phase_center_y", "phase_center_z"]
        )

        ax.scatter(
            centers["phase_center_x"],
            centers["phase_center_y"],
            centers["phase_center_z"],
            s=35,
            label="Phase centers",
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Without-Bifurcation End-Effector 3D Trajectory")
    ax.legend()

    save_and_close(fig, out_dir, "03_hard_switch_end_effector_3d_trajectory.png")


# ---------------------------------------------------------------------
# Plot 4: End-effector XY path with explicit circular references
# ---------------------------------------------------------------------

def plot_ee_xy_path(df: pd.DataFrame, out_dir: str):
    active = get_active_df(df)
    circle = get_circle_df(df)

    x = active["ee_x"].to_numpy(dtype=float)
    y = active["ee_y"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(x, y, linewidth=2, label="End-effector path")
    ax.scatter(x[0], y[0], s=50, label="Start")
    ax.scatter(x[-1], y[-1], s=50, label="End")

    # Plot only circular reference trajectories.
    if {"ref_x", "ref_y"}.issubset(circle.columns) and not circle.empty:
        ref_colors = ["#f0ad4e", "#2ca02c"]

        for idx, (phase_name, group) in enumerate(circle.groupby("phase_name")):
            group = clean_time_series(group)
            ref_color = ref_colors[idx % len(ref_colors)]
            display_name = clean_phase_display_name(phase_name)

            ax.plot(
                group["ref_x"],
                group["ref_y"],
                linestyle="--",
                linewidth=1.4,
                color=ref_color,
                label=f"{display_name} reference",
            )

    # IMPORTANT FIX:
    # Use circle dataframe for centers/labels, not active dataframe.
    # This removes labels like "move to polish disc 1 center".
    if {"phase_center_x", "phase_center_y", "phase_name"}.issubset(circle.columns) and not circle.empty:
        centers = circle[
            ["phase_center_x", "phase_center_y", "phase_name"]
        ].drop_duplicates(
            subset=["phase_center_x", "phase_center_y"]
        )

        ax.scatter(
            centers["phase_center_x"],
            centers["phase_center_y"],
            s=35,
            label="Phase centers",
        )

        for _, row in centers.iterrows():
            name = clean_phase_display_name(row["phase_name"])

            place_center_label(
                ax,
                float(row["phase_center_x"]),
                float(row["phase_center_y"]),
                name,
            )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Without-Bifurcation End-Effector XY Path")
    ax.axis("equal")
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()

    # Make reference trajectories appear solid in legend only,
    # while keeping them dashed in the actual plot.
    if any("reference" in label.lower() for label in labels):
        proxy_handles = []
        proxy_labels = []

        for handle, label in zip(handles, labels):
            if "reference" in label.lower():
                proxy_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=handle.get_color(),
                        linestyle="-",
                        linewidth=handle.get_linewidth(),
                    )
                )
            else:
                proxy_handles.append(handle)

            proxy_labels.append(label)

        ax.legend(proxy_handles, proxy_labels, fontsize=8)
    else:
        ax.legend(fontsize=8)

    save_and_close(fig, out_dir, "04_hard_switch_end_effector_xy_path.png")


# ---------------------------------------------------------------------
# Plot 5: Circular tracking radius error
# ---------------------------------------------------------------------

def plot_circle_radial_error(df: pd.DataFrame, out_dir: str):
    circle = get_circle_df(df)

    if circle.empty:
        print("[WARN] No circle phase found in CSV. Skipping radial error plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 4))

    for phase_name, group in circle.groupby("phase_name"):
        group = clean_time_series(group)

        t = group["time"].to_numpy(dtype=float)
        t_local = t - t[0]

        if "radial_error_xy" in group.columns:
            radial_error = group["radial_error_xy"].to_numpy(dtype=float)
        else:
            dx = group["ee_x"].to_numpy(dtype=float) - group["phase_center_x"].to_numpy(dtype=float)
            dy = group["ee_y"].to_numpy(dtype=float) - group["phase_center_y"].to_numpy(dtype=float)
            radius = get_radius_column(group)
            radial_error = np.sqrt(dx**2 + dy**2) - radius

        ax.plot(t_local, radial_error, linewidth=2, label=clean_phase_display_name(phase_name))

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Local circular polishing time [s]")
    ax.set_ylabel("Radial error [m]")
    ax.set_title("Radial Error During Without-Bifurcation Circular Polishing")
    ax.grid(True)
    ax.legend()

    save_and_close(fig, out_dir, "05_hard_switch_circle_radial_error.png")


# ---------------------------------------------------------------------
# Plot 6: Radius tracking during explicit circular polishing
# ---------------------------------------------------------------------

def plot_circle_radius_tracking(df: pd.DataFrame, out_dir: str):
    circle = get_circle_df(df)

    if circle.empty:
        print("[WARN] No circle phase found in CSV. Skipping radius tracking plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 4))

    for phase_name, group in circle.groupby("phase_name"):
        group = clean_time_series(group)
        display_name = clean_phase_display_name(phase_name)

        t = group["time"].to_numpy(dtype=float)
        t_local = t - t[0]

        if "xy_radius_about_center" in group.columns:
            radius_actual = group["xy_radius_about_center"].to_numpy(dtype=float)
        else:
            dx = group["ee_x"].to_numpy(dtype=float) - group["phase_center_x"].to_numpy(dtype=float)
            dy = group["ee_y"].to_numpy(dtype=float) - group["phase_center_y"].to_numpy(dtype=float)
            radius_actual = np.sqrt(dx**2 + dy**2)

        radius_target = get_radius_column(group)

        ax.plot(
            t_local,
            radius_actual,
            linewidth=2,
            label=f"{display_name} actual radius",
        )

        ax.plot(
            t_local,
            radius_target,
            linestyle="--",
            linewidth=1.2,
            label=f"{display_name} target radius",
        )

    ax.set_xlabel("Local circular polishing time [s]")
    ax.set_ylabel("XY radius about polish center [m]")
    ax.set_title("Without-Bifurcation Explicit Circular Radius Tracking")
    ax.grid(True)
    ax.legend(fontsize=8)

    save_and_close(fig, out_dir, "06_hard_switch_circle_radius_tracking.png")


# ---------------------------------------------------------------------
# Plot 7: Reference vs actual XY during circular phases
# ---------------------------------------------------------------------

def plot_circle_reference_tracking_xy(df: pd.DataFrame, out_dir: str):
    circle = get_circle_df(df)

    required = {"ref_x", "ref_y", "ee_x", "ee_y", "phase_name"}

    if circle.empty or not required.issubset(circle.columns):
        print("[WARN] Circle reference columns not found. Skipping reference tracking plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    for phase_name, group in circle.groupby("phase_name"):
        group = clean_time_series(group)
        display_name = clean_phase_display_name(phase_name)

        ax.plot(
            group["ref_x"],
            group["ref_y"],
            linestyle="--",
            linewidth=1.8,
            label=f"{display_name} reference",
        )

        ax.plot(
            group["ee_x"],
            group["ee_y"],
            linewidth=2,
            label=f"{display_name} actual",
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Without-Bifurcation Circular Reference vs Actual EE Path")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(fontsize=8)

    save_and_close(fig, out_dir, "07_hard_switch_circle_reference_vs_actual_xy.png")


# ---------------------------------------------------------------------
# Plot 8: Gripper command and phase timeline
# ---------------------------------------------------------------------

def plot_gripper_and_phase(df: pd.DataFrame, out_dir: str):
    active = clean_time_series(df.copy())
    changes = phase_change_times(active)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(active["time"], active["gripper_cmd"], linewidth=2)
    axes[0].set_ylabel("Gripper command [m]")
    axes[0].set_title("Without-Bifurcation Gripper Command and Phase Timeline")
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

    save_and_close(fig, out_dir, "08_hard_switch_gripper_and_phase_timeline.png")


# ---------------------------------------------------------------------
# Plot 9: Angular command norm
# ---------------------------------------------------------------------

def plot_angular_velocity(df: pd.DataFrame, out_dir: str):
    active = get_active_df(df)
    changes = phase_change_times(active)

    t = active["time"].to_numpy(dtype=float)
    w_norm = get_angular_speed(active)

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(t, w_norm, linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|\omega_{cmd}\|$ [rad/s]")
    ax.set_title("Without-Bifurcation Commanded Angular Velocity Norm")
    ax.grid(True)

    add_phase_lines(ax, changes, add_labels=True)

    save_and_close(fig, out_dir, "09_hard_switch_commanded_angular_velocity.png")


# ---------------------------------------------------------------------
# Plot 10: Cartesian velocity components
# ---------------------------------------------------------------------

def plot_velocity_components(df: pd.DataFrame, out_dir: str):
    active = get_active_df(df)
    changes = phase_change_times(active)

    t = active["time"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(t, active["v_cmd_x"], linewidth=1.5, label=r"$v_x$")
    ax.plot(t, active["v_cmd_y"], linewidth=1.5, label=r"$v_y$")
    ax.plot(t, active["v_cmd_z"], linewidth=1.5, label=r"$v_z$")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Commanded velocity [m/s]")
    ax.set_title("Without-Bifurcation Cartesian Velocity Components")
    ax.grid(True)
    ax.legend()

    add_phase_lines(ax, changes, add_labels=True)

    save_and_close(fig, out_dir, "10_hard_switch_velocity_components.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    csv_path = get_csv_path()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n\n"
            "Pass the CSV path as:\n"
            "python plot_hard_switch_mujoco_log.py "
            "/path/to/mujoco_hard_switch_pick_polish_place_log.csv"
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
        "gripper_cmd",
    ]

    require_columns(df, required_cols)

    print("[INFO] Loaded hard-switch CSV:", csv_path)
    print("[INFO] Number of samples:", len(df))
    print("[INFO] Time range:", df["time"].iloc[0], "to", df["time"].iloc[-1])
    print("[INFO] Output directory:", out_dir)

    if "phase_mode" in df.columns:
        print("[INFO] Phase modes found:", sorted(df["phase_mode"].astype(str).unique()))

    plot_command_smoothness(df, out_dir)
    plot_hard_switch_velocity_jumps(df, out_dir)
    plot_ee_trajectory_3d(df, out_dir)
    plot_ee_xy_path(df, out_dir)
    plot_circle_radial_error(df, out_dir)
    plot_circle_radius_tracking(df, out_dir)
    plot_circle_reference_tracking_xy(df, out_dir)
    plot_gripper_and_phase(df, out_dir)
    plot_angular_velocity(df, out_dir)
    plot_velocity_components(df, out_dir)

    print("[INFO] All without-bifurcation plots generated.")


if __name__ == "__main__":
    main()