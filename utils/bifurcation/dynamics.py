from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BifurcationParams:
    """Parameters for the Hopf-inspired point/limit-cycle dynamical system."""

    center: np.ndarray
    rho0: float = 0.0
    M: float = 4.0
    R: float = 4.0


@dataclass(frozen=True)
class HardSwitchParams:
    """Parameters for the baseline that switches between unrelated controllers."""

    center: np.ndarray
    radius: float = 0.04
    reach_gain: float = 1.6
    cycle_period: float = 6.0


def smoothstep(s: float) -> float:
    """Cubic smoothstep on [0, 1]."""
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def limit_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale a vector down if its norm is above max_norm."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm > max_norm and norm > 1e-12:
        return v * (max_norm / norm)
    return v


def _radial_tangent_2d(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rho = np.linalg.norm(y)
    if rho < 1e-9:
        radial = np.array([1.0, 0.0])
    else:
        radial = y / rho
    tangent = np.array([-radial[1], radial[0]])
    return radial, tangent, rho


def ds_2d(x: np.ndarray, params: BifurcationParams) -> np.ndarray:
    """Hopf-inspired 2D DS from the notebook.

    rho0 = 0 gives a stable point attractor at center.
    rho0 > 0 gives a stable limit cycle of radius rho0 around center.
    """
    x = np.asarray(x, dtype=float)
    center = np.asarray(params.center, dtype=float)
    y = x - center[:2]

    radial, tangent, rho = _radial_tangent_2d(y)
    rho_dot = -np.sqrt(params.M) * (rho - params.rho0)
    theta_dot = params.R * np.exp(-(params.M**2) * (rho - params.rho0) ** 2)

    return rho_dot * radial + rho * theta_dot * tangent


def ds_3d(x: np.ndarray, params: BifurcationParams) -> np.ndarray:
    """3D version: bifurcation dynamics in XY, stable convergence in Z."""
    x = np.asarray(x, dtype=float)
    center = np.asarray(params.center, dtype=float)

    xy_dot = ds_2d(x[:2], params)
    z_dot = -np.sqrt(params.M) * (x[2] - center[2])
    return np.array([xy_dot[0], xy_dot[1], z_dot])


def linear_reach_velocity(x: np.ndarray, goal: np.ndarray, gain: float) -> np.ndarray:
    """Separate point-to-point controller used by the hard-switch baseline."""
    return -gain * (np.asarray(x, dtype=float) - np.asarray(goal, dtype=float))


def circle_reference(
    center: np.ndarray,
    radius: float,
    cycle_period: float,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return position and feed-forward velocity on a horizontal circle."""
    center = np.asarray(center, dtype=float)
    omega = 2.0 * np.pi / cycle_period
    theta = omega * t
    offset = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
    tangent = radius * omega * np.array([-np.sin(theta), np.cos(theta), 0.0])
    return center + offset, tangent


def hard_switched_velocity(
    x: np.ndarray,
    t: float,
    switch_time: float,
    params: HardSwitchParams,
    cycle_gain: float = 2.0,
) -> np.ndarray:
    """Baseline velocity with a discontinuous controller switch."""
    if t < switch_time:
        return linear_reach_velocity(x, params.center, params.reach_gain)

    ref_pos, ref_vel = circle_reference(
        params.center,
        params.radius,
        params.cycle_period,
        t - switch_time,
    )
    return ref_vel + cycle_gain * (ref_pos - np.asarray(x, dtype=float))
