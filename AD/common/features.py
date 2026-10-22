"""What a decision model sees: the robot's Observation turned into numbers (for small networks)
or into a short labelled text (for text models), plus the 5 discrete actions as labels.

Nothing here reads ground truth: only the Observation every driver gets (lidar, encoders, the
goal sensor, contact). The environment itself (robot_env) is not changed."""

from __future__ import annotations

import math

import numpy as np

from robot_env import config as C
from robot_env.types import Command, Observation

# The models' discrete actions, within speed level 2's caps (0.30 m/s, 1.0 rad/s), the level the
# reference driver is recorded at. Besides the environment's 5 basic actions (robot_env/actions.py)
# they include a slow creep and two arcs, because the reference driver uses both a lot; with only
# the 5, its 0.06 m/s creep would be labelled full speed and its arcs as spins on the spot.
LABEL_LEVEL = 1  # index into config.SPEED_LEVELS
ACTIONS: dict[str, Command] = {
    "forward": Command(0.30, 0.0),
    "forward_left": Command(0.25, 0.5),
    "forward_right": Command(0.25, -0.5),
    "slow_forward": Command(0.08, 0.0),
    "turn_left": Command(0.0, 1.0),
    "turn_right": Command(0.0, -1.0),
    "stop": Command(0.0, 0.0),
    "back_up": Command(-0.15, 0.0),
}
ACTION_NAMES: tuple[str, ...] = tuple(ACTIONS)
FEATURE_VERSION = 1  # vector() and scene_text() layout; recordings store it
LABEL_VERSION = 2  # 2: nearest of the 8 actions (1 was a threshold rule over the 5 basic ones)
LIDAR_BINS = 72  # 5 degree sectors: the minimum range in each
SECTORS = (  # name, centre (deg, + left), half width (deg): the scene text's sectors
    ("front", 0, 22.5), ("front_left", 45, 22.5), ("left", 90, 22.5), ("back_left", 135, 22.5),
    ("rear", 180, 22.5), ("back_right", -135, 22.5), ("right", -90, 22.5), ("front_right", -45, 22.5),
)
FEATURES = LIDAR_BINS + 6  # binned lidar, goal (distance, sin, cos), speed, turn rate, contact
# The measured speed and turn rate. Cloned policies can learn "keep doing what the wheels already
# do" from them (they spin in place); models may blank them out (no-odometry runs).
ODOMETRY = slice(LIDAR_BINS + 3, LIDAR_BINS + 5)


def lidar_bins(obs: Observation, bins: int = LIDAR_BINS) -> np.ndarray:
    """Minimum range per sector, ray 0 at -180 degrees; an invalid ray counts as no return (the
    safety layer handles unknown readings; a model gets the same view in every episode)."""
    r = np.where(obs.lidar_valid, obs.lidar, C.LIDAR_RANGE).astype(np.float32)
    return r.reshape(bins, -1).min(axis=1)


def vector(obs: Observation) -> np.ndarray:
    """A fixed-length float vector, every value roughly in [-1, 1]."""
    v, w = obs.velocity_estimate
    out = np.empty(FEATURES, np.float32)
    out[:LIDAR_BINS] = lidar_bins(obs) / C.LIDAR_RANGE
    out[LIDAR_BINS:] = (min(obs.goal_distance, 10.0) / 10.0, math.sin(obs.goal_bearing), math.cos(obs.goal_bearing),
                        v / C.MAX_LINEAR_SPEED, w / C.MAX_ANGULAR_SPEED, float(obs.contact))
    return out


def _sector_min(obs: Observation, centre: float, half: float) -> float:
    ang = (np.degrees(obs.lidar_angles) - centre + 180.0) % 360.0 - 180.0
    sel = np.abs(ang) <= half
    r = np.where(obs.lidar_valid[sel], obs.lidar[sel], C.LIDAR_RANGE)
    return float(r.min()) if r.size else C.LIDAR_RANGE


def scene_text(obs: Observation) -> str:
    """Labelled fields with units, as the scene description for a text model. The lidar cannot
    tell a wall from a box or a cat, so the text never names what is there, only how far."""
    parts = []
    for name, centre, half in SECTORS:
        d = _sector_min(obs, centre, half)
        parts.append(f"{name}_min_m: {'none' if d >= C.LIDAR_RANGE - 1e-6 else f'{d:.2f}'}")
    unknown = int((~obs.lidar_valid).sum())
    v, w = obs.velocity_estimate
    parts += [f"unknown_rays: {unknown}", f"forward_speed_m_s: {v:.2f}", f"turn_rate_rad_s: {w:.2f}",
              f"goal_distance_m: {obs.goal_distance:.2f}", f"goal_bearing_deg: {math.degrees(obs.goal_bearing):.0f}",
              f"contact: {'yes' if obs.contact else 'none'}"]
    return "; ".join(parts)


def _scaled(v: float, w: float) -> tuple[float, float]:
    vmax, wmax = C.SPEED_LEVELS[LABEL_LEVEL]
    return v / vmax, w / wmax


_ACTION_POINTS = np.array([_scaled(c.v, c.omega) for c in ACTIONS.values()])


def action_of(command: Command | None) -> str:
    """The action nearest to a continuous command, measuring speed as a share of level 2's
    0.30 m/s and turn rate as a share of its 1.0 rad/s (so both count alike)."""
    if command is None:
        return "stop"
    return ACTION_NAMES[labels_of(np.array([[command.v, command.omega]]))[0]]


def labels_of(cmd: np.ndarray) -> np.ndarray:
    """Action indices nearest to an (N, 2) array of (v, omega) commands."""
    vmax, wmax = C.SPEED_LEVELS[LABEL_LEVEL]
    pts = np.asarray(cmd, float) / np.array([vmax, wmax])
    return np.argmin(((pts[:, None, :] - _ACTION_POINTS[None]) ** 2).sum(-1), axis=1).astype(np.int64)


def to_command(action: str | int) -> Command:
    return ACTIONS[ACTION_NAMES[action]] if isinstance(action, (int, np.integer)) else ACTIONS[action]


ACTION_INDEX = {a: i for i, a in enumerate(ACTION_NAMES)}
__all__ = ["ACTIONS", "ACTION_NAMES", "ACTION_INDEX", "FEATURES", "FEATURE_VERSION", "LABEL_LEVEL", "LABEL_VERSION",
           "action_of", "labels_of", "scene_text", "to_command", "vector"]
