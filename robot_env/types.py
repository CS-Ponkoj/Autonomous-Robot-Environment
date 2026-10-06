"""Data passed between the world, the safety layer, and any driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Command:
    """Requested motion. Positive omega turns left."""

    v: float = 0.0  # m/s, forward positive
    omega: float = 0.0  # rad/s


STOP = Command(0.0, 0.0)


@dataclass(frozen=True)
class Observation:
    """What a driver may see. No ground truth except the stated ideal goal sensor."""

    seq: int
    time: float  # simulated seconds since reset
    lidar: np.ndarray  # (LIDAR_RAYS,) meters; LIDAR_RANGE means no return
    lidar_valid: np.ndarray  # (LIDAR_RAYS,) bool; False means unknown reading
    lidar_angles: np.ndarray  # (LIDAR_RAYS,) radians relative to the heading
    scan_time: float  # simulated time the scan was taken
    velocity_estimate: tuple[float, float]  # (v, omega) from wheel encoders
    goal_distance: float  # ideal goal sensor (privileged simulation assumption)
    goal_bearing: float  # radians, positive means the goal is to the left
    contact: bool  # touching a wall or obstacle now
    camera: np.ndarray | None = field(default=None, repr=False)
    # forward depth sensor (robot_env/sensing.py): ranges along fixed rays, their validity, and
    # when the frame was taken; None where an observation has no depth (synthetic tests)
    depth: np.ndarray | None = field(default=None, repr=False)  # (DEPTH_ROWS, DEPTH_COLS) m
    depth_valid: np.ndarray | None = field(default=None, repr=False)  # False: closer than DEPTH_MIN
    depth_time: float | None = None
    # the IMU's gravity reading when the frame was taken: world up as a unit vector in the body
    # frame (the body pitches a few degrees when it speeds up or brakes); levels the depth frame
    depth_up: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Decision:
    command: Command
    obs_seq: int
    confidence: float | None = None


class Driver(Protocol):
    name: str

    def decide(self, obs: Observation) -> Decision | None: ...
