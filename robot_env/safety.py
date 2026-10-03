"""The safety layer. Every command from every driver passes through SafetyLayer.filter.

Order of checks (the first hard stop wins and nothing can undo it):
  1. invalid_input    requested values are not finite numbers
  2. emergency_brake  operator pressed the brake
  3. focus_lost       the window lost focus
  4. episode_over     the episode ended (goal, collision, or time limit)
  5. released         manual mode and no drive input is held
  6. no_command       nothing requested yet
  7. command_expired  the newest command is older than COMMAND_LIFETIME
  8. stale_scan       the newest lidar scan is older than SCAN_MAX_AGE
Then the command is limited (speed_limit) and checked for clearance (clearance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config as C
from .types import STOP, Command, Observation

# Reasons that count against the driver (interventions). Operator stops and
# "no_command" are recorded as safety events but are not interventions.
INTERVENTION_REASONS = frozenset({"invalid_input", "command_expired", "stale_scan", "speed_limit", "clearance"})
OPERATOR_REASONS = frozenset({"emergency_brake", "focus_lost", "episode_over", "released"})


@dataclass
class SafetyFlags:
    emergency_brake: bool = False
    focus_lost: bool = False
    episode_over: bool = False
    manual_mode: bool = False
    manual_input_held: bool = False


@dataclass(frozen=True)
class SafetyResult:
    command: Command
    reasons: tuple[str, ...] = ()

    @property
    def intervened(self) -> bool:
        return bool(INTERVENTION_REASONS.intersection(self.reasons))


class SafetyLayer:
    def __init__(self, clearance_enabled: bool = True):
        self.clearance_enabled = clearance_enabled

    def filter(self, requested: Command | None, issued_at: float | None, obs: Observation,
               now: float, flags: SafetyFlags) -> SafetyResult:
        if requested is not None and not (math.isfinite(requested.v) and math.isfinite(requested.omega)):
            return SafetyResult(STOP, ("invalid_input",))
        if flags.emergency_brake:
            return SafetyResult(STOP, ("emergency_brake",))
        if flags.focus_lost:
            return SafetyResult(STOP, ("focus_lost",))
        if flags.episode_over:
            return SafetyResult(STOP, ("episode_over",))
        if flags.manual_mode and not flags.manual_input_held:
            return SafetyResult(STOP, ("released",))
        if requested is None or issued_at is None:
            return SafetyResult(STOP, ("no_command",))
        if now - issued_at > C.COMMAND_LIFETIME + 1e-9:
            return SafetyResult(STOP, ("command_expired",))
        if now - obs.scan_time > C.SCAN_MAX_AGE + 1e-9:
            return SafetyResult(STOP, ("stale_scan",))

        reasons: list[str] = []
        v = min(max(requested.v, -C.MAX_LINEAR_SPEED), C.MAX_LINEAR_SPEED)
        w = min(max(requested.omega, -C.MAX_ANGULAR_SPEED), C.MAX_ANGULAR_SPEED)
        if (v, w) != (requested.v, requested.omega):
            reasons.append("speed_limit")
        command = Command(v, w)
        if self.clearance_enabled and command != STOP:
            safe = clearance_filter(command, obs)
            if safe != command:
                reasons.append("clearance")
                command = safe
        return SafetyResult(command, tuple(reasons))


# ----- clearance -----

def obstacle_points(obs: Observation) -> np.ndarray:
    """Obstacle points in the robot frame. Invalid rays become a band of virtual
    points just outside the footprint across that ray's whole sector, so motion
    into an unknown sector is blocked while motion away stays allowed."""
    angles = obs.lidar_angles
    hit = obs.lidar_valid & (obs.lidar < C.LIDAR_RANGE)
    pts = [np.column_stack((obs.lidar[hit] * np.cos(angles[hit]), obs.lidar[hit] * np.sin(angles[hit])))]
    half_sector = math.pi / C.LIDAR_RAYS
    for a in angles[~obs.lidar_valid]:
        for da in np.linspace(-half_sector, half_sector, 5):
            ang = a + da
            r = _footprint_boundary(ang) + C.SAFETY_BUFFER * 0.5
            pts.append(np.array([[r * math.cos(ang), r * math.sin(ang)]]))
    p = np.vstack(pts) if pts else np.zeros((0, 2))
    return p[np.hypot(p[:, 0], p[:, 1]) <= C.SAFETY_CONSIDER_RADIUS]


def _footprint_boundary(angle: float) -> float:
    c, s = abs(math.cos(angle)), abs(math.sin(angle))
    return min(C.FOOTPRINT_HALF_LENGTH / c if c > 1e-9 else math.inf,
               C.FOOTPRINT_HALF_WIDTH / s if s > 1e-9 else math.inf)


def footprint_distance(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Distance from points (robot frame) to the footprint rectangle; 0 inside."""
    dx = np.maximum(np.abs(px) - C.FOOTPRINT_HALF_LENGTH, 0.0)
    dy = np.maximum(np.abs(py) - C.FOOTPRINT_HALF_WIDTH, 0.0)
    return np.hypot(dx, dy)


def _approach(current: float, target: float, accel: float, decel: float, dt: float) -> float:
    if current == target:
        return current
    speeding_up = current * target >= 0 and abs(target) > abs(current)
    rate = accel if speeding_up else decel
    step = rate * dt
    if target > current:
        nxt = min(current + step, target)
    else:
        nxt = max(current - step, target)
    if current != 0 and nxt * current < 0:  # crossing zero: stop first
        return 0.0
    return nxt


def predict_poses(v0: float, w0: float, v: float, w: float) -> np.ndarray:
    """Poses (x, y, theta) over: delay at current motion, ramp to the request and hold
    it, then brake to zero. Start pose is (0, 0, 0)."""
    dt = C.SAFETY_PREDICT_DT
    x = y = th = 0.0
    cv, cw = v0, w0
    poses = [(0.0, 0.0, 0.0)]
    t = 0.0
    phase = "delay"
    hold_left = C.SAFETY_HOLD
    for _ in range(400):
        if phase == "delay":
            if t >= C.SAFETY_DELAY:
                phase = "ramp"
        if phase == "ramp":
            cv = _approach(cv, v, C.SAFETY_ACCEL, C.SAFETY_DECEL, dt)
            cw = _approach(cw, w, C.SAFETY_ANG_ACCEL, C.SAFETY_ANG_DECEL, dt)
            if cv == v and cw == w:
                hold_left -= dt
                if hold_left <= 0:
                    phase = "brake"
        elif phase == "brake":
            cv = _approach(cv, 0.0, C.SAFETY_ACCEL, C.SAFETY_DECEL, dt)
            cw = _approach(cw, 0.0, C.SAFETY_ANG_ACCEL, C.SAFETY_ANG_DECEL, dt)
            if cv == 0.0 and cw == 0.0:
                poses.append((x, y, th))
                break
        x += cv * math.cos(th) * dt
        y += cv * math.sin(th) * dt
        th += cw * dt
        t += dt
        poses.append((x, y, th))
    return np.array(poses)


def is_safe(command: Command, obs: Observation, points: np.ndarray | None = None) -> bool:
    pts = obstacle_points(obs) if points is None else points
    if len(pts) == 0:
        return True
    v0, w0 = obs.velocity_estimate
    poses = predict_poses(v0, w0, command.v, command.omega)
    X, Y, TH = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
    dx, dy = pts[None, :, 0] - X, pts[None, :, 1] - Y
    c, s = np.cos(TH), np.sin(TH)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    dist = footprint_distance(lx, ly)  # (poses, points)
    d0 = footprint_distance(pts[:, 0], pts[:, 1])
    dmin = dist.min(axis=0)
    outside = d0 >= C.SAFETY_BUFFER
    ok_outside = dmin[outside] >= C.SAFETY_BUFFER
    ok_inside = dmin[~outside] >= d0[~outside] - 1e-4  # escaping must not get closer to anything
    return bool(ok_outside.all() and ok_inside.all())


MIN_SCALE = 0.1  # below this fraction of the requested speed, stop instead of creeping


def _largest_safe(make, obs: Observation, pts: np.ndarray) -> float:
    """Largest k in [0, 1] for which make(k) passes the clearance check (bisection)."""
    if is_safe(make(1.0), obs, pts):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(7):
        mid = (lo + hi) / 2
        if is_safe(make(mid), obs, pts):
            lo = mid
        else:
            hi = mid
    return lo


def clearance_filter(command: Command, obs: Observation) -> Command:
    """Every candidate passes the same check, in this order:
      1. the requested command;
      2. full turning with less forward speed (keeps steering away possible);
      3. the whole command slowed down (smooth approach instead of stop-go);
      4. stop."""
    pts = obstacle_points(obs)
    if is_safe(command, obs, pts):
        return command
    v, w = command.v, command.omega
    if w != 0.0:
        k = _largest_safe(lambda k: Command(v * k, w), obs, pts)
        if is_safe(Command(v * k, w), obs, pts):
            return Command(v * k, w)
    k = _largest_safe(lambda k: Command(v * k, w * k), obs, pts)
    if k >= MIN_SCALE:
        return Command(v * k, w * k)
    return STOP
