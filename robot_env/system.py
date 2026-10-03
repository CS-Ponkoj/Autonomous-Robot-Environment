"""RobotSystem: the one path from any driver to the wheels.

    driver -> drive(v, omega) -> [50 Hz control tick: sense -> SafetyLayer] -> wheel motors
                                  [500 Hz physics steps with contact detection]

The GUI, the Gymnasium environment, and the tests all use this class.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from . import config as C
from .safety import INTERVENTION_REASONS, SafetyFlags, SafetyLayer, SafetyResult
from .sim import RobotSim
from .types import STOP, Command, Decision, Observation


@dataclass(frozen=True)
class SafetyEvent:
    """A change in what the safety layer is doing (recorded for review)."""

    time: float
    reasons: tuple[str, ...]
    requested: Command | None
    executed: Command


class RobotSystem:
    def __init__(self, sim: RobotSim | None = None, safety: SafetyLayer | None = None):
        self.sim = sim or RobotSim()
        self.safety = safety or SafetyLayer()
        self.flags = SafetyFlags()
        self.scan_enabled = True  # test hook: False freezes the lidar to make it stale
        self._ctrl_every = round(C.CONTROL_PERIOD / C.PHYSICS_DT)
        self.reset(0.0, 0.0, 0.0, (2.0, 2.0))

    def reset(self, x: float, y: float, yaw: float, goal_xy: tuple[float, float]) -> None:
        # Zero every command first, then restore the complete state.
        self.sim.set_wheel_targets(0.0, 0.0)
        self._requested: Command | None = None
        self._issued_at: float | None = None
        self.last_decision: Decision | None = None
        self._wheel_v = 0.0
        self._wheel_w = 0.0
        self.sim.reset(x, y, yaw, goal_xy)
        self.scan_enabled = True
        self._step = 0
        self._carry = 0.0
        self._seq = 0
        self._contact = False
        self._last_contact_time = -1.0
        self.collisions = 0
        self.collision_times: list[float] = []
        self.intervention_events = 0
        self.intervention_time = 0.0
        self._intervening = False
        self.safety_log: deque[SafetyEvent] = deque(maxlen=2000)
        self.last_result = SafetyResult(STOP, ("no_command",))
        self._scan, self._scan_valid = self.sim.scan()
        self._scan_time = self.sim.time
        self._obs = self._build_observation()

    # ----- driver interface -----
    def drive(self, v: float, omega: float) -> None:
        """Request a motion. Refreshes the command lifetime. Safety decides what runs."""
        try:
            command = Command(float(v), float(omega))
        except (TypeError, ValueError):
            command = Command(float("nan"), float("nan"))
        self._requested = command
        self._issued_at = self.sim.time

    def apply(self, decision: Decision | None) -> None:
        """Apply a driver's decision. None means no fresh command (it expires, then stops)."""
        if decision is None:
            return
        self.last_decision = decision
        self.drive(decision.command.v, decision.command.omega)

    @property
    def command_age(self) -> float | None:
        return None if self._issued_at is None else self.sim.time - self._issued_at

    def observe(self, with_camera: bool = False) -> Observation:
        if not with_camera:
            return self._obs
        o = self._obs
        return Observation(o.seq, o.time, o.lidar, o.lidar_valid, o.lidar_angles, o.scan_time,
                           o.velocity_estimate, o.goal_distance, o.goal_bearing, o.contact,
                           camera=self.sim.render_camera())

    @property
    def time(self) -> float:
        return self.sim.time

    @property
    def requested(self) -> Command | None:
        return self._requested

    # ----- simulation -----
    def advance(self, seconds: float) -> None:
        """Advance simulated time in fixed physics steps (fractions carry over)."""
        total = seconds + self._carry
        steps = int(total / C.PHYSICS_DT + 1e-9)
        self._carry = total - steps * C.PHYSICS_DT
        for _ in range(steps):
            if self._step % self._ctrl_every == 0:
                self._control_tick()
            contact = self.sim.physics_step()
            self._step += 1
            if contact:
                # Contact is checked every physics step. Flicker while pressed against
                # something is one collision: a new one needs a contact-free gap first.
                if self.sim.time - self._last_contact_time > C.COLLISION_EVENT_GAP:
                    self.collisions += 1
                    self.collision_times.append(self.sim.time)
                self._last_contact_time = self.sim.time
            self._contact = contact

    def _control_tick(self) -> None:
        now = self.sim.time
        if self.scan_enabled:
            self._scan, self._scan_valid = self.sim.scan()
            self._scan_time = now
        self._obs = self._build_observation()
        result = self.safety.filter(self._requested, self._issued_at, self._obs, now, self.flags)
        self._wheel_v = _smooth(self._wheel_v, result.command.v, C.SMOOTH_ACCEL * C.CONTROL_PERIOD)
        self._wheel_w = _smooth(self._wheel_w, result.command.omega, C.SMOOTH_ANG_ACCEL * C.CONTROL_PERIOD)
        self.sim.set_wheel_targets(self._wheel_v, self._wheel_w)
        if result.reasons != self.last_result.reasons or result.command != self.last_result.command:
            if result.reasons or self.last_result.reasons:
                self.safety_log.append(SafetyEvent(now, result.reasons, self._requested, result.command))
        intervening = bool(INTERVENTION_REASONS.intersection(result.reasons))
        if intervening and not self._intervening:
            self.intervention_events += 1
        if intervening:
            self.intervention_time += C.CONTROL_PERIOD
        self._intervening = intervening
        self.last_result = result

    def _build_observation(self) -> Observation:
        self._seq += 1
        dist, bearing = self.sim.goal_sensor()
        return Observation(
            seq=self._seq,
            time=self.sim.time,
            lidar=self._scan.copy(),
            lidar_valid=self._scan_valid.copy(),
            lidar_angles=self.sim.lidar_angles,
            scan_time=self._scan_time,
            velocity_estimate=self.sim.velocity_estimate(),
            goal_distance=dist,
            goal_bearing=bearing,
            contact=self.in_contact,
        )

    @property
    def in_contact(self) -> bool:
        """Touching something now, debounced over COLLISION_EVENT_GAP (contact flickers while pressed)."""
        return self.sim.time - self._last_contact_time <= C.COLLISION_EVENT_GAP + 1e-9

    # ----- evaluation (ground truth) -----
    def ground_truth(self) -> dict:
        x, y, yaw = self.sim.true_pose()
        v, w = self.sim.true_velocity()
        return {"pose": (x, y, yaw), "velocity": (v, w), "goal": tuple(self.sim.goal)}

    def close(self) -> None:
        self.sim.close()


def _smooth(current: float, target: float, max_increase: float) -> float:
    """Velocity smoother: speeding up is rate limited; slowing down is immediate."""
    if target * current < 0:
        current = 0.0  # reversing direction: drop to zero at once
    if abs(target) <= abs(current):
        return target
    step = min(abs(target) - abs(current), max_increase)
    return current + step if target > 0 else current - step
