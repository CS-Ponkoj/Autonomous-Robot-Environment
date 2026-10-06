"""RobotSystem: the one path from any driver to the wheels.

    driver -> drive(v, omega) -> [50 Hz control tick: sense -> SafetyLayer -> approved target]
           -> [every 2 kHz physics step: velocity smoother -> wheel motors -> physics, contacts]

The GUI, the Gymnasium environment, and the tests all use this class.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from dataclasses import dataclass

from . import config as C
from .safety import HARD_STOP_REASONS, INTERVENTION_REASONS, SafetyFlags, SafetyLayer, SafetyResult
from .sim import RobotSim
from .types import STOP, Command, Decision, Observation


@dataclass(frozen=True)
class SafetyEvent:
    """A change in what the safety layer is doing (recorded for review)."""

    time: float
    reasons: tuple[str, ...]
    requested: Command | None  # from the driver
    approved: Command  # after the safety layer
    applied: Command  # motor command at the event: the value before this tick's smoothing
    # for an ordinary change, zero for a hard stop (hard stops zero the motors at once)


@dataclass
class SensorFaults:
    """Seeded lidar faults for evaluation (all off by default). The random draws restart from
    `seed` at every reset, so a run repeats exactly and paired runs see the same faults."""
    dropout: float = 0.0  # chance each reading is dropped (invalid), per scan
    noise: float = 0.0  # m: standard deviation of the range noise on each return
    bias: float = 0.0  # m added to every return
    outage: tuple[float, float] | None = None  # (start, end) s after reset: no new scans (they go stale)
    seed: int = 0

    def active(self) -> bool:
        return bool(self.dropout or self.noise or self.bias or self.outage)


class RobotSystem:
    def __init__(self, sim: RobotSim | None = None, safety: SafetyLayer | None = None, cats: int = 0,
                 cat_seed: int = 0):
        from .cats import CatHerd, cat_world
        if sim is None:
            sim = RobotSim(extra=cat_world(cats)) if cats else RobotSim()
        elif cats:
            raise ValueError("pass cats only when the system builds its own world")
        self.sim = sim
        # Wandering cats (off by default): a separate model with cat bodies is used only when
        # cats > 0, so the default world is unchanged. Order every physics step: control tick
        # (cat behavior, lidar scan, safety), robot motors, cat motion, physics and contacts.
        self.cats = CatHerd(sim, cats, cat_seed) if cats else None
        self.cat_contacts = 0
        self.cat_contact_events: list[dict] = []  # one per contact (open ones lack duration_s)
        self._cat_open: dict[int, dict] = {}
        # cats whose contact was finished early (episode end) while still touching: that same
        # touch is not counted again until the cat has separated for COLLISION_EVENT_GAP
        self._cat_done: set[int] = set()
        self._cat_done_apart: dict[int, float] = {}
        self.safety = safety or SafetyLayer()
        self.flags = SafetyFlags()
        self.scan_enabled = True  # test hook: False freezes the lidar to make it stale
        self.faults = SensorFaults()  # evaluation hook: seeded lidar faults (none by default)
        self._fault_rng = np.random.default_rng(0)
        self._ctrl_every = round(C.CONTROL_PERIOD / C.PHYSICS_DT)
        self.log = None  # optional DriveLog (robot_env/drive_log.py); written after each tick's decisions
        self.reset(0.0, 0.0, 0.0, (2.0, 2.0))

    def reset(self, x: float, y: float, yaw: float, goal_xy: tuple[float, float]) -> None:
        # Zero every command first, then restore the complete state. System-owned flags are
        # cleared; operator flags (emergency brake, focus) are kept: only the operator
        # (or the window reporting the operator's state) may change them.
        if getattr(self, "_cat_open", None):
            self.finish_cat_contacts("reset")
        if self.log is not None and self.log.started:
            # One log per episode (time and tick numbers restart at reset): finish the old log
            # and detach it; attach a new DriveLog for the new episode.
            self.log.close()
            self.log = None
        self._cat_done, self._cat_done_apart = set(), {}
        self.sim.set_wheel_targets(0.0, 0.0)
        self.flags.episode_over = False
        self.flags.manual_mode = False
        self.flags.manual_input_held = False
        self._requested: Command | None = None
        self._issued_at: float | None = None
        self.last_decision: Decision | None = None
        self._wheel_v = 0.0
        self._wheel_w = 0.0
        self._target = STOP  # latest approved command; the smoother moves toward it every physics step
        self.applied = STOP
        self.sim.reset(x, y, yaw, goal_xy)
        self._fault_rng = np.random.default_rng(self.faults.seed)  # the same faults every run
        self._reset_time = self.sim.time
        if self.cats is not None:
            self.cats.reset((x, y), goal_xy)
            mujoco_forward(self.sim)
        self.cat_contacts = 0
        self.cat_contact_events = []
        self._cat_open = {}
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
        if self.log is not None:
            self.log.bind(self)  # the header always describes this system (cats, seed)
            self.log.event("reset", self.sim.time, start=[x, y, yaw], goal=[float(goal_xy[0]), float(goal_xy[1])])

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

    def _waiting_forward(self) -> bool:
        """The robot has a fresh forward request that its safety layer is holding back (from the
        last control tick): it is waiting for its way to clear."""
        req, age = self._requested, self.command_age
        if req is None or age is None or age > C.COMMAND_LIFETIME or not req.v > 0.02:
            return False
        held = self.last_result.command.v < req.v - 0.02
        return held and "clearance" in self.last_result.reasons

    def end_episode(self) -> None:
        """The episode ended (goal, collision, or time limit): latch the shared stop and make it
        take effect now, not at the next control tick. The request and the last decision are
        dropped, the motors are zeroed, and the stop is this tick's result. reset() clears it."""
        self.flags.episode_over = True
        requested = self._requested
        self._requested = None
        self._issued_at = None
        self.last_decision = None
        self._target = STOP
        self._wheel_v = self._wheel_w = 0.0
        self.sim.set_wheel_targets(0.0, 0.0)
        self.applied = STOP
        result = SafetyResult(STOP, ("episode_over",))
        if result.reasons != self.last_result.reasons or result.command != self.last_result.command:
            self.safety_log.append(SafetyEvent(self.sim.time, result.reasons, requested, STOP, STOP))
        self.last_result = result

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
                if self.cats is not None:
                    self.cats.tick(robot_blocked=self._waiting_forward())
                    if self.log is not None:
                        for index, _t, state in self.cats.new_transitions():
                            # stamped with the simulation clock shared by every log record
                            self.log.event("cat_state", self.sim.time,
                                           evaluation_only_truth={"cat": index, "state": state})
                self._control_tick()
            self._motor_step()
            if self.cats is not None:
                self.cats.step(C.PHYSICS_DT)
            contact = self.sim.physics_step()
            self._step += 1
            counted_now = False
            if contact:
                # Contact is checked every physics step. Flicker while pressed against
                # something is one collision: a new one needs a contact-free gap first.
                if self.sim.time - self._last_contact_time > C.COLLISION_EVENT_GAP:
                    self.collisions += 1
                    self.collision_times.append(self.sim.time)
                    counted_now = True
                self._last_contact_time = self.sim.time
            if self.cats is not None:
                self._update_cat_contacts(counted_now)
            self._contact = contact
        if steps:
            # Endpoint snapshot: goal, encoders, contact, and time are current; the lidar
            # keeps its real scan_time (scans happen only on 50 Hz control ticks).
            self._obs = self._build_observation()

    CONTACT_TOL = 0.02  # m/s: closing speeds closer than this cannot tell who moved into whom

    def _update_cat_contacts(self, counted_now: bool) -> None:
        """Per-cat contact tracking, independent of the global collision debounce: every cat that
        starts touching the robot is a cat contact and a collision (the generic count made this
        step stands for the first one only). Each contact records who moved into whom (from
        velocities along the contact normal), peak normal force, impulse, and duration. A
        separation shorter than COLLISION_EVENT_GAP is the same contact, so a contact logs one
        start and one end. A touching cat freezes and stays still until it is clear of the
        robot's circle again."""
        touching = self.sim.cat_contacts_now()
        if not (touching or self._cat_done or self._cat_open or any(c.frozen for c in self.cats.cats)):
            return  # nothing touching, nothing open or frozen: every step below would do nothing
        import mujoco
        m, d = self.sim.model, self.sim.data
        f6 = np.zeros(6)
        now = self.sim.time
        generic_unused = counted_now
        for index in list(self._cat_done):
            if index in touching:
                self._cat_done_apart.pop(index, None)
            elif now - self._cat_done_apart.setdefault(index, now) > C.COLLISION_EVENT_GAP:
                self._cat_done.discard(index)
                self._cat_done_apart.pop(index, None)
        for index, contacts in touching.items():
            if index in self._cat_done:
                continue  # the touch already finished in the log; the cat stays frozen
            force = 0.0
            for c, _sign in contacts:
                mujoco.mj_contactForce(m, d, c, f6)
                force += float(f6[0])
            event = self._cat_open.get(index)
            if event is None:
                event = {"t": now, "cat": index, **self._initiator(index, contacts),
                         "peak_force_n": 0.0, "impulse_ns": 0.0}
                self._cat_open[index] = event
                self.cat_contact_events.append(event)  # updated in place until the contact ends
                self.cat_contacts += 1
                if generic_unused:
                    generic_unused = False
                else:
                    self.collisions += 1
                    self.collision_times.append(now)
                self._log_cat_event("cat_contact_start", now, event)
            event.pop("separated_t", None)  # touching again: any short separation is forgiven
            self.cats.freeze(index)
            event["peak_force_n"] = max(event["peak_force_n"], force)
            event["impulse_ns"] += force * C.PHYSICS_DT
        for index, event in list(self._cat_open.items()):
            if index in touching:
                continue
            event.setdefault("separated_t", now)
            if now - event["separated_t"] > C.COLLISION_EVENT_GAP:
                self._end_cat_contact(index, "separated")
        for cat in self.cats.cats:
            if (cat.frozen and cat.index not in self._cat_open and cat.index not in self._cat_done
                    and self.cats.gaps(cat, cat.x, cat.y, cat.yaw)[1] > 0.0):
                self.cats.freeze(cat.index, False)

    def _end_cat_contact(self, index: int, reason: str) -> None:
        event = self._cat_open.pop(index)
        end = event.pop("separated_t", self.sim.time)
        event["duration_s"] = end - event["t"]
        event["end_reason"] = reason  # separated, episode_end, reset, or close
        self._log_cat_event("cat_contact_end", self.sim.time, event)

    def finish_cat_contacts(self, reason: str) -> None:
        """Close every open cat contact (episode end, reset, close) so its force, impulse, and
        duration reach the log; a contact still touching says so in end_reason."""
        for index in list(self._cat_open):
            if "separated_t" not in self._cat_open[index]:
                self._cat_done.add(index)  # still touching
            self._end_cat_contact(index, reason)

    def _log_cat_event(self, name: str, t: float, event: dict) -> None:
        """Cat identity, initiator, and forces are privileged truth: only under evaluation_only_truth."""
        if self.log is not None:
            truth = {("start_t" if k == "t" else k): v for k, v in event.items() if k != "separated_t"}
            self.log.event(name, t, evaluation_only_truth=truth)

    def _initiator(self, index: int, contacts) -> dict:
        """Closing speeds along the contact normal at the contact point (robot -> cat positive)."""
        import mujoco
        m, d = self.sim.model, self.sim.data
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.sim.robot_body, vel, 0)
        ang, lin = vel[:3], vel[3:]
        robot_origin = d.xpos[self.sim.robot_body]
        robot_in, cat_in = 0.0, 0.0
        for c, sign in contacts:
            p = d.contact.pos[c]
            n = sign * d.contact.frame[c][:3]  # robot -> cat
            v_robot = lin + np.cross(ang, p - robot_origin)
            v_cat = self.cats.contact_velocity(index, p)  # the cat's surface there (limbs move too)
            robot_in = max(robot_in, float(v_robot @ n))
            cat_in = max(cat_in, float(-(v_cat @ n)))
        tol = self.CONTACT_TOL
        if robot_in <= tol and cat_in <= tol:
            who = "indeterminate"
        elif robot_in - cat_in > tol:
            who = "robot"
        elif cat_in - robot_in > tol:
            who = "cat"
        else:
            who = "both"
        return {"robot_closing_speed": robot_in, "cat_closing_speed": cat_in, "initiator": who}

    def _control_tick(self) -> None:
        now = self.sim.time
        f = self.faults
        out = f.outage is not None and f.outage[0] <= now - self._reset_time < f.outage[1]
        if self.scan_enabled and not out:
            self._scan, self._scan_valid = self.sim.scan()
            self._scan_time = now
            if f.active():
                self._scan, self._scan_valid = self._apply_faults(self._scan, self._scan_valid)
        self._obs = self._build_observation()
        result = self.safety.filter(self._requested, self._issued_at, self._obs, now, self.flags)
        self._target = result.command
        if is_hard_stop(result):
            # Safety stops bypass the comfort smoothing: motors are zero before the next physics step.
            self._wheel_v = self._wheel_w = 0.0
            self.sim.set_wheel_targets(0.0, 0.0)
            self.applied = STOP
        if result.reasons != self.last_result.reasons or result.command != self.last_result.command:
            if result.reasons or self.last_result.reasons:
                self.safety_log.append(SafetyEvent(now, result.reasons, self._requested, result.command, self.applied))
        intervening = bool(INTERVENTION_REASONS.intersection(result.reasons))
        if intervening and not self._intervening:
            self.intervention_events += 1
        if intervening:
            self.intervention_time += C.CONTROL_PERIOD
        self._intervening = intervening
        self.last_result = result
        if self.log is not None:
            self.log.tick(self)  # after every decision of this tick: logging cannot change them

    def _apply_faults(self, scan: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f, rng = self.faults, self._fault_rng
        hit = valid & (scan < C.LIDAR_RANGE)
        scan = scan.copy()
        if f.noise or f.bias:
            scan[hit] = np.clip(scan[hit] + f.bias + f.noise * rng.standard_normal(int(hit.sum())), 0.0, C.LIDAR_RANGE)
        if f.dropout:
            valid = valid & ~(rng.random(len(valid)) < f.dropout)
        return scan, valid

    def _motor_step(self) -> None:
        """Velocity smoother, once before every physics step: the motor command moves toward the
        approved target at the smoother's rates (a 50 Hz staircase would excite the tire contact)."""
        dt = C.PHYSICS_DT
        self._wheel_v = _smooth(self._wheel_v, self._target.v, C.SMOOTH_ACCEL * dt, C.SMOOTH_DECEL * dt)
        self._wheel_w = _smooth(self._wheel_w, self._target.omega, C.SMOOTH_ANG_ACCEL * dt, C.SMOOTH_ANG_DECEL * dt)
        self.sim.set_wheel_targets(self._wheel_v, self._wheel_w)
        self.applied = Command(self._wheel_v, self._wheel_w)

    def _build_observation(self) -> Observation:
        self._seq += 1
        dist, bearing = self.sim.goal_sensor()
        return Observation(
            seq=self._seq,
            time=self.sim.time,
            lidar=_frozen(self._scan),
            lidar_valid=_frozen(self._scan_valid),
            lidar_angles=_frozen(self.sim.lidar_angles),
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
        if self.cats is not None:
            self.finish_cat_contacts("close")
        if self.log is not None:
            self.log.close()  # finalize an attached drive log
        self.sim.close()


def is_hard_stop(result: SafetyResult) -> bool:
    """Stops that command zero at once: every stop except the manual driver releasing its keys."""
    if result.command != STOP:
        return False
    return bool(HARD_STOP_REASONS.intersection(result.reasons))


def _smooth(current: float, target: float, max_increase: float, max_decrease: float) -> float:
    """Velocity smoother: speeding up and ordinary slowing down are rate limited. Reversing
    direction ramps down to zero first."""
    goal = 0.0 if target * current < 0 else target
    if abs(goal) > abs(current):
        step = min(abs(goal) - abs(current), max_increase)
        return current + step if goal > 0 else current - step
    step = min(abs(current) - abs(goal), max_decrease)
    return current - step if current > 0 else current + step


def _frozen(a: np.ndarray) -> np.ndarray:
    """A driver-facing snapshot: its own memory, read-only (in-place preprocessing raises
    instead of changing the simulator or a later observation)."""
    out = np.array(a, copy=True)
    out.flags.writeable = False
    return out


def mujoco_forward(sim) -> None:
    """Recompute positions after moving mocap bodies outside a physics step (reset)."""
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
