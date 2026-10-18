"""Track B: behavior selection above the built-in driver.

A selector (a model, or the labelling oracle) picks one of 5 behaviors; the built-in driver keeps
running every tick (so its map and odometry stay current) and the behavior shapes its output.
Behaviors are options with fixed semantics, identical in labelling and in deployment:

- FOLLOW:  the built-in driver's command as is (re-selected every SELECT_PERIOD).
- SLOW:    the same, forward speed capped at SLOW_V, committed for HOLD seconds.
- WAIT:    stop, committed for HOLD seconds.
- REPLAN:  one shot (then FOLLOW): forget refused space and briefly seen (moving) obstacles,
           keep confirmed static ones, and restart planning and stuck detection. Repeat requests
           within REPLAN_COOLDOWN are ignored.
- RECOVER: committed for RECOVER_BACK + RECOVER_TURN seconds: back up (only if the lidar sees the
           space behind clear), then turn toward the more open side, then FOLLOW.

While a behavior overrides the driver's motion (WAIT, RECOVER), the driver's own stuck detector
is held off, since it would read the commanded-but-absent motion as being blocked. Selections
are ignored while a committed option runs. Nothing in robot_env changes: the driver is wrapped,
and everything goes through the same decide(observation) plug and safety layer as any driver.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_env import config as C
from robot_env.baseline import CONFIRM_SPAN, DYNAMIC_TTL, REAR_CLEAR, BaselineDriver
from robot_env.types import Command, Decision, Observation

from .features import LABEL_LEVEL

BEHAVIORS: tuple[str, ...] = ("FOLLOW", "SLOW", "WAIT", "REPLAN", "RECOVER")
BEHAVIOR_INDEX = {b: i for i, b in enumerate(BEHAVIORS)}
SELECT_PERIOD = 0.5  # s between selections (when no option is committed)
HOLD = 1.5  # s a SLOW or WAIT option runs once chosen
SLOW_V = 0.12  # m/s
RECOVER_BACK, RECOVER_TURN = 1.0, 1.0  # s
RECOVER_V, RECOVER_W = -0.12, 0.8
REPLAN_COOLDOWN = 3.0  # s
HISTORY = 3  # previous behaviors a selector may see
DURATION = {"FOLLOW": SELECT_PERIOD, "SLOW": HOLD, "WAIT": HOLD, "REPLAN": SELECT_PERIOD,
            "RECOVER": RECOVER_BACK + RECOVER_TURN}


def _ranges(obs: Observation) -> np.ndarray:
    return np.where(obs.lidar_valid, obs.lidar, np.nan)  # unknown readings do not count as open


def open_side(obs: Observation) -> float:
    """+1 when the left side (positive lidar angles) has more known free space than the right, else -1."""
    r, ang = _ranges(obs), obs.lidar_angles
    left, right = r[(ang > 0.2) & (ang < 2.9)], r[(ang < -0.2) & (ang > -2.9)]
    lm = float(np.nanmean(left)) if np.isfinite(left).any() else 0.0
    rm = float(np.nanmean(right)) if np.isfinite(right).any() else 0.0
    return 1.0 if lm >= rm else -1.0


def rear_clear(obs: Observation) -> bool:
    """The lidar sees the lane straight behind the robot clear for REAR_CLEAR (every reading known)."""
    r, ang = _ranges(obs), obs.lidar_angles
    behind = (np.cos(ang) < 0) & (np.abs(np.nan_to_num(r * np.sin(ang), nan=0.0)) < C.FOOTPRINT_HALF_WIDTH + 0.04)
    if not behind.any() or np.isnan(r[behind]).any():
        return False
    along = r[behind] * np.cos(ang[behind])
    return float(np.min(-along)) - C.FOOTPRINT_HALF_LENGTH >= REAR_CLEAR


class BehaviorExecutor:
    """Runs the built-in driver and applies the current option to its output."""

    def __init__(self):
        self.base = BaselineDriver(C.SPEED_LEVELS[LABEL_LEVEL])
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self.behavior, self.since, self.turn, self.back = "FOLLOW", 0.0, 1.0, True
        self.last_replan = -np.inf

    def busy(self, t: float) -> bool:
        """A committed option is still running (selections are ignored)."""
        return self.behavior in ("SLOW", "WAIT", "RECOVER") and t < self.since + DURATION[self.behavior] - 1e-9

    def set(self, behavior: str, obs: Observation) -> str:
        """Start an option; returns the behavior actually in force (a REPLAN within its cooldown
        is ignored and becomes FOLLOW)."""
        if behavior not in BEHAVIOR_INDEX:
            raise ValueError(behavior)
        if behavior == "REPLAN":
            if obs.time - self.last_replan < REPLAN_COOLDOWN:
                behavior = "FOLLOW"
            else:
                self._replan(obs.time)
                self.last_replan = obs.time
        if behavior == "RECOVER":
            self.turn, self.back = open_side(obs), rear_clear(obs)
        self.behavior, self.since = behavior, obs.time
        return behavior

    def _replan(self, now: float) -> None:
        b = self.base
        brief = (b._last - b._first) < CONFIRM_SPAN  # seen only briefly: moving things, not walls
        b._first[brief], b._last[brief] = np.inf, -np.inf
        b._refused[:] = -np.inf
        b.occupied = (b._last - b._first >= CONFIRM_SPAN) | (now - b._last < DYNAMIC_TTL) | (b._refused > now)
        b.dist = None
        b._next_plan = 0.0
        b._turn_until = b._backup_until = -1.0
        b._still_since, b._last_v = None, 0.0

    def command(self, obs: Observation) -> Decision:
        b, age = self.behavior, obs.time - self.since
        override = (b == "WAIT" and age < HOLD) or (b == "RECOVER" and age < DURATION["RECOVER"])
        if override:  # the driver's stuck detector would read the held robot as blocked
            self.base._still_since = None
        d = self.base.decide(obs)  # always, so the driver's map and odometry stay current
        cmd = d.command if d is not None else Command(0.0, 0.0)
        if b == "SLOW" and age < HOLD:
            cmd = Command(min(cmd.v, SLOW_V), cmd.omega)
        elif b == "WAIT" and age < HOLD:
            cmd = Command(0.0, 0.0)
        elif b == "RECOVER" and age < RECOVER_BACK:
            cmd = Command(RECOVER_V, 0.0) if self.back else Command(0.0, RECOVER_W * self.turn)
        elif b == "RECOVER" and age < DURATION["RECOVER"]:
            cmd = Command(0.0, RECOVER_W * self.turn)
        if override:
            self.base._still_since = None
            self.base._backup_until = self.base._turn_until = -1.0  # no recovery of its own meanwhile
        return Decision(cmd, obs.seq)


class BehaviorDriver:
    """A driver whose selector picks the next option whenever none is committed, at most every
    SELECT_PERIOD seconds. selector(obs, history) -> behavior name; history is the last HISTORY
    behaviors in force, oldest first."""

    name = "behavior"

    def __init__(self, selector):
        self.selector = selector
        self.exec = BehaviorExecutor()
        self.reset()

    def reset(self) -> None:
        self.exec.reset()
        self.history: list[str] = ["FOLLOW"] * HISTORY
        self.next_at = 0.0

    def choose(self, obs: Observation, behavior: str) -> None:
        """Put a chosen behavior in force (also used by the labelling oracle)."""
        b = self.exec.set(behavior, obs)
        self.history = self.history[1:] + [b]
        self.next_at = obs.time + SELECT_PERIOD

    def due(self, obs: Observation) -> bool:
        return obs.time >= self.next_at - 1e-9 and not self.exec.busy(obs.time)

    def decide(self, obs: Observation) -> Decision:
        if self.due(obs):
            self.choose(obs, self.selector(obs, tuple(self.history)))
        return self.exec.command(obs)


def follow_only(obs, history) -> str:
    """The plain built-in driver expressed as a selector (the Track B reference)."""
    return "FOLLOW"


# ----- data and benchmark for behavior models -----
LABELS = Path(__file__).resolve().parent.parent / "data" / "behavior"


def load(seeds) -> dict:
    """The labelled decision points of the given seeds (those labelled): x (N, FEATURES), text
    (N scene texts), label (N,), history (N, HISTORY) previous behaviors as indices (oldest first),
    branched (N,) whether the label came from trying every behavior, seed (N,)."""
    xs, texts, labels, hist, br, sd = [], [], [], [], [], []
    for s in seeds:
        f = LABELS / f"seed{s}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        if not len(d["label"]):
            continue
        lines = (LABELS / f"seed{s}.txt").read_text(encoding="utf-8").split("\n")
        if len(lines) != len(d["label"]):
            raise ValueError(f"seed{s}.txt has {len(lines)} lines for {len(d['label'])} points")
        xs.append(d["x"]), texts.extend(lines), labels.append(d["label"].astype(np.int64))
        hist.append(d["history"].astype(np.int64)), br.append(d["branched"]), sd.append(np.full(len(d["label"]), s))
    if not xs:
        raise FileNotFoundError(f"no behavior labels for these seeds in {LABELS}")
    return {"x": np.concatenate(xs), "text": texts, "label": np.concatenate(labels), "history": np.concatenate(hist),
            "branched": np.concatenate(br), "seed": np.concatenate(sd)}


def _driver_from_selector(make_selector):
    sel = make_selector()
    pick = sel.select if hasattr(sel, "select") else sel
    return BehaviorDriver(lambda obs, history: pick(obs, list(history)))


def run_selector(make_selector, final: bool = False, workers: int = 1) -> dict:
    """The driving benchmark (common.benchmark) of a behavior selector. make_selector() is a picklable
    callable returning an object with select(obs, history: list[str]) -> behavior name (or a plain
    function of the same form)."""
    import functools
    from . import benchmark
    return benchmark.run(functools.partial(_driver_from_selector, make_selector), workers=workers, final=final)
