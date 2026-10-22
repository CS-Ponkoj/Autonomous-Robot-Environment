"""Counterfactual behavior labels for Track B, and the oracle check that tells whether Track B is
worth it.

    .venv\\Scripts\\python -m AD.common.behavior_labels --seeds 11000-11159 --workers 6
    .venv\\Scripts\\python -m AD.common.behavior_labels --oracle --workers 6     # the go/no-go check

Labelling drives a training layout with the same BehaviorDriver and option timing as deployment.
At every selection point (when no option is committed) that is hard (little route progress in the
last STRUGGLE_WINDOW seconds, or a cat close by) or randomly sampled (RANDOM_SHARE of the others),
it deep-copies the whole simulation (robot, cats, the driver's memory: a copy replays identically)
and tries every behavior: the option, then FOLLOW, for HORIZON seconds in total, stopping at the
goal or a collision. Each branch is scored on a fixed reference: the shortest route from the robot
to the goal on the static floor map (privileged, used for labels only, never as a model input):
    reached the goal: GOAL_VALUE + seconds left;  a collision: -GOAL_VALUE;
    otherwise: route metres gained - INTERVENTION_COST per safety intervention.
FOLLOW is kept unless another behavior beats it by MARGIN. The episode then continues with the
chosen behavior, so later states and the history reflect good choices.

--oracle runs the practice layouts (2000-2009, without cats and with 4 cats; never the held-out
test layouts) twice: the labelling oracle (branching at every hard point) against FOLLOW only. If
the oracle does not clearly beat FOLLOW, no learned selector can, and Track B is not worth labels.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "behavior"
HORIZON = 10.0  # s per branch (the option, then FOLLOW)
STRUGGLE_WINDOW = 3.0  # s over which route progress is measured
STRUGGLE_PROGRESS = 0.3  # m of route progress below which a point counts as hard
CAT_NEAR = 1.0  # m: a cat this close also makes a point hard (labelling only; models never see it)
RANDOM_SHARE = 0.1  # share of ordinary points also branched (coverage beyond the hard ones)
GOAL_VALUE, INTERVENTION_COST, MARGIN = 100.0, 0.02, 0.1
TEST_SEEDS = set(range(1000, 1010)) | set(range(2000, 2020)) | set(range(5000, 5030)) | {4002, 4016}
DEV = tuple(range(2000, 2010))


class Route:
    """Shortest route length (m) to the goal over the static floor map, from anywhere."""

    def __init__(self, model, goal):
        from robot_env import config as C
        from robot_env import kernels
        from robot_env.baseline import _STEP_ARRAY
        from robot_env.layout import RoomMap
        self.C, room = C, RoomMap(model)
        self.room = room
        passable = room.clear >= C.CIRCUMSCRIBED_RADIUS  # the robot fits (no extra planning margin)
        src = np.zeros_like(passable)
        src[room.cell(*goal)] = True
        self.field = kernels.wavefront(src, ~passable, -1, -1, _STEP_ARRAY, 0, 4 * room.n) * C.GRID_RESOLUTION

    def __call__(self, x: float, y: float) -> float:
        i, j = self.room.cell(x, y)
        win = self.field[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4]  # the robot may stand just inside a margin
        v = float(win.min())
        return v + 3 * self.C.GRID_RESOLUTION if not np.isfinite(self.field[i, j]) and np.isfinite(v) else v


def _run(system, drv, seconds, selector, route):
    """Drive for `seconds` with `selector`; stops at the goal or a collision. Returns (outcome, end time)."""
    from robot_env import config as C
    end = min(system.time + seconds, C.EPISODE_TIME_LIMIT)
    drv.selector = selector
    while system.time < end - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            return "success", system.time
        system.apply(drv.decide(obs))
        system.advance(min(C.DECISION_PERIOD, end - system.time))
        if system.collisions:
            return "collision", system.time
    return ("success" if system.observe().goal_distance <= C.GOAL_RADIUS else "running"), system.time


def _value(system, drv, behavior, route):
    """Outcome of choosing `behavior` now, then FOLLOW, for HORIZON seconds (on deep copies)."""
    from .behavior import follow_only
    s, d = copy.deepcopy((system, drv))
    x, y = s.sim.true_pose()[:2]
    before, i0, t0 = route(x, y), s.intervention_events, s.time
    d.choose(s.observe(), behavior)
    outcome, t = _run(s, d, HORIZON, follow_only, route)
    x, y = s.sim.true_pose()[:2]
    after = route(x, y)
    interventions = s.intervention_events - i0
    s.close()
    del s, d
    gc.collect()  # the copies hold reference cycles: without this, 10 branches leak about 1.7 GB
    if outcome == "success":
        return GOAL_VALUE + (HORIZON - (t - t0))
    if outcome == "collision":
        return -GOAL_VALUE
    gain = (before - after) if np.isfinite(before) and np.isfinite(after) else 0.0
    return gain - INTERVENTION_COST * interventions


def episode(seed: int, cats: int | None = None, oracle: bool = True, record: bool = True) -> dict:
    """One episode. oracle: label (and drive) by trying every behavior at hard points; otherwise
    FOLLOW only. record: save the decision points as training data."""
    from robot_env import config as C
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem

    from .behavior import BEHAVIOR_INDEX, BEHAVIORS, BehaviorDriver, follow_only
    from .features import scene_text, vector

    cats = seed % 5 if cats is None else cats
    rng = np.random.default_rng(seed)
    system = RobotSystem(cats=cats, cat_seed=seed)
    task = RoomMap(system.sim.model).sample_task(seed)
    system.reset(*task.start, task.goal)
    route = Route(system.sim.model, task.goal)
    drv = BehaviorDriver(follow_only)
    drv.reset()
    trail = []
    X, T, H, Y, V, B = [], [], [], [], [], []
    outcome, branched, start = "timeout", 0, time.time()
    while system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome = "success"
            break
        if drv.due(obs):
            x, y = system.sim.true_pose()[:2]
            here = route(x, y)
            trail = [p for p in trail if p[0] >= obs.time - STRUGGLE_WINDOW - 1e-9] + [(obs.time, here)]
            hard = (obs.time >= STRUGGLE_WINDOW and trail[0][1] - here < STRUGGLE_PROGRESS) or _cat_near(system)
            values = np.full(len(BEHAVIORS), np.nan)
            label = 0
            if oracle and (hard or rng.random() < RANDOM_SHARE):
                values = np.array([_value(system, drv, b, route) for b in BEHAVIORS])
                best = int(np.argmax(values))
                label = best if values[best] > values[0] + MARGIN else 0
                branched += 1
            if record:
                X.append(vector(obs)), T.append(scene_text(obs)), H.append([BEHAVIOR_INDEX[h] for h in drv.history])
                Y.append(label), V.append(values), B.append(not np.isnan(values[0]))
            drv.choose(obs, BEHAVIORS[label])
        system.apply(drv.exec.command(obs))
        system.advance(C.DECISION_PERIOD)
        if system.collisions:
            outcome = "collision"
            break
    if outcome == "timeout" and system.observe().goal_distance <= C.GOAL_RADIUS:
        outcome = "success"
    result = {"seed": seed, "cats": cats, "oracle": oracle, "outcome": outcome, "time_s": round(system.time, 1),
              "interventions": system.intervention_events, "branched": branched, "points": len(Y),
              "wall_s": round(time.time() - start)}
    system.close()
    if record:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"seed{seed}.txt").write_text("\n".join(T), encoding="utf-8")
        tmp = OUT / f"seed{seed}.tmp.npz"
        np.savez_compressed(tmp, x=np.array(X, np.float32), history=np.array(H, np.int8), label=np.array(Y, np.int8),
                            values=np.array(V, np.float32), branched=np.array(B), outcome=outcome, cats=cats)
        tmp.replace(OUT / f"seed{seed}.npz")
        counts = np.bincount(np.array(Y, int), minlength=len(BEHAVIORS))
        result["labels"] = {b: int(n) for b, n in zip(BEHAVIORS, counts)}
    return result


def _cat_near(system) -> bool:
    if system.cats is None:
        return False
    x, y = system.sim.true_pose()[:2]
    return any(math.hypot(c.x - x, c.y - y) < CAT_NEAR for c in system.cats.cats)


def _label(seed):
    return episode(seed)


def _check(args):
    seed, cats, oracle = args
    return episode(seed, cats=cats, oracle=oracle, record=False)


def main(argv=None) -> int:
    from .record import parse_seeds
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="11000-11159")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--oracle", action="store_true", help="the go/no-go check on the practice layouts")
    a = p.parse_args(argv)
    start = time.time()
    if a.oracle:
        jobs = [(s, c, o) for o in (False, True) for c in (0, 4) for s in DEV]
        with ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as pool:
            res = list(pool.map(_check, jobs))
        summary = {}
        for o in (False, True):
            for c in (0, 4):
                sel = [r for r in res if r["oracle"] == o and r["cats"] == c]
                summary[f"{'oracle' if o else 'follow'}_{c}cats"] = {
                    "success": sum(r["outcome"] == "success" for r in sel), "of": len(sel),
                    "collisions": sum(r["outcome"] == "collision" for r in sel),
                    "interventions": sum(r["interventions"] for r in sel)}
        out = {"time": time.strftime("%Y-%m-%d %H:%M"), "summary": summary, "episodes": res,
               "wall_s": round(time.time() - start)}
        (OUT.parent.parent / "private" / "trackb_oracle_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(summary), flush=True)
        return 0
    seeds = parse_seeds(a.seeds)
    if TEST_SEEDS & set(seeds):
        p.error("these seeds include test seeds")
    todo = [s for s in seeds if not (OUT / f"seed{s}.npz").exists()]
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as pool, \
            open(OUT / "episodes.jsonl", "a", encoding="utf-8") as log:
        for i, r in enumerate(pool.map(_label, todo), 1):
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(f"{i}/{len(todo)} {r} {time.time() - start:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
