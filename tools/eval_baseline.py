"""Evaluate the rule-based baseline driver headless, on fixed simulated-time decision ticks
(independent of any rendering). The evaluator may use ground truth (path length, planned path);
the driver never does.

    .venv\\Scripts\\python tools\\eval_baseline.py --set dev        # tuning seeds 2000-2019
    .venv\\Scripts\\python tools\\eval_baseline.py --set heldout    # frozen evaluation: seeds 1000-1009
    options: --levels 1 2 5   --out qa_output/baseline_heldout.json   --logs qa_output/baseline_logs
    faults (seeded, the same in every run): --dropout 0.05  --noise 0.02  --bias 0.03
             --outage 5 6 (s: no new scans)  --decision-period 0.5  --fault-seed 7
             --obstacle-on-path 0.10 (m: a box that tall on the planned route, halfway)
             --paired (each episode run clean and with the faults, and the difference reported)

The driver's thresholds are frozen on the dev set before the held-out set is run; the held-out
result is saved with its schema version and the exact command.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_env import config as C  # noqa: E402
from robot_env.baseline import BaselineDriver  # noqa: E402
from robot_env.drive_log import DriveLog, config_hash  # noqa: E402
from robot_env.layout import RoomMap  # noqa: E402
from robot_env.sim import RobotSim  # noqa: E402
from robot_env.system import RobotSystem, SensorFaults  # noqa: E402

SCHEMA = 2  # 2: "path_efficiency" renamed "reference_path_ratio" (it can exceed 1)
SEED_SETS = {"dev": tuple(range(2000, 2020)), "heldout": tuple(C.HELDOUT_SEEDS),
             "trapped": (4002, 4016),  # the QA report's traps: no route (4002), a slow route with blind backups (4016)
             "unseen": tuple(range(5000, 5030))}  # never used for tuning: the generalisation check


def _drift(driver: BaselineDriver, start, true_pose) -> float:
    """Distance between the driver's integrated pose and the true pose, in the start frame."""
    x0, y0, th0 = start
    dx, dy = true_pose[0] - x0, true_pose[1] - y0
    tx, ty = dx * math.cos(th0) + dy * math.sin(th0), -dx * math.sin(th0) + dy * math.cos(th0)
    return math.hypot(driver.pose[0] - tx, driver.pose[1] - ty)


def _obstacle_on_path(task, height: float) -> str:
    """A 0.1 m box `height` tall on the task's planned route, halfway along it."""
    pts, lengths = task.path, [math.dist(a, b) for a, b in zip(task.path, task.path[1:])]
    half, run = sum(lengths) / 2, 0.0
    for (a, b), d in zip(zip(pts, pts[1:]), lengths):
        if run + d >= half:
            f = (half - run) / d if d else 0.0
            x, y = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            return (f'<geom name="injected_obstacle" type="box" pos="{x:.3f} {y:.3f} {height / 2:.3f}" '
                    f'size="0.05 0.05 {height / 2:.3f}" rgba="1 0 1 1"/>')
        run += d
    return ""


def run_episode(system: RobotSystem, room: RoomMap, seed: int, level: int, log_dir: Path | None = None,
                decision_period: float = C.DECISION_PERIOD) -> dict:
    task = room.sample_task(seed)
    driver = BaselineDriver(C.SPEED_LEVELS[level])
    if log_dir is not None:
        system.log = DriveLog(log_dir / f"seed{seed}_level{level + 1}.jsonl", task_seed=seed, note="baseline")
    system.reset(*task.start, task.goal)
    path, prev = 0.0, system.sim.true_pose()[:2]
    start_goal, still = system.observe().goal_distance, 0.0
    outcome, reason = "timeout", "time limit reached"
    while system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome, reason = "success", ""
            break
        system.apply(driver.decide(obs))
        system.advance(decision_period)
        v, w = system.observe().velocity_estimate
        if abs(v) < 0.02 and abs(w) < 0.05:
            still += decision_period
        pos = system.sim.true_pose()[:2]
        path += math.dist(prev, pos)
        prev = pos
        if system.collisions:
            outcome, reason = "collision", f"contact at t={system.collision_times[0]:.2f} s"
            break
    if outcome == "timeout" and system.observe().goal_distance <= C.GOAL_RADIUS:
        outcome, reason = "success", ""
    planned = sum(math.dist(a, b) for a, b in zip(task.path, task.path[1:]))
    result = {
        "seed": seed, "level": level + 1, "outcome": outcome, "failure_reason": reason,
        "time_s": round(system.time, 3), "collisions": system.collisions,
        "interventions": system.intervention_events, "intervention_time_s": round(system.intervention_time, 3),
        "path_m": round(path, 3), "planned_path_m": round(planned, 3),
        # Reference path ratio: the planner's grid path to the goal CENTER divided by the driven
        # path, which stops anywhere inside the goal radius. Not an efficiency in [0, 1]: it can
        # exceed 1 when the robot stops at the near edge of the goal region or cuts a grid corner.
        "reference_path_ratio": round(planned / path, 3) if outcome == "success" and path > 0 else None,
        "final_goal_distance_m": round(system.observe().goal_distance, 3),
        "progress_m": round(start_goal - system.observe().goal_distance, 3),  # toward the goal
        "stationary_time_s": round(still, 3),
        # odometry drift: where the driver believes it is against where it is, from its start
        "odometry_drift_m": round(_drift(driver, task.start, system.sim.true_pose()), 3),
    }
    if system.log is not None:
        system.log.event("episode_" + outcome, system.time, **{k: result[k] for k in ("seed", "level")})
        system.log.close()
        system.log = None
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--set", choices=sorted(SEED_SETS), default="dev")
    p.add_argument("--levels", type=int, nargs="+", default=[C.DEFAULT_SPEED_LEVEL + 1],
                   choices=range(1, len(C.SPEED_LEVELS) + 1))
    p.add_argument("--out", type=Path, help="write the JSON result here")
    p.add_argument("--logs", type=Path, help="write one drive log per episode into this folder")
    p.add_argument("--dropout", type=float, default=0.0, help="chance each lidar reading is dropped, per scan")
    p.add_argument("--noise", type=float, default=0.0, help="m: range noise (standard deviation) on each return")
    p.add_argument("--bias", type=float, default=0.0, help="m added to every return")
    p.add_argument("--outage", type=float, nargs=2, metavar=("START", "END"), help="s: no new scans in this window")
    p.add_argument("--fault-seed", type=int, default=0)
    p.add_argument("--decision-period", type=float, default=C.DECISION_PERIOD, help="s between driver decisions")
    p.add_argument("--obstacle-on-path", type=float, metavar="HEIGHT", help="m: a 0.1 m box this tall halfway along the route")
    p.add_argument("--paired", action="store_true", help="run each episode clean and with the faults")
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=None,
                   help="release gate: refuse unless the commit is known, the tree clean, and nothing changes during the run (default: on for --set heldout)")
    a = p.parse_args(argv)
    from robot_env import provenance
    strict = a.strict if a.strict is not None else a.set == "heldout"
    args = sys.argv if argv is None else ["eval_baseline.py", *argv]
    before = provenance.begin(__file__, args, strict, a.out, model_sha256=provenance.model_of(), seed_set=a.set)
    faults = SensorFaults(a.dropout, a.noise, a.bias, tuple(a.outage) if a.outage else None, a.fault_seed)
    system = RobotSystem()
    room = RoomMap(system.sim.model)  # tasks come from the world without any injected obstacle
    episodes = []
    variants = [("clean", False), ("faults", True)] if a.paired else [("faults" if faults.active() or a.obstacle_on_path or a.decision_period != C.DECISION_PERIOD else "clean", True)]
    for level in a.levels:
        for seed in SEED_SETS[a.set]:
            for name, faulty in variants:
                run_on, period = system, C.DECISION_PERIOD
                if faulty:
                    period = a.decision_period
                    if a.obstacle_on_path:
                        run_on = RobotSystem(sim=RobotSim(extra_world_xml=_obstacle_on_path(room.sample_task(seed), a.obstacle_on_path)))
                    run_on.faults = faults
                else:
                    run_on.faults = SensorFaults()
                r = run_episode(run_on, room, seed, level - 1, a.logs, period)
                r["variant"] = name
                if run_on is not system:
                    run_on.close()
                episodes.append(r)
                print(f"level {r['level']} seed {seed} {name:6s}: {r['outcome']:9s} t {r['time_s']:6.1f} s  path {r['path_m']:5.1f} m  "
                      f"progress {r['progress_m']:5.2f} m  still {r['stationary_time_s']:5.1f} s  drift {r['odometry_drift_m']:.3f} m  "
                      f"interventions {r['interventions']}  {r['failure_reason']}", flush=True)
    summary = {}
    for level, name in ((lv, n) for lv in a.levels for n, _ in variants):
        rows = [e for e in episodes if e["level"] == level and e["variant"] == name]
        ok = [e for e in rows if e["outcome"] == "success"]
        key = f"level{level}" + (f"_{name}" if a.paired else "")
        summary[key] = {
            "episodes": len(rows), "success": len(ok),
            "collisions": sum(e["outcome"] == "collision" for e in rows),
            "timeouts": sum(e["outcome"] == "timeout" for e in rows),
            "mean_time_s": round(sum(e["time_s"] for e in ok) / len(ok), 2) if ok else None,
            "mean_reference_path_ratio": round(sum(e["reference_path_ratio"] for e in ok) / len(ok), 3) if ok else None,
            "interventions": sum(e["interventions"] for e in rows),
            "mean_progress_m": round(sum(e["progress_m"] for e in rows) / len(rows), 3),
            "stationary_time_s": round(sum(e["stationary_time_s"] for e in rows), 1),
            "max_odometry_drift_m": max(e["odometry_drift_m"] for e in rows),
        }
        print(f"level {level}{' ' + name if a.paired else ''}: {summary[key]}")
    system.close()
    after = provenance.end(before, __file__, args, strict, a.out, model_sha256=provenance.model_of(), seed_set=a.set)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            **after, "schema": SCHEMA, "command": "python tools/eval_baseline.py " + " ".join(sys.argv[1:]),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seed_set": a.set, "seeds": list(SEED_SETS[a.set]),
            "config_sha256": config_hash(), "driver": BaselineDriver.name,
            "faults": {"dropout": a.dropout, "noise": a.noise, "bias": a.bias, "outage": a.outage, "seed": a.fault_seed,
                       "decision_period": a.decision_period, "obstacle_on_path": a.obstacle_on_path, "paired": a.paired},
            "summary": summary, "episodes": episodes}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
