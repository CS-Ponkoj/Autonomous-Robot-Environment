"""Evaluate the rule-based baseline driver headless, on fixed simulated-time decision ticks
(independent of any rendering). The evaluator may use ground truth (path length, planned path);
the driver never does.

    .venv\\Scripts\\python tools\\eval_baseline.py --set dev        # tuning seeds 2000-2019
    .venv\\Scripts\\python tools\\eval_baseline.py --set heldout    # frozen evaluation: seeds 1000-1009
    options: --levels 1 2 5   --out qa_output/baseline_heldout.json   --logs qa_output/baseline_logs

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
from robot_env.system import RobotSystem  # noqa: E402

SCHEMA = 2  # 2: "path_efficiency" renamed "reference_path_ratio" (it can exceed 1)
SEED_SETS = {"dev": tuple(range(2000, 2020)), "heldout": tuple(C.HELDOUT_SEEDS)}


def run_episode(system: RobotSystem, room: RoomMap, seed: int, level: int, log_dir: Path | None = None) -> dict:
    task = room.sample_task(seed)
    driver = BaselineDriver(C.SPEED_LEVELS[level])
    if log_dir is not None:
        system.log = DriveLog(log_dir / f"seed{seed}_level{level + 1}.jsonl", task_seed=seed, note="baseline")
    system.reset(*task.start, task.goal)
    path, prev = 0.0, system.sim.true_pose()[:2]
    outcome, reason = "timeout", "time limit reached"
    while system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome, reason = "success", ""
            break
        system.apply(driver.decide(obs))
        system.advance(C.DECISION_PERIOD)
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
    a = p.parse_args(argv)
    system = RobotSystem()
    room = RoomMap(system.sim.model)
    episodes = []
    for level in a.levels:
        for seed in SEED_SETS[a.set]:
            r = run_episode(system, room, seed, level - 1, a.logs)
            episodes.append(r)
            print(f"level {r['level']} seed {seed}: {r['outcome']:9s} t {r['time_s']:6.1f} s  path {r['path_m']:5.1f} m  "
                  f"ratio {r['reference_path_ratio']}  interventions {r['interventions']}  {r['failure_reason']}", flush=True)
    summary = {}
    for level in a.levels:
        rows = [e for e in episodes if e["level"] == level]
        ok = [e for e in rows if e["outcome"] == "success"]
        summary[f"level{level}"] = {
            "episodes": len(rows), "success": len(ok),
            "collisions": sum(e["outcome"] == "collision" for e in rows),
            "timeouts": sum(e["outcome"] == "timeout" for e in rows),
            "mean_time_s": round(sum(e["time_s"] for e in ok) / len(ok), 2) if ok else None,
            "mean_reference_path_ratio": round(sum(e["reference_path_ratio"] for e in ok) / len(ok), 3) if ok else None,
            "interventions": sum(e["interventions"] for e in rows),
        }
        print(f"level {level}: {summary[f'level{level}']}")
    system.close()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "schema": SCHEMA, "command": "python tools/eval_baseline.py " + " ".join(sys.argv[1:]),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seed_set": a.set, "seeds": list(SEED_SETS[a.set]),
            "config_sha256": config_hash(), "driver": BaselineDriver.name,
            "summary": summary, "episodes": episodes}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
